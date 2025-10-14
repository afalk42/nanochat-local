#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Detect how many processes to launch with torchrun. Default to 1 when CUDA is unavailable.
NPROC=$(python - <<'PY'
import torch
def detect_nproc():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        return count if count > 0 else 1
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 1
    return 1
print(detect_nproc())
PY
)
echo "Using torchrun with $NPROC process(es)."
TORCHRUN="torchrun --standalone --nproc_per_node=$NPROC"
run_module() {
    local module="$1"
    shift
    if [ "$NPROC" -eq 1 ]; then
        python -m "$module" "$@"
    else
        if [ "$#" -gt 0 ]; then
            $TORCHRUN -m "$module" -- "$@"
        else
            $TORCHRUN -m "$module"
        fi
    fi
}

ACCELERATOR=$(python - <<'PY'
import torch
if torch.cuda.is_available():
    print("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)
echo "Detected accelerator backend: $ACCELERATOR"

if [ "$ACCELERATOR" = "cuda" ]; then
    TOKENIZER_SHARDS=8
    PRETRAIN_SHARDS=240
    TOK_MAX_CHARS=2000000000
    BASE_TRAIN_ARGS=(--depth=20 --device_batch_size=32 --total_batch_size=524288)
    BASE_LOSS_ARGS=()
    RUN_CORE_EVAL=1
    MID_TRAIN_ARGS=(--device_batch_size=32 --total_batch_size=524288)
    MID_CHAT_EVAL_ARGS=(-i mid)
    SFT_ARGS=(--device_batch_size=4)
    SFT_CHAT_EVAL_ARGS=(-i sft)
else
    TOKENIZER_SHARDS=1
    PRETRAIN_SHARDS=8
    TOK_MAX_CHARS=200000000
    BASE_TRAIN_ARGS=(--depth=8 --device_batch_size=2 --total_batch_size=8192 --max_seq_len=512 --num_iterations=200 --eval_every=50 --eval_tokens=16384 --core_metric_every=1000000 --sample_every=1000000)
    BASE_LOSS_ARGS=(--device_batch_size=2 --split_tokens=8192)
    RUN_CORE_EVAL=0
    MID_TRAIN_ARGS=(--device_batch_size=2 --total_batch_size=4096 --eval_every=50 --eval_tokens=16384)
    MID_CHAT_EVAL_ARGS=(-i mid -a GSM8K)
    SFT_ARGS=(--device_batch_size=2 --target_examples_per_step=8 --eval_every=20 --eval_metrics_every=80)
    SFT_CHAT_EVAL_ARGS=(-i sft -a GSM8K)
fi

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download a seed set of shards for tokenizer training (~250M chars per shard, ~100MB compressed).
python -m nanochat.dataset -n $TOKENIZER_SHARDS
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why PRETRAIN_SHARDS is selected
python -m nanochat.dataset -n $PRETRAIN_SHARDS &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on a configurable number of characters
python -m scripts.tok_train --max_chars=$TOK_MAX_CHARS
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Download the eval_bundle from s3 to evaluate CORE metric during training (~162MB)
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# The H100 profile targets a 561M parameter model and therefore downloads 240 shards (~24GB).
# Local profiles dramatically reduce PRETRAIN_SHARDS to keep disk usage and runtime reasonable.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# pretrain the d20 (or downsized) model
run_module scripts.base_train "${BASE_TRAIN_ARGS[@]}" --run=$WANDB_RUN
# evaluate the model on a larger chunk of train/val data and draw some samples
if [ ${#BASE_LOSS_ARGS[@]} -gt 0 ]; then
    run_module scripts.base_loss "${BASE_LOSS_ARGS[@]}"
else
    run_module scripts.base_loss
fi
# evaluate the model on CORE tasks (skip on local profiles)
if [ "$RUN_CORE_EVAL" -eq 1 ]; then
    run_module scripts.base_eval
fi

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# run midtraining and eval the model
run_module scripts.mid_train "${MID_TRAIN_ARGS[@]}" --run=$WANDB_RUN
run_module scripts.chat_eval "${MID_CHAT_EVAL_ARGS[@]}"

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
run_module scripts.chat_sft "${SFT_ARGS[@]}" --run=$WANDB_RUN
run_module scripts.chat_eval "${SFT_CHAT_EVAL_ARGS[@]}"

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# run_module scripts.chat_rl --run=$WANDB_RUN
# eval the RL model only on GSM8K
# run_module scripts.chat_eval -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
