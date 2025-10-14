# Repository Guidelines

## Project Structure & Module Organization
- `nanochat/` holds core model code such as `gpt.py`, `engine.py`, and shared utilities in `common.py`.
- Pipeline entry points belong in `scripts/` (`base_train.py`, `chat_web.py`, `speedrun.sh`), mirroring the staged workflow.
- `rustbpe/` houses the PyO3 tokenizer; run-time artifacts land under `~/.cache/nanochat` unless `NANOCHAT_BASE_DIR` is set.
- Tests live in `tests/`; `dev/` stores UI assets and exploratory notebooks.

## Build, Test, and Development Commands
- Prepare the environment with `uv venv` (first run) and `uv sync`; `source .venv/bin/activate` activates the environment for manual commands.
- Rebuild the tokenizer bridge using `uv run maturin develop --release --manifest-path rustbpe/Cargo.toml`.
- Kick off the full training loop on multi-GPU nodes with `bash speedrun.sh`; adjust model depth or batch size within the relevant script instead of editing the shell wrapper.
- Launch quick iterations via `uv run python -m scripts.chat_web` for the UI or `uv run python -m scripts.chat_cli -p "prompt"` for CLI checks.
- Run fast validations with `uv run pytest tests -m "not slow" -v`; add `cargo test -p rustbpe` when modifying Rust.
- Apple Silicon falls back to Metal (MPS) automatically; `speedrun.sh` drops to a single `torchrun` process and the model defaults to FP16 activations—shrink depth/batch sizes accordingly.

## Coding Style & Naming Conventions
- Follow PEP 8, 4-space indents, and `snake_case` identifiers; keep dataclasses in configs and give defaults that match `speedrun.sh`.
- Keep modules import-light, colocate new helpers in `nanochat/common.py`, and use explicit device/dtype handling as in `nanochat/gpt.py`.
- For Rust, run `cargo fmt`/`cargo clippy` before committing and surface only snake_case functions through PyO3.

## Testing Guidelines
- Name new tests `test_<feature>.py` and add `@pytest.mark.slow` to long runners; prefer deterministic fixtures that stub `torch.distributed`.
- Track tokenizer regressions with vocab fixtures or compression metrics and note expected deltas in the PR.
- When changes affect training stages, include the exact command (e.g., `torchrun --standalone ... scripts.mid_train`) and summarize loss or CORE shifts.

## Commit & Pull Request Guidelines
- Write present-tense, <72 character commit titles (e.g., `tighten kv cache fallback`); reference issues with `Fixes #123` when relevant.
- PRs should outline scope, testing evidence, and any artifacts or WANDB runs; attach UI screenshots for `chat_web` changes.
- Confirm `pytest` and `cargo test` pass before requesting review and avoid mixing formatting-only commits with feature work.

## Security & Configuration Tips
- Store WANDB tokens and dataset credentials in env vars, never in source.
- Clean only your own caches under `~/.cache/nanochat` and confirm `print0` logs show aligned `torchrun` ranks across concurrent jobs.
