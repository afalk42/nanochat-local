# nanochat training report

Generated: 2025-10-14 18:24:08

## Environment

### Git Information
- Branch: master
- Commit: 699d324 (dirty)
- Message: initial port to local (macOS) version

### Hardware
- Platform: Darwin
- CPUs: 16 cores (16 logical)
- Memory: 128.0 GB
- Accelerators: 1x Apple M-series (MPS)
- Accelerator Backend: MPS
- Accelerator Memory: 128.0 GB total
- Hourly Rate: $2.00/hour

### Software
- Python: 3.10.16
- PyTorch: 2.8.0


### Bloat
- Characters: 346,242
- Lines: 8,520
- Files: 44
- Tokens (approx): 86,560
- Dependencies (uv.lock lines): 2,014

Run started: 2025-10-14 18:24:08

---

## Tokenizer training
timestamp: 2025-10-14 18:24:17

- max_chars: 200,000,000
- doc_cap: 10,000
- vocab_size: 65,536
- train_time: 5.1929
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 32
- token_bytes_mean: 6.9076
- token_bytes_std: 2.8717


## Tokenizer evaluation
timestamp: 2025-10-14 18:24:19

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 382 | 4.76 | +5.4% |
| korean | 893 | 745 | 1.20 | 672 | 1.33 | +9.8% |
| code | 1259 | 576 | 2.19 | 492 | 2.56 | +14.6% |
| math | 1834 | 936 | 1.96 | 976 | 1.88 | -4.3% |
| science | 1112 | 260 | 4.28 | 229 | 4.86 | +11.9% |
| fwe-train | 4208518 | 900364 | 4.67 | 855629 | 4.92 | +5.0% |
| fwe-val | 4908443 | 1059062 | 4.63 | 1011926 | 4.85 | +4.5% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 382 | 4.76 | +1.3% |
| korean | 893 | 364 | 2.45 | 672 | 1.33 | -84.6% |
| code | 1259 | 309 | 4.07 | 492 | 2.56 | -59.2% |
| math | 1834 | 832 | 2.20 | 976 | 1.88 | -17.3% |
| science | 1112 | 249 | 4.47 | 229 | 4.86 | +8.0% |
| fwe-train | 4208518 | 874799 | 4.81 | 855629 | 4.92 | +2.2% |
| fwe-val | 4908443 | 1029691 | 4.77 | 1011926 | 4.85 | +1.7% |


## Summary

- Characters: 346,242
- Lines: 8,520
- Files: 44
- Tokens (approx): 86,560
- Dependencies (uv.lock lines): 2,014

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|

Total wall clock time: 0h0m
