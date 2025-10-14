# nanochat training report

Generated: 2025-10-14 17:43:48

## Environment

### Git Information
- Branch: master
- Commit: dd6ff9a (dirty)
- Message: fix bug in fallback case of find_largest_model

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
- Characters: 342,054
- Lines: 8,369
- Files: 43
- Tokens (approx): 85,513
- Dependencies (uv.lock lines): 2,014

Run started: 2025-10-14 17:43:48

---

## Tokenizer training
timestamp: 2025-10-14 17:44:40

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 65,536
- train_time: 29.3434
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 32
- token_bytes_mean: 6.9197
- token_bytes_std: 2.8748


## Tokenizer evaluation
timestamp: 2025-10-14 17:44:46

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 375 | 4.85 | +7.2% |
| korean | 893 | 745 | 1.20 | 712 | 1.25 | +4.4% |
| code | 1259 | 576 | 2.19 | 492 | 2.56 | +14.6% |
| math | 1834 | 936 | 1.96 | 966 | 1.90 | -3.2% |
| science | 1112 | 260 | 4.28 | 228 | 4.88 | +12.3% |
| fwe-train | 4208518 | 900364 | 4.67 | 856883 | 4.91 | +4.8% |
| fwe-val | 5459959 | 1184822 | 4.61 | 1134403 | 4.81 | +4.3% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 375 | 4.85 | +3.1% |
| korean | 893 | 364 | 2.45 | 712 | 1.25 | -95.6% |
| code | 1259 | 309 | 4.07 | 492 | 2.56 | -59.2% |
| math | 1834 | 832 | 2.20 | 966 | 1.90 | -16.1% |
| science | 1112 | 249 | 4.47 | 228 | 4.88 | +8.4% |
| fwe-train | 4208518 | 874799 | 4.81 | 856883 | 4.91 | +2.0% |
| fwe-val | 5459959 | 1155368 | 4.73 | 1134403 | 4.81 | +1.8% |


## Summary

- Characters: 342,054
- Lines: 8,369
- Files: 43
- Tokens (approx): 85,513
- Dependencies (uv.lock lines): 2,014

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|

Total wall clock time: 0h0m
