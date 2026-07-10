# CUDA Graph Memory Comparison

This note records the observed CUDA graph memory overhead between the baseline
standalone speculative decoding path and the suffix + dynamic-K path.

## Test Setup

- Target model: `/models/models/Qwen/Qwen2.5-72B-Instruct-AWQ`
- Draft model: `/models/models/Qwen/Qwen3-0.6B`
- Tensor parallel size: `4`
- CUDA graph max batch size: `16`
- Baseline: standalone speculative decoding, `K=4`
- Dynamic-K: suffix enabled, normal `K=4`, long suffix `K=8`

## Baseline Standalone

Log excerpts:

```text
Capture cuda graph end. mem usage=1.14 GB. avail mem=7.63 GB.
Capture draft cuda graph end. mem usage=0.43 GB. avail mem=5.37 GB.
Capture draft extend cuda graph end. mem usage=0.24 GB. avail mem=5.13 GB.
```

Memory summary per GPU:

| Component | Memory |
| --- | ---: |
| Target verify CUDA graph | 1.14 GB |
| Draft CUDA graph | 0.43 GB |
| Draft extend CUDA graph | 0.24 GB |
| Total CUDA graph memory | 1.81 GB |
| Final available memory | 5.13 GB |

## Suffix + Dynamic-K

Log excerpts:

```text
Capture cuda graph end. mem usage=1.39 GB. avail mem=7.38 GB.
Capture draft cuda graph end. mem usage=0.41 GB. avail mem=5.11 GB.
Capture draft extend cuda graph end. mem usage=0.30 GB. avail mem=4.81 GB.
```

Memory summary per GPU:

| Component | Memory |
| --- | ---: |
| Target verify CUDA graph | 1.39 GB |
| Draft CUDA graph | 0.41 GB |
| Draft extend CUDA graph | 0.30 GB |
| Total CUDA graph memory | 2.10 GB |
| Final available memory | 4.81 GB |

## Delta

| Component | Baseline | Dynamic-K | Delta |
| --- | ---: | ---: | ---: |
| Target verify CUDA graph | 1.14 GB | 1.39 GB | +0.25 GB |
| Draft CUDA graph | 0.43 GB | 0.41 GB | -0.02 GB |
| Draft extend CUDA graph | 0.24 GB | 0.30 GB | +0.06 GB |
| Total CUDA graph memory | 1.81 GB | 2.10 GB | +0.29 GB |
| Final available memory | 5.13 GB | 4.81 GB | -0.32 GB |

## Conclusion

The suffix + dynamic-K path adds about `0.29 GB/GPU` of CUDA graph memory based
on the reported `mem usage` values. The final available memory drops by about
`0.32 GB/GPU`, which is consistent with the CUDA graph delta plus minor allocator
and measurement variance.

Most of the additional memory comes from capturing the extra `K=8` target verify
graph. The draft extend graph also adds a smaller amount because it captures both
the normal `K=4` and long-suffix `K=8` shapes.

## K Sweep Plan

To measure how CUDA graph memory grows as long-suffix `K` increases, keep all
launch parameters fixed and only change `--speculative-long-suffix-draft-token-num`.

Recommended sweep:

| Run | Normal K | Long-Suffix K | Notes |
| --- | ---: | ---: | --- |
| baseline | 4 | N/A | Standalone only, suffix and dynamic-K disabled |
| dynamic-k-6 | 4 | 6 | Captures K=4 and K=6 graph buckets |
| dynamic-k-8 | 4 | 8 | Current tested configuration |
| dynamic-k-10 | 4 | 10 | Captures K=4 and K=10 graph buckets |
| dynamic-k-12 | 4 | 12 | Captures K=4 and K=12 graph buckets |

Keep these parameters identical across all runs:

```text
--model-path /models/models/Qwen/Qwen2.5-72B-Instruct-AWQ
--speculative-draft-model-path /models/models/Qwen/Qwen3-0.6B
--speculative-algorithm STANDALONE
--speculative-num-steps 3
--speculative-num-draft-tokens 4
--speculative-eagle-topk 1
--speculative-normal-draft-token-num 4
--cuda-graph-max-bs 16
--tp-size 4
```

For each dynamic-K run, change only this value:

```text
--speculative-long-suffix-draft-token-num <K>
```

Suggested command template:

```bash
K=8
LOG=logs/cuda_graph_dynamic_k_${K}.log
mkdir -p logs

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m sglang.launch_server \
  --model-path /models/models/Qwen/Qwen2.5-72B-Instruct-AWQ \
  --speculative-draft-model-path /models/models/Qwen/Qwen3-0.6B \
  --speculative-algorithm STANDALONE \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 \
  --speculative-suffix-enable \
  --speculative-dynamic-k-enable \
  --speculative-normal-draft-token-num 4 \
  --speculative-long-suffix-draft-token-num ${K} \
  --speculative-long-suffix-min-match-len 7 \
  --speculative-long-suffix-max-bs 8 \
  --speculative-high-bs-threshold 10 \
  --cuda-graph-max-bs 16 \
  --tp-size 4 \
  --host 0.0.0.0 \
  --port 30000 2>&1 | tee "${LOG}"
```

After the server reaches `Application startup complete`, stop it and run the next
`K`. Parse the logs with:

```bash
python scripts/parse_cuda_graph_memory.py \
  logs/cuda_graph_standalone_k4.log \
  logs/cuda_graph_dynamic_k_6.log \
  logs/cuda_graph_dynamic_k_8.log \
  logs/cuda_graph_dynamic_k_10.log \
  logs/cuda_graph_dynamic_k_12.log
```

Result table:

| Run | Target Verify | Draft Decode | Draft Extend | Total Graph | Final Available | Delta vs Baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline standalone K=4 | 1.14 GB | 0.43 GB | 0.24 GB | 1.81 GB | 5.13 GB | 0.00 GB |
| dynamic-k long K=6 | 1.34 GB | 0.41 GB | 0.31 GB | 2.06 GB | 4.84 GB | +0.25 GB |
| dynamic-k long K=8 | 1.39 GB | 0.41 GB | 0.30 GB | 2.10 GB | 4.81 GB | +0.29 GB |
| dynamic-k long K=10 | 1.45 GB | 0.43 GB | 0.33 GB | 2.21 GB | 4.73 GB | +0.40 GB |
| dynamic-k long K=12 | 1.49 GB | 0.43 GB | 0.33 GB | 2.25 GB | 4.69 GB | +0.44 GB |

Measured deltas by component:

| Run | Target Verify Delta | Draft Decode Delta | Draft Extend Delta | Total Graph Delta | Final Available Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| dynamic-k long K=6 | +0.20 GB | -0.02 GB | +0.07 GB | +0.25 GB | -0.29 GB |
| dynamic-k long K=8 | +0.25 GB | -0.02 GB | +0.06 GB | +0.29 GB | -0.32 GB |
| dynamic-k long K=10 | +0.31 GB | +0.00 GB | +0.09 GB | +0.40 GB | -0.40 GB |
| dynamic-k long K=12 | +0.35 GB | +0.00 GB | +0.09 GB | +0.44 GB | -0.44 GB |

Observed trend:

- Increasing long-suffix K from 6 to 12 raises total CUDA graph memory from
  `2.06 GB/GPU` to `2.25 GB/GPU`.
- Compared with standalone K=4, the measured extra graph memory is about
  `+0.25 GB/GPU` at K=6 and `+0.44 GB/GPU` at K=12.
- Most of the growth comes from the target verify CUDA graph. Draft decode is
  effectively unchanged, and draft extend grows only slightly.

## Automated Sweep

Use this script to launch each case, wait until the graph memory summary line is
printed, stop it, and parse all logs:

```bash
bash scripts/run_cuda_graph_memory_sweep.sh
```

Defaults:

```text
NORMAL_K=4
LONG_K_LIST="6 8 10 12"
CUDA_GRAPH_MAX_BS=16
TP_SIZE=4
PORT=30000
LOG_DIR=logs
```

The script stops a run after seeing either `max_total_num_tokens=...available_mem=...`
or the normal server startup message. This is intentional: CUDA graph memory is
already measurable once the `max_total_num_tokens` line is printed, so a later API
server startup hang does not block the memory sweep.

Override values with environment variables:

```bash
LONG_K_LIST="5 6 8 10 12" PORT=30001 bash scripts/run_cuda_graph_memory_sweep.sh
```
