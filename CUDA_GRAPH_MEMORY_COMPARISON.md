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

The measurements below were repeated after fixing the FlashInfer CUDA graph
metadata key so that K=4 and long-K plans for the same batch size no longer
overwrite each other. Earlier dynamic-K measurements are invalid because they
did not retain both graph-compatible FlashInfer plans.

Log excerpts:

```text
Capture cuda graph end. mem usage=2.26 GB. avail mem=6.51 GB.
Capture draft cuda graph end. mem usage=0.41 GB. avail mem=4.24 GB.
Capture draft extend cuda graph end. mem usage=0.42 GB. avail mem=3.82 GB.
```

Memory summary per GPU:

| Component | Memory |
| --- | ---: |
| Target verify CUDA graph | 2.26 GB |
| Draft CUDA graph | 0.41 GB |
| Draft extend CUDA graph | 0.42 GB |
| Total CUDA graph memory | 3.09 GB |
| Final available memory | 3.82 GB |

## Delta

| Component | Baseline | Dynamic-K | Delta |
| --- | ---: | ---: | ---: |
| Target verify CUDA graph | 1.14 GB | 2.26 GB | +1.12 GB |
| Draft CUDA graph | 0.43 GB | 0.41 GB | -0.02 GB |
| Draft extend CUDA graph | 0.24 GB | 0.42 GB | +0.18 GB |
| Total CUDA graph memory | 1.81 GB | 3.09 GB | +1.28 GB |
| Final available memory | 5.13 GB | 3.82 GB | -1.31 GB |

## Conclusion

With correct FlashInfer plan retention, the suffix + dynamic-K K=8 path adds
about `1.28 GB/GPU` of CUDA graph memory. The final available memory drops by
about `1.31 GB/GPU`, which is consistent with the graph-memory delta plus minor
allocator and measurement variance.

Most of the additional memory comes from retaining separate K=4 and K=8
FlashInfer target-verify plans. The draft-extend graph also retains both normal
and long-suffix shapes. This is required for safe CUDA graph replay: a K=4 graph
cannot reuse the K=8 FlashInfer wrapper for the same batch size.

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
| dynamic-k long K=6 | 2.21 GB | 0.41 GB | 0.43 GB | 3.05 GB | 3.85 GB | +1.24 GB |
| dynamic-k long K=8 | 2.26 GB | 0.41 GB | 0.42 GB | 3.09 GB | 3.82 GB | +1.28 GB |
| dynamic-k long K=10 | 2.32 GB | 0.43 GB | 0.45 GB | 3.20 GB | 3.74 GB | +1.39 GB |
| dynamic-k long K=12 | 2.36 GB | 0.43 GB | 0.45 GB | 3.24 GB | 3.70 GB | +1.43 GB |

Measured deltas by component:

| Run | Target Verify Delta | Draft Decode Delta | Draft Extend Delta | Total Graph Delta | Final Available Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| dynamic-k long K=6 | +1.07 GB | -0.02 GB | +0.19 GB | +1.24 GB | -1.28 GB |
| dynamic-k long K=8 | +1.12 GB | -0.02 GB | +0.18 GB | +1.28 GB | -1.31 GB |
| dynamic-k long K=10 | +1.18 GB | +0.00 GB | +0.21 GB | +1.39 GB | -1.39 GB |
| dynamic-k long K=12 | +1.22 GB | +0.00 GB | +0.21 GB | +1.43 GB | -1.43 GB |

Observed trend:

- The fixed cost of retaining both K=4 and long-K FlashInfer plans is about
  `+1.24 GB/GPU` at K=6 relative to standalone K=4.
- Increasing long-suffix K from 6 to 12 raises total CUDA graph memory from
  `3.05 GB/GPU` to `3.24 GB/GPU`; K=8 adds only `0.04 GB/GPU` over K=6.
- Most of the growth comes from the target verify CUDA graph. Draft decode is
  effectively unchanged, and draft extend adds about `0.18-0.21 GB/GPU`.

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

## FlashInfer vs FA3 Comparison

After changing FA3 to retain separate K=4 and K=8 CUDA-graph metadata, use the
following comparison script to remeasure both attention backends with identical
model, TP, concurrency, and sampling settings:

```bash
bash scripts/run_attention_backend_memory_comparison.sh
```

It runs these four cases and writes one startup log per case under
`logs/attention_backend_memory/`:

| Case | Attention backend | Verify widths |
| --- | --- | --- |
| `flashinfer_static_k4` | FlashInfer | K=4 |
| `flashinfer_dynamic_k4_8` | FlashInfer | K=4 and K=8 |
| `fa3_static_k4` | FlashAttention 3 | K=4 |
| `fa3_dynamic_k4_8` | FlashAttention 3 | K=4 and K=8 |

All cases set `--sampling-backend pytorch` so FlashInfer sampling kernels do not
affect the attention-backend comparison. Override paths and test dimensions as
needed:

```bash
MODEL_PATH=/models/target \
DRAFT_MODEL_PATH=/models/draft \
TP_SIZE=4 CUDA_GRAPH_MAX_BS=16 NORMAL_K=4 LONG_K=8 \
bash scripts/run_attention_backend_memory_comparison.sh
```
