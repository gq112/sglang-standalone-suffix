# Standalone + Suffix Dynamic-K Speculative Decoding

## Background

This repository adds a hybrid speculative decoding path for SGLang that combines:

- **Standalone drafting**: uses the standalone draft model as the default drafter.
- **Suffix drafting**: uses ArcticInference-style suffix cache hits to replace standalone draft tokens when the suffix proposal is stronger.
- **Dynamic-K verification**: uses different target verification widths per request group, while keeping the draft layout chain-based rather than tree-based.

The goal is to keep standalone+suffix useful under realistic online concurrency. Low concurrency can still benefit from long suffix hits, while higher concurrency avoids spending too much target verification time on long verify windows.

## Core Strategy

`K` is the target verification width and includes the root/current token.

- `K=4` means `[root + 3 draft tokens]`.
- `K=8` means `[root + 7 draft tokens]`.

The runtime policy is:

- **Normal group, K=4**
  - standalone requests
  - suffix-short requests
  - high-concurrency requests
- **Long suffix group, K=8**
  - low-concurrency requests
  - suffix proposal has enough match depth
  - suffix proposal can provide enough tokens for the long verify window

For one request in one decode round, only one drafter path is used:

- If it enters the long suffix group, suffix overrides standalone.
- Otherwise it uses the normal K=4 path, where suffix-short may still override standalone within the fixed K=4 width.

This keeps the implementation chain-based and avoids tree verification complexity.

## Runtime Flow

1. Run the standalone draft model to produce the default draft tokens.
2. Query the suffix proposer for each request.
3. Classify requests:
   - if `batch_size >= speculative_high_bs_threshold`, all requests use K=4.
   - otherwise, requests with suffix `match_len >= speculative_long_suffix_min_match_len` and enough suffix tokens enter the K=8 group.
   - all remaining requests enter the K=4 group.
4. Build verify inputs:
   - K=4 group uses standalone draft tokens, with suffix-short row overrides when applicable.
   - K=8 group uses a linear suffix chain.
5. Verify the sub-batches serially:
   - K=4 sub-batch uses the K=4 target verify path.
   - K=8 sub-batch uses the K=8 target verify path.
6. Merge verify results back in original request order:
   - `req.output_ids`
   - `seq_lens`
   - `seq_lens_cpu`
   - `out_cache_loc`
   - `verified_id`
   - `accept_length`
   - next-round `EagleDraftInput`

## CUDA Graph Handling

The target verify CUDA graph is extended from a single batch-size key to a `(K, bs_bucket)` key.

For dynamic-K standalone+suffix, the graph runner captures and replays at least:

- K=4 graphs for normal requests.
- K=8 graphs for low-batch long suffix requests.

The graph shape is fixed per `(K, bs_bucket)`, but token values remain dynamic. This means suffix token contents can change every round while still reusing the same CUDA graph, as long as the verify width and padded batch bucket match.

The implementation allocates graph buffers according to the maximum configured K, then slices by the active K during capture and replay.

## Configuration

New flags:

```bash
--speculative-dynamic-k-enable
--speculative-normal-draft-token-num 4
--speculative-long-suffix-draft-token-num 8
--speculative-long-suffix-min-match-len 7
--speculative-high-bs-threshold 20
```

Recommended starting point:

```bash
--speculative-algorithm STANDALONE \
--speculative-suffix-enable \
--speculative-dynamic-k-enable \
--speculative-normal-draft-token-num 4 \
--speculative-long-suffix-draft-token-num 8 \
--speculative-long-suffix-min-match-len 7 \
--speculative-high-bs-threshold 20
```

Interpretation:

- standalone drafts 3 tokens by default (`K=4`).
- long suffix hits can verify 7 suffix tokens (`K=8`).
- with the current benchmark setting `--speculative-high-bs-threshold 20`, an
  active decode batch below 20 can use K=8; batches at or above 20 use K=4.

## Benchmark Record (2026-07-15)

### Scope and method

The operational comparison focuses on concurrent request counts **10, 20, and
24**. Batch 30 is only a saturation reference and is not a dynamic-K decision
point. All configurations use Qwen2.5-72B-Instruct-AWQ, Qwen3-0.6B draft
model, TP=4, FA3, `max_running_requests=32`, `mem_fraction_static=0.72`, and
greedy sampling (`temperature=0`).

| Name | Standalone draft | Suffix cache | Dynamic K |
| --- | --- | --- | --- |
| No speculation | No | No | No |
| Standalone K=4 | Yes | No | No |
| Suffix static K=4 | Yes | Yes | No |
| Dynamic split K=4/4 | Yes | Yes | Yes, with long width fixed to 4 |
| Dynamic K=4/8 | Yes | Yes | Yes |

### End-to-end output throughput

The one-shot warm-cache experiment recorded the following total output
throughput (token/s). Each row was run independently and uses the same fixed
output-length workload within that concurrency.

| Concurrent requests | No speculation | Standalone K=4 | Suffix static K=4 | Dynamic K=4/8 |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 257.72 | **367.09** | 354.16 | 343.20 |
| 20 | 298.42 | **384.79** | 373.30 | 374.71 |
| 24 | 309.56 | **382.94** | 378.10 | 369.72 |

At these loads, standalone K=4 is the current end-to-end throughput baseline.
Dynamic K is faster than no speculation, but does not exceed standalone K=4.
Against suffix static K=4, dynamic K is `-3.09%`, `+0.38%`, and `-2.22%` at
10, 20, and 24 concurrency respectively. The 20-concurrency difference is
within normal benchmark variation; dynamic K has no demonstrated net
throughput win in the 10--24 operating range.

The corresponding dynamic-versus-static latency comparison is:

| Concurrent requests | Dynamic TTFT (ms) | Static TTFT (ms) | Dynamic TPOT (ms) | Static TPOT (ms) | Dynamic ITL (ms) | Static ITL (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2,209.56 | 2,087.91 | 25.43 | 24.44 | 140.08 | 91.09 |
| 20 | 1,872.03 | 2,085.07 | 52.50 | 53.20 | 180.11 | 172.74 |
| 24 | 3,585.73 | 3,157.25 | 65.32 | 65.47 | 212.62 | 198.85 |

### Dynamic-K instrumentation run

The 2026-07-15 warm-cache run added suffix/K=8 counters. The K=8 probe used
120 prompts at concurrency 8 after an identical concurrency-8 warmup; it is a
mechanism check rather than a 10--24 end-to-end result.

| Phase | Suffix proposals | K=4 suffix overrides | K=8 request rounds | K=8 committed tokens | K=8 verify tokens | K=8 efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warmup | 17,428 | 3,462 | 5,186 | 35,905 | 41,280 | 0.870 |
| K=8 probe | 15,472 | 3,319 | 6,382 | 45,064 | 50,720 | 0.888 |

`K=8 efficiency = committed tokens / K=8 verify tokens`. During the probe,
K=8 commits about `45,064 / 6,382 = 7.06` tokens per K=8 request round. This
confirms that the repeated-data workload produces real, high-quality long
suffix hits; poor K=8 acceptance is not the reason for the end-to-end result.

The per-concurrency counters are:

| Concurrent requests | K=8 request rounds | K=8 committed tokens | K=8 verify tokens | K=8 efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 3,526 | 24,357 | 27,352 | 0.891 |
| 20 | 813 | 5,857 | 6,488 | 0.903 |
| 24 | 1,234 | 7,938 | 8,888 | 0.893 |

K=8 appears at external concurrency 20 and 24 because the policy checks the
*current active decode batch*. Once requests finish and the tail batch drops
below 20, qualifying suffix requests can enter K=8.

### Decision criteria for 10--24 concurrency

At concurrency 10, K=8 can participate throughout the active decode batch. At
concurrency 20 and 24, the policy falls back to K=4 while the full batch is at
or above the threshold; K=8 may only appear during the request tail as the
active batch shrinks. Record, for each concurrency separately:

1. `dynamic_k8_request_total`: whether K=8 actually participated.
2. `dynamic_k8_output_token_total / dynamic_k8_draft_token_total`: K=8
   verification efficiency.
3. Output throughput, mean TPOT, and mean ITL for dynamic K versus suffix
   static K=4.

Dynamic K is a net win only if its K=8 hit quality remains high **and** the
dynamic-K result exceeds suffix static K=4 in the same 10--24 concurrency row.
The experiment script writes this comparison automatically as
`throughput_comparison.tsv` and `throughput_comparison.md` in its results
directory, parsing the terminal summary emitted by `test_req.py` after each
concurrency run.

### Verifying split/merge overhead with low-interference counters

The deployment experiment also runs a `dynamic_k4_k4` ablation. It keeps the
dynamic request classification and suffix-long path, but configures both paths
as K=4. This isolates the cost of request splitting, serial target verifies,
and result merging from the benefit of the K=8 width.

| Comparison | What it isolates |
| --- | --- |
| `dynamic_k4_k4` vs `suffix_static_k4` | K=4/K=4 sub-batch split, serial target calls, and merge overhead |
| `dynamic_k4_k8` vs `dynamic_k4_k4` | Incremental value/cost of widening the long suffix path from K=4 to K=8 |
| `dynamic_k4_k8` vs `suffix_static_k4` | Overall production impact |

The following counters are accumulated with integer additions only; they do
not add CUDA events, GPU synchronization, or per-token logging:

- `dynamic_k_verify_batch_total`: parent batches using a dynamic long-suffix
  verify path.
- `dynamic_k_mixed_verify_batch_total`: parent batches split into both K=4 and
  K=8 verifies. Every such batch submits two serial target forwards.
- `dynamic_k_normal_verify_call_total` and
  `dynamic_k_long_verify_call_total`: actual normal-path and long-suffix
  target-verify calls. In the K=4/4 ablation, the latter is also K=4.

For a mixed batch the expected relation is:

```text
normal_verify_calls + long_verify_calls
  = dynamic_verify_batches + mixed_verify_batches
```

Thus `mixed_verify_batches / dynamic_verify_batches` is the direct measure of
how often the dynamic implementation pays for an additional serial target
forward. Combine it with the end-to-end A/B/C throughput comparison; no GPU
timing in the serving hot path is needed for this decision.

## Correctness Boundaries

The first implementation intentionally keeps the dynamic-K path conservative:

- only supports `topk=1`.
- only enables sub-batch dynamic-K for greedy sampling.
- disables dynamic-K split when grammar or return-logprob is active.
- disables overlap schedule for this mode.
- each request enters only one sub-batch per decode round.
- sub-batches are merged back by original request index.

These constraints avoid:

- random sampling order changes.
- logprob alignment errors.
- grammar state mismatch.
- KV cache double-free.
- next-round draft input misalignment.

## Difference From Previous Fixed-K Fusion

Previous standalone+suffix fusion used a single global verify width:

```text
whole batch: K=4 or K=8
```

That caused one of two problems:

- if global K=4, long suffix hits were truncated.
- if global K=8, standalone/default requests paid extra target verify cost.

Dynamic-K changes this to:

```text
suffix-long requests -> K=8 graph
standalone/default requests -> K=4 graph
```

This is the key improvement: long suffix hits keep their value without forcing the whole batch into the larger verify window.

## Implementation Files

Main files changed:

- `python/sglang/srt/speculative/eagle_worker.py`
  - request-level dynamic-K classification
  - K=4/K=8 verify input construction
  - sub-batch verify and result merge
- `python/sglang/srt/model_executor/cuda_graph_runner.py`
  - CUDA graph keyed by `(K, bs_bucket)`
  - active K-aware capture and replay
- `python/sglang/srt/server_args.py`
  - dynamic-K flags and validation

## Validation

Static checks performed:

```bash
python3 -m py_compile \
  python/sglang/srt/speculative/eagle_worker.py \
  python/sglang/srt/model_executor/cuda_graph_runner.py \
  python/sglang/srt/server_args.py

git diff --check -- \
  python/sglang/srt/speculative/eagle_worker.py \
  python/sglang/srt/model_executor/cuda_graph_runner.py \
  python/sglang/srt/server_args.py
```

Both checks pass.

The local unit test entrypoint `test/srt/test_speculative_registry.py` currently requires missing environment dependencies such as `numpy`, so it was not completed in this environment.
