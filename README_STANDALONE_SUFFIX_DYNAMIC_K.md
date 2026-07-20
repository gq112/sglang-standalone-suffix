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
   - a homogeneous batch uses the fixed K=4 or K=8 input.
   - a supported mixed FA3 batch uses one flattened ragged input: K=4
     rows use the standalone/suffix-short chain and K=8 rows use the linear
     suffix chain.
5. Target verify:
   - fixed-width inputs retain their normal CUDA-graph path.
   - a ragged mixed input issues **one** FA3 target forward; it does not split
     the batch or merge two verify results.
   - when at least 75% of rows are K=8, the mixed input is padded per row to
     K=8 and replays the existing fixed-K=8 CUDA Graph. The synthetic K=4
     tails are never accepted and their KV slots are released immediately.
6. Greedy acceptance maps the flattened accepted indices back to each request,
   then advances `seq_lens`, KV ownership, `verified_id`, and the next-round
   `EagleDraftInput`.

## CUDA Graph Handling

Fixed-width target verify CUDA graphs are keyed by `(K, bs_bucket)`.

For dynamic-K standalone+suffix, the graph runner captures and replays at least:

- K=4 graphs for normal requests.
- K=8 graphs for homogeneous low-batch long suffix requests.

The graph shape is fixed per `(K, bs_bucket)`, but token values remain dynamic.
This means suffix token contents can change every round while still reusing the
same CUDA graph, as long as the verify width and padded batch bucket match.

Mixed ragged batches use eager FA3 by default. A bounded graph path reuses the
existing K=8 graph when the K=8 row ratio is at least 75%; it pads only the
unobserved causal suffix of K=4 rows and frees those cache slots after verify.
Set `SGLANG_RAGGED_CUDA_GRAPH_MIN_LONG_RATIO` to tune this
coverage/per-row-padding trade-off. The production-safe default is now `1.0`
(pure eager Ragged); lower values are experimental padding-graph policies.
The graph/eager coverage counters below are the rollout gate.

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

### Latest FA3 ragged result (2026-07-16)

The current implementation was re-measured with the one-command deployment
benchmark on the repeated-data workload, FA3, TP=4, a K=4 normal width, a K=8
long-suffix width, and `--speculative-high-bs-threshold 20`. The primary
metric is total end-to-end output throughput. Positive deltas favor ragged
K=4/8 over suffix static K=4.

| Concurrent requests | Suffix static K=4 | Dynamic K=4/4 control | FA3 ragged K=4/8 | Ragged vs static | Ragged vs K=4/4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 308.60 | 272.62 | **322.98** | **+4.66%** | +18.47% |
| 20 | 438.78 | 433.32 | **442.07** | **+0.75%** | +2.02% |
| 24 | 374.35 | 372.45 | **383.15** | **+2.35%** | +2.87% |

The dynamic K=4/4 control is slower than static K=4 (`-11.66%`, `-1.24%`,
and `-0.51%`), which measures the request-level classification and ragged
eager overhead. Widening suffix-long rows to K=8 more than repays that cost
in this workload, producing a net throughput gain at every measured
concurrency.

The K=8 probe confirmed 6,243 K=8 request rounds, 44,149 committed output
tokens, and 49,944 K=8 target-verify tokens, for `0.884` committed-token
efficiency. `dynamic_k_mixed_verify_batch_total` remained zero and
`dynamic_k_long_verify_call_total == dynamic_k_verify_batch_total == 2,055`:
mixed K=4/K=8 work used one ragged target forward rather than the retired
two-forward split.

Latency remains the trade-off: ragged K=4/8 raised mean TTFT/ITL at 10, 20,
and 24 concurrency despite increasing aggregate output throughput. Therefore
the current policy is throughput-oriented. The next optimization target is
ragged CUDA Graph replay; track
`ragged_verify_cuda_graph_batch_total / (ragged_verify_cuda_graph_batch_total
+ ragged_verify_eager_batch_total)` after it is enabled.

### Accuracy validation (2026-07-16)

The official labeled GSM8K test split is stored at `datasets/gsm8k_test.jsonl`.
On the first 200 questions with TP=4, greedy decoding, and active K=8 coverage,
suffix static K=4 and FA3 ragged K=4/8 both achieved `0.945` accuracy
(189/200). The dynamic run recorded 1,130 K=8 request rounds and `0.891`
K=8 committed-token efficiency, so this is an exercised K=8 correctness
result rather than a K=4 fallback result.

### Threshold tuning update (2026-07-17)

The threshold controls whether the *active decode batch* can use K=8. Raising
it from 20 to 24 allowed K=8 to remain active for the external-concurrency-20
workload rather than only its tail. On the same repeated-data benchmark,
`HIGH_BS_THRESHOLD=24` produced the following ragged-versus-static-K=4 output
throughput deltas:

| Concurrent requests | Ragged K=4/8 vs suffix static K=4 | K=8 request rounds | K=8 efficiency |
| ---: | ---: | ---: | ---: |
| 10 | +2.78% | 2,158 | 0.873 |
| 20 | **+5.05%** | 4,919 | 0.888 |
| 24 | +2.70% | 1,859 | 0.902 |

The benchmark default in `scripts/run_dynamic_k_experiment.sh` is therefore
24 for throughput-oriented testing. This is not a latency default: at
concurrency 20, mean TTFT rose from 1,213.81 ms (static K=4) to 1,714.24 ms
(ragged K=4/8), and mean ITL rose from 143.46 ms to 194.27 ms. Deploy with
`--speculative-high-bs-threshold 24` only when aggregate throughput is the
priority; retain 20 or lower when latency is more important.

### Bounded CUDA-Graph validation (2026-07-17)

The padded K=8 graph path was first checked with `SGLANG_RAGGED_CUDA_GRAPH_MIN_LONG_RATIO=0` on 20 labeled GSM8K questions. Static K=4 and dynamic K=4/8 both achieved `0.600` accuracy, while the dynamic run recorded 86 ragged CUDA-graph batches and no eager ragged batches. This validates that graph replay preserves greedy answers in the exercised run.

On the repeated-data throughput workload with ratio `0.75`, the initial run
at `dynamic_k_20260717_173642` measured:

| Concurrent requests | Static K=4 tok/s | Ragged K=4/8 tok/s | Dynamic vs static | Graph hit rate |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 363.44 | 385.88 | **+6.17%** | 13.94% |
| 20 | 377.17 | 407.90 | **+8.15%** | 12.95% |
| 24 | 371.04 | 381.73 | **+2.88%** | 17.54% |

The graph rate is low because K=8 proposals are distributed across many mixed
batches; few batches reach 75% K=8 rows. These are promising end-to-end
results, but not an isolated graph-speedup claim.

The completed ratio sweep `ragged_cuda_graph_ratio_20260717_192114` compared
identical workloads across eager and graph policies:

| Min K=8 ratio | Graph hit rate (10 / 20 / 24) | Ragged throughput at 10 / 20 / 24 (tok/s) |
| ---: | ---: | ---: |
| 1.00 (eager control) | 0.00% / 0.00% / 0.00% | **387.48 / 416.68 / 383.18** |
| 0.75 | 11.44% / 20.34% / 13.70% | 382.59 / 409.16 / 371.02 |
| 0.60 | 44.94% / 39.37% / 49.76% | 383.10 / 396.57 / 376.59 |
| 0.50 | 68.27% / 56.95% / 74.91% | 381.75 / 398.35 / 371.00 |

More graph coverage did not improve throughput: padding K=4 rows to K=8 costs
more than the current graph replay saves. Therefore the default is `1.0`
(pure eager Ragged) until a compact, true variable-K CUDA Graph is available.
Use `scripts/run_ragged_cuda_graph_ratio_sweep.sh` for future A/B checks.

### Compact true variable-K CUDA Graph

`SGLANG_RAGGED_VARLEN_CUDA_GRAPH_PATTERNS` enables opt-in compact FA3 graphs
for exact observed mixed shapes. Its format is a comma-separated list of
`batch_size:k8_request_count` pairs:

```bash
SGLANG_RAGGED_VARLEN_CUDA_GRAPH_PATTERNS="10:5,20:10,24:8" \
GPU_IDS=0,1,2,3 TP_SIZE=4 \
bash scripts/run_dynamic_k_experiment.sh
```

For a matched pattern, the graph input has exactly
`4 * batch_size + 4 * k8_request_count` query tokens. FA3 receives the real
`cu_seqlens_q`, such as `[0, 4, 12, 16]`; no K=4 row is padded to K=8. Graphs
are captured once at server startup and unmatched shapes safely use eager
Ragged FA3. Patterns are opt-in because capturing all possible `(bs, k8_count)`
combinations would consume excessive CUDA-graph memory. Start with a small set
of hot shapes and inspect `ragged_verify_cuda_graph_batch_total` for hits.
`ragged_verify_varlen_cuda_graph_batch_total` is the stricter proof that a
compact true-varlen graph, rather than the retired K=8-padding graph, replayed.

### Recorded validation results (2026-07-19 to 2026-07-20)

All measurements below use Qwen2.5-72B-Instruct-AWQ, Qwen3-0.6B draft, TP=4,
FA3, greedy decoding, and the repeated SpecForge workload. Result directories
are retained under `/workspace/SpecForge/results/`.

**Compact true-varlen CUDA Graph.** In
`varlen_eager_20260719_222455` versus
`varlen_graph_20260719_223510`, both with `max_running_requests=16` and
`mem_fraction_static=0.68`, concurrency-10 output throughput improved from
`390.60` to `405.82 tok/s` (**+3.90%**). The compact `10:5` graph replayed
121 times out of 692 ragged verify batches (17.5% coverage). A GSM8K run at
`varlen_gsm8k_20260719_220654` recorded two compact graph replays and achieved
0.930 accuracy versus 0.910 for suffix-static K=4; this is a correctness
smoke test, not a quality-improvement claim.

**Binary dynamic-K scaling.** The eager-only sweep in
`dynamic_k_scaling_20260719_232837` established the following output-throughput
deltas against suffix-static K=4:

| Concurrency | K=4/8 | K=4/12 | K=4/16 | Best |
| ---: | ---: | ---: | ---: | --- |
| 10 | +7.65% | +7.06% | +6.97% | K=4/8 |
| 20 | +10.57% | +16.39% | **+17.37%** | K=4/16 |
| 24 | +0.69% | -0.64% | **+3.64%** | K=4/16 |

K=16 reduced long-K committed-token efficiency to roughly 0.82--0.86, but
still won at concurrency 20 because its target verify batching was more
efficient. The current default candidate for deployment is therefore binary
K=4/16 with `long_suffix_min_match_len=15`; it must still be validated with
the match-threshold sweep below.

**Adaptive K=4/8/16 policy.** The first policy used
`SGLANG_DYNAMIC_K_TIERS="8:7,16:15"` and
`SGLANG_DYNAMIC_K_BATCH_POLICY="12:8,22:16,24:16"`. In
`dynamic_k_multitier_20260720_104930` it underperformed binary K=4/16:

| Concurrency | Binary K=4/16 | Adaptive K=4/8/16 |
| ---: | ---: | ---: |
| 10 | +7.21% | +5.18% |
| 20 | +16.08% | +13.85% |
| 24 | +4.61% | +1.40% |

The final TP0 metric snapshot contained 17,308 K=8 requests and only 2,693
K=16 requests. Active decode batches often shrink below 12 during request
tails even under external concurrency 20/24, so the batch-size rule selected
K=8 for 86.5% of long requests. Do not use that adaptive policy as a default.
`dynamic_k_tier_request_total{draft_tokens="..."}` is the authoritative
metric for future tier-selection validation.

**Next validation.** Run `scripts/run_dynamic_k16_match_sweep.sh` to compare
K=16 minimum suffix-match lengths 15, 19, and 23 at concurrency 20 and 24.
This isolates whether stricter, higher-confidence K=16 candidates improve the
remaining high-concurrency bottleneck before attempting K=20 or K=24.

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

| Concurrent requests | No speculation | Standalone K=4 | Suffix static K=4 | Dynamic K=4/4 | Dynamic K=4/8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 260.33 | **366.11** | 355.58 | 301.49 | 336.22 |
| 20 | 291.78 | **388.70** | 373.36 | 368.07 | 379.29 |
| 24 | 301.49 | **388.08** | 372.82 | 359.96 | 355.78 |

At these loads, standalone K=4 is the current end-to-end throughput baseline.
Dynamic K is faster than no speculation, but does not exceed standalone K=4.
The K=4/4 ablation is `-15.21%`, `-1.42%`, and `-3.45%` versus suffix static
K=4 at 10, 20, and 24 concurrency. This is the direct cost of splitting and
serializing the verify work. Widening long suffixes from K=4 to K=8 recovers
`+11.52%`, `+3.05%`, and `-1.16%` versus the K=4/4 ablation, but the final
K=4/8 result is still `-5.44%`, `+1.59%`, and `-4.57%` versus suffix static
K=4. The 20-concurrency gain is small enough to treat as benchmark variation,
so dynamic K has no demonstrated stable net throughput win in the 10--24
operating range.

The corresponding dynamic-versus-static latency comparison is:

| Concurrent requests | Dynamic TTFT (ms) | Static TTFT (ms) | Dynamic TPOT (ms) | Static TPOT (ms) | Dynamic ITL (ms) | Static ITL (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2,185.22 | 2,078.41 | 25.83 | 24.43 | 139.25 | 90.65 |
| 20 | 2,077.08 | 1,870.86 | 52.47 | 52.82 | 177.18 | 173.71 |
| 24 | 3,774.38 | 3,334.35 | 66.34 | 66.04 | 217.10 | 200.73 |

### Dynamic-K instrumentation run

The 2026-07-15 warm-cache run added suffix/K=8 counters. The K=8 probe used
120 prompts at concurrency 8 after an identical concurrency-8 warmup; it is a
mechanism check rather than a 10--24 end-to-end result.

| Phase | Suffix proposals | K=4 suffix overrides | K=8 request rounds | K=8 committed tokens | K=8 verify tokens | K=8 efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warmup | 17,432 | 3,416 | 5,171 | 35,813 | 41,368 | 0.866 |
| K=8 probe | 15,933 | 3,519 | 6,291 | 44,427 | 50,328 | 0.883 |

`K=8 efficiency = committed tokens / K=8 verify tokens`. During the probe,
K=8 commits about `44,427 / 6,291 = 7.06` tokens per K=8 request round. This
confirms that the repeated-data workload produces real, high-quality long
suffix hits; poor K=8 acceptance is not the reason for the end-to-end result.

The per-concurrency counters are:

| Concurrent requests | K=8 request rounds | K=8 committed tokens | K=8 verify tokens | K=8 efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 3,418 | 24,286 | 27,344 | 0.888 |
| 20 | 754 | 5,393 | 6,032 | 0.894 |
| 24 | 1,262 | 9,102 | 10,096 | 0.902 |

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

The latest counter evidence confirms the split cost:

| Concurrency | Config | Dynamic parent batches | Mixed batches | Target verify calls | Calls per parent batch |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10 | K=4/4 | 1,055 | 839 | 1,894 | 1.80 |
| 10 | K=4/8 | 730 | 677 | 1,407 | 1.93 |
| 20 | K=4/4 | 170 | 155 | 325 | 1.91 |
| 20 | K=4/8 | 123 | 117 | 240 | 1.95 |
| 24 | K=4/4 | 320 | 211 | 531 | 1.66 |
| 24 | K=4/8 | 199 | 190 | 389 | 1.95 |

`Target verify calls = normal verify calls + long-suffix verify calls`. The
K=8 path reduces parent batches at concurrency 10 from 1,055 to 730 and
recovers 11.52% throughput versus K=4/4, but almost every remaining parent
batch is still split into two serial target forwards. This directly explains
why high K=8 acceptance does not yet turn into a net production gain.

### FA3 ragged variable-K: stage one

The homogeneous-only policy was a safe interim rollout. FA3 stage one now
re-enables a mixed K=4/K=8 batch when all of the following are true:

- `--attention-backend fa3`;
- default `page_size=1`;
- `topk=1`, greedy sampling, no grammar, no return-logprob.

Rather than split the batch, the target query is flattened with per-request
lengths and passed once to FA3 through `cu_seqlens_q`:

```text
K per request: [4, 8, 4]
cu_seqlens_q:  [0, 4, 12, 16]
```

Only small integer acceptance tables are padded to `[batch, 8]`; the
vocabulary-sized target logits remain ragged. Thus a mixed batch has exactly
one target forward and no result merge. The existing
`dynamic_k_mixed_verify_batch_total` remains zero because it measures the
retired *two-forward split* implementation, not a ragged mixed batch.

The bounded graph path does not create a graph for every `(batch_size, K=8
request count)` pair. When K=8 coverage reaches the configured ratio, it lays
out each row as K=8, preserves the real K=4/K=8 acceptance mask, and replays
the ordinary K=8 graph. Causal attention makes each real K=4 prefix identical
to its eager ragged computation; padded tail KV slots are evicted before the
next round. Lower-coverage mixed batches remain eager so they do not pay the
full K=8 compute cost.

### GSM8K accuracy regression

Run `scripts/run_dynamic_k_gsm8k_accuracy.sh` on the serving host to compare
suffix static K=4 with FA3 ragged K=4/8. It defaults to
`datasets/gsm8k_test.jsonl` (the official labeled GSM8K test split), uses
greedy decoding, and writes
`accuracy_comparison.md` below its result directory. The dynamic run first
warms the cache with even-indexed questions, then evaluates the original
question order so cached and uncached requests are interleaved in a batch.
Require `dynamic_k8_request_total > 0` in the final metrics snapshot; otherwise
the reported score did not exercise K=8/ragged verification.
If the JSONL contains `question` and `answer`, the script reports GSM8K
accuracy. A prompt-only SpecForge JSONL with `turns` automatically uses the
same static-K=4 output as the temperature-zero baseline and writes
`greedy_output_comparison.md`; any token-id/text mismatch fails the run.

## Correctness Boundaries

The first implementation intentionally keeps the dynamic-K path conservative:

- only supports `topk=1`.
- only enables ragged dynamic-K for FA3 with `page_size=1` and greedy
  sampling.
- falls back to the homogeneous K=4/K=8 policy for other backends, grammar,
  or return-logprob.
- disables overlap schedule for this mode.
- every request remains in its original scheduler batch and occupies one
  variable-length FA3 query segment.

These constraints avoid:

- random sampling order changes.
- logprob alignment errors.
- grammar state mismatch.
- paged-KV eviction before its ragged page-alignment implementation.
- CUDA Graph shape/cache explosion before bounded ragged graph caching.

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

With stage-one FA3 ragged verification, this becomes one variable-length FA3
forward rather than two serial graphs. Long suffix hits keep their value
without forcing normal requests into a K=8 target query.

## Implementation Files

Main files changed:

- `python/sglang/srt/speculative/eagle_worker.py`
  - request-level dynamic-K classification
  - ragged K=4/K=8 verify input construction
- `python/sglang/srt/speculative/eagle_info.py`
  - ragged greedy acceptance and page-size-one KV reclamation
- `python/sglang/srt/layers/attention/flashattention_backend.py`
  - FA3 target-verify `cu_seqlens_q` metadata
- `python/sglang/srt/model_executor/cuda_graph_runner.py`
  - bounded fixed-K graph eligibility for padded ragged verification
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
