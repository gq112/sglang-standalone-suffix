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
--speculative-high-bs-threshold 10
```

Recommended starting point:

```bash
--speculative-algorithm STANDALONE \
--speculative-suffix-enable \
--speculative-dynamic-k-enable \
--speculative-normal-draft-token-num 4 \
--speculative-long-suffix-draft-token-num 8 \
--speculative-long-suffix-min-match-len 7 \
--speculative-high-bs-threshold 10
```

Interpretation:

- standalone drafts 3 tokens by default (`K=4`).
- long suffix hits can verify 7 suffix tokens (`K=8`).
- at batch size 10 or above, dynamic long suffix is disabled and all requests stay on K=4.

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
