#!/usr/bin/env bash
# Compare dynamic-K policies across 10/20/24/30 external concurrency.
#
# Policies:
#   fixed K=4
#   K=4/8 (K=8 allowed through active batch 32)
#   K=4/16 (K=16 below active batch 24, K=4 at/above it)
#   K=4/16 below 24 plus suffix K=8 fallback at/above 24

set -Eeuo pipefail

SGLANG_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPEC_FORGE_DIR="${SPEC_FORGE_DIR:-/workspace/SpecForge}"
RESULTS_DIR="${RESULTS_DIR:-${SPEC_FORGE_DIR}/results/dynamic_k_high_batch_$(date +%Y%m%d_%H%M%S)}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MEASURE_PROMPTS="${MEASURE_PROMPTS:-40 80 96 120}"
MEASURE_CONCURRENCY="${MEASURE_CONCURRENCY:-10 20 24 30}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-120}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-8}"
FIXED_OUTPUT_LEN="${FIXED_OUTPUT_LEN:-2048}"
SHUFFLE="${SHUFFLE:-0}"

run_case() {
    local label="$1"
    shift
    echo "========== high-batch dynamic-K: ${label} =========="
    env -u SGLANG_RAGGED_VARLEN_CUDA_GRAPH_PATTERNS \
        -u SGLANG_DYNAMIC_K_TIERS \
        -u SGLANG_DYNAMIC_K_BATCH_POLICY \
        -u SGLANG_DYNAMIC_K_HIGH_BATCH_FALLBACK \
        RESULTS_DIR="${RESULTS_DIR}/${label}" \
        SGLANG_RAGGED_CUDA_GRAPH_MIN_LONG_RATIO=1.0 \
        GPU_IDS="${GPU_IDS}" \
        TP_SIZE="${TP_SIZE}" \
        MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
        MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
        MEASURE_PROMPTS="${MEASURE_PROMPTS}" \
        MEASURE_CONCURRENCY="${MEASURE_CONCURRENCY}" \
        WARMUP_PROMPTS="${WARMUP_PROMPTS}" \
        WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY}" \
        FIXED_OUTPUT_LEN="${FIXED_OUTPUT_LEN}" \
        SHUFFLE="${SHUFFLE}" \
        "$@" \
        bash "${SGLANG_DIR}/scripts/run_dynamic_k_experiment.sh"
}

mkdir -p "${RESULTS_DIR}"

run_case fixed_k4 EXPERIMENTS=suffix_static_k4

run_case dynamic_k4_k8 \
    EXPERIMENTS=dynamic_k4_k8 \
    DYNAMIC_EXPERIMENT_NAME=dynamic_k4_k8 \
    DYNAMIC_LONG_DRAFT_TOKENS=8 \
    DYNAMIC_LONG_SUFFIX_MIN_MATCH_LEN=8 \
    HIGH_BS_THRESHOLD=33

run_case dynamic_k4_k16 \
    EXPERIMENTS=dynamic_k4_k16 \
    DYNAMIC_EXPERIMENT_NAME=dynamic_k4_k16 \
    DYNAMIC_LONG_DRAFT_TOKENS=16 \
    DYNAMIC_LONG_SUFFIX_MIN_MATCH_LEN=23 \
    HIGH_BS_THRESHOLD=24

run_case dynamic_k4_k16_high_k8 \
    EXPERIMENTS=dynamic_k4_k16 \
    DYNAMIC_EXPERIMENT_NAME=dynamic_k4_k16 \
    DYNAMIC_LONG_DRAFT_TOKENS=16 \
    DYNAMIC_LONG_SUFFIX_MIN_MATCH_LEN=23 \
    HIGH_BS_THRESHOLD=24 \
    SGLANG_DYNAMIC_K_HIGH_BATCH_FALLBACK=8:8

python "${SGLANG_DIR}/scripts/summarize_dynamic_k_scaling_sweep.py" "${RESULTS_DIR}" \
    | tee "${RESULTS_DIR}/high_batch_fallback_summary.md"

echo "Completed. Results: ${RESULTS_DIR}"
