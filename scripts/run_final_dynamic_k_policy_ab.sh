#!/usr/bin/env bash
# Final alternating A/B validation for the deployable dynamic-K policy.
#
# Baseline: suffix static K=4.
# Candidate: K=16 when active batch < 24 and suffix match >= 23; at active
# batch >= 24, suffix K=8 when match >= 8; all other rows remain K=4.
#
# Each round launches a fresh server. Odd rounds execute baseline then
# candidate, even rounds candidate then baseline. The companion summarizer
# reports per-concurrency medians across runs and validates K=8/K=16 tier use.

set -Eeuo pipefail

SGLANG_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPEC_FORGE_DIR="${SPEC_FORGE_DIR:-/workspace/SpecForge}"
RESULTS_DIR="${RESULTS_DIR:-${SPEC_FORGE_DIR}/results/final_dynamic_k_ab_$(date +%Y%m%d_%H%M%S)}"

REPEATS="${REPEATS:-3}"
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

if ! [[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPEATS must be a positive integer, got ${REPEATS}" >&2
    exit 2
fi

run_case() {
    local round="$1"
    local policy="$2"
    local label
    if [[ "${policy}" == "fixed" ]]; then
        label="r$(printf '%02d' "${round}")_fixed_k4"
        echo "========== round ${round}: fixed suffix K=4 =========="
        env -u SGLANG_RAGGED_VARLEN_CUDA_GRAPH_PATTERNS \
            -u SGLANG_DYNAMIC_K_TIERS \
            -u SGLANG_DYNAMIC_K_BATCH_POLICY \
            -u SGLANG_DYNAMIC_K_HIGH_BATCH_FALLBACK \
            SGLANG_RAGGED_CUDA_GRAPH_MIN_LONG_RATIO=1.0 \
            RESULTS_DIR="${RESULTS_DIR}/${label}" \
            GPU_IDS="${GPU_IDS}" TP_SIZE="${TP_SIZE}" \
            MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
            MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
            MEASURE_PROMPTS="${MEASURE_PROMPTS}" \
            MEASURE_CONCURRENCY="${MEASURE_CONCURRENCY}" \
            WARMUP_PROMPTS="${WARMUP_PROMPTS}" \
            WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY}" \
            FIXED_OUTPUT_LEN="${FIXED_OUTPUT_LEN}" SHUFFLE="${SHUFFLE}" \
            EXPERIMENTS=suffix_static_k4 \
            bash "${SGLANG_DIR}/scripts/run_dynamic_k_experiment.sh"
        return
    fi

    label="r$(printf '%02d' "${round}")_final_policy"
    echo "========== round ${round}: final K=4/16 + high-batch K=8 =========="
    env -u SGLANG_RAGGED_VARLEN_CUDA_GRAPH_PATTERNS \
        -u SGLANG_DYNAMIC_K_TIERS \
        -u SGLANG_DYNAMIC_K_BATCH_POLICY \
        SGLANG_RAGGED_CUDA_GRAPH_MIN_LONG_RATIO=1.0 \
        RESULTS_DIR="${RESULTS_DIR}/${label}" \
        GPU_IDS="${GPU_IDS}" TP_SIZE="${TP_SIZE}" \
        MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
        MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
        MEASURE_PROMPTS="${MEASURE_PROMPTS}" \
        MEASURE_CONCURRENCY="${MEASURE_CONCURRENCY}" \
        WARMUP_PROMPTS="${WARMUP_PROMPTS}" \
        WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY}" \
        FIXED_OUTPUT_LEN="${FIXED_OUTPUT_LEN}" SHUFFLE="${SHUFFLE}" \
        EXPERIMENTS=dynamic_k4_k16 \
        DYNAMIC_EXPERIMENT_NAME=dynamic_k4_k16 \
        DYNAMIC_LONG_DRAFT_TOKENS=16 \
        DYNAMIC_LONG_SUFFIX_MIN_MATCH_LEN=23 \
        HIGH_BS_THRESHOLD=24 \
        SGLANG_DYNAMIC_K_HIGH_BATCH_FALLBACK=8:8 \
        bash "${SGLANG_DIR}/scripts/run_dynamic_k_experiment.sh"
}

mkdir -p "${RESULTS_DIR}"

for ((round = 1; round <= REPEATS; ++round)); do
    if (( round % 2 )); then
        run_case "${round}" fixed
        run_case "${round}" dynamic
    else
        run_case "${round}" dynamic
        run_case "${round}" fixed
    fi
done

python "${SGLANG_DIR}/scripts/summarize_final_dynamic_k_policy_ab.py" "${RESULTS_DIR}" \
    | tee "${RESULTS_DIR}/final_ab_summary.md"

echo "Completed. Results: ${RESULTS_DIR}"
