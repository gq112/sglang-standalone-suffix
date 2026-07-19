#!/usr/bin/env bash
# Sweep fixed K=4 and suffix dynamic K=4/{8,12,16} on one identical workload.
#
# This isolates the algorithmic gain of dynamic step length. Compact varlen
# CUDA graphs are deliberately disabled by default; run the graph sweep only
# after selecting the best K policy.
#
# Example:
#   GPU_IDS=0,1,2,3 TP_SIZE=4 bash scripts/run_dynamic_k_scaling_sweep.sh

set -Eeuo pipefail

SGLANG_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPEC_FORGE_DIR="${SPEC_FORGE_DIR:-/workspace/SpecForge}"
RESULTS_DIR="${RESULTS_DIR:-${SPEC_FORGE_DIR}/results/dynamic_k_scaling_$(date +%Y%m%d_%H%M%S)}"
K_VALUES=( ${K_VALUES:-8 12 16} )

# Keep these identical across all configurations. They are passed through to
# run_dynamic_k_experiment.sh and can be overridden by the caller.
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
HIGH_BS_THRESHOLD="${HIGH_BS_THRESHOLD:-24}"
MEASURE_PROMPTS="${MEASURE_PROMPTS:-40 80 96}"
MEASURE_CONCURRENCY="${MEASURE_CONCURRENCY:-10 20 24}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-120}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-8}"
FIXED_OUTPUT_LEN="${FIXED_OUTPUT_LEN:-2048}"
SHUFFLE="${SHUFFLE:-0}"

run_case() {
    local label="$1"
    shift
    echo "========== dynamic-K scaling: ${label} =========="
    env -u SGLANG_RAGGED_VARLEN_CUDA_GRAPH_PATTERNS \
        RESULTS_DIR="${RESULTS_DIR}/${label}" \
        SGLANG_RAGGED_CUDA_GRAPH_MIN_LONG_RATIO=1.0 \
        GPU_IDS="${GPU_IDS}" \
        TP_SIZE="${TP_SIZE}" \
        MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
        MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
        HIGH_BS_THRESHOLD="${HIGH_BS_THRESHOLD}" \
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

# Fixed suffix K=4 is the user-facing baseline, while dynamic K=4/4 exposes
# the policy/ragged bookkeeping cost separately.
run_case fixed_k4 EXPERIMENTS=suffix_static_k4
run_case dynamic_k4_control EXPERIMENTS=dynamic_k4_k4

for k in "${K_VALUES[@]}"; do
    if (( k <= 4 )); then
        echo "K_VALUES entries must be greater than 4, got ${k}" >&2
        exit 2
    fi
    # A long K is eligible only when the suffix match is at least K-1 tokens.
    # This tests high-confidence widening rather than intentionally feeding a
    # K=16 verifier with a K=7 suffix continuation.
    run_case "dynamic_k4_k${k}" \
        EXPERIMENTS="dynamic_k4_k${k}" \
        DYNAMIC_EXPERIMENT_NAME="dynamic_k4_k${k}" \
        DYNAMIC_LONG_DRAFT_TOKENS="${k}" \
        DYNAMIC_LONG_SUFFIX_MIN_MATCH_LEN="$((k - 1))"
done

python "${SGLANG_DIR}/scripts/summarize_dynamic_k_scaling_sweep.py" "${RESULTS_DIR}" \
    | tee "${RESULTS_DIR}/dynamic_k_scaling_summary.md"

echo "Completed. Results: ${RESULTS_DIR}"
