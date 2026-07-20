#!/usr/bin/env bash
# Sweep high-confidence K=16 suffix thresholds at concurrency 20 and 24.
#
# The prior K sweep identified K=4/16 as the best binary policy. This script
# determines whether stricter suffix matching improves its high-concurrency
# throughput and latency without changing the model, workload, or K width.

set -Eeuo pipefail

SGLANG_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPEC_FORGE_DIR="${SPEC_FORGE_DIR:-/workspace/SpecForge}"
RESULTS_DIR="${RESULTS_DIR:-${SPEC_FORGE_DIR}/results/dynamic_k16_match_$(date +%Y%m%d_%H%M%S)}"
MATCH_LENGTHS=( ${MATCH_LENGTHS:-15 19 23} )

GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
HIGH_BS_THRESHOLD="${HIGH_BS_THRESHOLD:-24}"
MEASURE_PROMPTS="${MEASURE_PROMPTS:-80 96}"
MEASURE_CONCURRENCY="${MEASURE_CONCURRENCY:-20 24}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-120}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-8}"
FIXED_OUTPUT_LEN="${FIXED_OUTPUT_LEN:-2048}"
SHUFFLE="${SHUFFLE:-0}"

run_case() {
    local label="$1"
    shift
    echo "========== K=16 match sweep: ${label} =========="
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
run_case fixed_k4 EXPERIMENTS=suffix_static_k4

for match_len in "${MATCH_LENGTHS[@]}"; do
    run_case "dynamic_k4_k16_m${match_len}" \
        EXPERIMENTS=dynamic_k4_k16 \
        DYNAMIC_EXPERIMENT_NAME=dynamic_k4_k16 \
        DYNAMIC_LONG_DRAFT_TOKENS=16 \
        DYNAMIC_LONG_SUFFIX_MIN_MATCH_LEN="${match_len}"
done

python "${SGLANG_DIR}/scripts/summarize_dynamic_k_scaling_sweep.py" "${RESULTS_DIR}" \
    | tee "${RESULTS_DIR}/k16_match_summary.md"

echo "Completed. Results: ${RESULTS_DIR}"
