#!/usr/bin/env bash
# Isolate the effect of the optional SRI stable dataset tree.
#
# Compare the existing Arctic cache, SRI forest with only a rolling global
# tree, and SRI forest with an additional stable dataset tree. Every run uses
# the already-validated final K=4/16/8 policy and FA3 ragged verification.

set -Eeuo pipefail

SGLANG_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPEC_FORGE_DIR="${SPEC_FORGE_DIR:-/workspace/SpecForge}"
RESULTS_DIR="${RESULTS_DIR:-${SPEC_FORGE_DIR}/results/sri_dataset_tree_$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-3}"
DATASET_TREE_CAPACITY="${DATASET_TREE_CAPACITY:-256}"

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

PYTHONPATH="${SGLANG_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}" python -c \
  "import sglang.srt.speculative.sri_forest._sglang_sri_suffix_tree" || {
    echo "Build the SRI suffix-tree extension first: bash scripts/build_sri_suffix_tree.sh" >&2
    exit 1
  }

run_case() {
    local round="$1"
    local policy="$2"
    local backend="arctic"
    local dataset_capacity=0
    case "${policy}" in
        sri_global) backend="sri_forest" ;;
        sri_dataset)
            backend="sri_forest"
            dataset_capacity="${DATASET_TREE_CAPACITY}"
            ;;
    esac
    echo "========== round ${round}: ${policy} =========="
    env -u SGLANG_RAGGED_VARLEN_CUDA_GRAPH_PATTERNS \
        -u SGLANG_DYNAMIC_K_TIERS \
        -u SGLANG_DYNAMIC_K_BATCH_POLICY \
        RESULTS_DIR="${RESULTS_DIR}/r$(printf '%02d' "${round}")_${policy}" \
        SGLANG_RAGGED_CUDA_GRAPH_MIN_LONG_RATIO=1.0 \
        GPU_IDS="${GPU_IDS}" TP_SIZE="${TP_SIZE}" \
        MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
        MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
        MEASURE_PROMPTS="${MEASURE_PROMPTS}" \
        MEASURE_CONCURRENCY="${MEASURE_CONCURRENCY}" \
        WARMUP_PROMPTS="${WARMUP_PROMPTS}" \
        WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY}" \
        FIXED_OUTPUT_LEN="${FIXED_OUTPUT_LEN}" SHUFFLE="${SHUFFLE}" \
        SUFFIX_BACKEND="${backend}" \
        SUFFIX_DATASET_CACHE_MAX_REQUESTS="${dataset_capacity}" \
        EXPERIMENTS=dynamic_k4_k16 \
        DYNAMIC_EXPERIMENT_NAME=dynamic_k4_k16 \
        DYNAMIC_LONG_DRAFT_TOKENS=16 \
        DYNAMIC_LONG_SUFFIX_MIN_MATCH_LEN=23 \
        HIGH_BS_THRESHOLD=24 \
        SGLANG_DYNAMIC_K_HIGH_BATCH_FALLBACK=8:8 \
        bash "${SGLANG_DIR}/scripts/run_dynamic_k_experiment.sh"
}

mkdir -p "${RESULTS_DIR}"
policies=(arctic sri_global sri_dataset)
for ((round = 1; round <= REPEATS; ++round)); do
    offset=$(((round - 1) % ${#policies[@]}))
    for ((index = 0; index < ${#policies[@]}; ++index)); do
        run_case "${round}" "${policies[$(((index + offset) % ${#policies[@]}))]}"
    done
done

python "${SGLANG_DIR}/scripts/summarize_sri_forest_dataset_tree.py" "${RESULTS_DIR}" \
    | tee "${RESULTS_DIR}/sri_dataset_tree_summary.md"

echo "Completed. Results: ${RESULTS_DIR}"
