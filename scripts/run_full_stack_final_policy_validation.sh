#!/usr/bin/env bash
# End-to-end validation of the complete speculative-decoding stack.
#
# Configurations:
#   no_speculation       Target-model-only baseline
#   standalone_k4        Fixed K=4 standalone speculation
#   suffix_static_k4     ArcticInference suffix fusion with fixed K=4
#   dynamic_final        K=4/16 below active batch 24, K=4/8 at/above 24
#
# Four cyclic rounds are the default so every configuration occupies every
# launch position once. This controls slow time-dependent GPU/cache drift.

set -Eeuo pipefail

SGLANG_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPEC_FORGE_DIR="${SPEC_FORGE_DIR:-/workspace/SpecForge}"
RESULTS_DIR="${RESULTS_DIR:-${SPEC_FORGE_DIR}/results/full_stack_final_$(date +%Y%m%d_%H%M%S)}"

REPEATS="${REPEATS:-4}"
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

run_config() {
    local round="$1"
    local policy="$2"
    local label="r$(printf '%02d' "${round}")_${policy}"
    local common=(
        -u SGLANG_RAGGED_VARLEN_CUDA_GRAPH_PATTERNS
        -u SGLANG_DYNAMIC_K_TIERS
        -u SGLANG_DYNAMIC_K_BATCH_POLICY
        RESULTS_DIR="${RESULTS_DIR}/${label}"
        SGLANG_RAGGED_CUDA_GRAPH_MIN_LONG_RATIO=1.0
        GPU_IDS="${GPU_IDS}"
        TP_SIZE="${TP_SIZE}"
        MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}"
        MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}"
        MEASURE_PROMPTS="${MEASURE_PROMPTS}"
        MEASURE_CONCURRENCY="${MEASURE_CONCURRENCY}"
        WARMUP_PROMPTS="${WARMUP_PROMPTS}"
        WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY}"
        FIXED_OUTPUT_LEN="${FIXED_OUTPUT_LEN}"
        SHUFFLE="${SHUFFLE}"
    )

    case "${policy}" in
        no_speculation|standalone_k4|suffix_static_k4)
            echo "========== round ${round}: ${policy} =========="
            env -u SGLANG_DYNAMIC_K_HIGH_BATCH_FALLBACK "${common[@]}" \
                EXPERIMENTS="${policy}" \
                bash "${SGLANG_DIR}/scripts/run_dynamic_k_experiment.sh"
            ;;
        dynamic_final)
            echo "========== round ${round}: final dynamic K policy =========="
            env "${common[@]}" \
                EXPERIMENTS=dynamic_k4_k16 \
                DYNAMIC_EXPERIMENT_NAME=dynamic_k4_k16 \
                DYNAMIC_LONG_DRAFT_TOKENS=16 \
                DYNAMIC_LONG_SUFFIX_MIN_MATCH_LEN=23 \
                HIGH_BS_THRESHOLD=24 \
                SGLANG_DYNAMIC_K_HIGH_BATCH_FALLBACK=8:8 \
                bash "${SGLANG_DIR}/scripts/run_dynamic_k_experiment.sh"
            ;;
        *)
            echo "Unknown policy: ${policy}" >&2
            exit 2
            ;;
    esac
}

mkdir -p "${RESULTS_DIR}"
policies=(no_speculation standalone_k4 suffix_static_k4 dynamic_final)
for ((round = 1; round <= REPEATS; ++round)); do
    offset=$(((round - 1) % ${#policies[@]}))
    for ((index = 0; index < ${#policies[@]}; ++index)); do
        run_config "${round}" "${policies[$(((index + offset) % ${#policies[@]}))]}"
    done
done

python "${SGLANG_DIR}/scripts/summarize_full_stack_final_policy.py" "${RESULTS_DIR}" \
    | tee "${RESULTS_DIR}/full_stack_summary.md"

echo "Completed. Results: ${RESULTS_DIR}"
