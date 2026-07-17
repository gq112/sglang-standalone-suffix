#!/usr/bin/env bash
# Compare the bounded ragged CUDA-graph policy against eager Ragged FA3.
#
# The child benchmark keeps all workload parameters identical. Only
# SGLANG_RAGGED_CUDA_GRAPH_MIN_LONG_RATIO changes between runs.
#
# Example:
#   cd /workspace/sglang-standalone-suffix
#   GPU_IDS=0,1,2,3 TP_SIZE=4 bash scripts/run_ragged_cuda_graph_ratio_sweep.sh

set -Eeuo pipefail

SGLANG_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPEC_FORGE_DIR="${SPEC_FORGE_DIR:-/workspace/SpecForge}"
RATIOS=( ${RATIOS:-1.0 0.75 0.60 0.50} )
SWEEP_DIR="${SWEEP_DIR:-${SPEC_FORGE_DIR}/results/ragged_cuda_graph_ratio_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${SWEEP_DIR}"

for ratio in "${RATIOS[@]}"; do
    label="${ratio//./_}"
    result_dir="${SWEEP_DIR}/ratio_${label}"
    echo "========== ragged graph ratio=${ratio} =========="
    SGLANG_RAGGED_CUDA_GRAPH_MIN_LONG_RATIO="${ratio}" \
        RESULTS_DIR="${result_dir}" \
        bash "${SGLANG_DIR}/scripts/run_dynamic_k_experiment.sh"
done

python "${SGLANG_DIR}/scripts/summarize_ragged_cuda_graph_ratio_sweep.py" \
    "${SWEEP_DIR}"

echo "Completed. Results: ${SWEEP_DIR}"
