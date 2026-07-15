#!/usr/bin/env bash
# Compare no speculation, standalone K=4, suffix K=4, and suffix dynamic K=4/8.
#
# Default layout matches the deployment environment:
#   SGLang source:   /workspace/sglang-standalone-suffix
#   benchmark tool:  /workspace/SpecForge/test_req.py
#
# Example:
#   cd /workspace/sglang-standalone-suffix
#   bash scripts/run_dynamic_k_experiment.sh
#
# Environment variables are intentionally used instead of editing this file:
#   RESULTS_DIR=/workspace/SpecForge/results/dynamic_k_$(date +%F_%H%M%S)
#   GPU_IDS=0,1,2,3 TP_SIZE=4 PORT=30000 \
#   DATASET_PATH=donghuayiwei_fixed.jsonl \
#   bash scripts/run_dynamic_k_experiment.sh

set -Eeuo pipefail

SGLANG_DIR="${SGLANG_DIR:-/workspace/sglang-standalone-suffix}"
SPEC_FORGE_DIR="${SPEC_FORGE_DIR:-/workspace/SpecForge}"
TEST_SCRIPT="${TEST_SCRIPT:-${SPEC_FORGE_DIR}/test_req.py}"
DATASET_NAME="${DATASET_NAME:-openai-chat}"
DATASET_PATH="${DATASET_PATH:-donghuayiwei_fixed.jsonl}"
MODEL_PATH="${MODEL_PATH:-/models/models/Qwen/Qwen2.5-72B-Instruct-AWQ}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/models/models/Qwen/Qwen3-0.6B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"

HOST="${HOST:-0.0.0.0}"
CLIENT_BASE_URL="${CLIENT_BASE_URL:-http://127.0.0.1:30000}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-4}"
GPU_IDS="${GPU_IDS:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
PRELOAD_LIBSTDCXX="${PRELOAD_LIBSTDCXX:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"

# The measured workload is kept identical to the command supplied by the user.
MEASURE_PROMPTS=( ${MEASURE_PROMPTS:-40 80 96} )
MEASURE_CONCURRENCY=( ${MEASURE_CONCURRENCY:-10 20 24} )
WARMUP_PROMPTS="${WARMUP_PROMPTS:-120}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-8}"
FIXED_OUTPUT_LEN="${FIXED_OUTPUT_LEN:-2048}"
SHUFFLE="${SHUFFLE:-0}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
RESULTS_DIR="${RESULTS_DIR:-${SPEC_FORGE_DIR}/results/dynamic_k_$(date +%Y%m%d_%H%M%S)}"

SERVER_PID=""

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file does not exist: $1" >&2
        exit 1
    fi
}

cleanup_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Stopping server pid=${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}

trap cleanup_server EXIT INT TERM

wait_for_server() {
    local deadline=$((SECONDS + SERVER_START_TIMEOUT))
    while (( SECONDS < deadline )); do
        if curl --fail --silent --show-error "${CLIENT_BASE_URL}/health" >/dev/null 2>&1; then
            return 0
        fi
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "Server exited before becoming healthy. See ${CURRENT_DIR}/server.log" >&2
            return 1
        fi
        sleep 2
    done
    echo "Timed out waiting for ${CLIENT_BASE_URL}/health. See ${CURRENT_DIR}/server.log" >&2
    return 1
}

snapshot_metrics() {
    local name="$1"
    curl --fail --silent --show-error "${CLIENT_BASE_URL}/metrics" > "${CURRENT_DIR}/metrics_${name}.prom"
    grep -E '^sglang:(suffix_|dynamic_k8_|spec_accept_)' "${CURRENT_DIR}/metrics_${name}.prom" \
        > "${CURRENT_DIR}/metrics_${name}_focus.prom" || true
}

run_client() {
    local log_file="$1"
    shift
    local stage_dir="${CURRENT_DIR}/${log_file%.log}_artifacts"
    local resolved_dataset_path="${DATASET_PATH}"
    if [[ "${resolved_dataset_path}" != /* ]]; then
        resolved_dataset_path="${SPEC_FORGE_DIR}/${resolved_dataset_path}"
    fi
    mkdir -p "${stage_dir}"
    local args=(
        python "${TEST_SCRIPT}"
        --base-url "${CLIENT_BASE_URL}"
        --model "${MODEL_PATH}"
        --dataset-name "${DATASET_NAME}"
        --dataset-path "${resolved_dataset_path}"
        --tokenizer-path "${TOKENIZER_PATH}"
        --temperature 0.0
        --fixed-output-len "${FIXED_OUTPUT_LEN}"
        --disable-ignore-eos
    )
    if [[ "${SHUFFLE}" == "1" ]]; then
        args+=(--shuffle)
    fi
    args+=("$@")
    (
        cd "${stage_dir}"
        "${args[@]}"
    ) 2>&1 | tee "${CURRENT_DIR}/${log_file}"
}

start_server() {
    local experiment="$1"
    shift
    CURRENT_DIR="${RESULTS_DIR}/${experiment}"
    mkdir -p "${CURRENT_DIR}"

    local args=(
        python -m sglang.launch_server
        --model-path "${MODEL_PATH}"
        --max-running-requests "${MAX_RUNNING_REQUESTS}"
        --attention-backend "${ATTENTION_BACKEND}"
        --mem-fraction-static "${MEM_FRACTION_STATIC}"
        --tp-size "${TP_SIZE}"
        --host "${HOST}"
        --port "${PORT}"
        --enable-metrics
    )
    if [[ -n "${GPU_IDS}" ]]; then
        args=(env "CUDA_VISIBLE_DEVICES=${GPU_IDS}" "${args[@]}")
    fi
    if [[ -n "${PRELOAD_LIBSTDCXX}" ]]; then
        args=(env "LD_PRELOAD=${PRELOAD_LIBSTDCXX}" "${args[@]}")
    fi
    args+=("$@")

    printf '%q ' "${args[@]}" > "${CURRENT_DIR}/server_command.sh"
    printf '\n' >> "${CURRENT_DIR}/server_command.sh"
    (
        cd "${SGLANG_DIR}"
        "${args[@]}"
    ) > "${CURRENT_DIR}/server.log" 2>&1 &
    SERVER_PID=$!
    wait_for_server
    snapshot_metrics "startup"
}

run_experiment() {
    local experiment="$1"
    shift
    echo "========== ${experiment} =========="
    start_server "${experiment}" "$@"

    # Every configuration gets the same warmup. For suffix configurations this
    # creates prior completed responses in the suffix cache; it also equalizes
    # target prefix-cache state across all four configurations.
    run_client "warmup.log" \
        --num-prompts "${WARMUP_PROMPTS}" \
        --max-concurrency "${WARMUP_CONCURRENCY}"
    snapshot_metrics "after_warmup"

    # This low-concurrency probe is the main dynamic-K measurement. With the
    # default threshold 20, K=8 is only eligible below 20 active requests.
    run_client "k8_probe.log" \
        --num-prompts "${WARMUP_PROMPTS}" \
        --max-concurrency "${WARMUP_CONCURRENCY}"
    snapshot_metrics "after_k8_probe"

    if [[ "${#MEASURE_PROMPTS[@]}" -ne "${#MEASURE_CONCURRENCY[@]}" ]]; then
        echo "MEASURE_PROMPTS and MEASURE_CONCURRENCY must have the same length" >&2
        return 1
    fi
    local i
    for i in "${!MEASURE_CONCURRENCY[@]}"; do
        local concurrency="${MEASURE_CONCURRENCY[$i]}"
        local prompts="${MEASURE_PROMPTS[$i]}"
        run_client "measurement_bs${concurrency}_n${prompts}.log" \
            --num-prompts "${prompts}" \
            --max-concurrency "${concurrency}"
        snapshot_metrics "after_measurement_bs${concurrency}"
    done

    cleanup_server
}

require_file "${TEST_SCRIPT}"
require_file "${SGLANG_DIR}/python/sglang/version.py"
mkdir -p "${RESULTS_DIR}"

cat > "${RESULTS_DIR}/README.txt" <<EOF
Dynamic-K experiment results
============================

Each configuration has a server.log, warmup.log, k8_probe.log,
per-concurrency measurement logs, Prometheus snapshots, and isolated CSV
artifacts under *_artifacts/.

Interpret the counters in metrics_after_k8_probe_focus.prom (filter tp_rank="0"):
  sglang:dynamic_k8_request_total
      Must increase in the dynamic_k4_k8 run. Zero means K=8 never triggered.
  sglang:dynamic_k8_output_token_total / sglang:dynamic_k8_draft_token_total
      K=8 target verification efficiency. A value near 1 is strong; a low
      value means long suffix candidates are being rejected.
  sglang:suffix_override_total / sglang:suffix_proposal_total
      Fraction of suffix proposals that were strong enough to replace K=4
      standalone draft tokens.

Primary comparisons:
  standalone_k4 vs no_speculation: standalone speculative-decoding benefit.
  suffix_static_k4 vs standalone_k4: suffix-cache net benefit/cost.
  dynamic_k4_k8 vs suffix_static_k4: dynamic-K net benefit. Use the individual
  measurement_bs10, measurement_bs20, and measurement_bs24 results. Batches
  >=20 disable K=8 until the decode tail falls below 20.
EOF

run_experiment "no_speculation"
run_experiment "standalone_k4" \
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
    --speculative-algorithm STANDALONE \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4 \
    --speculative-eagle-topk 1
run_experiment "suffix_static_k4" \
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
    --speculative-algorithm STANDALONE \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4 \
    --speculative-eagle-topk 1 \
    --speculative-suffix-enable
run_experiment "dynamic_k4_k8" \
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
    --speculative-algorithm STANDALONE \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4 \
    --speculative-eagle-topk 1 \
    --speculative-suffix-enable \
    --speculative-dynamic-k-enable \
    --speculative-normal-draft-token-num 4 \
    --speculative-long-suffix-draft-token-num 8 \
    --speculative-long-suffix-min-match-len 7 \
    --speculative-high-bs-threshold 20

python "${SGLANG_DIR}/scripts/summarize_dynamic_k_experiment.py" "${RESULTS_DIR}" \
    | tee "${RESULTS_DIR}/dynamic_k_metric_summary.tsv"

echo "Completed. Results: ${RESULTS_DIR}"
