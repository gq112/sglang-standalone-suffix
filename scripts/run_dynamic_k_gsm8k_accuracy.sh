#!/usr/bin/env bash
# Compare GSM8K accuracy between suffix static K=4 and FA3 ragged K=4/8.
#
# The dynamic run first evaluates an interleaved warmup subset (the first five
# few-shot examples plus even-indexed questions). The full original-order run
# then interleaves cached and uncached requests, exercising mixed ragged K.
#
# Example:
#   cd /workspace/sglang-standalone-suffix
#   GPU_IDS=0,1,2,3 TP_SIZE=4 bash scripts/run_dynamic_k_gsm8k_accuracy.sh

set -Eeuo pipefail

SGLANG_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPEC_FORGE_DIR="${SPEC_FORGE_DIR:-/workspace/SpecForge}"
GSM8K_PATH="${GSM8K_PATH:-${SPEC_FORGE_DIR}/gsm8k.jsonl}"
MODEL_PATH="${MODEL_PATH:-/models/models/Qwen/Qwen2.5-72B-Instruct-AWQ}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/models/models/Qwen/Qwen3-0.6B}"

HOST="${HOST:-0.0.0.0}"
CLIENT_HOST="${CLIENT_HOST:-http://127.0.0.1}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-4}"
GPU_IDS="${GPU_IDS:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
NUM_QUESTIONS="${NUM_QUESTIONS:-200}"
PARALLEL="${PARALLEL:-10}"
NUM_SHOTS="${NUM_SHOTS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
PRELOAD_LIBSTDCXX="${PRELOAD_LIBSTDCXX:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
RESULTS_DIR="${RESULTS_DIR:-${SPEC_FORGE_DIR}/results/dynamic_k_gsm8k_$(date +%Y%m%d_%H%M%S)}"

SERVER_PID=""
CURRENT_DIR=""
CLIENT_BASE_URL="${CLIENT_HOST}:${PORT}"

require_file() {
    [[ -f "$1" ]] || { echo "Required file does not exist: $1" >&2; exit 1; }
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
    grep -E '^sglang:(suffix_|dynamic_k|spec_accept_)' "${CURRENT_DIR}/metrics_${name}.prom" \
        > "${CURRENT_DIR}/metrics_${name}_focus.prom" || true
}

start_server() {
    local experiment="$1"
    shift
    CURRENT_DIR="${RESULTS_DIR}/${experiment}"
    mkdir -p "${CURRENT_DIR}"

    local args=(
        python -m sglang.launch_server
        --model-path "${MODEL_PATH}"
        --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
        --speculative-algorithm STANDALONE
        --speculative-num-steps 3
        --speculative-num-draft-tokens 4
        --speculative-eagle-topk 1
        --speculative-suffix-enable
        --max-running-requests "${MAX_RUNNING_REQUESTS}"
        --attention-backend fa3
        --enable-deterministic-inference
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
    snapshot_metrics startup
}

run_eval() {
    local data_path="$1"
    local num_questions="$2"
    local log_file="$3"
    (
        cd "${CURRENT_DIR}"
        python -m sglang.test.few_shot_gsm8k \
            --data-path "${data_path}" \
            --num-questions "${num_questions}" \
            --num-shots "${NUM_SHOTS}" \
            --max-new-tokens "${MAX_NEW_TOKENS}" \
            --parallel "${PARALLEL}" \
            --host "${CLIENT_HOST}" \
            --port "${PORT}" \
            --temperature 0.0
    ) 2>&1 | tee "${CURRENT_DIR}/${log_file}"
}

run_prompt_comparison_client() {
    local data_path="$1"
    local num_prompts="$2"
    local output_path="$3"
    python "${SGLANG_DIR}/scripts/check_greedy_output_consistency.py" run \
        --base-url "${CLIENT_BASE_URL}" \
        --dataset-path "${data_path}" \
        --num-prompts "${num_prompts}" \
        --max-concurrency "${PARALLEL}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --output "${output_path}" | tee "${CURRENT_DIR}/greedy_compare.log"
}

require_file "${GSM8K_PATH}"
require_file "${SGLANG_DIR}/python/sglang/version.py"
require_file "${SGLANG_DIR}/scripts/check_greedy_output_consistency.py"
mkdir -p "${RESULTS_DIR}"

DATASET_MODE="$(python - "${GSM8K_PATH}" <<'PY'
import json
import sys
from pathlib import Path

first_line = next(line for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip())
record = json.loads(first_line)
print("labeled" if "question" in record and "answer" in record else "prompt_only")
PY
)"
echo "Dataset mode: ${DATASET_MODE}"

WARMUP_PATH="${RESULTS_DIR}/gsm8k_warmup_interleaved.jsonl"
WARMUP_QUESTIONS="$(python - "${GSM8K_PATH}" "${WARMUP_PATH}" "${NUM_QUESTIONS}" "${NUM_SHOTS}" <<'PY'
import sys
from pathlib import Path

source, destination, limit, num_shots = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lines = Path(source).read_text(encoding="utf-8").splitlines()
selected = [
    line
    for index, line in enumerate(lines[:limit])
    if index < num_shots or index % 2 == 0
]
Path(destination).write_text("\n".join(selected) + "\n", encoding="utf-8")
print(len(selected))
PY
)"

echo "========== suffix_static_k4 =========="
start_server suffix_static_k4
if [[ "${DATASET_MODE}" == "labeled" ]]; then
    run_eval "${GSM8K_PATH}" "${NUM_QUESTIONS}" accuracy.log
else
    run_prompt_comparison_client "${GSM8K_PATH}" "${NUM_QUESTIONS}" "${CURRENT_DIR}/outputs.jsonl"
fi
snapshot_metrics after_accuracy
cleanup_server

echo "========== ragged_dynamic_k4_k8 =========="
start_server ragged_dynamic_k4_k8 \
    --speculative-dynamic-k-enable \
    --speculative-normal-draft-token-num 4 \
    --speculative-long-suffix-draft-token-num 8 \
    --speculative-long-suffix-min-match-len 7 \
    --speculative-high-bs-threshold 20
if [[ "${DATASET_MODE}" == "labeled" ]]; then
    run_eval "${WARMUP_PATH}" "${WARMUP_QUESTIONS}" warmup_interleaved.log
else
    run_prompt_comparison_client "${WARMUP_PATH}" "${WARMUP_QUESTIONS}" "${CURRENT_DIR}/warmup_outputs.jsonl"
fi
snapshot_metrics after_warmup
if [[ "${DATASET_MODE}" == "labeled" ]]; then
    run_eval "${GSM8K_PATH}" "${NUM_QUESTIONS}" accuracy.log
else
    run_prompt_comparison_client "${GSM8K_PATH}" "${NUM_QUESTIONS}" "${CURRENT_DIR}/outputs.jsonl"
fi
snapshot_metrics after_accuracy
cleanup_server

if [[ "${DATASET_MODE}" == "labeled" ]]; then
python - "${RESULTS_DIR}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

def accuracy(name):
    text = (root / name / "accuracy.log").read_text(encoding="utf-8")
    match = re.search(r"^Accuracy:\s*([\d.]+)", text, re.MULTILINE)
    if not match:
        raise SystemExit(f"Accuracy was not found in {name}/accuracy.log")
    return float(match.group(1))

static = accuracy("suffix_static_k4")
ragged = accuracy("ragged_dynamic_k4_k8")
report = root / "accuracy_comparison.md"
report.write_text(
    "# Ragged Dynamic-K GSM8K Accuracy\n\n"
    "| Config | Accuracy |\n| --- | ---: |\n"
    f"| Suffix static K=4 | {static:.3f} |\n"
    f"| FA3 ragged dynamic K=4/8 | {ragged:.3f} |\n\n"
    f"Delta: {(ragged - static) * 100:+.2f} percentage points.\n\n"
    "Ragged coverage: inspect `ragged_dynamic_k4_k8/metrics_after_accuracy_focus.prom`; "
    "`dynamic_k8_request_total` must be nonzero.\n",
    encoding="utf-8",
)
print(report.read_text(encoding="utf-8"), end="")
PY
else
    python "${SGLANG_DIR}/scripts/check_greedy_output_consistency.py" compare \
        --reference "${RESULTS_DIR}/suffix_static_k4/outputs.jsonl" \
        --candidate "${RESULTS_DIR}/ragged_dynamic_k4_k8/outputs.jsonl" \
        --report "${RESULTS_DIR}/greedy_output_comparison.md"
fi

echo "Completed. Results: ${RESULTS_DIR}"
