#!/usr/bin/env bash
set -euo pipefail

# Compare CUDA-graph memory for FlashInfer and FlashAttention 3 with the same
# standalone speculative-decoding configuration. Each backend runs a fixed-K
# baseline followed by a dynamic-K run that captures both K=4 and K=8 graphs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-/models/models/Qwen/Qwen2.5-72B-Instruct-AWQ}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/models/models/Qwen/Qwen3-0.6B}"
ATTENTION_BACKENDS="${ATTENTION_BACKENDS:-flashinfer fa3}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-48}"
NORMAL_K="${NORMAL_K:-4}"
LONG_K="${LONG_K:-8}"
LOG_DIR="${LOG_DIR:-logs/attention_backend_memory}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-600}"
SHUTDOWN_GRACE_SEC="${SHUTDOWN_GRACE_SEC:-8}"
LD_PRELOAD_PATH="${LD_PRELOAD_PATH:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
READY_PATTERN="max_total_num_tokens=.*available_(gpu_)?mem=|Application startup complete|The server is fired up"

mkdir -p "${LOG_DIR}"

RUN_PID=""

cleanup() {
  if [[ -n "${RUN_PID}" ]] && kill -0 "${RUN_PID}" 2>/dev/null; then
    kill -TERM "-${RUN_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL "-${RUN_PID}" 2>/dev/null || true
    wait "${RUN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_graph_capture() {
  local log_file="$1"
  local started_at
  started_at="$(date +%s)"

  while true; do
    if grep -qE "${READY_PATTERN}" "${log_file}" 2>/dev/null; then
      return 0
    fi

    if grep -qE "Traceback|Received sigquit|Killed$|RuntimeError|AssertionError|ImportError|Exception:" "${log_file}" 2>/dev/null; then
      echo "Detected failure while launching. See ${log_file}" >&2
      return 1
    fi

    if ! kill -0 "${RUN_PID}" 2>/dev/null; then
      echo "Server exited before CUDA graph capture. See ${log_file}" >&2
      return 1
    fi

    if (( "$(date +%s)" - started_at > STARTUP_TIMEOUT_SEC )); then
      echo "Timed out after ${STARTUP_TIMEOUT_SEC}s. See ${log_file}" >&2
      tail -n 80 "${log_file}" >&2 || true
      return 1
    fi

    sleep 2
  done
}

stop_server() {
  if [[ -n "${RUN_PID}" ]] && kill -0 "${RUN_PID}" 2>/dev/null; then
    kill -TERM "-${RUN_PID}" 2>/dev/null || true
    for _ in $(seq 1 "${SHUTDOWN_GRACE_SEC}"); do
      if ! kill -0 "${RUN_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${RUN_PID}" 2>/dev/null; then
      kill -KILL "-${RUN_PID}" 2>/dev/null || true
    fi
    wait "${RUN_PID}" 2>/dev/null || true
  fi
  RUN_PID=""
  sleep 3
}

run_case() {
  local name="$1"
  local command="$2"
  local log_file="${LOG_DIR}/${name}.log"

  echo
  echo "==== Running ${name}; log: ${log_file} ===="
  rm -f "${log_file}"
  setsid bash -c "${command} 2>&1 | tee '${log_file}'" &
  RUN_PID="$!"

  if wait_for_graph_capture "${log_file}"; then
    echo "CUDA graph capture completed for ${name}."
    stop_server
  else
    stop_server
    return 1
  fi
}

logs=()
for attention_backend in ${ATTENTION_BACKENDS}; do
  case "${attention_backend}" in
    flashinfer|fa3) ;;
    *)
      echo "Unsupported backend '${attention_backend}'. Use flashinfer and/or fa3." >&2
      exit 2
      ;;
  esac

  base_command="LD_PRELOAD='${LD_PRELOAD_PATH}' python -m sglang.launch_server \
    --model-path '${MODEL_PATH}' \
    --speculative-draft-model-path '${DRAFT_MODEL_PATH}' \
    --speculative-algorithm STANDALONE \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens ${NORMAL_K} \
    --speculative-eagle-topk 1 \
    --attention-backend ${attention_backend} \
    --sampling-backend pytorch \
    --max-running-requests ${MAX_RUNNING_REQUESTS} \
    --cuda-graph-max-bs ${CUDA_GRAPH_MAX_BS} \
    --tp-size ${TP_SIZE} \
    --host ${HOST} \
    --port ${PORT}"

  static_name="${attention_backend}_static_k${NORMAL_K}"
  run_case "${static_name}" "${base_command}"
  logs+=("${LOG_DIR}/${static_name}.log")

  dynamic_name="${attention_backend}_dynamic_k${NORMAL_K}_${LONG_K}"
  dynamic_command="${base_command} \
    --speculative-suffix-enable \
    --speculative-dynamic-k-enable \
    --speculative-normal-draft-token-num ${NORMAL_K} \
    --speculative-long-suffix-draft-token-num ${LONG_K} \
    --speculative-long-suffix-min-match-len 7 \
    --speculative-high-bs-threshold 10"
  run_case "${dynamic_name}" "${dynamic_command}"
  logs+=("${LOG_DIR}/${dynamic_name}.log")
done

echo
echo "==== CUDA graph memory summary ===="
python scripts/parse_cuda_graph_memory.py "${logs[@]}"
