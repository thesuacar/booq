#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[dev-stack] Please 'source venv/bin/activate' first so python/streamlit come from the project venv." >&2
  exit 1
fi

export BOOQ_STORAGE_ROOT=${BOOQ_STORAGE_ROOT:-$ROOT_DIR/storage}
export BOOQ_USE_FAKE_TTS=${BOOQ_USE_FAKE_TTS:-false}
export BOOQ_AI_SERVICE_URL=${BOOQ_AI_SERVICE_URL:-http://127.0.0.1:8101}
export BOOQ_ORCHESTRATOR_URL=${BOOQ_ORCHESTRATOR_URL:-http://127.0.0.1:8100}
export BOOQ_INTERFACE_URL=${BOOQ_INTERFACE_URL:-http://127.0.0.1:8099}
export BOOQ_UI_PORT=${BOOQ_UI_PORT:-8501}
export BOOQ_DEV_STACK_START_UI=${BOOQ_DEV_STACK_START_UI:-true}

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/AI${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$BOOQ_STORAGE_ROOT"

pids=()
free_port() {
  local port=$1
  if command -v lsof >/dev/null 2>&1; then
    local existing
    existing=$(lsof -ti ":$port" || true)
    if [[ -n "$existing" ]]; then
      echo "[dev-stack] freeing port $port (pids: $existing)"
      kill $existing 2>/dev/null || true
    fi
  fi
}

free_port 8101
free_port 8100
free_port 8099
free_port "$BOOQ_UI_PORT"

cleanup() {
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "[dev-stack] stopping pid $pid"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

start_service() {
  local name="$1"
  shift
  echo "[dev-stack] starting $name → $*"
  "$@" &
  local pid=$!
  pids+=("$pid")
  echo "[dev-stack] $name (pid $pid)"
}

start_service ai-service python -m uvicorn services.ai_service:app --port 8101
start_service orchestrator python -m uvicorn services.orchestrator_server:app --port 8100
start_service interface python -m uvicorn services.interface_server:app --port 8099

if [[ "$BOOQ_DEV_STACK_START_UI" != "false" ]]; then
  start_service streamlit streamlit run UI/audiobook_app.py --server.port "$BOOQ_UI_PORT"
else
  echo "[dev-stack] BOOQ_DEV_STACK_START_UI=false → not starting Streamlit UI"
fi

echo "[dev-stack] stack running. Press Ctrl+C to stop everything."
wait "${pids[@]}" 2>/dev/null || true
