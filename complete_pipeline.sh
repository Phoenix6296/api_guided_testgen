#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load local .env values (if present), including MODEL_NAME.
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PY="/Users/krishna/Documents/api_guided_testgen/.demo/bin/python"
ITER_SUFFIX="${1:-}"
MAX_APIS="${2:-}"
MODEL_ARG="${3:-}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-300}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"

if [ -n "$MODEL_ARG" ]; then
  export OLLAMA_MODEL="$MODEL_ARG"
elif [ -n "${MODEL_NAME:-}" ]; then
  export OLLAMA_MODEL="$MODEL_NAME"
else
  export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5-coder:7b-instruct}"
fi

export OLLAMA_AUTO_PULL="${OLLAMA_AUTO_PULL:-true}"
export PATH="/Users/krishna/Documents/api_guided_testgen/.demo/bin:$PATH"

LLM_NAME="$(printf '%s' "$OLLAMA_MODEL" | sed 's/[^A-Za-z0-9._-]/_/g')"
if [ -n "$ITER_SUFFIX" ]; then
  ITER="$LLM_NAME/$ITER_SUFFIX"
else
  ITER="$LLM_NAME"
fi

LIBS=(tf torch sklearn xgb jax)
METHODS=(
  basic_rag_all
  basic_rag_apidoc
  basic_rag_sos
  basic_rag_issues
  basic_rag_repos
  similarity
  zero_shot
  api_rag_all
  api_rag_apidoc
  api_rag_issues
  api_rag_sos
  api_rag_repos
)

mkdir -p "log/$ITER"
LOG_FILE="log/$ITER/full_alllibs_allapis.log"

ensure_ollama_ready() {
  local base_url
  local server_url

  base_url="$OLLAMA_BASE_URL"
  server_url="${base_url%/v1}"

  if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: ollama command not found in PATH." | tee -a "$LOG_FILE"
    exit 1
  fi

  if curl -fsS "$server_url/api/tags" >/dev/null 2>&1; then
    echo "Ollama server already running at $server_url" | tee -a "$LOG_FILE"
  else
    echo "Ollama server not reachable. Starting ollama serve..." | tee -a "$LOG_FILE"
    nohup ollama serve > "log/$ITER/ollama_serve.log" 2>&1 &

    local ok=0
    for _ in $(seq 1 30); do
      sleep 1
      if curl -fsS "$server_url/api/tags" >/dev/null 2>&1; then
        ok=1
        break
      fi
    done

    if [ "$ok" -ne 1 ]; then
      echo "ERROR: Could not start/connect to Ollama at $server_url" | tee -a "$LOG_FILE"
      exit 1
    fi

    echo "Ollama server started at $server_url" | tee -a "$LOG_FILE"
  fi

  if ollama list | awk '{print $1}' | grep -Fx "$OLLAMA_MODEL" >/dev/null 2>&1; then
    echo "Ollama model available: $OLLAMA_MODEL" | tee -a "$LOG_FILE"
  else
    if [ "$OLLAMA_AUTO_PULL" = "true" ]; then
      echo "Model $OLLAMA_MODEL not found locally. Pulling..." | tee -a "$LOG_FILE"
      ollama pull "$OLLAMA_MODEL" 2>&1 | tee -a "$LOG_FILE"
    else
      echo "ERROR: Model $OLLAMA_MODEL not found and OLLAMA_AUTO_PULL=false" | tee -a "$LOG_FILE"
      exit 1
    fi
  fi
}

echo "ITER=$ITER" | tee -a "$LOG_FILE"
echo "LLM_NAME=$LLM_NAME" | tee -a "$LOG_FILE"
echo "OLLAMA_MODEL=$OLLAMA_MODEL" | tee -a "$LOG_FILE"
if [ -n "$MAX_APIS" ]; then
  echo "MAX_APIS=$MAX_APIS (limited run)" | tee -a "$LOG_FILE"
else
  echo "MAX_APIS=ALL (full run)" | tee -a "$LOG_FILE"
fi

echo "=== PRECHECK: OLLAMA ===" | tee -a "$LOG_FILE"
ensure_ollama_ready

echo "=== STEP 1: BUILD data/api_db + api_db ===" | tee -a "$LOG_FILE"
"$PY" rebuild_api_db_from_lists.py 2>&1 | tee -a "$LOG_FILE"

run_pair() {
  local lib="$1"
  local method="$2"

  echo "=== GENERATE $lib $method ===" | tee -a "$LOG_FILE"
  if [ -n "$MAX_APIS" ]; then
    "$PY" api_rag.py "$lib" "$method" "$ITER" ollama-small "$MAX_APIS" 2>&1 | tee -a "$LOG_FILE"
  else
    "$PY" api_rag.py "$lib" "$method" "$ITER" ollama-small 2>&1 | tee -a "$LOG_FILE"
  fi

  echo "=== EVALUATE $lib $method ===" | tee -a "$LOG_FILE"
  if [ -n "$MAX_APIS" ]; then
    "$PY" evaluate.py "$lib" "$method" "$ITER" "$MAX_APIS" 2>&1 | tee -a "$LOG_FILE"
  else
    "$PY" evaluate.py "$lib" "$method" "$ITER" 2>&1 | tee -a "$LOG_FILE"
  fi

  echo "=== COVERAGE $lib $method ===" | tee -a "$LOG_FILE"
  "$PY" coverage.py "$lib" "$method" "$ITER" 2>&1 | tee -a "$LOG_FILE"
}

echo "=== STEP 2: GENERATE + EVALUATE + COVERAGE ===" | tee -a "$LOG_FILE"
for lib in "${LIBS[@]}"; do
  for method in "${METHODS[@]}"; do
    run_pair "$lib" "$method"
  done
done

echo "FULL_ALLLIBS_ALLAPIS_DONE" | tee -a "$LOG_FILE"
