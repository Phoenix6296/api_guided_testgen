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
MAX_APIS="${2:-20}"
MODEL_ARG="${3:-}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_MAX_NEW_TOKENS="${HF_MAX_NEW_TOKENS:-768}"
export HF_DO_SAMPLE="${HF_DO_SAMPLE:-false}"
export HF_TEMPERATURE="${HF_TEMPERATURE:-0.0}"
export HF_TOP_P="${HF_TOP_P:-0.95}"
export HF_TIMEOUT="${HF_TIMEOUT:-120}"

if [ -n "$MODEL_ARG" ]; then
  export HF_MODEL_ID="$MODEL_ARG"
elif [ -n "${MODEL_NAME:-}" ]; then
  export HF_MODEL_ID="$MODEL_NAME"
else
  export HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
fi

if [ "$HF_MODEL_ID" = "qwen2.5-coder:7b-instruct" ] || [ "$HF_MODEL_ID" = "qwen2.5:7b-instruct" ] || [ "$HF_MODEL_ID" = "qwen2.5-7b" ]; then
  export HF_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
fi

if [ -z "${HF_API_TOKEN:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
  export HF_API_TOKEN="$HF_TOKEN"
fi

export PATH="/Users/krishna/Documents/api_guided_testgen/.demo/bin:$PATH"

LLM_NAME="$(printf '%s' "$HF_MODEL_ID" | sed 's/[^A-Za-z0-9._-]/_/g')"
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
  diversity
  hybrid
  zero_shot
  api_rag_all
  api_rag_apidoc
  api_rag_issues
  api_rag_sos
  api_rag_repos
)

mkdir -p "log/$ITER"
LOG_FILE="log/$ITER/full_alllibs_allapis.log"

echo "ITER=$ITER" | tee -a "$LOG_FILE"
echo "LLM_NAME=$LLM_NAME" | tee -a "$LOG_FILE"
echo "HF_MODEL_ID=$HF_MODEL_ID" | tee -a "$LOG_FILE"
if [ -n "${HF_API_TOKEN:-}" ] || [ -n "${HUGGINGFACEHUB_API_TOKEN:-}" ] || [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_API_TOKEN=SET" | tee -a "$LOG_FILE"
else
  echo "ERROR: Set HF_API_TOKEN (or HUGGINGFACEHUB_API_TOKEN) for hosted Hugging Face inference." | tee -a "$LOG_FILE"
  exit 1
fi
if [ -n "$MAX_APIS" ]; then
  echo "MAX_APIS=$MAX_APIS (limited run)" | tee -a "$LOG_FILE"
else
  echo "MAX_APIS=ALL (full run)" | tee -a "$LOG_FILE"
fi

echo "=== STEP 1: BUILD data/api_db + api_db ===" | tee -a "$LOG_FILE"
"$PY" rebuild_api_db_from_lists.py 2>&1 | tee -a "$LOG_FILE"

run_pair() {
  local lib="$1"
  local method="$2"

  echo "=== GENERATE $lib $method ===" | tee -a "$LOG_FILE"
  if [ -n "$MAX_APIS" ]; then
    "$PY" api_rag.py "$lib" "$method" "$ITER" "$HF_MODEL_ID" "$MAX_APIS" 2>&1 | tee -a "$LOG_FILE"
  else
    "$PY" api_rag.py "$lib" "$method" "$ITER" "$HF_MODEL_ID" 2>&1 | tee -a "$LOG_FILE"
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
