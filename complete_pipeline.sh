#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load local .env values (if present), including HF_MODEL.
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PY="/Users/krishna/Documents/api_guided_testgen/.demo/bin/python"
ITER_SUFFIX="${1:-}"
MAX_APIS="${2:-3}"
MODEL_ARG="${3:-}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [ -n "$MODEL_ARG" ]; then
  export HF_MODEL="$MODEL_ARG"
elif [ -n "${HF_MODEL:-}" ]; then
  export HF_MODEL="$HF_MODEL"
else
  export HF_MODEL="${HF_MODEL:-Qwen/Qwen2.5-7B}"
fi

HF_MODEL_LC="$(printf '%s' "$HF_MODEL" | tr '[:upper:]' '[:lower:]')"
case "$HF_MODEL_LC" in
  qwen2.5:7b|qwen2.5-coder:7b|qwen2.5-coder:7b-instruct)
    HF_MODEL="Qwen/Qwen2.5-7B"
    ;;
esac

export PATH="/Users/krishna/Documents/api_guided_testgen/.demo/bin:$PATH"

LLM_NAME="$(printf '%s' "$HF_MODEL" | sed 's/[^A-Za-z0-9._-]/_/g')"
if [ -n "$ITER_SUFFIX" ]; then
  ITER="$LLM_NAME/$ITER_SUFFIX"
else
  ITER="$LLM_NAME"
fi

LIBS=(sklearn)
METHODS=(
  similarity
  diversity
  hybrid
  zero_shot
  basic_rag_all
)

if [ ! -d "log/$ITER" ]; then
  echo "ERROR: Missing required directory log/$ITER"
  echo "Run: bash init_project_structure.sh \"$ITER\""
  exit 1
fi
LOG_FILE="log/$ITER/full_alllibs_allapis.log"

echo "ITER=$ITER" | tee -a "$LOG_FILE"
echo "LLM_NAME=$LLM_NAME" | tee -a "$LOG_FILE"
echo "HF_MODEL=$HF_MODEL" | tee -a "$LOG_FILE"
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
    "$PY" api_rag.py "$lib" "$method" "$ITER" "transformers:$HF_MODEL" "$MAX_APIS" 2>&1 | tee -a "$LOG_FILE"
  else
    "$PY" api_rag.py "$lib" "$method" "$ITER" "transformers:$HF_MODEL" 2>&1 | tee -a "$LOG_FILE"
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
