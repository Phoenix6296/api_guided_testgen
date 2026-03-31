#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load local .env values (if present), including MODEL_NAME/OLLAMA_MODEL.
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

ITER_SUFFIX="${1:-}"
MODEL_ARG="${2:-}"

if [ -n "$MODEL_ARG" ]; then
  OLLAMA_MODEL="$MODEL_ARG"
elif [ -n "${MODEL_NAME:-}" ]; then
  OLLAMA_MODEL="$MODEL_NAME"
else
  OLLAMA_MODEL="${OLLAMA_MODEL:-gpt-oss:20b}"
fi

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
mkdir -p "out/$ITER/coverage"

for method in "${METHODS[@]}"; do
  for lib in "${LIBS[@]}"; do
    mkdir -p "out/$ITER/generated/$method/$lib"
    mkdir -p "out/$ITER/prompt/$method/$lib"
    mkdir -p "out/$ITER/exec/$method/$lib"
  done
done

echo "Created required folders for ITER=$ITER"
