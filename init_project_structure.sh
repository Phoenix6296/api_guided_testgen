#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LIBS=(tf torch sklearn xgb jax)
BASELINES=(
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

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

default_model="${HF_MODEL:-Qwen/Qwen2.5-7B}"
default_iter="$(printf '%s' "$default_model" | sed 's/[^A-Za-z0-9._-]/_/g')"
ITER_NAME="${1:-$default_iter}"

echo "Creating data/api_db scaffolding..."
mkdir -p data/api_db
mkdir -p api_db

echo "Creating out/$ITER_NAME directory tree..."
mkdir -p "out/$ITER_NAME"/{coverage,plots}

echo "Creating log directory tree..."
mkdir -p "log/$ITER_NAME"

for baseline in "${BASELINES[@]}"; do
  for lib in "${LIBS[@]}"; do
    mkdir -p "out/$ITER_NAME/prompt/$baseline/$lib"
    mkdir -p "out/$ITER_NAME/generated/$baseline/$lib"
    mkdir -p "out/$ITER_NAME/exec/$baseline/$lib"
    mkdir -p "out/$ITER_NAME/cov/$baseline/$lib"
  done
done

echo "Done."
echo "Created directories: data/api_db, api_db, out/$ITER_NAME, and log/$ITER_NAME"
