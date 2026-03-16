# Quick Start: Local Ollama End-to-End

This guide runs the project fully local (Ollama only), including:

1. Creating required `.jsonl` files
2. Generating tests
3. Evaluating tests
4. Coverage and utility metrics

All commands should be run from the repository root.

## 1) Create and activate Python environment

```bash
cd /Users/krishna/Documents/api_guided_testgen

python -m venv .demo
source .demo/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Set model in .env

The full pipeline script reads `.env` automatically and uses `MODEL_NAME`.
Current default in this repo:

```dotenv
MODEL_NAME=qwen2.5-coder:7b-instruct
```

You can change this value any time before starting a run.

## 3) Ensure Ollama is running and model is available

```bash
ollama serve
```

In another terminal (optional pre-pull):

```bash
ollama pull qwen2.5-coder:7b-instruct
ollama list
```

## 4) Set environment variables for local generation

```bash
source .demo/bin/activate

export OLLAMA_BASE_URL="http://localhost:11434/v1"
export OLLAMA_TIMEOUT="300"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1
```

Note: `complete_pipeline.sh` sets `OLLAMA_MODEL` from
`MODEL_NAME` in `.env` (or from CLI arg 3 if provided), so you usually do not
need to export `OLLAMA_MODEL` manually.

## 5) Run full pipeline (generation + evaluation + coverage + util)

### Recommended: one-command pipeline via complete_pipeline.sh

```bash
bash complete_pipeline.sh
```

