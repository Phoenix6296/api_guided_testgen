# Quick Start: Local Transformers End-to-End

This guide runs the project fully local using Hugging Face Transformers, including:

1. Creating required `.jsonl` files
2. Generating tests
3. Evaluating tests
4. Coverage and utility metrics

All commands should be run from the repository root.

## 1) Create and activate Python environment

```bash
cd /path/to/api_guided_testgen

python -m venv .demo
source .demo/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Set model in .env

The full pipeline script reads `.env` automatically and uses `HF_MODEL`.
Current default in this repo:

```dotenv
HF_MODEL=Qwen/Qwen2.5-7B
```

You can change this value any time before starting a run.

## 3) Optional environment variables for local generation

```bash
source .demo/bin/activate

export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1
```

You can also set model and generation length in `.env`, for example:

```dotenv
HF_MODEL=Qwen/Qwen2.5-7B
HF_MAX_NEW_TOKENS=768
```

## 4) Run full pipeline (generation + evaluation + coverage + util)

### Recommended: one-command pipeline via complete_pipeline.sh

```bash
bash complete_pipeline.sh
```

