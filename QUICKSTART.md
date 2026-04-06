# Quick Start: Hosted Hugging Face End-to-End

This guide runs the project with hosted Hugging Face inference (no local model download), including:

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

## 2) Set model and token in .env

The full pipeline script reads `.env` automatically and uses `MODEL_NAME`.
Current default in this repo:

```dotenv
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
HF_API_TOKEN=your_huggingface_token
```

You can change this value any time before starting a run.

## 3) Set environment variables for hosted generation

```bash
source .demo/bin/activate

export HF_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
export HF_MAX_NEW_TOKENS="768"
export HF_DO_SAMPLE="false"
export HF_TEMPERATURE="0.0"
export HF_TOP_P="0.95"
export HF_TIMEOUT="120"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1
```

Note: `complete_pipeline.sh` sets `HF_MODEL_ID` from
`MODEL_NAME` in `.env` (or from CLI arg 3 if provided), so you usually do not
need to export `HF_MODEL_ID` manually. It also requires `HF_API_TOKEN`
(or `HUGGINGFACEHUB_API_TOKEN`) to call hosted inference.

## 4) Run full pipeline (generation + evaluation + coverage + util)

### Recommended: one-command pipeline via complete_pipeline.sh

```bash
bash complete_pipeline.sh
```

