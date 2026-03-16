# Commands for Local Ollama Generation + Evaluation

Run everything from the repository root:

```bash
cd /Users/krishna/Documents/api_guided_testgen
```

## 1) Copy the `out/` folder

If your generated outputs are in another location, copy them into this repo as `./out`:

## 2) Create and activate a Python environment

```bash
python -m venv .demo
source .demo/bin/activate
python -m pip install --upgrade pip
```

## 3) Set .env variables


## 4) Install dependencies

```bash
pip install -r requirements.txt
```

## 5) Generate tests locally with Ollama

Make sure Ollama is running and the model is pulled first:

```bash
ollama serve
ollama pull qwen2.5-coder:7b-instruct
```

Set Ollama env vars and generate:

```bash
export OLLAMA_BASE_URL="http://localhost:11434/v1"
export OLLAMA_MODEL="qwen2.5-coder:7b-instruct"

ITER=local_ollama
LIBS=(tf torch sklearn jax xgb)
BASELINES=(zero_shot basic_rag_all basic_rag_apidoc basic_rag_issues basic_rag_sos api_rag_all api_rag_apidoc api_rag_issues api_rag_sos)

for b in "${BASELINES[@]}"; do
  for l in "${LIBS[@]}"; do
    python api_rag.py "$l" "$b" "$ITER" ollama-small
  done
done
```

## 6) Evaluate generated tests

```bash
ITER=local_ollama
LIBS=(tf torch sklearn jax xgb)
BASELINES=(zero_shot basic_rag_all basic_rag_apidoc basic_rag_issues basic_rag_sos api_rag_all api_rag_apidoc api_rag_issues api_rag_sos)

mkdir -p "logs/${ITER}"
printf "baseline\tlib\tparse\texec\tpass\n" > "logs/${ITER}/metrics.tsv"

for b in "${BASELINES[@]}"; do
  for l in "${LIBS[@]}"; do
    LOG="logs/${ITER}/${b}__${l}.log"
    python evaluate.py "$l" "$b" "$ITER" | tee "$LOG"
    PARSE="$(grep -a '^parse rate:' "$LOG" | tail -1 | sed 's/^parse rate: //')"
    EXEC="$(grep -a '^exec rate:' "$LOG" | tail -1 | sed 's/^exec rate: //')"
    PASS="$(grep -a '^pass rate:' "$LOG" | tail -1 | sed 's/^pass rate: //')"
    printf "%s\t%s\t%s\t%s\t%s\n" "$b" "$l" "$PARSE" "$EXEC" "$PASS" >> "logs/${ITER}/metrics.tsv"
  done
done
```

## 7) Run posthoc test

```bash
python posthoc_test.py > "logs/${ITER}/posthoc_metrics.txt"
```

## 8) Quick output checks

```bash
cat logs/local_ollama/posthoc_metrics.txt
```
