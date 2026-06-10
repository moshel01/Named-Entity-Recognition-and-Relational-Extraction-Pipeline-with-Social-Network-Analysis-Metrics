#!/usr/bin/env bash
# Run the Abel-papers pipeline with a given Ollama model and print a summary.
#
#   bash scripts/run_ollama_test.sh gemma4:26b 20
#   bash scripts/run_ollama_test.sh qwen3.6:27b 20
#
# Prereqs on this machine:
#   - The code (git pull) AND the data: copy data/abel_papers/ here (RTFs +
#     "Nazi metadata.xlsx") - it is gitignored, so it is NOT in the repo.
#   - pip install -r requirements.txt ; python -m spacy download de_core_news_lg
#   - Ollama installed and the model pullable.
set -uo pipefail
MODEL="${1:?usage: run_ollama_test.sh <ollama-model> [limit]}"
LIMIT="${2:-20}"
cd "$(dirname "$0")/.." || exit 1

[ -d data/abel_papers ] || { echo "ERROR: data/abel_papers/ not found - copy the Abel data here first."; exit 1; }

echo "[1/3] Ensuring ollama + model '$MODEL'..."
curl -s http://localhost:11434/api/tags >/dev/null 2>&1 || { nohup ollama serve >/tmp/ollama_serve.log 2>&1 & sleep 6; }
ollama list | grep -q "$MODEL" || ollama pull "$MODEL"

# Prefer the project venv interpreter if present.
PY=python
[ -x .venv/Scripts/python.exe ] && PY=.venv/Scripts/python.exe
[ -x .venv/bin/python ] && PY=.venv/bin/python

TAG=$(echo "$MODEL" | tr ':/.' '___')
LOG="run_${TAG}.log"
echo "[2/3] Running $MODEL on $LIMIT docs -> output/abel_${TAG}/ (log: $LOG)..."
# --resume is a no-op on a fresh run; on a rerun it skips already-extracted
# docs (the checkpoint is per run-name, so it never mixes models).
PYTHONUTF8=1 "$PY" main.py --config domain/nazi_era/config_nazi_era.yaml \
  --mode ollama --ollama-model "$MODEL" --run-name "abel_${TAG}" --limit "$LIMIT" \
  --resume > "$LOG" 2>&1
echo "    exit code: $?"

echo "[3/3] === SUMMARY ($MODEL) ==="
grep -aiE "Aggregated|oversized|Built graph|Graph QA|Done|Error" "$LOG" | grep -av compatible | tail -6
PYTHONUTF8=1 "$PY" - "output/abel_${TAG}/" <<'PY'
import csv, collections, sys, os
d = sys.argv[1]
if not os.path.exists(d+'gephi_nodes.csv'):
    print("no output found at", d); raise SystemExit
N=list(csv.DictReader(open(d+'gephi_nodes.csv',encoding='utf-8')))
E=list(csv.DictReader(open(d+'gephi_edges.csv',encoding='utf-8')))
tc=collections.Counter(r['tie_class'] for r in E)
pol=collections.Counter(r['polarity'] for r in E)
ns=[n for n in N if n['Label']=='NSDAP']
print("nodes:", len(N), "| edges:", len(E), "| persons:", sum(1 for r in N if r['type']=='PERSON'))
print("tie_class:", dict(tc))
print("polarity:", dict(pol))
print("NSDAP aliases:", len(ns[0]['aliases'].split(';')) if ns and ns[0]['aliases'] else 0)
PY
echo ""
echo "Run again with the other model, then compare the two summaries."
