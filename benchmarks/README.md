# Benchmarks

Score the pipeline against standard information-extraction datasets. Each adapter
converts a dataset into pipeline inputs + a gold file, then the evaluation
harness reports entity and relation P/R/F1.

## Availability

| Dataset | Source | Status |
|---------|--------|--------|
| **Re-DocRED** | `tonytan48/Re-DocRED` (HF) | ✅ auto-downloads, document-level |
| **DWIE** | `DFKI-SLT/DWIE` (HF) | ✅ auto-downloads, document-level |
| **ACE 2005** | LDC2006T06 | ⚠️ license required; not on HF - bring local JSON |
| **TACRED / TAC-KBP** | LDC2018T24 | ⚠️ license required; bring local JSON, sentence-level |

## One-command runs (HF datasets)

```powershell
# Prepare + run + evaluate, 50 Re-DocRED test docs, offline rules mode:
python -m benchmarks.run_benchmark --dataset redocred --split test --limit 50 --run --eval

# DWIE with a local LLM (better relations):
python -m benchmarks.run_benchmark --dataset dwie --limit 50 --mode ollama `
    --ollama-model qwen2.5:7b-instruct --run --eval
```

Drop `--run --eval` to only **prepare** the data (inputs + gold + a tuned
config); the command then prints the exact `RUN:` and `EVAL:` commands to run
yourself.

### Useful flags

| Flag | Effect |
|------|--------|
| `--types PERSON,ORG,LOCATION` | Score only these entity types (default per dataset). Trims both the GLiNER labels the pipeline is given **and** the gold, so the metric is apples-to-apples. Dropping noisy types (NUM/MISC/DATE) raises the headline F1. |
| `--constrain-relations` | Inject the dataset's relation inventory as an ontology so the **LLM** emits those exact labels. Makes **typed** relation F1 comparable. Meaningful only with `--mode ollama`/`api` and readable labels (DWIE good; Re-DocRED uses opaque `Pxxx` codes, so skip it there). |
| `--mode ollama --ollama-model qwen2.5:7b-instruct` | Use a local LLM for relation extraction. |

## ACE 2005 / TACRED (local, licensed)

After obtaining the data from the LDC and preprocessing to JSON:

```powershell
# ACE2005 in OneIE/DyGIE JSONL:
python -m benchmarks.run_benchmark --dataset ace2005 --path ace_test.jsonl --limit 100 --run --eval
# TACRED standard JSON:
python -m benchmarks.run_benchmark --dataset tacred --path tacred_test.json --limit 500 --run --eval
```

See the docstrings in `ace2005.py` / `tacred.py` for the exact expected JSON
shape (small, easy to adjust if your preprocessing differs).

## What gets written

```
data/bench/
  <dataset>.gold.json          # gold annotations (entity clusters + relations)
  <dataset>.config.yaml        # pipeline config tuned for this dataset
  <dataset>/inputs/*.txt       # one plaintext file per document
  <dataset>/output/<run>/      # pipeline output (entities.json, gephi_edges.csv, ...)
  <dataset>.report.<tier>.json # eval reports (conservative / moderate / full)
```

The benchmark config uses the **generic domain** with coreference, canonical
inference, and mandatory-membership all **off** - those are Abel-specific helpers
that would add non-gold nodes/edges and depress precision here.

## Reading the numbers (important)

- **Entity P/R/F1** - directly comparable to published numbers. Matching is
  alias-aware and entity-linking-based (gold mention clusters -> one node). Look
  at `entities` (typed) and the per-type table.
- **Relations - use the UNTYPED metric for cross-system comparison.** Re-DocRED
  labels are Wikidata property ids (`P17`, `P131`...) and DWIE/ACE/TACRED use their
  own inventories; the pipeline emits free-text / dependency relation types, so
  **typed** relation F1 against these datasets is *not* meaningful unless you
  constrain extraction to the dataset's label set (LLM mode + a custom prompt).
  `relations_untyped` measures whether the right entity *pairs* are connected.
- **Tiers** - `conservative` (text-stated edges only) is the precision-oriented
  headline; `moderate` adds co-occurrence (higher recall, lower precision);
  `full` adds inference. Compare all three.

### Rough expectations (orientation, not targets)
- `python_only`: solid entity recall (GLiNER), modest untyped-relation recall
  (dependency + co-occurrence), good precision at the `conservative` tier.
- `ollama` / `api`: higher relation recall and better entity boundaries.
- To improve entity recall on these English datasets, keep
  `gliner_model: urchade/gliner_large-v2.1` and tune `gliner_threshold`.

## Caveats

- Typed relation comparison needs label alignment (future work: an LLM prompt
  constrained to the dataset's relation inventory).
- Detokenized text (Re-DocRED/ACE/TACRED are tokenized) reads slightly unnaturally;
  this can marginally affect NER vs. the original raw text.
- DWIE on HF exposes a single `train` split (~700 docs).
