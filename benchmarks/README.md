# Benchmarks

Score the pipeline against standard information-extraction datasets. Each adapter
converts a dataset into pipeline inputs + a gold file, then the evaluation
harness reports entity and relation P/R/F1.

## Availability

| Dataset | Source | Status |
|---------|--------|--------|
| **Re-DocRED** | `tonytan48/Re-DocRED` (HF) | ✅ auto-downloads, document-level |
| **DWIE** | `DFKI-SLT/DWIE` (HF) | ✅ auto-downloads, document-level |
| **CLEF HIPE-2022** | `hipe-eval/HIPE-2022-data` (GitHub) | ✅ auto-downloads; German historical newspapers, **NER only** (no relation gold). Uses `de_core_news_lg` + `gliner2-multi` automatically. Closest public proxy for the Abel corpus. |
| **DialogRE** | `nlpdata/dialogre` (GitHub) | ✅ auto-downloads; interpersonal relations in dialogue. "Speaker N" slots get deterministic per-dialogue names ("Alan Abbott") in text and gold, and every speaker counts as a gold PERSON (gold pairs alone are non-exhaustive). Pair with `--constrain-relations`. |
| **GermEval 2014** | `gwlms/germeval2014` (HF) | ✅ auto-downloads; modern German NER (news/wiki), **NER only**. Pairs with HIPE: modern + historical German bracket the Abel register. Sentences grouped into ~25-sentence pseudo-docs. |
| **CoNLL-2003** | `eriktks/conll2003` (HF) | ✅ auto-downloads (parquet); English newswire, **NER only** (PER/ORG/LOC). The classic baseline. |
| **OntoNotes 5.0** | `tner/ontonotes5` (HF) | ✅ auto-downloads (parquet); English multi-genre (news/web/broadcast), **NER only**. Scores PERSON/ORG/GPE+LOC. Tag map read from the dataset README. |
| **WNUT-17** | `tner/wnut2017` (HF) | ✅ auto-downloads (parquet); noisy user text (Twitter/Reddit/YouTube), **NER only**. Emerging/OOV entities - a recall stress test, the opposite register from CoNLL. |
| **Universal NER** | github.com/UniversalNER | ⚠️ HF repo is script-only (won't load on datasets 4.x). Download a treebank `.iob2` and pass it via `--path` (PER/ORG/LOC, many languages). Set `--spacy-model` to the language. |
| **ACE 2005** | LDC2006T06 | ⚠️ license required; not on HF - bring local JSON |
| **TACRED / TAC-KBP** | LDC2018T24 | ⚠️ license required; bring local JSON, sentence-level |

> NER datasets that once shipped as loading scripts (CoNLL-2003, tner's
> OntoNotes/WNUT, GermEval) no longer load by id under `datasets` 4.x. The
> adapters pull the auto-converted **parquet** from each dataset's
> `refs/convert/parquet` branch; ClassLabel names survive when the source
> declared them (CoNLL), and the tner mirrors' raw-int tag maps are scraped from
> the dataset README (a confirmed fallback map covers a README format change). If
> a mirror moves, override the HF id with `--path <hf-id>`.

## One-command runs (HF datasets)

```powershell
# Prepare + run + evaluate, 50 Re-DocRED test docs, offline rules mode:
python -m benchmarks.run_benchmark --dataset redocred --split test --limit 50 --run --eval

# DWIE with a local LLM (better relations):
python -m benchmarks.run_benchmark --dataset dwie --limit 50 --mode ollama `
    --ollama-model qwen2.5:7b-instruct --run --eval

# English NER baselines (entities only, offline):
python -m benchmarks.run_benchmark --dataset conll2003 --limit 50 --run --eval
python -m benchmarks.run_benchmark --dataset ontonotes5 --limit 50 --run --eval
python -m benchmarks.run_benchmark --dataset wnut17 --limit 50 --run --eval

# Universal NER: local .iob2 (download a treebank from github.com/UniversalNER):
python -m benchmarks.run_benchmark --dataset uner --path en_ewt-ud-test.iob2 --limit 50 --run --eval
```

Drop `--run --eval` to only **prepare** the data (inputs + gold + a tuned
config); the command then prints the exact `RUN:` and `EVAL:` commands to run
yourself.

### Useful flags

| Flag | Effect |
|------|--------|
| `--types PERSON,ORG,LOCATION` | Score only these entity types (default per dataset). Trims both the GLiNER labels the pipeline is given **and** the gold, so the metric is apples-to-apples. Dropping noisy types (NUM/MISC/DATE) raises the headline F1. |
| `--constrain-relations` | Inject the dataset's relation inventory as an ontology so the **LLM** emits those exact labels. Makes **typed** relation F1 comparable. Meaningful only with `--mode ollama`/`api`. All adapters now expose readable labels (Re-DocRED `Pxxx` codes are mapped to names via `REL_INFO`, e.g. `p17 -> country`). |
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
- **Relations - use the UNTYPED metric for cross-system comparison.** Each
  dataset has its own relation inventory; the unconstrained pipeline emits
  free-text / dependency relation types, so **typed** relation F1 is *not*
  meaningful unless you run with `--constrain-relations` (LLM mode), which
  injects the dataset's labels as the extraction ontology. Without it expect
  typed F1 near zero by construction. `relations_untyped` measures whether the
  right entity *pairs* are connected.
- **Tiers** - `conservative` (text-stated edges only) is the precision-oriented
  headline. `moderate` adds domain signal-detected membership (`canonical_inferred`)
  - none in the generic benchmark path, so it equals `conservative` here. `full`
  adds the co-occurrence layer (window proximity + cross-doc co-mention; higher
  recall, lower precision). Compare them to see co-occurrence's effect.

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
