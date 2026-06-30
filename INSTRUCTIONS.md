# INSTRUCTIONS - NER + SNA Extraction Pipeline

Everything you need to run the pipeline in any of the **3 modes**, on **English**
or **German (NSDAP / Abel Papers)** text, and how to adapt it to **other inputs**.

> If you only read one thing: install deps + models (§1), then run
> `python main.py --config domain/nazi_era/config_nazi_era.yaml --limit 2` to
> confirm it works on the bundled German samples.

---

## Table of contents
1. [Install](#1-install)
2. [The mental model](#2-the-mental-model)
3. [Quick start (bundled samples)](#3-quick-start-bundled-samples)
4. [The three modes](#4-the-three-modes)
5. [Running on English](#5-running-on-english)
6. [Running on German (NSDAP domain)](#6-running-on-german-nsdap-domain)
7. [Understanding the output](#7-understanding-the-output)
8. [Coreference & the narrator](#8-coreference--the-narrator)
9. [The mandatory-membership assumption](#9-the-mandatory-membership-assumption)
10. [Edge-source sensitivity analysis](#10-edge-source-sensitivity-analysis)
11. [Evaluating against gold data](#11-evaluating-against-gold-data)
12. [Adapting to other inputs / languages / domains](#12-adapting-to-other-inputs--languages--domains)
13. [CLI reference](#13-cli-reference)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Install

### Step 1 - Python packages
```powershell
# From the project root (Windows PowerShell shown; bash is analogous)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2 - ML models

There are **four** model families. Two are explicit downloads (spaCy); the other
three pull automatically from Hugging Face the first time you run.

**(a) spaCy** - explicit, pick per language:
```powershell
python -m spacy download en_core_web_trf       # English, best quality
python -m spacy download en_core_web_sm         # English, light/fast (fallback)
python -m spacy download de_core_news_lg        # German (for the NSDAP domain)
# Other languages, e.g.: fr_core_news_lg, es_core_news_lg, it_core_news_lg
```

**(b) GLiNER2** - automatic on first run; chosen by `foundation.gliner_model`.
The original GLiNER (urchade/*) is deprecated and no longer loads:
```text
Multilingual (default) .. fastino/gliner2-multi-v1   (100+ langs, mDeBERTa; EN+German)
English-only, heavier ... fastino/gliner2-large-v1   (best English NER; ~2x the load)
```

**(c) sentence-transformers** - automatic; only used by `mode: python_only`.
Set `intelligence.python_only.embedding_model` (the NSDAP config uses a
multilingual one). Default English: `all-MiniLM-L6-v2`.

**(d) fastcoref** - ONLY needed if you set `coreference.pronoun_resolution: true`:
```powershell
pip install fastcoref
```
The default **narrator/first-person** coreference needs nothing extra.

> The first run downloads (b)/(c)/(d) - expect a few minutes and a few GB on a
> cold cache. They are cached under `~/.cache/huggingface`; subsequent runs are
> offline-capable. For GPU, install a CUDA-enabled PyTorch and set
> `foundation.device: cuda` (and `coreference.device: cuda`).

### One-shot copy-paste (English + German, everything)
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_trf
python -m spacy download de_core_news_lg
pip install fastcoref            # optional (third-person coref)
# GLiNER + embeddings download on first pipeline run.
```

### Verify install (no models needed)
```powershell
python main.py --help                 # CLI loads => core packages OK
python -m evaluation.evaluate --help  # eval harness loads (stdlib only)
```
Then confirm the models load with a tiny real run (downloads GLiNER on first use):
```powershell
python main.py --config domain/nazi_era/config_nazi_era.yaml --limit 1 -v
```

---

## 2. The mental model

```
documents ─► FOUNDATION (always): spaCy + GLiNER + coref + dates
          ─► INTELLIGENCE (mode-dependent): relationships + refined entities
          ─► POST-PROCESS: aggregate ► dedup ► quality ► inference ► tag ► graph ► export
```

- **Foundation always runs** (GLiNER + spaCy). The *mode* only changes how
  relationships are found.
- A **domain** (e.g. `nazi_era`) plugs in aliases, labels, patterns, prompts,
  and inference rules without touching core code.
- Everything is **checkpointed** - re-run with `--resume` after an interruption.
  Documents whose extraction failed outright (timeout, unrepairable LLM output)
  are retried on `--resume`; completed ones are skipped.

---

## 3. Quick start (bundled samples)

Two English letters live in `data/sample_en/`; two German biograms in
`data/abel_papers/`.

```powershell
# German NSDAP domain, fully offline, first 2 docs:
python main.py --config domain/nazi_era/config_nazi_era.yaml --limit 2

# English generic:
#   1) copy the template and point it at the English samples
copy config_template.yaml config.yaml
#   2) edit config.yaml -> io.input_path: "./data/sample_en"
python main.py --config config.yaml
```

Look in `output/<run_name>/` for `gephi_nodes.csv`, `gephi_edges.csv`,
`network.gexf`, `entities.json`, `timeline.csv`.

---

## 4. The modes

Set `mode:` in the config, or override with `--mode`.

### Mode `python_only` (offline, no API, no GPU needed)
- Relationships from spaCy dependency parses + sentence co-occurrence +
  sentence-transformer similarity.
- **Best for:** privacy, cost, reproducibility, large batches.
- **Setup:** nothing beyond §1. (Embeddings model auto-downloads.)
- **Trade-off:** precision-oriented; lower relation recall than LLM modes.

### Mode `api` (Claude / OpenAI / Bedrock / any OpenAI-compatible host)
- Relationships + entity refinement + quality review by a hosted LLM. NER stays
  local/free (GLiNER); only relation extraction hits the API.
- **Setup:** set the key env var named in `intelligence.api.api_key_env`:
  ```powershell
  $env:ANTHROPIC_API_KEY = "sk-ant-..."      # or OPENAI_API_KEY, etc.
  ```
  Then in config: `intelligence.api.provider` (`anthropic|openai|bedrock`) and
  `model`. For Bedrock set `aws_region` and use standard AWS credentials.
- **Cheap path (recommended for cost):** point `provider: openai` at any
  OpenAI-compatible host with `base_url` + `json_mode`. DeepSeek example:
  ```yaml
  intelligence:
    api:
      provider: "openai"
      model: "deepseek-chat"            # V3 - NOT deepseek-reasoner (R1): R1 burns
      api_key_env: "DEEPSEEK_API_KEY"   # tokens on reasoning + breaks structured output
      base_url: "https://api.deepseek.com"
      json_mode: true                   # response_format=json_object, fewer repairs
  ```
  Same shape works for Together / Groq / OpenRouter / local vLLM - just change
  `base_url` + `model`.
- **Cost gate:** set `intelligence.skip_sparse_chunks: true` to skip the LLM call
  for chunks too sparse to hold a relation (no two entities within
  `sparse_window_words`). Free NER still runs and the co-occurrence floor is
  untouched, so it's zero recall loss for fewer tokens. Off by default; turn it on
  when paying per token.
- **Best for:** highest relation quality (frontier models), or cheapest tokens
  (deepseek-chat).
- **Trade-off:** cost + sends text to a third party.

### Mode `ollama` (local LLM)
- Same prompts as `api`, run locally.
- **Setup:**
  ```powershell
  # install Ollama from ollama.com, then:
  ollama serve
  ollama pull qwen3.5:9b         # or gemma4:12b, llama3.1, ...
  ```
  Config: `intelligence.ollama.model` + `host`.
- **Best for:** LLM quality without sending data out.
- **Trade-off:** needs a capable local machine.

### Mode `gemini_batch` (manual long-context batch)
Process the whole corpus in one paste-in pass instead of chunking. The model does
NER + relations over each document whole, so there's no chunk-boundary recall loss,
no API key, and a 2M-token window (Gemini) eats large documents fast.
```powershell
# 1. Write the prompt file(s). --batch-docs caps documents per file: the model's
#    JSON REPLY length scales with doc count, so this is the anti-truncation knob.
python main.py --config <cfg> --mode gemini_batch --stage extract --batch-docs 25
#    -> <run>/gemini_batch_prompt.001.txt, .002.txt, ...
# 2. In the model, SET MAX OUTPUT TOKENS TO THE MAXIMUM (AI Studio: 65536). Then
#    upload each prompt; it returns a JSON object keyed by doc id. Save each reply to
#    the run dir as gemini_batch_response.001.json, ... matching the prompt numbers.
# 3. Import + build the graph (globs all reply files; flags any doc no reply covered):
python main.py --config <cfg> --mode gemini_batch --stage analyze
#    --import-json <path-or-glob> if the replies are saved elsewhere.
```
Truncation is the one failure mode: a too-large batch comes back with only the
first few documents. Keep batches small (`--batch-docs 15-25` for dense first-person
sources) and max the output-token setting. `--stage analyze` prints `N of M
documents not covered` if a reply was cut off - re-export those at a smaller
`--batch-docs` and redo just them.

**`--submit` (no manual paste).** A free Google AI Studio API key turns the whole
thing into one command - it POSTs each batch to the Gemini API (setting the
output-token cap the chat UI hides) and runs analyze:
```powershell
# one-time: free key at https://aistudio.google.com/apikey
$env:GEMINI_API_KEY = "AIza..."
python main.py --config <cfg> --mode gemini_batch --stage extract --batch-docs 25 --submit
```
That writes the prompts, calls Gemini for each, writes the replies, imports, and
builds the graph. `--batch-model gemini-2.5-pro` for higher quality (lower free-tier
rate limit); default is `gemini-2.5-flash`. The free tier easily covers a
500-document corpus.

Thinking is OFF by default (`batch_thinking_budget: 0`): 2.5-flash otherwise burns
the output-token budget on reasoning tokens and truncates the JSON. Leave it off for
extraction; `--batch-thinking <n>` (or `<0` to restore the model default) if you ever
want it.

**Resume.** A failed/truncated batch is skipped and reported. Re-run the SAME command
with `--resume` and it skips every batch whose reply is already on disk and complete
(the reply file is the checkpoint), re-POSTing only the missing/truncated ones - so an
interrupted or rate-limited run continues without re-paying for done batches. Resume
assumes the same `--batch-docs`/`--batch-budget` (the batch boundaries must line up
with the saved files). `--stage analyze` still flags any doc no reply covered.
```powershell
python main.py --config <cfg> --mode gemini_batch --stage extract --batch-docs 25 --submit --resume
```

**LLM post-processing.** Extraction comes from the whole-document reply, so the
LLM-assisted post steps (`dedup.llm_assist`, `quality.llm_review`, `enrichment`) have
no live backend and are SKIPPED (you get a warning naming them; rule-based dedup/review
still run). For a domain that leans on LLM-dedup to merge entities across documents
(InfluenceWatch, OREM), either run it in `ollama`/`api` mode, or set
`intelligence.batch_post_llm: true` to route those steps through Gemini's
OpenAI-compatible endpoint with the same key (extra API calls at analyze time).
- The prompt carries the **same** relation ontology, guide, type hints, and qualifier
  schema as `api`/`ollama` - so domains (InfluenceWatch monetary_value, OREM
  jurisdiction) work unchanged. The reply tolerates ```code fences``` and split files.
- **Best for:** large/whole documents, top accuracy, using a subscription chat model
  with no API wiring.
- **Trade-off:** manual paste step; NER is the model's (no GLiNER spans, so the
  within-doc proximity floor is skipped); no live backend means enrichment/LLM-dedup
  are off (as in python_only).

Switch modes without re-extracting the foundation? Not yet - the mode changes
extraction, so a mode change means a fresh `extract`. But you can re-run only
post-processing after tuning dedup/quality/export with `--stage analyze`.

---

## 5. Running on English

```powershell
copy config_template.yaml config.yaml
```
Edit `config.yaml`:
```yaml
mode: "python_only"            # or api / ollama
io:
  input_path: "./data/sample_en"
foundation:
  spacy_model: "en_core_web_trf"
  gliner_model: "fastino/gliner2-multi-v1"
  gliner_labels: ["person", "organization", "location", "event"]
coreference:
  languages: ["en"]
```
Run:
```powershell
python main.py --config config.yaml
```

---

## 6. Running on German (NSDAP domain)

Use the pre-built config - it already selects the German spaCy model, the
**multilingual GLiNER**, the 24 domain labels, 200+ EntityRuler patterns, 500+
aliases, German date parsing, German narrator pronouns, and evidence-based
membership inference.

```powershell
python main.py --config domain/nazi_era/config_nazi_era.yaml
```

Point it at your corpus by editing `io.input_path` in that file (default
`./data/abel_papers`). Supported file types: `.txt .md .rtf .pdf .docx .html .epub`.

What the domain does automatically:
- merges `München`↔`Munich`, `SA`↔`Sturmabteilung`, `der Führer`↔`Adolf Hitler`, ...
- parses `Herbst 1923`, `4. März 1921`, `Frühling 1920` into real dates
- detects `Ich/mein/wir` first-person narration -> a `Narrator [file]` author node
- infers membership in NSDAP/SA/SS/Freikorps/... from textual evidence
- flags anachronisms (e.g. "SS in 1922")

In `ollama`/`api` mode the nazi config also runs two LLM post-passes (set in
`config_nazi_era.yaml`): `dedup.llm_assist: true` (merge same-entity nodes the
rules missed) and `enrichment.enabled: true` (subtype + rank/office on resolved
entities). Set them `false` to keep post-processing deterministic.

Abel files are named `<Author><hooverID>.rtf`, so the author of each document is
known: the narrator node is set to that real author name (and merges with the
in-text mention, flagged `is_author`).

**Validate a run without hand-annotated gold:**
```powershell
python -m domain.nazi_era.validate_run --run-dir output\abel_papers --inputs data\abel_papers --out validation.json
```
`--inputs` adds author coverage (how many of the processed letters' authors
surfaced as a PERSON / `is_author`). Also reports anachronisms, rank/org
consistency, alias application (a known alias and its canonical must not be two
nodes), known-entity coverage, and structural sanity (isolates, evidence %, edge
sources, membership counts).

**Metadata as network structure.** With `io.metadata_file` set, the spreadsheet
isn't just node attributes: each author also gets verified edges
(`born_in` -> place_of_birth, `resided_in` -> place_of_residence, `member_of` ->
NSDAP with membership#/join date, `member_of` -> prior party), tagged
`edge_source=metadata` - the most authoritative tier for filtering in Gephi.

## Direct commands (live progress bars)

Run these in a plain terminal (no `> log` redirect) to watch the rich progress
bars: per-document extraction rate, stage banners, and the final summary table.
The bash wrappers in `scripts/` log to a file instead - use these when you want
to *see* the run. `--run-name` keeps each model's output in its own directory;
`--resume` is always safe (no-op on a fresh run, continues an interrupted one).

```powershell
$env:PYTHONUTF8 = 1

# Abel pilot, choose the model (qwen3.5:9b fits an 8 GB GPU; gemma4:12b will
# spill and run ~5x slower but extracts richer stance/polarity edges)
python main.py --config domain/nazi_era/config_nazi_era.yaml --mode ollama `
  --ollama-model qwen3.5:9b --run-name abel_qwen3_5_9b --limit 25 --resume
python main.py --config domain/nazi_era/config_nazi_era.yaml --mode ollama `
  --ollama-model gemma4:12b --run-name abel_gemma4_12b --limit 25 --resume

# bigger machine (16 GB+ VRAM): same command, bigger model
python main.py --config domain/nazi_era/config_nazi_era.yaml --mode ollama `
  --ollama-model gemma4:26b  --run-name abel_gemma4_26b  --limit 20 --resume
python main.py --config domain/nazi_era/config_nazi_era.yaml --mode ollama `
  --ollama-model qwen3.6:27b --run-name abel_qwen3_6_27b --limit 20 --resume

# re-run post-processing only (dedup/quality/export tweaks; no re-extraction)
python main.py --config domain/nazi_era/config_nazi_era.yaml --mode ollama `
  --run-name abel_qwen3_5_9b --stage analyze

# score a run against gold annotations (entity/relation P/R/F1 by tier)
# Relations are scored DIRECTED by default (asymmetric relations must match
# orientation; symmetric ones match either way). Add --undirected-relations for
# the old direction-agnostic behavior. Entities use 1:1 matching (over-splitting
# costs precision).
python -m evaluation.evaluate --gold evaluation/gold_template.json `
  --run-dir output/abel_qwen3_5_9b --edge-sources conservative

# public IE benchmark: prepares inputs+gold, runs the pipeline, scores it
python -m benchmarks.run_benchmark --dataset redocred --limit 25 --run --eval
python -m benchmarks.run_benchmark --dataset redocred --limit 10 --mode ollama `
  --ollama-model qwen3.5:9b --constrain-relations --run --eval
# German historical NER (closest public proxy for the Abel corpus):
python -m benchmarks.run_benchmark --dataset hipe --limit 20 --run --eval
# modern German NER (pairs with hipe - the two bracket the Abel register):
python -m benchmarks.run_benchmark --dataset germeval --limit 20 --run --eval
# interpersonal ties in dialogue:
python -m benchmarks.run_benchmark --dataset dialogre --limit 30 --mode ollama `
  --ollama-model qwen3.5:9b --constrain-relations --run --eval

# book vs gold; --relation-guide gives the LLM contrastive label definitions
# (A/B the typed-relation accuracy vs the bare-label run):
python scripts/book_bench.py --book data/hobbit.txt --gold data/hobbit.gold.json `
  --mode ollama --ollama-model qwen3.5:9b --relation-guide data/hobbit.relation_guide.json

# are the confidence scores honest? reliability bins + ECE vs gold
python -m evaluation.calibration --gold data/bench/redocred.gold.json `
  --run-dir data/bench/redocred/output/redocred_test_ollama_constr
```

## Scaling up (larger test -> full corpus)

Extraction is the cost (ollama qwen3.5:9b on an 8 GB GPU is ~75 s/letter, so ~540
letters is roughly 11-12 h). It is checkpointed per document, so:

```powershell
# 1) larger smoke test first
python main.py --config domain/nazi_era/config_nazi_era.yaml --mode ollama --limit 50

# 2) full extraction (resumable - re-run with --resume if interrupted)
python main.py --config domain/nazi_era/config_nazi_era.yaml --mode ollama --stage extract
python main.py --config domain/nazi_era/config_nazi_era.yaml --mode ollama --stage extract --resume

# 3) analyze once extraction is complete (re-runnable without re-extracting)
python main.py --config domain/nazi_era/config_nazi_era.yaml --mode ollama --stage analyze
python -m domain.nazi_era.validate_run --run-dir output\abel_papers --inputs data\abel_papers --out validation.json
```

Notes:
- Split extract and analyze so you can re-tune dedup/quality/enrichment/export
  (`--stage analyze`) without paying for re-extraction.
- The analyze LLM passes (llm_dedup, llm_review, enrichment) add time proportional
  to entity count; set any of `dedup.llm_assist` / `quality.llm_review` /
  `enrichment.enabled` to `false` for a faster first pass, then turn on and re-run
  `--stage analyze`.
- A CUDA GPU + `foundation.device: cuda` cuts extraction time sharply.

**Provenance + metadata join.** Every output ties back to source text and the
Hoover LetterID:
- `documents.csv` - `doc_id, letter_id, author, filename` (the join key).
- nodes: `attr_evidence` (source sentence) + `attr_evidence_doc`; author nodes
  also have `attr_letter_id`.
- edges: `evidence` (source sentence) + `letter_id`; timeline: `description` + `letter_id`.

To join your `Nazi metadata.xlsx` (keyed by its LetterID column) to the network,
match `documents.csv.letter_id` to the spreadsheet LetterID, e.g. in pandas:
`docs.merge(meta, left_on="letter_id", right_on="LetterID")`. Keep the spreadsheet
out of `data/abel_papers/` (it's skipped as input, but cleaner under `data/`).

---

## 7. Understanding the output

In `output/<run_name>/`:

| File | What it is |
|------|-----------|
| `gephi_nodes.csv` | One row per entity. `type, mention_count, doc_count`, tie-class degree split (`deg_interaction, deg_affiliation, ...`), `tag_*`, `attr_*`. Centralities are not precomputed — Gephi does that. |
| `gephi_edges.csv` | One row per aggregated relationship. `Source, Target, Type, Label, tie_class, connection_type, polarity, Weight, n_mentions, n_sources, reciprocal, period, origin, edge_source, confidence, source_name, target_name, letter_id, evidence` |
| `network.gexf` | The whole graph in one file. |
| `graph_interaction.gexf` | **Interpersonal social network (open this for SNA).** |
| `graph_affiliation.gexf` / `graph_discourse.gexf` | Two-mode and discourse layers. |
| `entities.json` | Full entity records incl. aliases, attributes, tags. |
| `timeline.csv` | Dated events, chronological. |
| `codebook.xlsx` | Auto-generated SNA codebook: boundary spec, definition of every node/edge column, entity types, tie classes, the run's relation inventory with example evidence, evidence tiers. Hand this to anyone who has never seen the pipeline. |
| `run_meta.json` | Which model/mode/config produced this run, with timestamp. |
| `graph_report.json` | Graph-health QA + brokerage/bridge counts, plus a `quality_pillars` block (KGC-2026: provenance + consistency from real data; accuracy/completeness/timeliness as labelled coverage proxies). When `export.graph_metrics` is on. |
| `raw_extractions.jsonl` | Per-document extractions (provenance / re-analysis). |
| `checkpoints/` | Resume data. |

**Gephi import:** `File ▸ Open ▸ graph_interaction.gexf` for the social network,
or `network.gexf` for everything. Run `Statistics` (degree, betweenness,
modularity) — Gephi computes them on the loaded view. Color by the resulting
modularity class, size by betweenness. Filter edges by `tie_class` (§10a) and
`edge_source` (§10).

Node tags (in `entities.json` and as `tag_*` columns):
- `entity_scope`: `macro` (hubs/broad) vs `specific`
- `relevance_tier`: `core` / `secondary` / `peripheral`

---

## 8. Coreference & the narrator

Configured under `coreference:`.

- **`narrator_resolution: true`** (default) - detects first-person pronouns
  (English + German, set via `languages`) and creates one author node per
  document, `Narrator [filename]`, placed at every "I/me/my" (or `ich/mir/mein`)
  position. Because the narrator sits on the pronoun spans, statements like
  *"Ich trat in die SA ein"* link the **author -> SA**, and the author is flagged
  `is_author` for the membership rule (§9). This is essential for first-person
  sources like the Abel autobiographies.
- **`pronoun_resolution: true`** (optional, needs `fastcoref`) - resolves
  third-person pronouns ("he", "er") to named entities. English-oriented;
  silently no-ops if `fastcoref` isn't installed.
- **`service_url`** (optional) - run fastcoref out-of-process. It needs2
  `transformers <5`, which conflicts with the main env's GLiNER2; the microservice
  keeps it isolated and light. In a separate venv:
  `pip install -r services/requirements-coref.txt` then
  `uvicorn services.coref_service:app --port 8000`. **Start it first and wait for
  `Application startup complete`** - it preloads the model at boot, so that line
  means it's ready (no cold-start race). Then set
  `coreference.service_url: "http://127.0.0.1:8000"`. The pipeline warms the service
  once up front (generous budget) before extraction; if it's unreachable it falls
  back to in-process fastcoref, then the heuristic resolver, so it's safe to leave
  configured. NB: in-process fastcoref is broken in the main env (transformers 5.x:
  `all_tied_weights_keys`), so the service is the real path - if it's down you get
  the heuristic, not neural coref.

Turn it all off with `coreference.enabled: false`.

---

## 9. The mandatory-membership assumption

The Abel corpus consists of autobiographies **by NSDAP members**, so the author
of each document is a party member by construction. This is encoded - but
**scoped** so it doesn't wrongly tag *mentioned* people (Marx, Lenin, opponents):

`inference.mandatory_membership`:
- **`authors_only`** (default, recommended) - only the detected narrator/author
  nodes get a mandatory `member_of NSDAP` edge (`edge_source: pipeline_inferred`).
  Requires `coreference.narrator_resolution: true`.
- **`all`** - every PERSON gets it (legacy; over-connects; **not** recommended).
- **`off`** - no mandatory edges; NSDAP membership only when stated in text.

This is a documented modeling assumption, not an extraction. State it in any
write-up, and consider reporting an `off` network alongside.

---

## 10. Edge-source sensitivity analysis

Every edge carries `edge_source` so you can rebuild the network at increasing
evidentiary looseness:

| Network | `edge_source` values | Meaning |
|---------|----------------------|---------|
| **Conservative** | `llm_extracted`, `langextract_extracted`, `rule_extracted`, `metadata` | stated in the text or a verified record |
| **Moderate** | + `canonical_inferred` | + membership inferred from a detected textual signal |
| **Full** | + `rule_cooccurrence`, `pipeline_inferred` | + raw co-occurrence and the mandatory-membership assumption (weakest layers) |

The single source of truth for this mapping is `postprocess/evidence_tiers.py`;
the evaluator and the per-run codebook both import it.

In Gephi, filter edges on the `edge_source` column. For evaluation, pass
`--edge-sources conservative|moderate|full` (§11). **Report the conservative
network for empirical claims**, and show the others to quantify what inference
adds.

`edge_source` answers *how strongly we know an edge*; `tie_class` (§10a) answers
*what kind of relationship it is*. They are independent — filter on both.

---

## 10a. Tie class — what kind of relationship (the SNA question)

A mention is not a social tie. Every edge carries a `tie_class`:

| `tie_class` | Meaning | Use for |
|-------------|---------|---------|
| `interaction` | person↔person, actually narrated (met, served with, recruited, family) | **the interpersonal social network** |
| `affiliation` | person→org/unit (member_of, joined, led, founded) | two-mode membership network |
| `participation` | person→event (fought_in, participated_in) | two-mode event network |
| `biographical` | person→place/rank (born_in, resided_in) | attributes, geography |
| `stance` | attitude (supported, opposed, influenced_by) | discourse / ideology, **not** a social tie |
| `causal` | one thing brings about another (caused, caused_by, contributed_to, prevented) | driver→impact / cause→effect; substantive content, **not** interpersonal |
| `cooccurrence` | names co-present only | weakest layer, **not** a tie |

**Compute interpersonal centrality on `graph_interaction.gexf`** (run Gephi's
Statistics on it). Computing centrality on the full mixed graph instead inflates
whatever the corpus is *about* — Hitler's degree on the full graph is ~590 but
only ~22 in the interaction layer. Ready-made views: `graph_interaction.gexf`
(SNA), `graph_affiliation.gexf`, `graph_discourse.gexf`; or filter the combined
`gephi_edges.csv` on `tie_class`.

Public/historical figures carry `attr_reference_figure=true`. They are kept in the
graph; exclude them in Gephi when you want only authors' lived contacts.

Edge weight is **distinct corroborating documents** (`Weight`), with `n_mentions`
and `n_sources` (distinct letters) alongside — a tie attested by ten authors
outranks one repeated ten times in a single letter.

Other tags for different analyses:
- **`polarity`** (positive/negative/neutral) — signed-network / balance-theory
  analysis (supported/allied vs opposed/fought_against).
- **`period`** (imperial_ww1/weimar/nazi_rule) + edge **`year`** + node
  **`first_year`/`last_year`** — temporal slicing. `network_dynamic.gexf` carries
  `start` years for Gephi's timeline player.
- **`attr_wikidata_qid`/`_url`/`_label`** — present when `linking.enabled: true`
  (off by default; makes network calls). Disambiguates entities against Wikidata
  for cross-dataset joins.

---

## 10b. Faithfulness tags — catching hallucinated edges

Local (and cheap) models invent edges. Three deterministic guards run on every
LLM/text-asserted relation and **tag** suspect edges so you can filter them in
Gephi — they are not dropped (a real edge can trip a guard via coref or alias
variation), except where noted:

| Tag (edge attribute) | Set when | Why |
|----------------------|----------|-----|
| `evidence_unverified` | the model's evidence quote is not verbatim in the source chunk | the quote was paraphrased/fabricated |
| `evidence_ungrounded` | the evidence quote names **neither** endpoint (anchor check, AEVS-style) | the model attached a real sentence that doesn't mention the pair |
| `type_violation` | endpoint entity types contradict the relation's signature (`born_in`→org, `led`→place) | a likely misextraction |

`type_violation` is the one you can turn into a hard filter:
`ontology.drop_type_violations: true` drops instead of tags. The other two are
tag-only by design. In Gephi, filter these columns out for a high-precision view;
keep them in to audit what the model claimed. The narrator/author endpoint is
exempt from `evidence_ungrounded` (first-person evidence legitimately says "I").

All four flags (these three plus `suspect_membership`) ship as boolean columns in
`gephi_edges.csv` and as edge attributes in the GEXF, so you can filter on them in
either import path. `type_violation` is also counted in
`graph_report.json` → `quality_pillars.consistency` (`type_violations`,
`clean_pct`) — a fast way to gauge extraction precision without opening Gephi.

The single source of truth for relation signatures is
`postprocess/ontology.py` (`RELATION_TYPE_SIGNATURES`); loose relations
(`supported`, `met_with`) have no signature and are never type-flagged. If a guard
fires on edges you know are good, that relation's signature is too tight — loosen
it there. `graph_report.json` → `quality_pillars.consistency.type_violations_by_relation`
shows which relations violate most — one relation dominating usually means its
signature is too tight, not that the model is hallucinating.

To cut violations at the source instead of just tagging them, set
`intelligence.type_hints: true`. It shows the model each constrained relation's
argument types in the extraction prompt (`born_in (person->place)`), rendered from
the same `RELATION_TYPE_SIGNATURES`. Off by default; A/B it against
`type_violations_by_relation` to confirm it helps your model before leaving it on.

---

## 10c. Expanding an existing network (instead of starting fresh)

When you already have a curated network and want to grow it from new documents
*without it drifting* — no new relation types, no off-target entity kinds — turn on
expansion. It reads the schema of the existing network and locks this run to it.

```yaml
expansion:
  enabled: true
  source: "./output/abel_papers"   # a prior run dir, its gephi_edges.csv, or a network.gexf
  lock_relations: true             # only keep relation types already in the source
  drop_unmapped_relations: true    # strict: drop a new edge whose type isn't in that set
  lock_entity_types: true          # keep only the entity kinds already in the source
  entity_types: []                 # or pin explicitly, e.g. ["PERSON","ORG"]
```

What it does:
- **Strict edge vocabulary.** New documents can only produce relation types the
  existing network already uses. Synonyms still resolve (`"worked for"` →
  `employed_by` if that's in the set), so you keep recall on surface variants but
  never introduce a new edge type. Anything off-vocabulary is dropped (or tagged
  `ontology=unmapped` if you set `drop_unmapped_relations: false`).
- **Only certain kinds of entities.** Keeps only the entity types present in the
  source network (or exactly the `entity_types` you pin). Off-target kinds are
  dropped before dedup.

`source` can be a run directory (reads `entities.json` + `gephi_edges.csv`), a bare
edge CSV, or a `.gexf`. A missing/empty source makes the locks no-ops (the run
proceeds normally, with a warning). Works in `--stage analyze` too, so you can
re-lock an existing checkpoint without re-extracting. The network's own
`co_occurs_with` layer is never treated as a lockable relation type.

> Note: this constrains the *output* network. The LLM/NER still look broadly during
> extraction; the locks apply in post-processing. The resulting graph conforms to
> the existing schema either way.

---

## 10d. Affiliation projection & edge qualifiers

Two opt-in features for affiliation-dense and quantitative corpora.

**Two-mode (affiliation) projection.** When direct person-person ties are rare and
actors connect through shared groups — political boards/PACs, agencies sharing a
disaster-response event — turn on the projection: actors tied to the same
org/institution/event get a `co_affiliated` edge (Newman 1/(k-1) weighted by group
size, summed over shared groups). It's a co-presence, not a direct tie (full tier).

```yaml
inference:
  enable_affiliation_projection: true
  affiliation_min_shared: 1     # raise to 2 on a corpus with a universal group
                                # (e.g. every NSDAP author shares NSDAP) to drop the near-clique
```
Filter `gephi_edges.csv` to `rel_type = co_affiliated` (or `edge_source = affiliation_projected`), sort by `affiliation_strength`.

**Edge qualifiers.** To capture a typed value *on* a relation — a funding amount, a
jurisdiction, a spatiotemporal value — declare the field names; the LLM fills them
only when the text states them, and they ride through as `qual_<name>` edge columns.

```yaml
intelligence:
  edge_qualifiers: ["monetary_value", "jurisdiction"]   # InfluenceWatch / OREM
  # ["location", "time"] for spatiotemporal records; ["weapon", "setting"] for a script
```
Empty by default (no behavior change). The model is told to omit a field it can't find — it won't guess.

---

## 11. Evaluating against gold data

This is what turns the pipeline from "runs" into "validated." See
[evaluation/README.md](evaluation/README.md) for the full guide. Short version:

1. Hand-annotate ~20-30 documents in the gold JSON format
   ([evaluation/gold_template.json](evaluation/gold_template.json)).
2. Run the pipeline.
3. Score:
   ```powershell
   python -m evaluation.evaluate --gold gold.json --run-dir output/abel_papers
   # conservative-network edge precision (headline relation number):
   python -m evaluation.evaluate --gold gold.json --run-dir output/abel_papers --edge-sources conservative --out report.json
   ```

You get entity P/R/F1 (overall, type-agnostic, per type) and relation P/R/F1
(typed + untyped). Aliases are matched automatically. The harness uses only the
standard library, so it runs even without the ML stack installed.

---

## 12. Adapting to other inputs / languages / domains

### a) Just new documents, same language
Point `io.input_path` at your folder. Edit `foundation.gliner_labels` to the
entity types you care about (they are plain English phrases; GLiNER is zero-shot).

**Input sources** - mix freely:

| You have... | Do this |
|-----------|---------|
| A folder of files (`.txt .md .pdf .docx .rtf .html .epub`) | `io.input_path: "./mycorpus"` |
| One file | `io.input_path: "./book.pdf"` |
| A book (`.epub` native, or `.pdf`/`.txt`) | drop it in the folder; it's chunked automatically |
| A TV/movie/play script | drop the `.pdf`/`.txt` in the folder; add `--parse-scripts` for scene co-presence edges |
| A web page or online PDF | `--url https://...` (repeatable) or `io.urls: ["https://..."]` |
| Many URLs | put them (one per line) in a file -> `--urls-file urls.txt` or `io.urls_file` |
| A whole site + its subpages | `--crawl https://site/section/` or `io.crawl.{enabled,seeds}` (resumable; see below) |
| A wiki (Wikipedia / Fandom / any MediaWiki) | `--wiki "host:Page Title"` or `--wiki "host:Category:X"` or `io.wiki` |
| An influence graph (LittleSis: donors, boards, PACs) | `--littlesis "search:Koch Industries"` or `--littlesis id:28220` or `io.littlesis`. Imports curated typed edges (CC BY-SA — attribute it) |
| A subreddit / Mastodon / Bluesky / Telegram / etc. | `--social platform:target` or `io.social` (see README's social table) |
| A raw string / pasted text | `--text "Hitler met Goebbels in 1926."` |

```powershell
# A book:
python main.py --config config.yaml            # with io.input_path: "./books/mein_buch.pdf"
# A Wikipedia article:
python main.py --config config.yaml --url https://en.wikipedia.org/wiki/Weimar_Republic
# A batch of pages + your local corpus together:
python main.py --config config.yaml --urls-file sources.txt
# A whole site section (bounded crawl, one merged network):
python main.py --config config.yaml --crawl https://example.org/topic/ --crawl-max-pages 40
```

The **generic domain** (`domain: {name: "generic"}`) carries an English common-noun
stoplist and a general subtype vocabulary, and every structural/SNA step
(tie classes, polarity, corroboration weight, dedup, junk-name filter, multi-view
graphs, optional Wikidata linking) is domain-agnostic — so a novel or a scraped
page gets the same clean, fully-tagged output as the tuned Nazi-era corpus, minus
the domain-specific aliases. Use `--mode ollama`/`api` for best entity resolution
on fiction (no prebuilt alias list), e.g. merging "Bilbo"/"Mr. Baggins".

Caveats: `--url` fetches exactly the page(s) you give. To pull a **whole site**,
use `--crawl` (below). PDFs need a real text layer; **scanned/image PDFs require
OCR first** (e.g. `ocrmypdf` -> then feed the OCR'd PDF). Very large books work
but take proportionally longer; use `--limit` while tuning.

#### Books, scripts, wikis: one generic domain, not a domain per medium

The generic domain's **semantics** already cover these — the relation ontology
(`domain/generic/relationship_config.py`) carries the interpersonal / organizational /
biographical / spatial / stance / causal ties any narrative or article states, the
narrative-sequence net has a `fiction` element scheme, and the script parser adds scene
co-presence. So you do **not** make a new "book domain" or "wiki domain". What differs by
medium is two things: **ingestion** (how the bytes become clean text — handled by the file
extractors and the wiki/social connectors above) and a **thin config preset** that flips
the right levers. Make a domain only when you have real domain *knowledge* to inject
(aliases, special labels, inference rules), the way `nazi_era` / `influencewatch` do.

| Medium | Ingestion | Levers to set (in a config preset) |
|--------|-----------|-----------|
| **Novel / book** | `.epub` native, or `.pdf`/`.txt` | `export.narrative_scheme: fiction`; `coreference.pronoun_resolution: true` (third-person "he/she" → names — the recall ceiling for fiction; needs the fastcoref service, §8). `--mode ollama`/`api` for alias merging (Bilbo/Mr. Baggins). |
| **TV / movie / play script** | `.pdf`/`.txt` | `intelligence.parse_scripts: true` (`--parse-scripts`): scene co-presence edges. Dialogue lines still feed the LLM for who-speaks-to / interacts-with. |
| **Wiki (Wikipedia / Fandom)** | `--wiki host:Category:X` (API prose, clean) | nothing special; the generic ontology + dedup handle it. Crawl the HTML only if a wiki has no API. |
| **Scraped article site** | `--crawl` / `--url` | `io.crawl.boilerplate` to strip site nav leak; `inference.enable_affiliation_projection` if it's affiliation-dense (boards/orgs). |
| **Social platform** | `--social platform:target` | the connector emits the reply/mention/forwarded/posted_in graph as asserted edges; the post text still runs through NER/RE. |

The point: **one generic domain handles the relation vocabulary for all of them.** Add an
ingestion adapter (a connector / file reader) for a new *shape* of input, and a thin config
preset for a new *medium* — not a full domain. A full domain is for injected knowledge.

#### Importing LittleSis (curated influence graph)

LittleSis is a sourced graph of donations / boards / ownership / lobbying — import the edges
directly (they land as asserted `edge_source=littlesis` ties; donations carry
`qual_monetary_value`). Two ways:

- **Targeted (API).** `--littlesis "search:Koch Industries"` or `--littlesis id:28220` pulls
  an ego-network around those seeds. Cheap; good for specific actors.
- **Bulk (full dump).** Download `relationships.json.gz` + `entities.json.gz` from
  [littlesis.org/bulk_data](https://littlesis.org/bulk_data) (relationships carry the edges +
  endpoint names; entities adds blurb/types/website per node). `scripts/littlesis_bulk.py`
  streams them (GB-scale, no full load) and writes a snapshot. Three scope variants out of the
  ~1.7M-edge graph — `--entities` adds node attributes in every case:
  ```powershell
  # 1. CONNECTED to your scrape (1-hop): every LittleSis edge touching one of your entities.
  python -m scripts.littlesis_bulk data/littlesis/relationships.json --entities data/littlesis/entities.json `
    --names-from output/influencewatch_llm/entities.json --out output/ls_connected.jsonl
  # 2. INDUCED subgraph: only edges where BOTH endpoints are your entities.
  python -m scripts.littlesis_bulk data/littlesis/relationships.json --entities data/littlesis/entities.json `
    --names-from output/influencewatch_llm/entities.json --induced --out output/ls_induced.jsonl
  # 3. WHOLE dump: everything, incl. edge-less entities (heavy - the SNA metrics may not finish).
  python -m scripts.littlesis_bulk data/littlesis/relationships.json --entities data/littlesis/entities.json `
    --include-isolated --out output/ls_full.jsonl
  ```
  Then merge by combining the snapshot with your scrape and ingesting (dedup folds by name):
  ```powershell
  Get-Content output/influencewatch_llm/documents.jsonl, output/ls_connected.jsonl |
    Set-Content output/iw_ls_connected/documents.jsonl
  python main.py --config domain/influencewatch/config_influencewatch.yaml `
    --ingest-from output/iw_ls_connected/documents.jsonl --run-name iw_ls_connected --mode ollama --resume
  ```
  `--names-from <run>/entities.json` is the merge lever (keeps only LittleSis edges touching an
  entity you already have). In Gephi, filter edges by `edge_source=littlesis` and nodes by
  `attr_littlesis` to toggle the LittleSis layer on/off. **CC BY-SA 4.0 — attribute it.**

#### Crawling a whole site
`--crawl <seed>` (repeatable) expands a seed URL into its subpages and merges them
into one network (entities fold across pages: "Ford Foundation" on page A = page B).
It is **bounded and polite by default** so a crawl can't run away or hammer a host:

- **Bounds:** `max_pages` (document cap, default 50), `max_depth` (link hops, 3),
  `max_bytes` (per-page download ceiling). A hard request budget backstops all three.
- **Scope:** stays on the seed's host (`stay_on_host`, folds `www.`); optional
  `stay_under_path` keeps only the seed's directory; `allow`/`deny` are regex lists.
- **Politeness:** obeys `robots.txt` + `Crawl-delay`; `delay` (s) rate-limits per
  host; seeds from `sitemap.xml` when present. Identifies with a clear User-Agent.
- **Fetch-once:** the page read for links is the page kept (no double fetch). The
  discovered URL list is cached in the run dir, so `--stage analyze` never re-crawls.

```powershell
# CLI: crawl one section, cap at 40 pages, 2 hops deep.
python main.py --config config.yaml --crawl https://example.org/topic/ `
  --crawl-max-pages 40 --crawl-max-depth 2

# Or in the config (io.crawl) for repeatable runs + finer scope:
#   io:
#     crawl:
#       enabled: true
#       seeds: ["https://example.org/topic/"]
#       stay_under_path: true
#       deny: ["/tag/", "/author/"]
```

Respect the target's terms of service and robots.txt. Leave `respect_robots: true`
and a non-zero `delay` unless you own the site or have permission; an unthrottled
crawl can get your IP blocked.

### b) A different language
- Set `foundation.spacy_model` to that language's spaCy model
  (e.g. `fr_core_news_lg`), and install it.
- Set `foundation.gliner_model: "fastino/gliner2-multi-v1"` (100+ languages).
- Add the language's first-person pronouns: extend `_FIRST_PERSON` in
  [core/coreference.py](core/coreference.py) and add the code to
  `coreference.languages`.
- For non-English month/season dates, give your domain a `historical_context`
  module exposing `GERMAN_MONTHS`-style dicts (the date extractor will use them).

### c) A brand-new domain (recommended path)
Copy `domain/generic/` to `domain/<yourname>/` and fill in any of:

| File / attribute | Effect |
|------------------|--------|
| `aliases.ALIASES` | `{"surface form": "Canonical Name"}` merges. |
| `entity_config.LABEL_OVERRIDES` | force a type for specific surface forms. |
| `gliner_labels.LABELS` + `LABEL_TO_TYPE_MAP` | domain NER labels -> canonical types. |
| `spacy_patterns.PATTERNS` | EntityRuler patterns (strings or token specs). |
| `prompts_*.SYSTEM_EXTRACTION` / `SYSTEM_QUALITY_REVIEW` | LLM prompt overrides. |
| `inference_rules.infer_edges(entities, edges, options)` | add `origin="canonical"` edges. |
| `historical_context.GERMAN_MONTHS` / `GERMAN_SEASONS` | non-English date vocab. |

Then set `domain.name: <yourname>` in your config. No core changes needed; the
loader reflects over the package and uses whatever you provide. Use
`domain/nazi_era/` as a complete worked example.

### d) New entity types
Add the label to `gliner_labels.LABELS` and map it in `LABEL_TO_TYPE_MAP`. If the
canonical type is new (beyond PERSON/ORG/LOCATION/EVENT/RANK/DATE/INSTITUTION),
add it to `_CANONICAL_PASSTHROUGH` in [core/spacy_engine.py](core/spacy_engine.py)
so EntityRuler matches pass through, and add a dedup threshold for it in
`dedup.fuzzy_thresholds`.

---

## 13. CLI reference

```text
python main.py --config <path> [options]

  --config PATH      (required) YAML config file.
  --stage STAGE      all | fetch | ingest | extract | analyze   (default: all)
                     fetch = crawl/preprocess only, freeze to documents.jsonl, stop
                     (no models/GPU). analyze reuses the checkpoint - re-tune
                     dedup/quality/export without re-extracting.
  --resume           Skip documents already in the checkpoint. With --stage fetch,
                     continue an interrupted crawl from its frontier checkpoint.
  --ingest-from PATH Load documents straight from a frozen documents.jsonl - no
                     crawl/fetch/file walk. Pairs with --stage fetch (scrape once,
                     extract anywhere / on another machine).
  --mode MODE        api | python_only | ollama | gemini_batch  (override config).
  --limit N          Process only the first N documents (quick tests).
  --url URL          Fetch + analyze a web page / PDF URL (repeatable).
  --urls-file PATH   Newline-delimited list of URLs to fetch.
  --crawl URL        Crawl a site from this seed, analyze its subpages (repeatable;
                     enables crawling). Tune scope/bounds in io.crawl. Resumable: a
                     long crawl shows a progress bar, checkpoints its frontier, and
                     Ctrl-C + --resume continues it.
  --crawl-max-pages N   Override io.crawl.max_pages.
  --crawl-max-depth N   Override io.crawl.max_depth.
  --render-js        Render JS with headless Chromium during crawl (SPA sites).
  --wiki SPEC        MediaWiki source 'host:Target' (repeatable): clean article prose
                     via the API. 'en.wikipedia.org:Ada Lovelace' or
                     'en.wikipedia.org:Category:Physicists'.
  --wiki-limit N     Pages per --wiki source / category cap (override io.wiki_limit).
  --littlesis SPEC   LittleSis source 'search:term' or 'id:N' (repeatable): imports the
                     curated influence graph as asserted typed edges (donations carry
                     amounts). CC BY-SA - attribute it in any published network.
  --social SPEC      Social source 'platform:target' (repeatable): reddit:datascience,
                     bluesky:climate, telegram:durov, ... Pulls posts + the explicit
                     reply/mention/forwarded/posted_in graph.
  --parse-scripts    Add scene co-presence edges for screenplay/TV-script documents.
  --text "..."       Analyze a raw text string directly.
  -v, --verbose      DEBUG logging.

python -m evaluation.evaluate --gold <gold.json> (--run-dir DIR | --entities E --edges F)
  --edge-sources conservative | moderate | full   (default: full)
  --out report.json
```

Typical loop for tomorrow:
```powershell
python main.py --config domain/nazi_era/config_nazi_era.yaml --limit 5 -v   # small test
python main.py --config domain/nazi_era/config_nazi_era.yaml                # full run
python -m evaluation.evaluate --gold gold.json --run-dir output/abel_papers --edge-sources conservative
```

---

## 14. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `OSError: [E050] Can't find model 'de_core_news_lg'` | `python -m spacy download de_core_news_lg` (or set `spacy_model: de_core_news_sm`). It also auto-falls back to `en_core_web_sm`/blank. |
| `Segmentation fault` while "Loading foundation models" (CPU) | `en_core_web_trf` co-loaded with GLiNER2 clashes on a duplicate OpenMP runtime. `main.py` now sets `KMP_DUPLICATE_LIB_OK=TRUE` to allow it; for ollama/crawl runs (foundation on CPU) prefer a non-transformer spaCy model (`en_core_web_lg`/`_sm`) + `gliner2-multi-v1`. |
| Very few German entities | You're on an English-only GLiNER. Set `gliner_model: fastino/gliner2-multi-v1`. |
| `Env var ANTHROPIC_API_KEY is not set` | Export the key, or switch `--mode python_only`. |
| `Could not reach Ollama` | `ollama serve` running? `ollama pull <model>` done? Check `host`. |
| `No module named 'polars'` | `pip install -r requirements.txt`. CSV export falls back to stdlib; Parquet needs polars. |
| `fastcoref unavailable` warning | Only matters if `pronoun_resolution: true`; `pip install fastcoref` or leave it off (narrator still works). |
| Run crashed midway | Re-run the same command with `--resume`. Docs whose extraction failed outright are retried automatically. |
| Some docs lost chunks (`chunks_failed` in checkpoint meta, "JSON repair exhausted" / timeout warnings) | `python scripts/drop_failed_docs.py <run_dir>/checkpoints/<name>.extractions.jsonl`, then re-run with `--resume` - only those docs re-extract. |
| Want to re-tune without re-extracting | `--stage analyze`. |
| Author nodes named `Narrator [doc_...]` | Give files meaningful filenames; the node uses the filename stem. |
| Marx/Lenin showing as NSDAP members | You set `mandatory_membership: all`. Use `authors_only`. |

---

Questions this doesn't answer are almost always answered by the inline comments
in the relevant config (`config_template.yaml`, `domain/nazi_era/config_nazi_era.yaml`)
or [README.md](README.md) / [evaluation/README.md](evaluation/README.md).
