# NER + SNA Extraction Pipeline

A modular, pipeline that turns any unstructured text corpus
into **structured entities, relationship graphs, timelines, and Gephi-ready
network exports**.

The foundation - **GLiNER (zero-shot NER) + spaCy (linguistic analysis)** -
*always* runs. Three interchangeable "intelligence tiers" sit on top:

| Mode | Foundation | Intelligence Tier | Relationships | Network? |
|------|-----------|-------------------|---------------|----------|
| `api`         | GLiNER2 + spaCy | Claude / OpenAI / Bedrock | LLM, structured prompts | none |
| `python_only` | GLiNER2 + spaCy | rules + dependency parse + embeddings | SVO + co-occurrence | none |
| `ollama`      | GLiNER2 + spaCy | local LLM (qwen3.5, gemma4, ...) | LLM, same prompts as `api` | local |
| `langextract` | GLiNER2 + spaCy | LangExtract over Ollama/Gemini/OpenAI | LLM + char-level source grounding | local/cloud |
| `gemini_batch`| model does NER too | paste-in long-context model (Gemini 2M, Claude) | LLM, whole-document (no chunking) | none |

> **`gemini_batch` (manual batch):** `--stage extract` writes one self-contained
> prompt over the whole corpus; you paste it into a 2M-context model, save the JSON
> reply to the run dir, and `--stage analyze` imports it and builds the graph. No
> chunk-boundary recall loss, no API key. See [INSTRUCTIONS.md §4](INSTRUCTIONS.md#4-the-three-modes).

> **Optional add-ons:** `io.use_docling` for structure-aware PDF/DOCX/OCR
> ingestion; `export.graph_metrics` for SNA metrics Gephi can't compute
> (Burt's constraint/brokerage, bridges, articulation points + a `graph_report.json`
> health check with a KGC-2026 quality-pillar summary). Both off by default. NER is
> **GLiNER2** (`fastino/*`); GLiNER v1 is deprecated.

> **Cheap API path:** `api` mode talks to any OpenAI-compatible host - set
> `provider: openai` + `base_url` + `json_mode` to point at DeepSeek, Together, Groq,
> OpenRouter, or a local vLLM (NER stays local/free; only relations hit the API). The
> `intelligence.skip_sparse_chunks` cost gate skips the LLM call for chunks too sparse
> to hold a relation. See [INSTRUCTIONS.md §4](INSTRUCTIONS.md#4-the-three-modes).

---

## Why GLiNER + spaCy is always the base

1. **GLiNER** catches entities an LLM misses - a dedicated zero-shot NER model
   that accepts your custom labels at inference time.
2. **spaCy** supplies linguistic structure - sentence boundaries, dependency
   trees, POS tags, noun chunks, statistical NER.
3. **Consistency** - the same entity candidates feed every intelligence tier.
4. **Speed** - the foundation pre-filters so the LLM/rules only reason over
   confirmed spans.
5. **Validation** - entities found by *both* GLiNER and spaCy get a confidence
   boost.

---

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     |  *nix: source .venv/bin/activate
pip install -r requirements.txt
```

**Foundation models** (download what your corpus needs):

```bash
# spaCy (pick per language)
python -m spacy download en_core_web_trf      # English (best); en_core_web_sm = lighter
python -m spacy download de_core_news_lg       # German (for the nazi_era domain)

# GLiNER, sentence-transformers, and fastcoref weights all download
# automatically from Hugging Face on first run - no manual step. The GLiNER
# model is chosen in config: fastino/gliner2-multi-v1 (multilingual, the
# default) or fastino/gliner2-large-v1 (English-only, heavier). Backend
# (GLiNER vs GLiNER2) is auto-detected from the model name.
```

> Full step-by-step model install (incl. GPU, offline caching, and what each
> mode/language needs) is in [INSTRUCTIONS.md §1](INSTRUCTIONS.md#1-install).

Mode-specific extras:

- **`api`** - set the relevant key, e.g. `setx ANTHROPIC_API_KEY "sk-..."`
  (PowerShell) and configure `intelligence.api` in the YAML.
- **`ollama`** - install [Ollama](https://ollama.com), run `ollama serve`, and
  `ollama pull llama3.1:8b`.
- **`python_only`** - no network needed; sentence-transformers downloads its
  embedding model on first use.
- **coreference** - narrator resolution needs nothing extra; the optional
  third-person `pronoun_resolution` needs `pip install fastcoref`.

## Inputs: any unstructured text

The pipeline ingests, in any combination:

| Source | How |
|--------|-----|
| Text / Markdown files | `.txt`, `.md` in `io.input_path` (file or directory) |
| Books / documents | `.pdf` (PyMuPDF), `.docx`, `.rtf`, `.epub` (spine of chapters) |
| Saved web pages | `.html` / `.htm` (boilerplate stripped) |
| **Live web pages / PDFs** | `--url https://...` (repeatable), or `io.urls` / `io.urls_file` |
| **A whole site (subpages)** | `--crawl https://...` (bounded, polite, resumable); or `io.crawl`. Add `--render-js` for SPA/JS sites (needs Playwright) |
| **Wikis (MediaWiki)** | `--wiki host:Title` or `--wiki host:Category:X` (repeatable); or `io.wiki`. Clean API prose, not page HTML |
| **Influence graph (LittleSis)** | `--littlesis "search:Koch Industries"` or `--littlesis id:28220`; or `io.littlesis`. Imports curated typed edges (donations w/ amounts) directly. CC BY-SA |
| **Social media** | `--social platform:target` (repeatable); or `io.social` |
| **A portable corpus snapshot** | `--ingest-from documents.jsonl` (scrape once with `--stage fetch`, extract anywhere) |
| **Direct / pasted text** | `--text "..."` |

```bash
python main.py --config config.yaml --wiki "en.wikipedia.org:Category:Weimar Republic"
python main.py --config config.yaml --crawl https://example.org/topic/ --crawl-max-pages 40
python main.py --config domain/social/config_social.yaml --social reddit:datascience --social bluesky:climate
python main.py --config config.yaml --text "Hitler met Goebbels in Munich in 1926."
```

Large books (`.epub`/`.pdf`/`.txt`) are chunked automatically. **Crawling** a site follows links from the
seed(s) into one merged network - bounded (page/depth/size caps) and polite
(robots.txt, per-host rate limit, sitemap-aware) by default; tune in `io.crawl`.
**Limit:** PDFs must contain a real text layer (scanned/image-only PDFs need OCR
first, which is not bundled).

### Social-media networks

`--social platform:target` pulls posts **and the explicit social graph** (who replied
to / mentioned whom, who posted where) as asserted edges, then runs NER/relations over
the text. Users become PERSON nodes, communities ORG nodes, so co-posters project
together (`co_affiliated`). Fetched once, cached to `social_docs.jsonl`.

| Platform | `target` | Access |
|----------|----------|--------|
| `reddit` | `datascience` | public `.json` API |
| `hackernews` / `hn` | `top` \| `new` \| story id | official Firebase API |
| `lemmy` | `lemmy.world/technology` | open `/api/v3` |
| `mastodon` | `instance` \| `instance/tag/x` | open public timelines |
| `bluesky` / `bsky` | `from:alice.bsky.social` \| `climate` | AT-protocol public AppView |
| `telegram` / `tg` | `durov` (a public channel) | public channel preview `t.me/s/`; forwards are the edge |
| `truthsocial` / `truth` | *(blank)* \| `tag/news` | its own Mastodon API (gated; fails soft) |
| `twitter` / `x` | `from:nasa` \| `climate policy` | **official API v2**, `$TWITTER_BEARER_TOKEN` |

> **Only sanctioned access.** Connectors use documented/public/official endpoints only -
> no login-wall or anti-bot circumvention, no proxy/fingerprint/CAPTCHA evasion, no
> private-app impersonation. **Facebook** group/post scraping is **not supported** (it
> needs a credentialed session against Meta ToS): use the Graph API for public Pages,
> Meta's **Content Library** (academic research API), or a *Download Your Information*
> export ingested via `--ingest-from`. **Twitter/X** beyond the free API tier means the
> paid X API tiers - the connector won't scrape the login-walled UI. **Telegram** reads
> public broadcast channels via the open `t.me/s/` preview (the `forwarded_from` channel
> graph is the value); groups, full history, or non-public channels need the official
> MTProto/Bot API, not built in.

---

## Quick start

```bash
cp config_template.yaml config.yaml      # then edit: input_path, mode, labels
python main.py --config config.yaml
```

Outputs land in `output/<run_name>/`:

| File | Description |
|------|-------------|
| `documents.csv` | Per-doc manifest: `doc_id, letter_id, author, filename` (join key to external metadata). |
| `gephi_nodes.csv` | Nodes with `type`, `mention_count`, `doc_count`, `first_year`/`last_year`, a degree split by tie class (`deg_interaction`, `deg_affiliation`, ...), `tag_*`, `attr_*` (incl. `attr_wikidata_qid` when linking is on). Standard centralities are **not** precomputed — Gephi computes those on whichever view you load. |
| `gephi_edges.csv` | Edges with `tie_class`, `connection_type` (physical/ideological/organizational/biographical), `polarity`, `Weight` (distinct documents), `n_mentions`, `n_sources`, `reciprocal`, `period`, `year`, `origin`, `edge_source`, `letter_id`, `evidence`. Faithfulness/membership flags `suspect_membership`, `evidence_unverified`, `evidence_ungrounded`, `type_violation` (filter in Gephi; also GEXF edge attributes). Co-occurrence edges also carry `cooccur_strength` (Newman projection weight) and `disparity_alpha` (backbone significance); `co_affiliated` edges (two-mode projection) carry `affiliation_strength` and `shared_groups`. Declared per-edge qualifiers (e.g. `monetary_value`, `jurisdiction`) appear as `qual_*` columns. |
| `network.gexf` | Full single-file graph for Gephi / Cytoscape. |
| `network_dynamic.gexf` | Same graph with `start` years on nodes/edges for Gephi's timeline (when datable). |
| `graph_interaction.gexf` | **Interpersonal social network** (person↔person ties only) — the headline SNA. |
| `graph_affiliation.gexf` | Two-mode membership/biography network (person→org/event/place). |
| `graph_discourse.gexf` | Stance + co-occurrence layer (attitudes, co-presence). |
| `narrative.gexf` / `narrative_transitions.csv` | Bearman-Stovel narrative-sequence network: event-element transitions across the corpus (opt-in `export.narrative_network`). |
| `graph_report.json` | NetworkX QA: components, weighted brokerage, bridges, articulation points, signed structural balance, + a `quality_pillars` summary (provenance/consistency from real data; accuracy/completeness/timeliness as labelled coverage proxies). Opt-in `export.graph_metrics`. |
| `timeline.csv` | Chronological dated events (`letter_id` included). |
| `entities.json` | Full resolved entity records with aliases, tags, attributes. |
| `raw_extractions.jsonl` | Per-document extractions (provenance). |
| `checkpoints/` | Resumable progress (JSONL). |

Import `graph_interaction.gexf` for social-structure analysis, `network.gexf` for
everything, or load the two CSVs via the Data Laboratory (nodes first, then edges)
and filter on the `tie_class` column.

> **A mention is not a social tie.** In a corpus of NSDAP autobiographies almost
> everyone *co-occurs with* and *expresses an attitude toward* Hitler, but few
> actually knew him. Edges are therefore labelled by `tie_class` and headline
> centrality (`int_*`) is computed on the **interaction** layer only, so hubs
> reflect documented social relationships rather than the corpus topic. Public
> figures carry `attr_reference_figure=true` so you can include or exclude them.
> Run Gephi's Statistics on `graph_interaction.gexf` to get centrality/community
> computed on the social layer alone.

> **Full operating guide:** see [INSTRUCTIONS.md](INSTRUCTIONS.md) for the three
> modes, English vs German runs, coreference, the mandatory-membership
> assumption, evidentiary tiers, evaluation, and adapting to new inputs.
> **Validate output:** see [evaluation/README.md](evaluation/README.md).

---

## CLI

```bash
python main.py --config config.yaml                 # run everything
python main.py --config config.yaml --resume        # resume after a crash
python main.py --config config.yaml --stage extract # extraction only
python main.py --config config.yaml --stage analyze # re-run analysis on checkpoint
python main.py --config config.yaml --mode api      # override mode
python main.py --config config.yaml -v              # debug logging
```

Quality/recall toggles (all off by default, for A/B without editing YAML):

```bash
--structured-output   # schema-constrain extraction JSON (recommended for ollama)
--recall-pass         # re-prompt the whole doc for cross-chunk ties the pass missed
--verify-relations    # re-check each LLM edge against its evidence (tag/drop)
--link-authors        # fold a lone surname into the author it uniquely names
--batch-post-llm      # gemini_batch: route dedup/review/verify through the Gemini key
```

Stages: `ingest` -> `extract` -> `analyze`. `--stage analyze` reuses the
checkpoint so you can re-tune dedup/quality/export without re-extracting.

---

## Architecture

```
Input docs
  └─ core/preprocessor   PDF/DOCX/RTF/HTML/TXT -> normalized plaintext
  └─ core/chunker        sentence-aligned overlapping chunks
  └─ core/foundation     [ALWAYS] spaCy + GLiNER -> merge -> coref -> dates
        └─ core/spacy_engine, gliner_engine, entity_merger, coreference, date_extractor
  └─ intelligence/       [MODE] relationships + refined entities + timeline
        ├─ api_backend       (Claude/OpenAI/Bedrock)   prompts.py + json_repair.py
        ├─ ollama_backend    (local LLM)               prompts.py + json_repair.py
        └─ python_backend    relationship_patterns.py + embedding_utils.py
  └─ checkpoint/manager  crash-safe append-only JSONL, resumable
  └─ postprocess/
        ├─ aggregator         per-doc -> corpus tables
        ├─ deduplicator       3-layer: aliases -> exact -> bucketed fuzzy
        ├─ llm_dedup          [api/ollama, optional] LLM merges rules missed
        ├─ ontology           relation-type alignment (domain/config)
        ├─ enricher           [api/ollama, optional] subtype + attributes
        ├─ quality_review     rules (+ optional LLM); protects salient/high-degree nodes
        ├─ relation_verify    [opt-in] re-check each LLM edge against its evidence
        ├─ canonical_inference co-occurrence + domain edges
        ├─ bipartite          [opt-in] two-mode affiliation -> co_affiliated actors
        ├─ tagger             scope / relevance_tier / connection_quality
        ├─ gephi_builder      NetworkX metrics -> node/edge tables
        └─ exporter           CSV / Parquet / JSON / GEXF / JSONL
  └─ domain/               pluggable knowledge (aliases, inference rules)
```

LLM is called only in **extraction** (per chunk, optionally schema-constrained via
`structured_output`; an opt-in `recall_pass` re-prompts the whole doc for missed
cross-chunk ties) and these **opt-in** analyze passes: `verify_relations` (re-check
each edge against its evidence), `llm_dedup`, `enrichment`, and `quality_review`
(when `llm_review` is on). Dedup itself is rule-based; the LLM passes are additive
and gated by config. Every LLM-assisted step is guarded against hallucination
(merge-group caps, batch-drop caps, salient/high-degree entity protection).
Change history: [CHANGELOG.md](CHANGELOG.md).

### Deduplication

Three layers, in order:

1. **Known aliases** - `domain/<name>/aliases.py` (`alias -> canonical`).
2. **Exact** - identical normalized name + label.
3. **Bucketed fuzzy** - Levenshtein ratio within `(label, initial)` buckets,
   gated by per-type thresholds (`PERSON 0.85`, `ORG 0.72`, `LOCATION 0.82`,
   `EVENT 0.92`).

**Blocking rules** prevent over-merging: family members (shared surname, different
given name), events with mismatched years, and substring locations
(`Paris` ≠ `Paris, Texas`).

### Analytical tags

- **`entity_scope`** - `macro` (broad hubs / high reach) vs `specific`.
- **`relevance_tier`** - `core` / `secondary` / `peripheral` (blended mention
  frequency, document spread, degree).
- **`connection_quality`** (edges) - `direct` (text evidence), `structural`
  (inferred / canonical), `ideological` (affinity relations).

### Edge origins

- `extracted` - asserted in the text (LLM or dependency parse).
- `inferred` - co-occurrence: within-document window proximity (spans chunk
  boundaries) and cross-document co-mention.
- `canonical` - added by domain inference rules.

---

## Expanding an existing network

Already have a curated graph and want to grow it from new documents without it
drifting into new relation types or off-target entity kinds? Turn on `expansion`:
it reads the schema of a prior run (`source:` a run dir, its `gephi_edges.csv`, or
a `.gexf`) and locks this run to it — only the relation types already there
(synonyms still map, so `"worked for"` → `employed_by`), only the entity kinds
already there. Off-vocabulary edges are dropped (strict) or tagged. Works in
`--stage analyze`. See INSTRUCTIONS §10c.

---

## Adding a domain

Copy `domain/generic/` to `domain/<yourname>/`, fill in `aliases.py`,
`entity_config.py`, and (optionally) `inference_rules.infer_edges`, then set
`domain.name: <yourname>` in the config. No core code changes needed.

A domain package may also export, and the pipeline will automatically use:

| Module / attribute | Effect |
|--------------------|--------|
| `gliner_labels.LABELS` + `LABEL_TO_TYPE_MAP` | Override the zero-shot NER labels and their canonical-type mapping. |
| `spacy_patterns.PATTERNS` | EntityRuler patterns merged into the spaCy pipeline before statistical NER. |
| `prompts_*.SYSTEM_EXTRACTION` / `SYSTEM_QUALITY_REVIEW` | Replace the LLM system prompts (api/ollama modes). |
| `inference_rules.infer_edges(entities, edges)` | Add `origin="canonical"` edges. |

### Bundled domain: `nazi_era`

`domain/nazi_era/` is a worked example for **Weimar/Nazi-era German primary
sources (1919-1945)**, e.g. the Theodore Abel Papers (1934 NSDAP
autobiographies). It ships:

- **523 aliases** (German↔English, abbreviations, name variants),
- **24 GLiNER labels** + 209 spaCy EntityRuler patterns (ranks, units, events),
- full **SA / SS / Wehrmacht rank ladders** with an `identify_rank_org` resolver,
- **German NLP** (noble particles, umlaut transliteration, name parsing),
- **historical validation** (organization existence windows / anachronism flags),
- a **4-tier evidence membership-inference engine** (`canonical_inference.py`).

Run it with the pre-built config:

```bash
python main.py --config domain/nazi_era/config_nazi_era.yaml
```

### Bundled domains: `influencewatch`, `orem_opal`

Two affiliation-dense English domains, built on the generic-package contract:

- **`influencewatch`** — modern US political influence (dark money). PACs/shells/
  foundations as ORG nodes, money-flow (`funded`/`donated_to`/`granted`) +
  governance (`board_member_of`/`owns`/`subsidiary_of`) relations, funding amounts
  on the edge as `qual_monetary_value`. Affiliation projection ON: people on the
  same board/PAC get a `co_affiliated` edge.
- **`orem_opal`** — Oregon multi-agency disaster response. An *inter-organizational*
  network: agencies/NGOs/tribes are the nodes, and the projection makes them the
  actors sharing a disaster EVENT (`affiliation_actor_kinds: [ORG, INSTITUTION]`),
  so two agencies that responded to the same fire link directly. `qual_jurisdiction`
  / `qual_location` pin scope.

```bash
python main.py --config domain/influencewatch/config_influencewatch.yaml --mode ollama
python main.py --config domain/orem_opal/config_orem_opal.yaml --mode ollama
```

### Academic sensitivity analysis (`edge_source`)

Every edge in `gephi_edges.csv` / `network.gexf` carries an `edge_source` so you
can rebuild the network at progressively looser evidentiary thresholds:

| Network | Includes edge_source |
|---------|----------------------|
| **Conservative** | `llm_extracted`, `langextract_extracted`, `rule_extracted`, `metadata` (stated in text / verified record) |
| **Moderate** | + `canonical_inferred` (membership inferred from a detected signal) |
| **Full** | + `rule_cooccurrence`, `pipeline_inferred` (co-occurrence + mandatory-membership assumption; weakest layers) |

The pipeline extracts what the documents state and never fabricates claims;
inferred edges are labeled so they can be included or excluded explicitly.

---

## Robustness

- **JSON repair** - 5 escalating levels recover malformed LLM output.
- **Faithfulness guards** - hallucination signals tagged on edges (filter in Gephi),
  never silently dropped unless you opt in: `evidence_unverified` (the quote isn't
  verbatim in the chunk), `evidence_ungrounded` (the quote names neither endpoint -
  anchor check), `type_violation` (endpoint types contradict the relation's signature,
  e.g. `born_in` pointing at an org). Type violations drop with
  `ontology.drop_type_violations`.
- **Checkpointing** - append-only JSONL; `--resume` skips completed docs; a
  truncated final line from a crash is detected and discarded.
- **Fail-soft** - an unreadable file or a failed chunk logs a warning and falls
  back to foundation-only output instead of aborting the run.
- **Model fallbacks** - missing `en_core_web_trf` falls back to `en_core_web_sm`
  then a blank sentencizer; missing sentence-transformers degrades to neutral
  similarity scores.
```

---

## License

[MIT](LICENSE) - Copyright (c) 2026 moshel01
