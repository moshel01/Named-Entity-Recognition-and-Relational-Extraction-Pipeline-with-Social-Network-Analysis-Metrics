# Architecture

Modular, local, zero-shot NER + relation-extraction pipeline that turns text
into a Gephi-ready social network. Two paths share one codebase:

- **Generalized** — English; any plain text or a scraped site.
- **NSDAP / Nazi-era** — German; the Theodore Abel Papers (540 RTF
  autobiographies of early NSDAP members), with a metadata spreadsheet merged in.

State of play is in CHANGELOG.md (newest first); operating rules in AGENTS.md.

## Pipeline stages
`main.py` (click CLI: `--config --mode --limit --run-name --resume --stage`)
runs:

1. **Ingest / preprocess** (`core/preprocessor.py`) — RTF + mojibake repair,
   encoding detection. Books optionally via Docling (`io.use_docling`); web via
   URL ingestion. Author/narrator detection in `core/foundation.py`.
2. **Chunk** (`core/chunker.py`) — sentence-aligned, with a hard char split for
   boundary-less text. `max_chars` ~5-6k, `overlap_chars` 400-600 (~7-12%); the
   overlap is what lets a typed relation survive a chunk boundary.
3. **Foundation NER** (`core/gliner_engine.py` + spaCy) — GLiNER2 (`fastino/*`)
   zero-shot labels + spaCy NER, merged in `core/entity_merger.py`. This is the
   entity layer and is **model-independent of the LLM**.
4. **Coreference** (`core/coreference.py`) — first-person narrator resolution
   (EN+DE) always; optional third-person pronoun resolution via fastcoref, in
   process or through the coref microservice (below), with a nearest-antecedent
   heuristic fallback. Coref is **chunk-local** (see the recall-ceiling note).
5. **Relation extraction** (`intelligence/`) — one of four tiers selected by
   `--mode`: `api` (Anthropic/OpenAI/Bedrock), `ollama` (local; `think:false`
   for qwen3), `python_only` (dependency rules + sentence co-occurrence, no LLM),
   `langextract` (few-shot + char-grounded). The LLM only does relations.
6. **Aggregate + resolve** (`postprocess/aggregator.py`, `deduplicator.py`,
   `llm_dedup.py`, `identity_resolution.py`) — canonicalize and merge entities
   (author-fold, acronym/subset folds, demonym folding) under hallucination
   guards.
7. **Classify + infer** (`ontology.py`, `tie_classes.py`,
   `canonical_inference.py`) — align relations to the ontology; tag each edge
   with tie-class, connection-type, polarity; add inferred layers (canonical
   membership, window/cross-doc co-occurrence).
8. **Quality + tier** (`quality_review.py`, `evidence_tiers.py`) — guarded
   entity/edge filtering (tag, don't drop); map every `edge_source` to an
   evidence tier.
9. **Export** (`gephi_builder.py`, `exporter.py`, `graph_metrics.py`,
   `codebook.py`) — nodes/edges CSV, multi-view GEXF, NetworkX structural
   metrics, per-run codebook.xlsx.

## Tech stack (as built)
- **NER:** GLiNER2 (`fastino/*`) + spaCy (`en_core_web_trf` / `de_core_news_lg`).
  GLiNER v1 (`urchade/*`) is deprecated and does not load.
- **Coref:** fastcoref (`biu-nlp/f-coref`), in-process or microserviced.
- **Relation extraction:** Ollama (`qwen3.5:9b` on the 8 GB box; bigger models on
  the 16 GB machine) / Anthropic API / dependency rules / LangExtract.
- **Graph/SNA:** NetworkX → GEXF for Gephi. NetworkX computes only what Gephi
  cannot (Burt constraint, effective size, bridges, articulation points).
- **Ingestion:** static fetch (`requests`) + trafilatura main-content extraction
  (drops nav/ads/sidebars/boilerplate), BeautifulSoup fallback; Docling
  (books/PDF layout+OCR, optional). Not ScrapeGraphAI/Crawl4AI (see grounding).

## The fastcoref microservice
fastcoref needs `transformers <5`; the main env runs `transformers 5.x` for
GLiNER2. So coref can run out-of-process in an isolated, light env
(`services/coref_service.py`, FastAPI on :8000). The pipeline POSTs chunk text
and re-attaches the returned char-offset clusters — identical logic to the
in-process path. Enable with `coreference.service_url`; unreachable falls back to
in-process fastcoref, then the heuristic resolver. Setup:
`pip install -r services/requirements-coref.txt`, then
`uvicorn services.coref_service:app --port 8000`.

## Design invariants (don't undo)
- **Tag, don't filter.** Anything Gephi can filter stays in the graph with a tag
  (`suspect_membership`, tie_class, connection_type, polarity, edge_source,
  evidence tier).
- **Evidence tiers are one map.** `postprocess/evidence_tiers.py` is the single
  `edge_source -> tier` ladder (conservative -> moderate -> full); evaluator and
  codebook import it.
- **Mention != tie.** Tie-class taxonomy (interaction/affiliation/participation/
  biographical/stance/cooccurrence/other) + multi-view GEXF. Co-occurrence is two
  layers (within-doc window proximity, cross-doc co-mention), both
  `rule_cooccurrence`/full-tier, never a tie.
- **Recall ceiling.** Coref is chunk-local, so a typed relation between two third
  parties split across chunks is lost; window co-occurrence floors the weak part,
  `overlap_chars` the boundary-spanning part. The author hub is globally resolved.
- **Guards are load-bearing.** Every LLM-assisted step is guarded (merge-group
  caps, alias caps, plausibility checks, batch-drop caps, salient protection).

## Methodological grounding (research_context/)
- **Pipeline, not joint tagging (Zheng et al. 2017).** We extract entities first
  (GLiNER2), then relations (generative). The joint sequence-tagging scheme is
  deliberately not used — incompatible with zero-shot, swappable backends.
- **Narrative networks (Bearman & Stovel 2000, "Becoming a Nazi").** "Becoming"
  is a process: edges carry temporal markers (`period`, `year`), and we emit a
  timeline + dynamic GEXF. A full narrative-*sequence* network (life events as
  nodes, narrative order as arcs) is deferred — see Possible future additions.
- **Physical vs ideological connection (transnationalism, Toro 2024).**
  `tie_classes.connection_type` tags each edge physical / ideological /
  organizational / biographical — cross-cutting tie_class (e.g. `fought_against`
  is a stance but a physical connection; `influenced_by` is a stance but
  ideological).
- **Entity disambiguation (Tamper et al.).** Entities are canonicalized and
  merged (aggregator/deduplicator/identity_resolution) before they enter the
  graph.
- **Web IE (ScrapeGraphAI-100k 2026; WebScraper-MLLM).** LLM/agent scrapers for
  dynamic, JS-rendered, interactive sites. Not adopted: heavy deps, and they'd
  spend the LLM budget on scraping. The static path uses trafilatura main-content
  extraction instead; the LLM tier is reserved for relation extraction. Revisit
  if JS-rendered targets become a requirement.
- **KG construction (Choi & Jung 2025; Zavarella 2026).** Extraction ->
  canonicalization -> evaluation, with hallucination handling — the spine this
  pipeline already follows; the evaluation harness reports P/R/F1 by tier.
- **Membership-universe context (Bosshart et al. 2026, NBER).** The Abel corpus
  is an opt-in sample of committed early members, not a random draw — read the
  network as descriptive of this corpus, not inferential for the movement (the
  codebook note states this).

## Possible future additions (deferred)
Not on the current track; recorded so the rationale survives.

- **Bearman narrative-sequence network.** Life events as nodes, narrative order
  as arcs, built from the existing per-author timeline. Highest-value research
  addition on the table. Deferred: needs a node/edge-semantics decision (what
  counts as an event node, how arcs carry order vs. causation) before building —
  it's a second graph model, not a tweak to the current one.
- **JS-rendered / interactive web targets.** Current ingestion is static-fetch +
  trafilatura (server-rendered HTML). Dynamic SPA sites would need a headless
  browser or an LLM/agent scraper (ScrapeGraphAI, WebScraper-MLLM). Revisit only
  if such targets appear; the LLM budget stays on relation extraction.
- **KG link prediction / completion** is deliberately out of scope: inventing
  edges breaks the evidence-faithful, tag-don't-filter design. Listed here so
  it's a decision on record, not an oversight.
