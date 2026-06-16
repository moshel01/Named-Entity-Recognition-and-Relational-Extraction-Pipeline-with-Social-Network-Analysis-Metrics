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
   URL fetch + trafilatura main-content extraction, or a bounded whole-site crawl
   (`core/crawler.py`, `io.crawl` / `--crawl`). Author/narrator detection in
   `core/foundation.py`.
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
   for qwen3.5), `python_only` (dependency rules + sentence co-occurrence, no LLM),
   `langextract` (few-shot + char-grounded). The LLM only does relations.
6. **Aggregate + resolve** (`postprocess/aggregator.py`, `deduplicator.py`,
   `llm_dedup.py`, `identity_resolution.py`) — canonicalize and merge entities
   (author-fold, acronym/subset folds, demonym folding) under hallucination
   guards.
7. **Classify + infer** (`ontology.py`, `tie_classes.py`,
   `canonical_inference.py`, `backbone.py`) — align relations to the ontology; tag
   each edge with tie-class, connection-type, polarity; add inferred layers
   (canonical membership, window/cross-doc co-occurrence with Newman projection
   weighting); thin the dense co-occurrence layer with the disparity-filter backbone.
   Optional Wikidata QID identity consolidation when linking is on.
8. **Quality + tier** (`quality_review.py`, `evidence_tiers.py`) — guarded
   entity/edge filtering (tag by default, filter where the method demands it); map
   every `edge_source` to an evidence tier.
9. **Export** (`gephi_builder.py`, `exporter.py`, `graph_metrics.py`,
   `narrative.py`, `codebook.py`) — nodes/edges CSV, multi-view GEXF, NetworkX
   structural metrics (weighted brokerage, bridges, signed balance), the
   Bearman-Stovel narrative-sequence network, per-run codebook.xlsx.

## Tech stack (as built)
- **NER:** GLiNER2 (`fastino/*`) + spaCy (`en_core_web_trf` / `de_core_news_lg`).
  GLiNER v1 (`urchade/*`) is deprecated and does not load.
- **Coref:** fastcoref (`biu-nlp/f-coref`), in-process or microserviced.
- **Relation extraction:** Ollama (`qwen3.5:9b` on the 8 GB box; bigger models on
  the 16 GB machine) / Anthropic API / dependency rules / LangExtract.
- **Graph/SNA:** NetworkX → GEXF for Gephi. NetworkX computes only what Gephi
  cannot (weighted Burt constraint/effective size, bridges, articulation points,
  signed structural balance, disparity-filter backbone, Newman projection weights).
- **Ingestion:** static fetch (`requests`) + trafilatura main-content extraction
  (drops nav/ads/sidebars/boilerplate), BeautifulSoup fallback; Docling
  (books/PDF layout+OCR, optional). Whole-site crawl in `core/crawler.py`
  (sitemap+scoped-BFS, robots.txt, per-host rate limit, page/depth/size caps,
  fetch-once). Not ScrapeGraphAI/Crawl4AI (see grounding).

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
  timeline + dynamic GEXF. The narrative-*sequence* network (`postprocess/
  narrative.py`) builds it directly: event-element categories as nodes, narrative
  order (timeline year, then telling order) as directed transitions aggregated
  across the corpus -> narrative.gexf + narrative_transitions.csv. v1 element scheme
  is coarse keyword buckets, refinable per domain via `Domain.narrative_rules()`.
- **Weighted-network methods (Newman 2001; Serrano et al. 2009).** Co-mention is a
  one-mode projection of an entity x document bipartite graph: pairs are Newman-
  weighted 1/(k-1), and the dense layer is reduced to its disparity-filter backbone
  rather than by a global cutoff.
- **Signed networks (Cartwright & Harary).** Edge polarity feeds a structural-
  balance fraction (balanced friend/enemy triads) in graph_report.json.
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
- **Multi-agent KG enrichment (KARMA, Lu et al. 2025).** Their conflict-resolution
  agent flags contradictory edges before integration; we take the detection half as
  an offline rule (`graph_metrics._polarity_conflicts`): dyads that are both ally
  and enemy, which signed balance otherwise drops as net-zero. The full nine-agent
  verify/align/resolve loop is not adopted — it assumes API-scale compute, and
  schema alignment (`ontology.py`) + the hallucination guards + quality review
  already cover align/verify on the local box.
- **Verified extraction (Serdiukov et al. 2026).** Schema-guided JSON with a
  recovery module that rescues systematically corrupted output. Their finding —
  most LLM extraction failures are correctable formatting, not semantic — is
  exactly what `json_repair.py` does (the multi-`json`-block recovery included);
  their nine-model benchmark also lands on a Qwen model as the best
  efficiency/low-hallucination fit, matching the `qwen3.5:9b` default.
- **Two-stage scenario-prompt RE (Zhao et al. 2025).** Zero-shot document-level RE
  by constraining the LLM to a predefined relation schema instead of letting it
  free-form. We adopt the schema half: the generic (non-domain) path falls back to
  a default relation ontology (`ontology.GENERIC_RELATION_ONTOLOGY`) that constrains
  the extraction prompt and aligns the verbose tail, so a web crawl yields a usable
  edge vocabulary (funded/led/member_of...) instead of a unique verb phrase per edge.
  Domain configs still supply their own ontology; `ontology.enabled: false` opts out.
- **Type-consistency & anchor grounding (Tran et al. 2025, LLM+ASP; Yang et al.
  2026, AEVS).** Both fight hallucination by checking extractions against structure:
  ASP rejects relations whose argument types violate a signature; AEVS grounds every
  triplet element to a source-text span (anchor discovery -> grounded extraction ->
  restoration verification). We adopt the type half as a plain-Python consistency gate
  (`ontology.check_relation_types`: an endpoint type contradicting a relation's
  signature -> `type_violation`, running after the existing suspect_membership rule) -
  no clingo, a 14-relation check doesn't need an answer-set solver. The provenance half
  AEVS formalizes is mostly already here: local GLiNER spans are the anchors, every
  relation carries its evidence text, and `evidence_tiers` ranks faithfulness. The one
  net-new AEVS lever - verify a relation's endpoints actually occur in its evidence span -
  shipped as `intelligence.base._tag_ungrounded_evidence` (`evidence_ungrounded`,
  tag-only; author endpoint exempt for first-person coref). Complements the existing
  `evidence_unverified` (quote-not-in-chunk) check.
- **Membership-universe context (Bosshart et al. 2026, NBER).** The Abel corpus
  is an opt-in sample of committed early members, not a random draw — read the
  network as descriptive of this corpus, not inferential for the movement (the
  codebook note states this).

## Possible future additions (deferred)
Not on the current track; recorded so the rationale survives.

- **Bearman narrative-sequence network — BUILT (v1).** `postprocess/narrative.py`
  now emits it (element categories as nodes, timeline order as directed transitions
  aggregated across the corpus). The deferred node-semantics decision was resolved
  pragmatically for v1: an element is a coarse keyword-bucketed event category, an
  arc is consecutive-in-time succession (not causation). Refinements still open:
  finer/learned element abstraction, causal vs. sequential arcs, per-narrative
  structural-equivalence (Bearman's blockmodel of role positions).
- **JS-rendered / interactive web targets.** The whole-site crawler
  (`core/crawler.py`) handles server-rendered HTML. Dynamic SPA sites (content
  built client-side by JS) would still need a headless browser or an LLM/agent
  scraper (ScrapeGraphAI, WebScraper-MLLM). Revisit only if such targets appear;
  the LLM budget stays on relation extraction.
- **KG link prediction / completion** is deliberately out of scope: inventing
  edges breaks the evidence-faithful, tag-don't-filter design. Listed here so
  it's a decision on record, not an oversight.
- **Graph-indexed RAG (LightRAG, Guo et al. 2024).** A retrieval/QA system: it
  builds a KG index over a corpus for dual-level query answering. Different problem
  — this pipeline extracts a static SNA graph, it doesn't answer queries — so not
  adopted. The one transferable piece, incremental graph update without a full
  rebuild, is already covered by checkpoint/resume + the fetch-once crawl.
- **LLM entity linking (LELA, Haffoudhi et al. 2026).** A modular EL framework
  (zero-shot NER -> candidate generation -> LLM reranking/disambiguation against a
  KB). The disambiguation passes would spend the LLM budget reserved for relations,
  and entity resolution is already handled by dedup/identity_resolution + optional
  Wikidata QID linking. Not adopted as a framework; its lesson — fold article/plural
  ORG variants ("the Rockefeller Foundation", "Knight Foundations") — shipped in
  `deduplicator._clean_org_surfaces` (strip leading "the"; singularize a plural only
  when its singular already exists, so real plural names are kept).
- **Generating unseen temporal facts (Amalvy & Huang 2026).** A method for building
  contamination-free TKGE benchmarks by forecasting future quadruples then generating
  text for them. Useful for benchmark hygiene, but it's synthetic fact generation in
  the invent-unseen-edges space this pipeline excludes, and the harness already scores
  against real gold. Not adopted.
- **Code-based event extraction (SALE, Xu et al. 2026; AEC, Guo et al. 2026).** Both
  cast document-level EVENT extraction (triggers + argument roles) as Python class
  instantiation, AEC adding a multi-agent retrieve/plan/code/verify loop. Not adopted:
  this pipeline builds an actor-tie network, not an event-argument structure (the
  narrative-sequence net is the deliberately-light event layer), and the code-schema
  re-plumbing + multi-agent passes cost many more local LLM calls for a task we don't
  run. The convergent lesson — schema-as-code reduces structural violations — is
  already covered cheaply by the JSON schema + `json_repair` + ontology alignment.
- **Entity side information for zero-shot RE (DocZSRE-SI, Chanthran et al. 2026).**
  Feeding per-entity descriptions + hypernyms into the RE step lifts macro-F1 ~11.6%
  without synthetic data. A real recall lever and the most promising deferred item,
  but a faithful version needs an entity knowledge source during extraction. Concrete
  path: when Wikidata linking is on, move it ahead of relation extraction and pass the
  entity descriptions into the prompt as side information.
- **SLM-proxy knowledge mining (Falconer, Zhang et al. 2026).** An LLM plans an
  executable workflow and generates supervision to train small BERT proxies
  (`get_label`/`get_span`) that carry the bulk inference, ~90% cheaper and 20x faster
  at corpus scale. A scale play for "massive corpora"; our corpus is 540 docs and the
  small-model role is already filled by GLiNER (free local NER). Training per-task
  relation proxies would add a supervision/training loop for a cost we don't have at
  this scale. Revisit only if the corpus grows by orders of magnitude.
- **Hyper-relational KG construction (LLHKG, Zhu et al. 2026).** A PLM-for-KG survey
  plus a lightweight-LLM framework for n-ary / qualified facts (a triple plus qualifier
  key-values, Wikidata-statement style, claimed comparable to GPT-3.5). We already hang
  qualifiers on edges (`period`, `year`, evidence, `edge_source`); promoting them to
  first-class qualified statements is a data-model change for marginal SNA gain. Not
  adopted; the survey confirms the extract -> canonicalize -> evaluate spine we follow.
