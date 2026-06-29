# Aspirational pipeline — if compute/budget were not the constraint

What we'd build for the "perfect" version, recorded so the rationale survives. The
shipped pipeline is deliberately local-first (RTX 3070 8 GB, one box, free/cheap
inference) and evidence-faithful. Everything here is deferred because it spends a
resource we ration: GPU/VRAM, API tokens, latency, or infra. Newest thinking first.

Format per item: **Now** (what we do) / **Unlimited** (what we'd do) / **Why deferred**.

---

## Extraction quality

### Cross-chunk relation extraction (Chain-of-Agents, Google 2024)  [partially shipped]
- **Now.** Coref is chunk-local; a typed relation between two third parties split
  across chunks is lost. We floor the weak part with window co-occurrence and span
  the boundary with `chunking.overlap_chars`. The opt-in `intelligence.recall_pass`
  re-prompts over the REASSEMBLED whole document (entities + already-found relations
  in hand) for the ties the per-chunk pass missed - the recall half of an L3X
  generate-then-verify loop, one extra call per doc. gemini_batch sidesteps the
  ceiling entirely (whole-doc prompts). The chunk-local coref limit itself remains.
- **Unlimited.** A Chain-of-Agents pass: one worker agent per chunk, each passing a
  short "communication unit" (entities introduced, open referents) forward to the
  next, a manager agent synthesizing the document-level relation set. This dissolves
  the chunk-local ceiling directly — the right fix, not a floor.
- **Why deferred.** N sequential LLM calls per document with growing context; on a
  local 8 GB box that is minutes per doc × 540 docs. Viable only with a fast cheap
  API and parallelism.

### Entity side information (DocZSRE-SI, Chanthran et al. 2026)
- **Now.** Candidates go to the relation prompt as `"name" [TYPE]`.
- **Unlimited.** Attach a one-line description + hypernym per entity ("Brookings —
  centrist US think tank") so the model disambiguates relations it can't infer from
  the surface. +11.6% macro-F1 in the paper.
- **Why deferred.** Needs an entity knowledge source during extraction. Concrete
  path: turn Wikidata linking on and move it *ahead* of relation extraction, then
  pass the QID descriptions as side information. Today linking is post-extraction.

### Multi-agent verify / align / resolve (KARMA, Lu et al. 2025; AEC, Guo 2026)  [partially shipped]
- **Now.** Deterministic guardrails: `json_repair`, ontology alignment, hallucination
  guards, one optional LLM quality-review pass. The verification half of KARMA ships:
  `quality.verify_relations` re-checks each LLM edge against its own evidence (tag or
  drop), `ontology.check_functional_consistency` flags a subject with two different
  birthplaces/birthdates (knowledge alignment), the type-signature gate rejects edges
  whose endpoints contradict the relation's argument types, and conflict detection
  surfaces contradictory dyads (`graph_metrics._polarity_conflicts`). AEC's schema-
  violation guard ships as `intelligence.structured_output`: a JSON schema constrains
  generation at the grammar level so the model can't emit invalid structure at all
  (stronger than re-prompting after the fact). Caveat measured on the pilot: a weak
  local model is a weak SELF-verifier (it over-rejects valid edges), so verify is most
  trustworthy with a model at least as strong as the extractor.
- **Unlimited.** KARMA's full nine-agent loop (entity discovery → relation extraction
  → schema alignment → conflict resolution, each its own LLM with cross-agent
  verification), each step a separate, ideally STRONGER model than the extractor.
- **Why deferred.** The full nine-agent loop is many LLM calls per chunk; our own prior
  review (Serdiukov 2026) found iterative prompting trades latency for little gain on a
  small local model. The cheap, high-value half (one verify pass + schema constraint)
  shipped; the rest waits for API scale or a stronger verifier model.

### Code-based event extraction (SALE 2026; AEC 2026)
- **Now.** We extract actors + typed ties + a light timeline; the narrative-sequence
  net (Bearman-Stovel) is the event layer, coarse keyword buckets.
- **Unlimited.** Full document-level event extraction (triggers + argument roles) as
  Python-class instantiation with a Code-LLM, feeding a richer event graph.
- **Why deferred.** Different task from the actor-tie network; re-plumbs extraction
  around event structures we don't currently use. Revisit only if the research turns
  event-centric.

## Canonicalization & identity

### LLM clustering for canonicalization (KGGen, Stanford 2025)
- **Now.** Rule-based dedup (alias/exact/fuzzy, acronym/subset/demonym folds) + a
  guarded LLM merge pass; optional Wikidata QID consolidation.
- **Unlimited.** KGGen-style iterative LLM clustering over entities *and* relation
  types for aggressive canonicalization, benchmarked on something MINE-like.
- **Why deferred.** More LLM calls and a hallucination surface; our guards exist
  precisely because local models over-merge. Cheap-API budget could fund it with the
  guards kept load-bearing.

### Full LLM entity linking (LELA, Haffoudhi 2026)
- **Now.** Dedup + optional Wikidata QID linking.
- **Unlimited.** LELA's candidate-generation + LLM-reranking disambiguation against a
  KB for every salient mention.
- **Why deferred.** Spends the LLM budget reserved for relations. The cheap win (ORG
  article/plural folding) already shipped in `deduplicator._clean_org_surfaces`.

## Graph store, indexing & retrieval

### Neo4j backend + GDS (neo4j-labs/llm-graph-builder)
- **Now.** Export GEXF/CSV for Gephi; NetworkX computes what Gephi can't.
- **Unlimited.** Persist to Neo4j: Cypher queries, the GDS algorithm library,
  per-edge provenance as first-class, and incremental updates — alongside the GEXF
  export, not replacing it.
- **Why deferred.** Infra (a running DB) for a deliverable that is currently a static
  Gephi file. Pure additive value when a queryable KG is wanted.

### Community hierarchy + summaries (Microsoft GraphRAG)
- **Now.** Community detection is left to Gephi (one click); we don't summarize.
- **Unlimited.** Leiden community hierarchy + an LLM-generated summary per community
  ("this cluster is the Detroit philanthropy network, centered on Ford/Kresge"),
  giving an analytical layer and global+local query support.
- **Why deferred.** Community summaries are many LLM calls and drift toward RAG/QA,
  which is not this pipeline's purpose (see LightRAG below).

### Graph-indexed RAG / QA (LightRAG, Guo 2024)
- **Now.** We produce a graph; we don't answer questions over it.
- **Unlimited.** A dual-level retrieval layer so an analyst can ask the corpus
  questions and get graph-grounded answers.
- **Why deferred.** Different problem (retrieval/QA, not SNA extraction). Listed so
  the boundary is on record.

## Evaluation & QA

### KGC-2026 semantic-backbone quality pillars  [partially shipped]
- **Now.** `graph_report.json` carries a five-pillar block (`quality_pillars`):
  provenance (edge_source coverage) and consistency (polarity conflicts + type
  violations) from real data; accuracy/completeness/timeliness as labelled coverage
  proxies. Plus the eval harness (P/R/F1 by tier) for gold-scored runs.
- **Unlimited.** Replace the three proxies with gold-scored pillars: real recall
  (completeness) against an annotated subset, edge-recency tracking (timeliness), and
  a per-pillar trendline across runs in a dashboard.
- **Why deferred.** The proxy version was the cheap win and shipped; true accuracy/
  completeness scoring needs per-corpus gold we don't have at run time.

### RAG benchmarking & agentic GraphRAG (BenchmarkQED; Neo4j Agentic GraphRAG; Memgraph Atomic GraphRAG)
- **Now.** We build a graph and score extraction (P/R/F1 by tier); we don't answer
  questions over the graph or benchmark answer quality.
- **Unlimited.** A retrieval/QA layer on top of the network — schema-inferring agents
  that route between vector and graph traversal (Neo4j Agentic GraphRAG), a
  single-query execution plan that returns context plus a decision trace (Memgraph
  Atomic GraphRAG), and Microsoft BenchmarkQED (AutoQ synthetic local→global queries +
  automated answer scoring) to measure it.
- **Why deferred.** All three are retrieval/QA, not SNA extraction — same boundary as
  LightRAG below. They presume a served graph DB and an LLM answer loop; this pipeline
  ships a static Gephi/GEXF deliverable. Revisit only if a queryable, question-
  answering KG becomes a requirement (then pair with the Neo4j backend item above).

## Ingestion & sources

### Structured wiki ingestion (beyond prose)  [partially shipped]
- **Now.** `core/wiki.py` pulls clean article PROSE via the MediaWiki API
  (`prop=extracts&explaintext`) — the relation extractor reads it like any document.
  Infoboxes, the wikilink graph, and categories are dropped (prose only).
- **Unlimited.** Parse the infobox into structured edges directly (spouse / party /
  employer / founded → typed relations with no LLM call), treat categories as
  affiliation groups (the two-mode projection already exists), and read the curated
  wikilink graph as a weak co-occurrence layer. For Wikipedia specifically, pull the
  matching Wikidata statements (qualified facts) and fold them in as `metadata`-tier
  edges. A near-complete actor network from the structured wiki alone, the prose only
  filling gaps.
- **Why deferred.** Infobox/template parsing is per-wiki brittle (every template differs);
  raw wikilinks are too dense to be ties without heavy filtering. Marginal SNA gain over
  prose-extraction for most corpora; revisit for a wiki-centric project (a Fandom
  character network, a Wikipedia-derived domain graph).

### Chat / correspondence exports (mbox, Discord/Slack, subtitles)
- **Now.** Social connectors cover live platforms (reddit/hn/lemmy/mastodon/bluesky/
  telegram/truthsocial/twitter); plain `.txt` ingests anything dumped to text.
- **Unlimited.** First-class readers for email mbox (a correspondence network from the
  From/To/In-Reply-To headers — the structure is explicit, like the social graph),
  Discord/Slack export JSON, and subtitle/`.srt` tracks (a per-scene speaker network to
  complement the screenplay co-presence parser).
- **Why deferred.** Each is a small adapter, not a research bet — built on demand when a
  corpus of that shape appears. The pattern is set (return Documents + any explicit
  structure as asserted edges); only the parser differs.

## Infrastructure

### Bigger models, longer context, parallel calls
- **Now.** qwen3.5:9b on 8 GB, `think:false`, 5k-char chunks, sequential.
- **Unlimited.** A frontier model (or DeepSeek-V3 at scale) with 128k context to read
  whole documents (no chunking → no chunk-local ceiling), many requests in flight.
- **Why deferred.** VRAM and tokens. The cheap-API path (DeepSeek-chat) is the
  pragmatic middle: cheap enough to gate per chunk, good enough for structured JSON.

---

When an item here becomes cheap (a fast cheap API, a spare GPU, a queryable-KG
requirement), pull it into ARCHITECTURE.md's grounding section and ship it behind a
config flag. The default must stay local-first and evidence-faithful.
