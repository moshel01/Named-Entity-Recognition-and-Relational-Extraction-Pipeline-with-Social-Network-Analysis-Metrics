# Changelog

Sequential record of what shipped. Newest first. Terse on purpose.

---

## Type-signature consistency gate, evidence-grounding anchor check, KGC quality-pillar report

Three precision/QA guards from a paper pass; the rest triaged.

- **Evidence-grounding anchor check** (`intelligence/base.py`, `_tag_ungrounded_evidence`,
  after Yang et al. 2026 AEVS): a typed relation whose evidence quote names NEITHER
  endpoint is likely misattributed (the model picked a real sentence that doesn't
  mention the pair). Tag `evidence_ungrounded`; never drop - coref-resolved first-person
  evidence uses pronouns, so the author endpoint is exempt and this stays a filterable
  signal, not a filter. Token-level match (so "Goebbels" grounds "Joseph Goebbels",
  multilingual-safe). Complements the existing `evidence_unverified` (quote-not-in-chunk).

- **ASP-style relation type signatures** (`ontology.py`, `RELATION_TYPE_SIGNATURES`):
  a relation whose endpoint entity types contradict its signature ("led" or
  "born_in" pointing at a place, "founded_by" reversed) is a likely misextraction.
  Tag `type_violation` (filterable in Gephi); drop only with
  `ontology.drop_type_violations`. High-precision: only constraining relations get a
  signature; loose stance/interaction ones are exempt, and a non-core entity type is
  a wildcard (no false flags on domain labels). Generalizes the hand-rolled
  suspect_membership check already in main.py. After Tran et al. 2025 (LLM + ASP for
  joint entity-relation extraction): the ASP solver is overkill for a 14-relation
  type check, so the consistency rule is plain Python - no clingo dependency.
- **KGC-2026 quality pillars** (`graph_metrics.quality_pillars`): `graph_report.json`
  now carries a five-pillar summary - provenance (edge_source coverage) and
  consistency (polarity conflicts + type violations) from real data; accuracy,
  completeness, timeliness as labelled coverage proxies (no gold at run time).
  Reporting overlay only, fail-soft.

Three RAG/QA resources triaged into ASPIRATIONAL.md (different problem - retrieval,
not SNA extraction): Microsoft BenchmarkQED (RAG eval harness), Neo4j Agentic
GraphRAG (autonomous KG + adaptive retrieval), Memgraph Atomic GraphRAG (single-query
pipeline). LELA/KGGen/DocZSRE-SI already on record there.

Three more papers triaged into ARCHITECTURE.md grounding/deferred: AEVS
(anchor-constrained extraction + provenance, Yang 2026) - grounding; its net-new
lever (verify endpoints occur in the evidence span) shipped as the anchor check
above. Falconer (SLM-proxy mining, Zhang 2026) and LLHKG (hyper-relational KG,
Zhu 2026) - deferred (wrong scale / data-model change for marginal SNA gain).

---

## API build-out (cheap-endpoint ready), cost gate, edge consolidation, negative anchoring

Built the API path out for cheap-or-expensive endpoints, plus three asked-about
methodologies. New ASPIRATIONAL.md tracks the unlimited-resources "perfect pipeline".

- **OpenAI-compatible endpoints** (`api_backend.py`, `ApiConfig`): provider `openai`
  now takes `base_url` + `json_mode`, so any cheap OpenAI-compatible host works
  (DeepSeek, Together, Groq, OpenRouter, local vLLM). The documented DeepSeek path is
  `deepseek-chat` (V3) - NOT `deepseek-reasoner` (R1), which burns tokens on reasoning
  and breaks structured output the same way qwen3.5 did. NER stays local/free; only
  relation extraction hits the API (the RetriCo-style hybrid is already the design).
- **Sparse-chunk cost gate** (`intelligence.skip_sparse_chunks`, default off): a
  relation needs two entities co-occurring, so a chunk without two distinct entities
  inside `sparse_window_words` can't yield one - skip its LLM call. Free NER still
  runs and the co-occurrence floor is untouched; zero recall loss, tokens saved. LLM
  modes only (never python_only). `chunks_skipped` recorded in checkpoint meta.
- **Cross-chunk edge consolidation** (`aggregator.py`): `overlap_chars` puts the
  boundary sentence in two chunks, so the same relation was extracted twice and
  inflated edge weight. Now drop duplicates with identical doc/endpoints/type AND
  verbatim evidence; distinct-evidence repeats (real corroboration) and cross-doc
  repeats are kept.
- **Negative anchoring in the extraction prompt** (`prompts.py`): explicit "NEVER do
  these" examples (pronoun-as-entity, sentence-as-relation-type, inferred-not-stated
  relations, translated evidence, markdown fences) so a cheap model complies first try.

The disparity-filter backbone (asked about) was already shipped (`backbone.py`,
Serrano 2009). Papers triaged into ASPIRATIONAL.md with why each is deferred: KGGen,
Neo4j LLM graph builder, Microsoft GraphRAG, KGC-2026 quality pillars, KARMA,
Chain-of-Agents.

---

## ORG name folding, multilingual GLiNER default, paper triage

Two delegated follow-ups, plus a round of papers.

- **ORG display-name cleanup** (`deduplicator.py`, after LELA's entity-linking
  lesson): strip a leading "the" (always) and singularize an org-suffix plural
  only when the singular already exists as a node, so "the Lilly Endowment" /
  "Knight Foundations" fold onto the bare/singular form while a genuinely plural
  name (Open Society Foundations, Council on Foundations) is left alone. English
  "the" only - German der/die/das stays so party names ("Die Linke") survive.
  ORG/INSTITUTION only; runs after every other merge and re-folds collisions.
- **GLiNER default -> multilingual** (`config.py`): the generic default is now
  `fastino/gliner2-multi-v1` (serves English and German, fits 8GB, lighter on CPU)
  instead of the English-only `large-v1` - which was also the model in the
  foundation segfault combo. large-v1 stays available for English-only max-NER runs.

Papers reviewed:
- **SALE** (code-based DEE) and **AEC** (multi-agent code-EE) declined. Both target
  document-level EVENT extraction (triggers + argument roles), not the actor-tie
  network this pipeline builds; their shared code-as-class-schema + multi-agent
  refinement would mean re-plumbing extraction around event structures and many more
  local LLM calls (the verified-extraction paper already found iterative prompting
  trades latency for little gain on a small local model).
- **DocZSRE-SI** (entity side information for zero-shot RE) deferred, not declined.
  Per-entity descriptions/hypernyms in the RE prompt are a real recall lever (+11.6%
  F1 in the paper), but a faithful version needs a knowledge source we don't cheaply
  have during extraction. Path on record: feed Wikidata descriptions as side info if
  entity linking moves ahead of relation extraction.
- **"Pick a Document Extraction Platform 2026"** webinar: validation, nothing to
  build - it endorses pydantic schema enforcement, per-field source grounding (our
  per-edge evidence + tiers), layout-preserving preprocessing (Docling), and local
  quantized Qwen, all already in place.

---

## Generic relation ontology, smart-quote JSON repair, crawl data verified

The InfluenceWatch ollama crawl verified end-to-end: 392 typed edges (funding /
board / employment ties python_only can't produce - Henry Ford II trustee_of Ford
Foundation, Ford Foundation granted_to NAACP LDF), and reference stripping cut
publisher nodes to 5 vs the wiki run's 166. Two issues that run surfaced, both fixed.

- **Relation-type sprawl -> a generic relation ontology** (`postprocess/ontology.py`,
  after the two-stage scenario-prompt RE work, Zhao et al. 2025). The generic path
  had no relation schema, so qwen invented a verb phrase per edge
  (`sent_letters_to_requesting_compliance_info_about_funding_of_affiliates_of` ...) -
  unusable as an SNA edge vocabulary. `resolve_relation_ontology` now falls back to a
  31-relation default (canonical names aligned to the `tie_classes` maps; direction
  kept - funded vs funded_by separate) when neither config nor domain supplies one.
  It constrains the extraction prompt and fuzzy-aligns the verbose tail: on the real
  crawl, provided_funding_to / granted_to / donates_to -> funded, president_of /
  chairman_of -> led, trustee_of -> member_of, the 12-word letters one -> met_with.
  Off-switch: `ontology.enabled: false` -> free-form. `drop_unmapped` stays false so
  an unmatched relation still passes through.
- **JSON repair: smart-quote value delimiters** (`json_repair.py`): qwen opened an
  evidence value with a curly quote (`: "When ...,"`) instead of a straight one,
  losing a whole chunk (43 entities, 26 relations). A new level straightens smart
  double quotes then runs the inner-quote escaper; the captured dump recovers.

Papers reviewed: LELA (LLM entity linking) declined - a heavyweight EL framework for
a problem dedup/Wikidata already cover (the concrete win, article/plural ORG folding,
noted for dedup). "Beyond Known Facts" (generating unseen temporal facts) declined - a
benchmark-construction method in the invent-unseen-facts space the project excludes.
Two-stage scenario-prompt RE adopted as the generic ontology above.

---

## Foundation segfault guard, JSON multi-block recovery, directed-scoring baseline

From the qwen3.5 morning batch.

- **Intermittent segfault loading foundation models on CPU** (crawl/ollama runs):
  a transformer spaCy model (`en_core_web_trf`) co-resident with GLiNER2-large in
  one CPU process aborts on a duplicate OpenMP init (thinc vs torch) - a native
  crash, not a catchable error, so it only shows as "Segmentation fault" mid-load.
  Set `KMP_DUPLICATE_LIB_OK` / `TOKENIZERS_PARALLELISM` at the top of `main.py`
  before any torch/spacy import (thread count left alone - pinning it throttles CPU
  inference). The English ollama path runs foundation on CPU so it was exposed; the
  German path dodged it with a non-transformer spaCy model + the multilingual
  GLiNER. `scratch/crawl_influencewatch.yaml` now uses that same stable pairing
  (`en_core_web_lg` + `gliner2-multi-v1`).
- **JSON repair lost a whole dedup batch** (`intelligence/json_repair.py`): qwen3.5
  ignored `think:false`/`format:json` and emitted visible reasoning with two
  ```json blocks - a discarded first attempt then the corrected answer last. The
  repairer took the first (invalid) block. It now tries each fenced block
  last-first, then the whole response, and dumps only if all fail. All 8 captured
  failure fixtures recover.
- **LLM review logs a sample of dropped entity names** so an aggressive batch is
  auditable from the log (the oversized-drop guard already caps a hallucinated
  batch wholesale; one fired at 127/150 on the Abel pilot).
- **Directed vs undirected relation scoring, quantified** on the qwen3.5 runs:
  typed F1 barely moves (Re-DocRED 0.159 directed / 0.162 undirected; DialogRE
  0.046 / 0.059) - when the model gets a relation it orients it right, so the
  directed-scoring fix was correctness insurance, not a number correction. Untyped
  moves more (Re-DocRED 0.23 / 0.27) because undirected collapses reciprocal gold
  pairs. The real ceiling is relation recall (Re-DocRED typed fn 501 vs tp 62) and
  type-schema alignment (DialogRE's social-relation labels), not direction.
- **Crawl ollama runs timed out (~90% failures)**: `scratch/crawl_influencewatch.yaml`
  carried no `intelligence.ollama` block, so it inherited the 180s default
  request_timeout while using 8000-char chunks - each qwen3.5 call on dense web
  prose ran past 180s. Interspersed successes reset the consecutive-failure guard,
  so the run limped on near-empty instead of aborting. Now 5000-char chunks +
  `request_timeout: 600` (the Abel settings, which never time out).
- **Polarity-conflict detection** (`graph_metrics.py`, after KARMA's conflict-edge
  idea but as an offline rule, not a multi-agent LLM): dyads carrying both a
  positive and a negative tie (allied_with + fought_against on the same pair) are
  counted with a readable sample in `graph_report.json`. Signed balance drops them
  as net-zero, so they were invisible; either an extraction error or a real
  ambivalent/over-time relationship. Reported for review, not filtered.

---

## SNA/NER best-practice pass: backbone, projection weighting, balance, narrative networks, QID identity, directed scoring

A second audit against the research_context papers, then a batch of fixes and
research-grounded additions. "Tag, don't filter" relaxed where filtering is the
correct default.

Bug fixes:
- **Scorer was direction-agnostic** (`evaluation/scorer.py`): relation matching
  sorted endpoints, so a reversed prediction counted as correct - inflating F1 vs
  the official directed scorers. Now directed by default, per-relation: asymmetric
  relations must match orientation, symmetric ones (married_to, met_with) match
  either way. `--undirected-relations` restores the old behavior. Past benchmark
  numbers in this log were computed undirected; re-baseline before comparing.
- **Entity scoring was many-to-many** ("any overlap"), so 3 predicted nodes all
  hitting one gold entity scored 3 TP and didn't penalize over-segmentation. Now a
  greedy 1:1 match - precision charges the splits.
- **Burt brokerage was unweighted** (`graph_metrics.py`): `constraint` /
  `effective_size` now pass `weight="weight"` (corroboration), as Burt defines them.
- **Crawler `stay_under_path` could widen to the whole host**: a single-segment
  seed ('/docs') took parent '/'. Fixed; a page seed still scopes to its parent dir.

Additions (research-grounded):
- **Disparity-filter backbone** (Serrano, Boguna, Vespignani, PNAS 2009),
  `postprocess/backbone.py`: per-node significance test over the weighted
  co-occurrence layer. Every co_occurs_with edge gets `disparity_alpha`; with
  `inference.cooccurrence_backbone_alpha > 0`, non-backbone edges are dropped.
  On the Wikipedia crawl: 20,260 -> 4,040 co-occurrence edges at alpha 0.10, 855 at
  0.01 - principled where the global proximity floor was a blunt instrument.
- **Newman projection weighting** (PNAS 2001), `canonical_inference.py`: cross-doc
  co-mention is a one-mode projection of the entity x document bipartite graph; a
  pair sharing a k-entity document now contributes 1/(k-1) (`cooccur_strength`),
  so a 50-entity page no longer forges 1225 ties as strong as a tete-a-tete.
- **Signed structural balance** (Cartwright-Harary), `graph_metrics.py`: the edge
  `polarity` we already computed now feeds a balanced-triad fraction in
  graph_report.json (balanced = friend-of-friend / enemy-of-enemy triangles).
- **Narrative-sequence network** (Bearman & Stovel, Poetics 2000),
  `postprocess/narrative.py`: the Abel autobiographies are their exact case. Builds
  corpus-level element->element transitions from the timeline (war -> hardship ->
  politics) -> narrative.gexf + narrative_transitions.csv. Opt-in
  (`export.narrative_network`); on by default in the nazi_era config. v1: coarse
  keyword element scheme, domain-overridable via `Domain.narrative_rules()`.
- **Wikidata QID as identity** (`postprocess/wikidata.py`): linking was decorative
  (tagged qid, ran after dedup). A shared QID is now a high-precision cross-doc
  merge key - same-QID nodes fold and edges remap (`linking.consolidate_by_qid`).
- **Reference-section stripping** (`core/preprocessor.py`): cut the trailing
  References / Bibliography / External links / Einzelnachweise tail from web pages
  before NER, so publishers and cited-author names never become top-mention nodes
  in the first place. Back-half + half-length guards; the RTF path is untouched.
  The name-shape `citation_artifact` tagger stays as a backstop.

New edge columns: `cooccur_strength`, `disparity_alpha` (codebook updated). Ten new
offline tests; suite green.

## GEXF export no longer drops parallel edges

Audit catch: `_write_gexf` wrote into a plain `nx.Graph`/`DiGraph`, so when a pair
had two relation types (A met_with B *and* A supported B) the second `add_edge`
overwrote the first - one edge and its weight vanished from every .gexf, and
multi-relational pairs looked weaker than they were. graph_metrics.py already
summed parallel edges, so the exported graph and the QA metrics disagreed. Fixed:
on a repeat pair the exporter now sums the weight and unions rel_type / tie_class /
connection_type / polarity / origin / edge_source (matching graph_metrics).
The CSV edge table was always correct (one row per s,t,rel_type); this only
affected the GEXF views. Regression test reads a written GEXF back and checks the
summed weight + retained labels.

## Citation/bibliography artifacts tagged out of the actor network

Reviewing the 12-page Wikipedia crawl: the highest-mention "core" nodes were
reference-list debris - publishers (Oxford University Press, Routledge, DK),
archive services (the Wayback Machine, Google Books), and bibliographic author
forms (Weeks, Marcus / Todd M. / Ripple WJ). These are real proper nouns, so the
POS gate keeps them, and they accumulate mentions across every page's reference
section, floating to the top as bogus hubs. Per tag-don't-filter: new
`tag_citation_artifact` in `postprocess/tagger.py` (publisher-suffix + known
publisher/archive set for ORG; inverted-comma / trailing-initials name shapes for
PERSON). Filter `tag_citation_artifact=false` in Gephi for the substantive graph.
On the crawl run it tagged 166/1637 nodes - several of them the top-mention ones -
with no false positives on the real sociologists (Durkheim/Marx/Weber/Simmel) or
on initial-bearing real names (J. R. R. Tolkien, George W. Bush, Malcolm X).
Offline test asserts both the catches and the protected names.

## Default ollama model bumped to qwen3.5:9b

The shipped default was still `qwen3:8b`, so a plain `--mode ollama` ran the old
model. qwen3.5:9b is the verified best fit for the 8 GB box (beats qwen3:8b on
relations, still fits VRAM), so make it the default everywhere code picks a model:
`OllamaConfig.model`, `LangExtractConfig.model_id`, and the `--ollama-model`
defaults in `benchmarks/run_benchmark.py`, `benchmarks/common.py`,
`scripts/book_bench.py`. Doc command examples updated to match. Historical
CHANGELOG benchmark numbers keep their original model names (they record what ran).

## Coref microservice: no more cold-start race

First real use on a web run exposed a race: the service loaded its model lazily on
the first /resolve, that cold call outlasted the client's 30s timeout, one timeout
disabled the service for the whole run, and the in-process fallback is broken in
the main env (transformers 5.x `all_tied_weights_keys`) - so the run silently used
the heuristic resolver instead of neural coref.

- **Service preloads at startup** (`@app.on_event("startup")`, opt-out
  `COREF_PRELOAD=0`): `Application startup complete` now means the model is ready.
- **Client warms the service once, up front**, with a generous budget
  (`max(service_timeout, 180s)`) instead of racing the per-chunk timeout. A genuine
  unreachable service still falls back immediately; a slow cold load is waited out,
  not treated as failure. A per-chunk hiccup after warmup falls back for that chunk
  only, keeping the service for the next. Offline test (monkeypatched urlopen).

## Co-occurrence floor for dense corpora

A 12-page Wikipedia crawl exposed a scaling hole: the within-doc proximity layer
had no weight floor (cross-doc co-mention already had `min_shared_docs`), so dense
encyclopedic pages produced 148k co-occurrence edges - 86k of them weight-1 (a
single accidental within-window adjacency) - bloating the GEXF to 96 MB, past what
Gephi will load. Only 212 of the 148k edges were typed relations.

- New `inference.proximity_min_count` (default 1 = unchanged): drop proximity pairs
  co-occurring fewer than N times. Typed/asserted edges are never touched - this
  floors only the weakest layer. Set 2-3 on web/encyclopedic corpora.
- Tuning a crawl run (`proximity_min_count: 2`, `cooccurrence_min_shared_docs: 3`,
  `quality.min_entity_mentions: 2`) cut it from 5571 nodes / 148k edges / 96 MB to
  1614 nodes / 20k edges / 14 MB, keeping every typed edge. Documented in
  config_template.yaml; example in scratch/crawl_wiki_test.yaml.
- Reminder surfaced by the same run: web text is third-person, so `narrator_resolution`
  is a no-op - enable `coreference.pronoun_resolution` (via the coref microservice)
  to resolve he/she/it/"the organization" on scraped pages. test added.

## Whole-site crawler

`--url` only ever fetched the exact pages you named. `core/crawler.py` adds
bounded, polite whole-site ingestion: give it a seed and it expands into the
subpages and merges them into one network (entities fold across pages).

- **Discovery:** sitemap.xml first (incl. sitemap-index recursion), then scoped
  breadth-first link following. Fetch-once - the page read for links is the page
  kept, so the pipeline never double-fetches. Discovered URLs are cached in the
  run dir; `--stage analyze` rebuilds id-only stubs from the cache, never re-crawls.
- **Bounded:** `max_pages` (doc cap), `max_depth` (hops), `max_bytes` (per-page),
  plus a hard request budget backstop. Visited-set dedup with URL normalization
  (lowercase host, default-port + fragment + tracking-param strip, // collapse).
- **Scoped:** same-host (folds `www.`), optional seed path-prefix, allow/deny
  regex. Seeds are exempt from allow/deny/path (they define the scope); redirects
  are re-checked against scope on the destination.
- **Polite:** obeys robots.txt + `Crawl-delay`, per-host rate limit (`delay`),
  identifying User-Agent. Fail-soft per page - one bad URL is logged, never raised.
- Config `io.crawl` (documented in config_template.yaml); CLI `--crawl <seed>`
  (repeatable), `--crawl-max-pages`, `--crawl-max-depth`. Local `.html` mirrors
  already run through the same trafilatura cleaner, so wget/HTTrack + a folder is
  the alternative for sites that forbid crawling.
- 31 offline crawler tests (injected fetcher, no network): scope, depth, caps,
  robots, sitemap+index, redirects in/out of scope, non-html skip, dedup,
  doc_id parity with url ingestion. `fetch_url` also hardened for the requests
  ISO-8859-1 charset fallback.

## Web ingestion: trafilatura main-content extraction

The web path was `requests` + `BeautifulSoup.get_text` with a tag blocklist - it
kept sidebars/ads/related-links/captions, feeding boilerplate into NER/RE on
scraped pages. `_clean_html` now prefers **trafilatura** (main-content extraction:
drops the boilerplate, keeps article body + data tables), BeautifulSoup as
fail-soft fallback. Optional dep (in requirements; works without it).

- ARCHITECTURE.md corrected: the stack is `requests` + trafilatura, **not**
  ScrapeGraphAI/Crawl4AI (a Gemini-draft claim). Those are LLM/agent scrapers for
  dynamic JS sites - heavy, and they'd spend the LLM budget on scraping; revisit
  only if JS-rendered targets become a requirement. The LLM tier stays reserved
  for relation extraction.
- Coref microservice verified end-to-end (correct pronoun cluster on a Bilbo test,
  model loads lazily on first request). test_html_extraction added.

## Coref microservice; physical/ideological edge axis; architecture + papers review

Reviewed research_context/ (Bearman & Stovel narrative networks; Zheng 2017 joint
tagging; Choi & Jung / Zavarella KG-construction surveys; ScrapeGraphAI / WebScraper
web IE; Bosshart et al. NBER NSDAP membership universe) and reconciled
ARCHITECTURE.md (a Gemini draft) with the real pipeline.

- **fastcoref microservice (`services/coref_service.py`).** fastcoref needs
  transformers <5, which conflicts with the main env's GLiNER2 (transformers 5.x);
  an isolated FastAPI service keeps it out-of-process and light (no spaCy/GLiNER).
  The pipeline POSTs chunk text and re-attaches char-offset clusters with the same
  logic as in-process; enable via `coreference.service_url`, falls back to
  in-process fastcoref then the heuristic. `services/requirements-coref.txt` for
  the isolated env. Pipeline-side client is stdlib urllib (no new main-env dep).
- **connection_type edge axis (Toro 2024 / ARCHITECTURE guideline).**
  physical / ideological / organizational / biographical, orthogonal to tie_class:
  separates a direct material tie (meeting, funding, combat, kinship) from a
  shared/opposed-belief one. The cross-cut is the point - fought_against is a
  stance but physical; influenced_by is a stance but ideological. New
  `tie_classes.connection_type`; flows to gephi_edges.csv, GEXF, codebook.
- ARCHITECTURE.md rewritten to the as-built pipeline (GLiNER2-only, four RE tiers,
  real stage flow, the microservice, recall ceiling, paper grounding). The draft
  had GLiNER v1, langextract-only RE, and a fictional coref microservice.
- Best-practice check vs the KG surveys: the canonicalize -> typed extraction ->
  tiered evaluation spine with hallucination guards is already what we run.
  Proposed (not built): a Bearman-style narrative-sequence network (life events as
  nodes, narrative order as arcs) from the timeline. tests: coref re-attach +
  connection_type, offline.

## Four more NER benchmarks: CoNLL-2003, OntoNotes 5.0, WNUT-17, Universal NER

Widen the entity-side eval beyond German (GermEval/HIPE): English clean (CoNLL
newswire, OntoNotes multi-genre), noisy social media (WNUT), multilingual (UNER).

- All BIO token-classification. Decode + pseudo-doc grouping factored into
  `benchmarks/common.py` (`decode_bio`, `build_ner_docs`); adapters are thin.
- datasets 4.x dropped script loading, so the canonical NER sets no longer load
  by id. `common.load_token_dataset` falls back to the auto-converted parquet
  (refs/convert/parquet). ClassLabel names survive for CoNLL; the tner mirrors
  (OntoNotes/WNUT) store raw ints, so `hf_iob_label_map` scrapes label2id from
  the dataset README. The tner WNUT map orders O last (O=12), not first - a
  hardcoded guess decoded garbage, so README-scrape-with-fallback it is.
- UNER's HF repo is script-only with no data; the adapter reads a local UNER
  `.iob2` via --path (`common.parse_iob2`), like ace2005/tacred.
- Prepare validated end-to-end (download + decode + gold): CoNLL 139 ents/4 docs,
  OntoNotes 68, WNUT 25 - all real names, right types. The --run step (GLiNER2
  foundation) is the owner's; prepare writes gold + inputs + config. Tests:
  decode_bio / build_ner_docs / parse_iob2 offline.

## Proximity edges validated on the 15-doc run; org-name suspect guard

15-doc Abel run on qwen3.5:9b with the new proximity layer + GermEval/HIPE German
entity validation. Scored offline.

- **Window co-occurrence does what it was built for.** Metadata-gold untyped
  relation recall: conservative (text only) 0.628, full (with the proximity
  floor) **0.884** - the floor recovers 11 of the 16 author->place/org facts the
  LLM never typed, only 5 left unconnected. Recovered edges are all tagged
  rule_cooccurrence (full tier), so they stay filterable. Edge mix on the run:
  4849 rule_cooccurrence / 365 llm_extracted / 46 metadata. Entity recall on the
  metadata targets is 1.0 - every author/place/org node is found; the gap was
  never entities, only the tie between them.
- **GermEval 0.690 / HIPE 0.405 - both exactly at baseline under qwen3.5:9b.**
  Entities are foundation (GLiNER2+spaCy), model-independent, so the qwen3:8b ->
  qwen3.5:9b swap moves relations, not entities. Confirmed: no regression. HIPE
  stays OCR-bound (token splits, title-laden gold person spans); GermEval ORG
  precision (0.46) is modern-news compounds that won't appear in Abel.
- **suspect_common_noun no longer flags real orgs by their form.** German
  proper-org names are capitalized common nouns (spaCy tags NOUN not PROPN), so
  the propn-ratio gate flagged genuine parties/units - Deutschnationale
  Volkspartei, Sozialdemokratische Partei, Völkische Bewegung, Freikorps,
  Garde-Feldartillerie-Regiment. `quality_review._has_org_marker` exempts
  ORG/INSTITUTION whose name ends in a distinctive org-form marker
  (partei/bewegung/front/bund/verein/regiment/korps/... + English party/union/
  league/...). Only ever removes a false suspect flag - never drops a node, never
  adds one. 15 of 231 flags on the run corrected, all true orgs. The born/resided
  -> located_in "typed gap" is correct behaviour, not a bug: born_in/resided_in
  are metadata-only; located_in is the honest text label.

## Window co-occurrence; the cross-chunk recall ceiling; metadata hygiene

Top-to-bottom comb of both pipelines. Coref is chunk-local (fastcoref runs per
chunk), so a relation between two third parties split across chunks is never
seen - the cross-chunk recall ceiling. Three layers, fixed differently.

- **Within-document window co-occurrence (`enable_proximity_edges`, default on).**
  Links entities mentioned within `proximity_window_chars` (600) of each other.
  Positions are document-absolute, so a windowed pair spans chunk boundaries the
  LLM never saw across - a floor under the *weak/untyped* layer of the ceiling.
  It is also the only within-letter weak tie in ollama/api mode (python_only had
  sentence co-occurrence; the LLM modes had none). The character-network
  literature uses exactly this (a k-sentence / fixed-char window; physical
  divisions like pages/chunks miss co-occurrences). Stays the weakest evidence
  tier - co_occurs_with, full only - and far less noisy than the old whole-doc
  complete graph. New: `postprocess/canonical_inference.py:proximity_edges`,
  threaded `agg.mentions` + dedup name map into `InferenceEngine.run`.
- **Typed boundary-spanning relations:** raise `chunking.overlap_chars` (we ship
  400 ~= 7%; the chunking literature says 10-20%). Resume is doc-level, so
  bumping it only affects not-yet-extracted docs - safe to raise mid-corpus.
  Left the default alone (cost/output call); documented as the lever.
- **Typed long-range relations** (entities chapters apart) still need
  document-level coref or a second doc-level pass - scoped, not built. Bounded
  impact here: the author hub is already resolved in every chunk, so only
  third-party<->third-party pairs hit it.
- **Metadata mojibake hygiene.** `load_metadata` now runs the same
  `clean_surface` repair the text path uses, so a metadata-only place
  ("Stallup√∂nen") no longer mints a corrupted node / `attr_place_of_birth`
  column. Not a recall fix - `normalize_name` already repairs umlaut mojibake at
  match time, so node-merging and scoring were never broken by it; this is node
  hygiene for places the prose doesn't mention.

Reviewed and unchanged (already best-practice): `graph_metrics` (Burt
constraint + effective size, bridges, articulation, substantive-graph only),
`tie_classes` (signed polarity, person<->person interaction correction,
opposition-as-stance), the dedup guards, the aggregator mojibake repair.

---

## German relation gold from the spreadsheet (no hand-annotation)

The metadata xlsx already encodes verified relations - birthplace, residence,
prior party, NSDAP membership - one set per author. Turned that into a gold and
scored how much of it the *text* extraction recovers on its own. First German
relation number that doesn't need a hand-annotated set.

- **`scripts/metadata_gold.py`** reads a finished run's `entities.json` (metadata
  is already merged onto author nodes) and emits a gold of the spreadsheet's
  biographical relations for every matched author. Self-contained; pass
  `--metadata` only for runs predating the merge.
- **`--exclude-edge-source` on the evaluator.** Drops edges whose sources are
  *entirely* in the exclude set (a `metadata;llm_extracted` edge survives
  `--exclude metadata` because the text also asserts it). Scoring with
  `--edge-sources conservative --exclude-edge-source metadata` measures what the
  prose recovered vs the injected edges - otherwise the match is circular.
- **Baseline (qwen3.5:9b, 15-doc Abel run).** Untyped relation recall 0.581
  (25/43 verified author<->fact ties recovered from prose); `member_of` recall
  0.55. Read untyped recall as the headline - the text uses its own labels, so
  endpoints-only is the honest match. Precision is meaningless here (the prose
  asserts many true ties the four fields never list).
- **Finding it surfaced:** `born_in` / `resided_in` typed recall is 0 while the
  endpoints often *are* recovered (untyped > typed). The pipeline ties
  author->place generically (`located_in`); the metadata labels are
  birth/residence. Not a recall hole - a labeling gap, and metadata already
  covers those facts. The member_of 0.55 is the real signal: it validates the
  membership extraction that feeds the whole affiliation analysis.

This is a permanent regression gold - every future Abel run and model A/B can
score against it offline (pure stdlib, no GPU).

---

## Relation guide lands; evidence tiers rebuilt; two more JSON shapes

Ran the Hobbit A/B. The guide works.

- **Relation-guide result (qwen3.5:9b, Hobbit gold, conservative tier).** Typed
  relation F1 0.084 -> 0.203 (tp 23 -> 64, fn 165 -> 124); `associate` alone
  P0.41 / R0.36 / F1 0.39 where before it barely registered. Untyped F1 flat
  (0.238 -> 0.231) - as intended: the guide fixes labels, not which pairs get
  found. Both runs identical config bar the guide (run_meta confirms). Carries
  to Abel for free (RELATION_GUIDE already shipped).

- **Evidence tiers rebuilt - they had drifted from what the pipeline stamps.**
  The tier->edge_source map lived in two places (evaluator + codebook) and
  matched neither reality:
  - `langextract_extracted` and `metadata` were in no tier but `full`. So a
    langextract run scored `--edge-sources conservative` showed zero relations,
    and Abel's verified-spreadsheet edges (the most precise in the system) were
    excluded from the conservative network. Both now conservative.
  - Co-occurrence was double-branded - `sna_inferred` (ollama/api) vs
    `rule_cooccurrence` (python_only) - and rode in `moderate`. The 3445
    proximity edges flooded the middle tier and made `moderate == full`.
    Unified to `rule_cooccurrence` (legacy `sna_inferred` still recognised),
    demoted to `full`-only. Co-occurrence is the weakest layer (not a tie); it
    belongs in the widest network, not the middle one.
  - `gliner_extracted` was a phantom - referenced, emitted by nothing. Dropped.
  - One source of truth now: `postprocess/evidence_tiers.py`, imported by the
    evaluator and the codebook so they cannot drift again. Tested.
  Result: the three tiers are finally distinct. Hobbit guided, typed F1 -
  conservative 0.203, moderate 0.203 (no domain inference in the generic
  pipeline), full 0.031 (co-occurrence flood, now isolated). For Abel, moderate
  sits between (canonical_inferred membership edges).

- **JSON repair levels 5.5 / 5.6.** The longer guide prompt makes qwen wrap
  evidence in book quotes - two new malformations: a value opening with an
  escaped quote (`"evidence": \"...`), and escaped dialogue quotes with an
  embedded comma (`\"...quietly,\" said Gandalf.\""`) that the 4.6 segmenter
  mis-closes. 5.5 composes escaped-delim + inner-quote escaping; 5.6 strips the
  opening backslash-quote then lets the inner-quote escaper find the real close.
  Both captured dumps recover; suite green (7/7 real shapes).

---

## Relation guide: contrastive label definitions in the extraction prompt

The one weakness every dataset confirmed this round is typed-relation
accuracy: at 9B the model labels by intuition, not by the coding scheme
(Hobbit's `associate` - 62% of the gold - comes back as `friend`; DialogRE
the same). The fix is to give the model the definitions it never had.

- **`ontology.relation_guide`** (config) / **`RELATION_GUIDE`** (domain):
  `{label: one-line definition}`. When constraining relations, the prompt now
  renders each allowed label with its definition instead of a bare comma list,
  prefixed "the definitions are deliberate; follow them over your intuition."
  Labels without a definition still render bare. No guide -> old behavior.
- **Contrastive where it matters.** Hobbit guide
  (data/hobbit.relation_guide.json) pins the associate/friend boundary
  ("companionship is associate, NOT friend"). Abel `RELATION_GUIDE` (27
  labels) separates the pairs qwen confuses: joined/member_of/served_in,
  led/commanded, opposed/fought_against, participated_in/fought_in,
  met_with/co_occurs_with.
- **book_bench `--relation-guide <file.json>`**: ships definitions for the
  gold's labels, tags the run `_guide`, implies `--constrain-relations`. The
  baseline (`hobbit_ollama_constr`, typed F1 0.084) already exists, so the
  A/B is one new run.
- Wired generically: api + ollama backends, domain hook
  (`base_domain.relation_guide()` reads the package's `RELATION_GUIDE`), test
  coverage in tests/run_tests.py. Helps Abel, not just the benchmark.

A/B to run (one ollama run; baseline already on disk):
`python scripts/book_bench.py --book data/hobbit.txt --gold data/hobbit.gold.json
--mode ollama --ollama-model qwen3.5:9b --relation-guide data/hobbit.relation_guide.json`
then compare `output/hobbit_ollama_constr_guide/eval_report.*.json` typed F1
against `output/hobbit_ollama_constr`.

---

## Live scrape test: three dedup gaps, a junk-narrator fix, GermEval, tests/

Ran two live URLs (Wikipedia + InfluenceWatch) through python_only as a real
test of the scraping path. Pipeline held end to end; the entity list exposed
four issues, all fixed and verified on the same run:

- **Fuzzy bucketing never compared "the X" with "X".** Non-person buckets
  keyed on the raw first character, so a leading article isolated a name from
  its own variants. Buckets now key on the first content token ("the American
  Enterprise Institute" finally merged into "American Enterprise Institute").
- **Acronym fold**: an all-caps ORG folds into the unique org whose
  capitalized-word initials spell it (AEI -> American Enterprise Institute,
  69 mentions reunited). No _blocked check here - the distinctive-token rule
  always fires for an acronym vs its expansion; uniqueness is the guard.
  DVP/DNVP-style distinct acronyms unaffected (initials must match exactly).
- **Token-subset person fold**: middle-name variants merge into the unique
  longer name ("Theodore Abel" -> "Theodore Fred Abel"), running before the
  single-token fold so bare surnames see one target. Family blocking still
  holds: "Fred Abel" stays separate (could be a sibling).
- **Scraped pages no longer get a narrator.** Quoted first person on a web
  page synthesized "Narrator [https://...]" hub nodes; URL-sourced docs now
  skip narrator detection (a memoir fetched by URL loses it - save locally).
- **GermEval 2014 adapter** (gwlms parquet mirror; the original HF script
  dataset is dead under datasets 4.x). Modern German NER to pair with HIPE's
  historical OCR German - the two bracket the Abel register. Entities only.
  Baseline at 20 pseudo-docs: F1 0.690, R 0.819 (PERSON 0.81 / LOC 0.67 /
  ORG 0.57) vs HIPE 0.405 - the HIPE number was mostly OCR noise, the German
  stack is fine on clean text. Precision reads low by design: we resolve
  demonym derivations the GermEval gold deliberately excludes.
- **tests/run_tests.py**: offline regression suite - every json_repair shape
  incl. the real failure dumps, checkpoint failure scoring, all three dedup
  folds, the article-bucket merge. No models, no network. Run it after
  touching any of those modules.
- Hobbit + DWIE re-scored with the per-label table. The cross-dataset
  pattern: factual relations land (citizen_of P 0.78, head_of 0.60),
  interpersonal stance labels do not - qwen answers "friend" for the
  mentor's "associate" (71 fp vs R 0.03 on the gold's largest class).
  Hobbit moderate-tier untyped recall 0.622: detection is fine, typing is
  the weak layer at 9B.

---

## Abel qwen3.5:9b pilot (15 docs): two repair gaps closed, one of them a corruptor

First production run with the chunk-failure accounting: checkpoint meta
pinpointed both losses (doc_48a5c22097 1/2 chunks - the JSON dump;
doc_a54fc3fa51 1/6 - a transient ollama error). Both docs kept their good
chunks. scripts/drop_failed_docs.py prunes failed records so --resume
re-extracts only those docs.

- **JSON repair level 4.8**: several comma-separated strings as one value
  with no array brackets (`"evidence": "s1", "s2", "s3",` - qwen citing
  multiple passages). Merged with " ... " separators, which the verbatim
  checker already understands. The colon anchor keeps array elements out;
  the next key's colon stops the merge.
- **Level 3 was writing commas INTO strings.** The `\d` alternative in the
  missing-comma regex matched a digit inside a string right before its close
  quote ("born 1903" -> "born 1903,"). Any payload reaching level 3 with a
  digit-final string got silently polluted. Digits now count as value
  terminators only across a newline - the actual shape of qwen's missing
  commas. All 5 real dumps + 16-case suite green.
- **Metadata xlsx: literal "NA" no longer becomes a name.** The _ok filter
  (already used for metadata edges) now applies at field load, so the
  "NA Bartsch" author and birth_date="NA" attrs are gone on next analyze.
- Run health otherwise: 15/15 authors detected, hubs are NSDAP / Hitler /
  Germany / DNVP / Berlin, enrichment 24% subtype coverage with the domain
  vocabulary (one batch lost to the guarded failure), entity count
  proportional to the 06-10 25-doc baseline. Graph QA in the expected
  envelope for ego-network autobiographies (largest CC 43.5%).

---

## qwen3.5:9b constrained benchmarks; checkpoint failure accounting; breaker was dead code

Re-DocRED + DialogRE re-runs (qwen3.5:9b, --constrain-relations) reviewed.
Headline numbers, conservative tier:

- **Re-DocRED**: typed relation F1 0.021 -> 0.161 (tp 8 -> 62) from the
  P-code mapping + constrained ontology; untyped 0.196 -> 0.255. Entities
  flat at 0.822, as expected (NER path untouched).
- **DialogRE first honest scores**: entity F1 0.831, untyped relations 0.203,
  typed 0.058. Typed is weak for a reason the new per-label table makes
  visible: qwen calls everyone `friends` (44 fp) and never emits
  `positive_impression`, the most common gold label.
- **Scorer now reports per-label relation P/R/F1** (`per_type` under
  relations_typed, sorted by gold support). Re-DocRED's typed score turns out
  to be carried by geo/admin labels; `member_of` is P 0.06. Also visible:
  `has_part`/`part_of` inverses match untyped but miss typed - directionality,
  not detection.
- **DialogRE gold now counts every speaker as a PERSON entity.** Gold pairs
  only cover relation arguments, so the extractor was charged fp for
  correctly finding the other speakers we ourselves named into the text.
  Entity precision 0.545 -> 0.765 with no extraction change.
- **JSON repair level 4.7**: parenthetical commentary after a closed string
  (`"...house arrest" (implied residence),`). The shape cost one Re-DocRED
  doc its whole extraction twice (temperature 0 = deterministic failure,
  one md5-deduped dump). All 4 real dumps + 14-case suite recover.
- **Checkpoint failure accounting.** Backends now record
  n_chunks/chunks_failed in meta; a failed LLM chunk passes foundation
  mentions through, so "has mentions" never proved anything. Cleanest record
  wins on duplicate doc_ids (an API-error re-run had stomped a good pass:
  4 relationships recovered on re-analyze), full failures are retried by
  --resume instead of being skipped forever, and the "N completed documents"
  log counts docs, not lines.
- **The circuit breaker was dead code.** extract_document's per-chunk
  catch-all swallowed the RuntimeError, so a downed server would have
  degraded every chunk with a warning instead of aborting. Breaker now raises
  BackendUnavailable, which the assembly loop re-raises. Never fired in a
  real run only because the server stayed up.
- Codebook: the Multi-view line only mentions network_dynamic.gexf when the
  run produced one (dialogue corpora have no dated events).
- benchmarks/README DialogRE row updated (synthetic speaker names + speaker
  gold); AGENTS.md layout line now lists all six adapters.

Re-run cost after the fixes: --resume retries only the failed docs (1 per
dataset), not the whole benchmark.

---

## Five-run review: dialogre identity bug, doubled-quote repair, coref verdict

Reviewed runs 1-5 against the archived 06-10 baselines
(scratch/baseline_reports_2026-06-10).

- **Re-DocRED run #1 never executed** - run_meta.json shows the on-disk output
  is still the 06-10 run (started 18:02 that day). Caught purely by the
  provenance file; the "new" report was the old one. Re-run needed.
- **DWIE re-ran byte-identical** to 06-10. Expected, not a bug: temperature 0
  + unchanged prompts = deterministic model output; the span/dedup fixes
  didn't touch any DWIE chunk that mattered.
- **DialogRE corpus-identity bug** (found via the 30-dialogue collapse,
  F1 0.40 -> 0.14): 96% of gold relations involve a "Speaker N" slot and 44
  of 89 pairs collided across dialogues - corpus-level scoring merged
  different people who share the literal slot name. Adapter now assigns
  deterministic per-dialogue names ("Alan Abbott": given name by slot,
  surname by dialogue) in BOTH the transcript text and the gold.
  Cross-dialogue collisions now 0. Old dialogre scores are void; re-run.
- **Hobbit constrained run validates the JSON repair overhaul in production**:
  1 repair failure across 19 chapters (previous runs: 5+), conservative
  untyped relation recall 0.250 -> 0.339, and the first nonzero TYPED F1
  against the mentor's 8 labels (0.084, 23 tp). The one new failure was a
  third delimiter shape - content starting/ending with a straight quote
  doubles against the JSON delimiter (`: ""Now go on!" ...`) - fixed with
  pre-missing-comma doubled-quote escaping; all 3 real dumps + 11-case suite
  green, legit empty strings untouched.
- **Coref heuristic verdict: does not pay for fiction.** The load probe
  correctly catches the fastcoref/transformers-5.x break and falls back; the
  heuristic emitted 2650 mentions on the Hobbit but moved recall exactly 0
  and cost conservative relation precision (0.116 -> 0.082): it re-emits
  already-found entities, so it adds edges, never nodes. Keep
  pronoun_resolution default-off; narrator resolution (the Abel path) is
  unaffected and stays on.
- HIPE at 20 docs: entity F1 0.405 (0.419 at 8) - stable German-historical
  baseline. DWIE calibration matches redocred's pattern (top entity bin 0.82
  precise, ECE 0.22; edge confidence still junk) - the "weight by
  corroboration, not confidence" rule holds across datasets.
- codebook.xlsx + run_meta.json confirmed present and correct in all five new
  run dirs (constrained codebook classifies the mentor labels into tie
  classes correctly: geo->biographical, friend/enemy/kin->interaction).
- Docs: INSTRUCTIONS output table now lists codebook.xlsx / run_meta.json /
  graph_report.json; benchmark + calibration commands added. AGENTS.md and
  CLAUDE.md carry the comment-style rule (terse, pragmatic, owner's voice).

---

## Constrained Hobbit run triage: timeouts + JSON repair levels

The 19-chapter constrained run surfaced two robustness gaps (~7 of ~60 chunks
silently lost their relationships):

- **request_timeout 600 in bench/book configs** (was the 180 default): long
  8k-char chunks on qwen3.5:9b hit read timeouts and a 500 under load. The
  hardware notes already said 600 for big models; the generated configs now
  comply.
- **JSON repair overhaul** (`json_repair.py`), driven by the real failing
  payloads (identical chunks fail identically at temperature 0; unrepairable
  responses now dump to scratch/json_failures/, bounded 50). The actual
  failure shape: qwen3.5 emits string delimiters PRE-ESCAPED -
  `"evidence": \"text\",` and `"evidence\":` - invalid JSON at value/key
  position. A state-machine fixer distinguishes mis-escaped delimiters from
  legitimate escaped quotes in content (`\"` before a bare quote is content;
  before `:` it closes a key; before `,}]`/EOL it closes only strings that
  were opened escaped). Second bug: the missing-comma repair level treated
  the quote in `\""` as a value terminator and inserted a comma inside the
  string - negative lookbehind added. Further new levels: Python literals
  ("directed": False), unquoted bare enum values ("type": associate),
  inner-quote escaping, dangling-key trim before bracket closing. Both real
  dumps now fully recover (12 + 5 relationships with evidence); 12-case
  regression suite green, valid-JSON and legit-escape guards included.
- Run speed itself was NOT a bug: ~7-9 min/chapter is this card's rate for
  multi-chunk chapters (matches the first overnight run).

---

## Full-module audit + HIPE/DialogRE benchmarks + coref/linking/calibration

Line-level audit of every module in core/, intelligence/, postprocess/,
checkpoint/, benchmarks/, evaluation/. Two real bugs found and fixed; four new
evaluation capabilities added.

- **Coref gating bug** (`core/foundation.py`): `coref.resolve` only ran when a
  narrator was detected, so pronoun_resolution could never fire on third-person
  text (books, news) even when enabled. Now runs whenever coref is on;
  narrator emission guards itself against an empty narrator name.
  `book_bench --coref` added for on/off A/B runs.
- **Ollama circuit breaker** (`ollama_backend`): a downed server produced a
  "successful" run with mentions but zero relationships (every call failed
  soft, discovered via a DialogRE run against a dead server). After 5
  consecutive call failures the backend now aborts with a clear message;
  --resume continues after the server is back.
- **CLEF HIPE-2022 adapter** (`benchmarks/hipe.py`): German historical
  newspaper NER (hipe2020 subset, auto-download + cache), the closest public
  proxy for Abel-era German. Adapters can now declare DEFAULT_SPACY_MODEL /
  DEFAULT_GLINER_MODEL (HIPE uses de_core_news_lg + gliner2-multi); the
  runner picks those unless overridden. Baseline measured (8 docs, dev,
  python_only foundation): entity F1 0.42 typed / 0.43 agnostic - OCR-era
  text is hard; gold itself carries OCR fragmentation. Track this number.
- **DialogRE adapter** (`benchmarks/dialogre.py`): interpersonal-relation
  gold from dialogue (friends/siblings/boss), auto-download + cache.
  Speaker-slot entities kept literal ("Speaker 1"); STRING/VALUE args and
  unanswerable pairs dropped. Works with --constrain-relations (12 labels).
- **Demonym handling** (`core/demonyms.py`, spacy_engine, deduplicator):
  ~130-entry demonym->place table (EN + German incl. Abel-era: prussian,
  bavarian, soviet...). spaCy NORP mentions matching the table are relabeled
  LOCATION with `demonym_of`; dedup folds them into the place node as aliases
  (`dedup.fold_demonyms`, default on, domain aliases take precedence).
  Benchmark-neutral on existing checkpoints (+-0.002); graph-level it
  consolidates "American" into "United States" instead of duplicating actors.
- **Confidence calibration report** (`evaluation/calibration.py`): reliability
  bins + ECE for entities and edges vs gold. Measured on redocred/qwen3:8b:
  entities are usable (0.9 bin -> 90.4% precise, ECE 0.10); edge confidences
  are inflated and near-meaningless (0.9 bin -> 26.7%, ECE 0.67). Use
  corroboration (Weight = distinct docs), not confidence, to weight edges.
- **Entity linking evaluated** (existing `postprocess/wikidata.py`, kept
  off by default): top-40 redocred entities -> 10 linked, effectively 10/10
  correct (one hit is Wikidata's rename of the same object). High precision,
  conservative coverage; enable per-run via `linking.enabled` when online.
- **fastcoref is dead under transformers 5.x** (all_tied_weights_keys API
  change at predict time; GLiNER2 needs 5.x, so no downgrade). Added a load
  probe so the failure surfaces once, plus a conservative heuristic fallback
  (`coreference.heuristic_pronoun_mentions`): a third-person pronoun resolves
  to the single PERSON mention in the preceding 250 chars; any ambiguity =
  skip; tagged coref_heuristic, confidence 0.4. EN+DE pronoun sets.
- **DialogRE measured** (qwen3.5:9b, constrained, 10 dialogues): untyped
  relation F1 0.40 at P 0.667, typed 0.16. Misses concentrate on ties between
  unnamed "Speaker N" slots (dataset quirk). First-run zero was a downed
  ollama server - that's what motivated the circuit breaker above.

---

## Hobbit ollama audit: dedup kill chain fixed, span hygiene, verbatim segments

Audited the hobbit_ollama (19 ch, qwen3.5:9b) output against the mentor gold.
Conservative-tier relation recall 0.25 vs python_only's 0.028 - the LLM reads
narrative ties the dependency rules can't. But four bug classes surfaced:

- **Salient characters silently deleted (kill chain of three bugs).** Beorn
  (66 mentions), Gloin, Nori, Thror vanished from the graph. Chain:
  (1) `llm_dedup._plausible_alias` accepted any fuzzy ratio >= 0.5, letting
  qwen merge distinct characters (Beorn~bear 0.67, Thror~Thorin 0.73);
  (2) `Deduplicator._merge_into` let the absorbed junk OVERWRITE the canon's
  attributes, poisoning `propn_ratio` to 0.0; (3) the POS gate then silently
  hard-dropped the poisoned PERSON, destroying the alias trace. Fixes: fuzzy
  floor raised to 0.75 with a comment documenting the collision pairs,
  `_merge_into` now only fills attribute gaps (primary's signals win), and
  the POS gate warns when dropping a PERSON with >= 20 mentions so the
  failure class can never be invisible again. Verified on the same
  checkpoint: all eight probe characters survive as their own nodes; entity
  recall 0.667 -> 0.759, moderate relation recall 0.433 -> 0.633.
- **Conjunction/preposition NER spans become dedup attractors.** "Bofur and
  Bombur" (GLiNER span) swallowed Bofur + Bombur as aliases; "in Fili"
  swallowed Fili via the partial-person fold (last-token match +
  keep_primary_name). Fixes: `entity_merger.repair_spans` strips leading/
  trailing prepositions+conjunctions and splits two-name PERSON conjunctions
  ("Marie und Adolf Spanku" -> two mentions; "Thorin and Company" kept -
  generic conjunct blocklist; articles NOT stripped, "The Shire"/"Der
  Stahlhelm" are real names); `_fold_partial_persons` and llm_dedup refuse
  function-word names as fold/merge targets. Wired into foundation, so fresh
  extractions are clean; the dedup guards also protect old checkpoints.
- **Evidence verbatim check: 62% false-flag rate.** qwen stitches multiple
  verbatim spans with "..." - legitimate compression, not paraphrase. The
  check now folds unicode punctuation (curly quotes, dashes) and verifies
  each ellipsis-separated segment independently: unverified rate 62% -> 3%,
  and the remaining 3% are genuinely corrupt spans. Tag carries signal now.
- **book_bench `--constrain-relations`**: injects the gold's relation labels
  as the extraction ontology (same as run_benchmark) so typed relation F1 is
  meaningful against the mentor's 8-label codebook.
- hobbit.txt: trimmed 2.9k chars of publisher back matter (HarperCollins
  nodes were leaking into the graph).
- Regression check: redocred benchmark re-scored byte-identical after the
  dedup changes.

---

## Codebook export + Hobbit gold benchmark + qwen3.5:9b results

- **qwen3.5:9b benchmarked under the new code** (constrained, conservative
  tier): DWIE untyped relation F1 0.410 (qwen3:8b was 0.261; gemma4:12b 0.436
  on the older prompt), typed 0.163. Re-DocRED untyped 0.253, and the first
  meaningful typed score on that dataset (0.161) now that the P-code mapping
  feeds --constrain-relations. A 9B model on the 8 GB card is now within ~6%%
  of gemma4:12b on relations - the 540-doc corpus no longer requires the
  friend's machine.
- **`codebook.xlsx` auto-generated for every run** (`postprocess/codebook.py`,
  `export.codebook: true`): standard SNA codebook so outsiders can read the
  data - boundary specification, definition of every node/edge column actually
  present, entity types with subtype inventories, the tie-class taxonomy with
  this run's counts, the full relation inventory with example evidence, and
  the evidence-tier table. Modeled on the mentor's Hobbit codebook (Node List /
  Edge List / Code Book) extended with provenance + value inventories.
  Fail-soft; wired into the analyze stage after export.
- **The Hobbit gold benchmark prepared** (`data/hobbit.txt` +
  `data/hobbit.gold.json`): mentor's codebook xlsx converted to the
  evaluation gold schema (58 entities: 43 PERSON / 14 LOCATION / 1 ORG; 190
  undirected relations across 8 tie types, no orphan endpoints). Book PDF
  extracted to text (front matter/TOC cut, hyphenation repaired, 19 chapters
  detected by book_bench's splitter). Run via `python scripts/book_bench.py
  --book data/hobbit.txt --gold data/hobbit.gold.json [--mode ollama]`.
  book_bench input glob tightened to `ch_*.txt` - a stray .txt in the input
  dir (e.g. the book itself) was ingested as a 20th document and OOM'd the
  transformer on 123k tokens.
- **python_only Hobbit baseline:** entity recall 0.69 (typed). Precision is
  not meaningful against this gold - the mentor's boundary includes only
  tie-linked nodes (58), not every named entity, so most pipeline "FPs" are
  out-of-boundary, not wrong. Untyped relation recall 0.028 conservative ->
  0.578 moderate: dependency rules barely fire on narrative fiction; the
  mentor's dominant "Associate" ties behave like co-presence and are caught
  by the cooccurrence layer. Gold has two type slips to flag upstream
  (Elrond typed Place, Dale typed Group/Man).

---

## Two-machine A/B audit (qwen3:8b vs gemma4:12b): eval fixes, provenance, evidence guard

Compared identical runs (Re-DocRED 25, DWIE 15 constr, Abel 25) across the
8 GB machine (qwen3:8b) and the 16 GB machine (gemma4:12b). Gold files and
foundation outputs matched across machines; entity F1 within 0.003 (entities
come from the foundation, not the LLM). gemma4:12b is clearly stronger on
relations: DWIE untyped F1 0.436 vs 0.261, typed 0.217 vs 0.127; Re-DocRED
untyped 0.239 vs 0.196 at higher precision with fewer edges. On Abel, gemma
emitted half the llm edges (377 vs 755) with far fewer suspect_membership
hits (8 vs 29) - precision-leaning profile, consistent with the benchmarks.

- **Re-DocRED relation labels mapped to readable names** (`benchmarks/
  redocred.py` `REL_INFO`): gold relations carried opaque Wikidata codes
  (`p131`, `p17`), so typed relation F1 was 0.0 by construction on every run.
  Codes now map to snake_case names (`country`, `member_of`, ...) from the
  DocRED rel_info inventory, making `--constrain-relations` usable for this
  dataset. Existing unconstrained runs re-scored: untyped metrics unchanged.
- **Eval reports print names, not internal ids** (`evaluation/scorer.py`):
  relation FP/FN lists showed `g55` / `p::x` keys; now resolved to entity
  names so error analysis is possible without code spelunking.
- **`run_meta.json` written into every run dir** (`main.py`): run name, mode,
  model, stage, limit, timestamp, full effective config. Motivated by a run
  named `abel_gemma4_12b` that actually used qwen3:8b - outputs were not
  traceable to the model that produced them.
- **Evidence verbatim guard** (`api_backend._map_extraction`, gephi_builder):
  LLM relationship evidence that is not a whitespace-normalized substring of
  the source chunk is tagged `evidence_unverified` (kept, Gephi-filterable).
  gemma4:12b was observed inserting bracketed paraphrases into evidence.
  Prompt now also forbids translating entity names/evidence (qwen emitted
  "Seizure of Power (1933)" for a German passage).
- **suspect_common_noun restricted to proper-name types**
  (`quality_review.py`): the tag fired on DATE/EVENT/RANK nodes (134/146
  DATEs), which legitimately consist of common nouns; it now applies only to
  PERSON/ORG/LOCATION/GPE/INSTITUTION so the Gephi filter carries signal.
- German stopwords: sentence-initial adverbs/verbs the tagger marks PROPN and
  NER promotes to PERSON ("Heran", "Kehrte", ...) added to the nazi_era list.
- Known cross-machine nit: doc_ids hash the full source path, so Windows/mac
  runs of the same corpus get different ids (path separator). Compare by
  filename, not doc_id. Not changed - new ids would orphan checkpoints.

---

## Language-general POS gate + book gold benchmark

- **POS gate replaces per-corpus stopword curation** (`core/foundation.py`,
  `aggregator.py`, `quality_review.py`, `config.quality.pos_gate`): each
  mention now records the share of its tokens spaCy tags PROPN
  (`propn_ratio`, averaged per entity, exported as `attr_propn_ratio`).
  A PERSON that is never a proper noun across >=2 mentions ("Monsieur",
  "der Vater", "the soldier") is dropped as a category word; borderline
  entities (<0.5) are tagged `suspect_common_noun` and kept. Works for any
  spaCy language regardless of capitalization conventions (German nouns),
  authors immune, no-op without a POS tagger and on pre-gate checkpoints.
  Verified: EN+DE unit test (Vater/Soldat/Lehrer/Mutter dropped, names and
  authors kept) and an end-to-end sample run (28/29 nodes carry the ratio,
  junk "My cousin" tagged 0.333). The static stopword lists remain as a fast
  precision layer - spaCy tags capitalized foreign honorifics PROPN in
  English text, so the two layers catch different failure classes.
- **`scripts/book_bench.py`**: run + score the pipeline on any book against
  gold annotations (entity/relation P/R/F1 at all three evidentiary tiers).
  Tolerant chapter splitting with single-document fallback, fiction config
  switches applied, gold validated before the run. Gold format documented in
  the header; scoring is corpus-level so whole-book gold in one document works.
- DWIE benchmark adapter verified loadable (HF download, readable relation
  labels - the right target for `--constrain-relations` typed-relation F1).
- Verified no growing-context bug in the ollama backend: every request is
  stateless (system + one chunk), `num_ctx` is pinned from config, chunks are
  hard-capped - per-request context cannot grow over a run. Long big-model
  runs slow down from VRAM/RAM spill, not from the pipeline.

---

## Three-run audit (lesmis x2 + 25-doc Abel): honorific stopwords, fiction config

Audited the lesmis_python_only (60 ch), lesmis_ollama (30 ch, qwen3:8b) and
abel_qwen3_8b (25 docs) outputs end to end.

- **Bare honorifics dropped as entities** (`domain/generic/entity_config.py`):
  "Monsieur" / "Madame" / "Bishop" / "Herr" / "Captain" etc. were surviving as
  PERSON nodes - a bare-title node conflates many distinct referents, which is
  an entity-resolution error, not Gephi-filterable noise (one even ranked as a
  reference figure). Added French/German/English/clerical honorifics to the
  existing generic STOPWORDS (exact normalized-name match; "M. Myriel",
  "Father Madeleine", "Bishop of D" all verified kept).
- **Fiction needs two config switches, not code changes** (book test script):
  `coreference.narrator_resolution: false` (a novel's authorial "I" is not a
  character; with it on, 28-59 per-chapter Narrator nodes appear and one was
  mis-merged into Monseigneur Bienvenu) and `dedup.llm_assist: true` (off by
  default; without it Myriel stayed split across 7 title variants).
- Abel 25-doc run verified clean: 24/25 named authors with letter_ids, all 12
  reference figures genuine historical figures, NSDAP aliases uncontaminated,
  0 mojibake, full edge provenance (`edge_source` on 1817/1817; the 56
  evidence-less edges are all `edge_source=metadata`, correct - spreadsheet
  facts have no source sentence), SNA columns populated (523 constraint,
  32 articulation, 462 bridge-tagged edges). Both LLM guards fired correctly
  against qwen3 (oversized merge groups; two >50% review drop lists ignored).
- INSTRUCTIONS: new "Direct commands (live progress bars)" section - plain
  `python main.py ... --run-name X --resume` invocations per model, plus
  evaluation/benchmark commands, instead of the log-redirecting bash wrappers.

---

## Benchmark + evaluation path repaired and verified

The one area the earlier audits hadn't covered. Three real problems found:

- **`benchmarks/` was never in git**: `.gitignore` listed `benchmarks/`
  (meant for outputs, which actually live in `data/bench/` and were already
  covered by `data/`), so the whole source package - runner + 4 dataset
  adapters - was silently excluded. Anyone cloning the repo had no benchmarks.
  Removed from `.gitignore`.
- **Stale GLiNER v1 defaults**: `config.py`, `benchmarks/common.py`, and
  `run_benchmark.py` still defaulted to `urchade/gliner_large-v2.1`, which the
  GLiNER2-only engine can no longer load - any config omitting `gliner_model`
  (all benchmark-generated configs) would crash at model load. Defaults are now
  `fastino/gliner2-large-v1`; ollama defaults bumped `qwen2.5:7b-instruct` ->
  `qwen3:8b`. Doc claims of a "legacy urchade fallback" removed
  (config_template, nazi config, INSTRUCTIONS, README).
- **`run_ollama_test.sh` now passes `--resume`**: no-op on a fresh run; an
  interrupted multi-hour big-model run no longer restarts from zero.

Verified end to end: scorer hand-checked against a synthetic prediction with
planted FP/FN (every tp/fp/fn matched expectation, tier filter drops
`sna_inferred` edges); full `redocred --limit 3 --run --eval` loop produced
zero-shot entity F1 0.816 (PERSON 1.000) and per-tier reports. Typed-relation
F1 on Re-DocRED is 0 by design for `python_only` (gold labels are Wikidata
P-codes; use `--constrain-relations` with ollama/api for comparable numbers).

---

## Audit fixes: LLM-review guard, tag-not-drop membership, ontology, run-name

Found in a comprehensive pass over the analyze-critical path.

- **LLM quality-review drop guard** (`quality_review.py`): the LLM reviewer's
  `drop_entities`/`drop_edges` list was applied unguarded (only authors exempt) -
  a weak model could drop salient entities. Now: a batch asking to drop >50% of
  its entities is ignored, and salient entities (author / reference_figure /
  mention_count>=5 / >=3 docs) are never dropped on the LLM's say-so. Mirrors the
  llm_dedup guards.
- **Non-org membership: tag, don't delete** (`main.py`, `config.InferenceConfig`,
  `gephi_builder`): `member_of`/`joined`/`served_in` edges pointing at a non-org
  are now TAGGED `suspect_membership=true` (a Gephi-filterable edge column) and
  KEPT. `drop_nonorg_membership` defaults to False; set it True to delete instead.
- **Ontology matching hardened** (`ontology.py`): substring match replaced with
  whole-token containment, and fuzzy match now skips canonical keys shorter than
  5 chars - so `led` no longer captures `fled`/`settled`. Real terms still align.
- **Metadata edges now tagged**: tagging moved to run AFTER the metadata merge, so
  metadata-derived edges get `connection_quality` and count toward degree.
- **`--run-name` CLI override**: A/B different models into separate output dirs
  without overwriting (e.g. `--run-name abel_gemma4_26b`).
- **Chunker hard-split for boundary-less text** (`core/chunker.py`): a single
  "sentence" longer than `max_chars` (scraped nav junk, OCR dumps, minified
  pages) produced one unbounded chunk that silently overflowed the LLM context.
  Oversize spans are now hard-split with overlap so no chunk exceeds the cap.
  Verified: 30k-char boundary-less input -> 6 bounded chunks; normal text
  unchanged. Matters most for the generalized any-input path.

---

## German text-repair + interaction-layer precision + LLM-dedup guard (20-doc Abel pilot audit)

- **LLM-dedup over-merge guard** (`llm_dedup.py`): a weak local model (qwen-7B)
  can hallucinate a catastrophic merge group - in the pilot it proposed merging
  **238** unrelated orgs/places/dates into `NSDAP`, collapsing ORG nodes 153->27
  and making NSDAP a garbage magnet. Added: drop any suggestion group larger than
  16 aliases (a hallucinated mega-merge), cap accepted merges at 8 per canonical
  node, and reject numeric/date-like aliases. Verified: the 238-alias group is now
  dropped, NSDAP keeps only its 16 real variants, ORG count restored to 159. This
  is essential resilience for the full-corpus run. A second 15-doc run reproduced
  the failure mode (qwen proposed 105 items into `1930`) - guard held.
  - Plus a **plausibility guard**: the LLM can't merge string-dissimilar names
    (e.g. `Deutsches Reich` the state into `NSDAP` the party) - a merge needs a
    shared content word, an acronym relationship, or moderate fuzzy similarity.
    Legit acronym<->full-name pairs are covered by the domain alias dict.
- **Final-layer name repair**: `gephi_builder` also runs `_repair_text` on node
  labels / aliases / edge endpoints, so any mojibake that slips past ingestion
  (one `Alt D√∂bern` survived in the pilot) is still clean in the exported graph.


- **Umlaut mojibake + soft-hyphen repair** (`aggregator._repair_text`, applied in
  `clean_surface`, `normalize_name`, and `core.preprocessor.normalize_text`):
  misread RTF codepages produced names like `Bruno Th√ºrling` / `Stallup√∂nen`
  and hyphenation left soft hyphens (`Kaisers­lautern`), which corrupted labels
  and split dedup. Now repaired (`√º`->`ü`, ...) and zero-width/soft-hyphens
  stripped - both at ingestion (future runs) and analyze (existing checkpoints).
- **Interaction layer no longer inflated by mis-typed entities**: the
  person<->person promotion now fires only for genuinely interpersonal relations
  (`_PERSON_TO_PERSON`: led/commanded/recruited/mentored/appointed_by/...). A
  place/role/org mis-tagged PERSON with `located_in`/`studied_at`/`promoted_to`
  keeps its biographical/affiliation class instead of polluting `interaction`.
  Free-form person<->person verbs are still caught by the unknown-relation fallback.

## SNA-correctness fixes (12-doc EN + Abel pilot audit)

Found by auditing 12-doc pilots in both modes. All in the analyze stage; the
generic (no-ontology) path benefited most. Nothing filtered - noise stays tagged
for Gephi.

- **Narrator/author de-fragmentation** (the big generic-path fix): a first-person
  author was appearing twice - as `Narrator [doc]` (first-person ties) and as their
  named self (third-person mentions). Three coordinated changes now collapse them
  into one clean node:
  1. **Author-name detection from the text** (`core/foundation._detect_author_from_text`):
     reads the name from the opening ("The Memoir of X", "I am X", "I, X,",
     German "Ich bin X"/"Mein Name ist X") so the narrator node is a real person,
     not a placeholder. Case-sensitive name capture; no false hits on "I am sure..."
     or third-person prose.
  2. **Author-mention fold** (`deduplicator._fold_author_mentions`): a non-author
     PERSON merges into the same-named author when unambiguous (so the six "Emil"
     authors are never collapsed).
  3. **Identity-edge merge** (`postprocess/identity_resolution.py`): consumes any
     `is`/`self_reference` edges the LLM emits ("Narrator [doc] is Jane Doe") to
     merge + then drops them (incl. the wrong "narrator is narrator" hallucinations).
  Plus: a `Narrator [doc]` placeholder can never win canonical over a real name in
  a dedup merge. Verified on the 12-doc EN pilot: 0 placeholders, 13 clean person
  nodes (1 central figure + 12 unified authors), no splits. Abel unaffected (named
  via metadata).
- **Free-form relation polarity/symmetry** (`tie_classes.py`): the curated
  polarity/`SYMMETRIC` sets only covered the domain-normalized vocabulary, so
  open-vocabulary LLM relations (generic path) got wrong signs/directedness.
  Added substring heuristics: `dislikes`/`undermines`/`disagreed_with` -> negative;
  `partner_of`/`colleague_of`/`reached_agreement_with`/`*_with` -> symmetric
  (undirected). EN negative edges 0->6, Abel 23->41.
- **Opposition is a stance, not membership**: a person who `opposes`/`is against`
  an org/group was classed `affiliation` (looked like a member). Now -> `stance`.
- **Person<->person structural verbs -> interaction**: `promoted_to`/`led`
  between two PEOPLE were `biographical`/`affiliation`; now `interaction`
  (Abel interaction edges 13->19).

## Docling + LangExtract + NetworkX metrics; GLiNER v1 deprecated

(feature branch: `feature/docling-langextract-networkx`)

- **GLiNER v1 deprecated - GLiNER2 only.** `core/gliner_engine.py` is now
  GLiNER2-only (`fastino/*`); the dual-backend dispatch and `urchade/*` path are
  gone. Dropped the explicit `gliner` dep (still pulled transitively by gliner2;
  its `transformers<5.2` pin is a harmless warning since we don't use v1).
- **NetworkX SNA metrics Gephi can't compute** (`postprocess/graph_metrics.py`,
  `export.graph_metrics`, default off; on in the generic template): Burt's
  constraint + effective_size (structural-hole brokerage), bridges, articulation
  points, and a graph-health QA report (`graph_report.json`). Fail-soft. Adds node
  cols `sna_constraint`/`sna_effective_size`/`sna_is_articulation`, edge `is_bridge`.
  Standard centrality/community stay Gephi's job.
- **Docling structure-aware ingestion** (`io.use_docling`, default off): routes
  PDF/DOCX/PPTX/images through Docling (tables/reading-order/OCR -> markdown) with
  fail-soft fallback to the lightweight extractors. Best for papers/complex PDFs.
  Heavy; upgrades `transformers` to 5.x (fine for GLiNER2).
- **LangExtract mode** (`mode: langextract`): a new extraction backend
  (`intelligence/langextract_backend.py`) that drives Ollama/Gemini/OpenAI via
  Google LangExtract with few-shot examples + char-level source grounding. An
  *alternative* to the ollama/api backends to A/B, not a replacement. Foundation
  GLiNER2 entities are always kept; LangExtract adds grounded entities + relations.

## GLiNER2 + dedup/whitespace fixes

- **GLiNER2 is now the default foundation NER.** `core/gliner_engine.py` supports
  both the original GLiNER (`urchade/*`) and GLiNER2 (`fastino/*`); the backend is
  auto-detected from the model name. Defaults: `fastino/gliner2-large-v1`
  (English) and `fastino/gliner2-multi-v1` (multilingual / German, mDeBERTa,
  100+ langs). Legacy `urchade/*` models still load unchanged. Added `gliner2` to
  requirements.
  - **Windows guard**: GLiNER2 prints an emoji banner on load that crashes a
    cp1252 console (`UnicodeEncodeError`); the loader/predict paths redirect
    stdout so it runs without any `PYTHONUTF8` workaround. Verified on EN + DE.
- **Dedup: bare first/last names now fold into their full name.** Surname-initial
  bucketing kept "Eleanor" and "Eleanor Vance" in different buckets, so they never
  merged - fragmenting people into two nodes (masked in the curated German domain
  by the alias list). New `_fold_partial_persons` merges a single-token PERSON into
  the *unique* full name whose first/last token matches (ambiguous bare names left
  alone; authors/narrators never folded). Verified: 23->19 entities on the EN test.
- **Whitespace in entity names.** A span straddling a line break became the label
  ("Robert\\nChen") and a phantom alias. `aggregator.clean_surface` collapses
  internal whitespace in the stored display name.
- **Known limitation surfaced**: `python_only` mode yields ~0 interpersonal
  *interaction* edges (rule SVO can't capture conjoined/reciprocal/pronominal
  ties like "Eleanor and Marcus met"). Rich interpersonal SNA needs `api`/`ollama`.

## Generalization + Wikidata + temporal

- **Generic domain hardened** so any unstructured text (books, articles, scraped
  pages) gets clean output without domain tuning: a 120-word English stopword list
  (`the man / the road / morning` no longer become nodes) and a general-purpose
  subtype vocabulary (PERSON: leader/official/family_member/...; ORG/LOCATION/
  EVENT/INSTITUTION). All the SNA machinery (tie_class, polarity, corroboration
  weight, junk filter, dedup, reference-figure tagging, multi-view exports) already
  lives in the domain-agnostic core, so it applies to every input automatically.
- **Optional Wikidata entity linking** (`linking.enabled`, off by default):
  adds `wikidata_qid`/`wikidata_url`/`wikidata_label` to high-signal entities
  (Hitler->Q352, Berlin->Q64, NSDAP->Q7320). Bounded, cached, fail-soft - network
  errors just leave entities unlinked. stdlib only.
- **Temporal / dynamic network**: nodes carry `first_year`/`last_year` (from
  datable incident edges) and an additional `network_dynamic.gexf` stamps `start`
  years on nodes/edges for Gephi's timeline. Best-effort, never fatal.
- **Subtype membership guard**: a subtype is accepted only if it belongs to the
  entity type's own vocabulary (no more PERSON tagged `nazi_organization`).

## Tagging + extraction-quality fixes (100-doc re-analyze)

- **Enrichment subtype was broken** - the prompt used a literal `"string"`
  placeholder and gave no vocabulary, so weak models echoed the type back
  (`tag_subtype` = LOCATION/person/"string"). Now the domain's `ENTITY_SUBTYPES`
  are passed as a controlled vocabulary per entity, and a guard drops any subtype
  that equals the type or is placeholder junk.
- **Junk PERSON filter**: OCR fragments / abbreviations mislabeled as people
  (`lch`, `cht`, `Nie`, `Pg.`, `W.`) dropped - single-token <=3 chars or
  all-lowercase names. Authors always kept.
- **`alias_of` edges dropped** (a dedup artifact, never a social tie).
- **Edge `polarity`** (positive / negative / neutral) for signed-network analysis:
  supported/allied -> positive, opposed/fought_against -> negative.
- `_DIRECTED_RELATIONS` removed from main; directedness is now
  `not tie_classes.is_symmetric(rel_type)` (single source of truth).

## SNA methodology overhaul (mention != tie)

The graph conflated three different things into one adjacency matrix: bare
co-occurrence (64% of edges), stance/attitude, and actual social ties. Centrality
on that mix is not interpretable. Edges are now classified and the social network
is a derived view - nothing is deleted.

- **`tie_class` on every edge**: interaction (person<->person) / affiliation
  (person->org) / participation (person->event) / biographical (person->place) /
  stance (attitude) / cooccurrence. `postprocess/tie_classes.py`.
- **Multi-view exports** so the social network is its own graph: `graph_interaction.gexf`
  (the SNA), `graph_affiliation.gexf` (two-mode membership/biography),
  `graph_discourse.gexf` (stance + co-occurrence). Combined `gephi_edges.csv`
  carries the `tie_class` column for manual filtering. Hitler's degree is ~590 on
  the full graph but ~22 in the interaction layer.
- **Stopped precomputing centralities** (degree/betweenness/eigenvector/pagerank/
  community/clustering/k-core): Gephi computes those in one click on whichever view
  you load, and dropping them removes the slowest pipeline stage at corpus scale.
  Nodes keep what Gephi can't derive: a per-tie-class degree split (`deg_*`) and
  the semantic `tag_*`/`attr_*` columns.
- **`reference_figure` tag** on public/historical figures (known list + cross-doc
  recurrence). Kept in the graph, flagged so the symbolic-reference network can be
  separated from authors' lived contacts.
- **Edge weight = distinct corroborating documents**, not raw mention count;
  `n_mentions` and `n_sources` (distinct letters) kept as separate columns.
- **Directedness follows tie semantics**: met_with/family_of/allied_with/
  co_occurs_with undirected; hierarchical/flow ties directed. Plus `reciprocal`.
- **`period` edge tag** (imperial_ww1 / weimar / nazi_rule) when the evidence
  carries a year, for temporal slicing.

## 100-doc pilot fixes

- **Per-letter author identity**: authors now key on their home document in the
  exact-merge layer, so first-name-only filenames ("Emil237442.rtf" ...) no longer
  collapse six different people into one node. Each keeps its letter_id and gets
  its real name from metadata (Emil Krug / Hanf / Groh / ...).
- **Person dedup buckets on surname**, not first name: "Joseph Goebbels" /
  "Dr. Goebbels" / "Goebbels" and K/C variants ("Karl"/"Carl Liebknecht") now land
  in the same fuzzy bucket and merge.
- **German genitive fallback** in alias resolution: "Hitlers"/"Führers"/
  "Deutschlands" fold into the canonical when the s-stripped form is a known alias.
- **Demonyms/abstractions -> canonical** (alias): Franzosen->France, Vaterland->
  Germany, Republik->Weimar Republic, Militär->German Army, Deutschnationalen->DNVP,
  Nationalsozialist/Führer->NSDAP/Hitler.
- **Stoplist round 3**: Pg/Arbeiter/Verwaltung/Kreise/Heil Hitler, bare ranks
  (General/Sturmführer/...), generic schools (Volksschule/Gymnasium/...). Plus a
  determiner-stripper so "mein Vater"/"der Soldat" hit the bare stopword.

## 16-doc re-analyze follow-ups

- **LLM dedup respects the acronym block**: the LLM was merging distinct Nazi-org
  acronyms (NSV, N.S.B.O., N.S.B.A.) into NSDAP; now vetoed like the rule layer.
- **Stoplist + alias additions** from the re-analyze: Staat/Volksgenossen/
  Parteigenossen/Ortsgruppe/Parteien dropped; "Nationalistische Deutsche
  Arbeiterpartei" (typo) / "Partei Hitler" -> NSDAP.

## 16-doc cross-document fixes

- **Acronym over-merge blocked**: distinct acronyms (DVP vs DNVP, USP vs USPD)
  no longer fuzzy-merge; NSDAP/N.S.D.A.P. still merge. Dedup-rule bug the LLM
  can't undo.
- **Third-person pronouns dropped**: "er/sie/ihn/..." no longer become nodes;
  relations with a third-person endpoint are dropped (unresolvable). First-person
  still remaps to the author.
- **Central party consolidation**: added German inflected/spaced + generic aliases
  (Partei, Bewegung, Nationalsozialismus, "Nationalsozialistischen Deutschen
  Arbeiter Partei", ...) -> NSDAP; kommunistische Partei -> KPD. "Krieg" -> World
  War I, "Revolution" -> November Revolution.
- **German generic-noun stoplist** (`entity_config.STOPWORDS`, domain hook): drops
  Stadt/Sohn/Schule/Soldat/Regierung/... mislabeled as entities. Exact-name match,
  so specific names ("Volksschule Berlin") are kept. Authors never dropped.

## pilot 3 polish

- **Pronoun entities dropped**: bare first-person tokens (e.g. "ich") are no
  longer kept as nodes; the named narrator node represents the author.
- **Directed relations forced**: asymmetric types (member_of, born_in, led, ...)
  are always directed, so the graph builder no longer flips endpoints on display
  ("<org> member_of <person>").
- Verified on a fresh 5-doc run: timeline 1870-1934 (no out-of-period), 5 authors
  at 100% coverage, 15 metadata edges, clean relation vocabulary.

## metadata edges + date + validation fixes (pilot 2)

- **Verified edges from metadata**: each author gets born_in -> place_of_birth,
  resided_in -> place_of_residence, member_of -> NSDAP (with membership#/join
  date), member_of -> prior_party; birthplaces/residences become LOCATION nodes.
  edge_source=metadata (most authoritative tier).
- **Membership target typing** (`inference.drop_nonorg_membership`): drop
  member_of/joined/served_in edges whose target isn't ORG/INSTITUTION (kills
  reversed and common-noun-target noise).
- **Date 2-digit pivot + non-date rejection** (`normalize_date`, shared by the
  regex and LLM-timeline paths): "17.2.22"->1922, "1. April 34"->1934,
  "30.09.76"->1876; drops junk like "6 Jahre alt"->2026. Uses the domain study
  period (PERIOD_END) as the century pivot.
- **validate_run**: tolerant of doubled run paths; author coverage now uses the
  run's documents.csv and matches node aliases (was comparing 5 docs vs all 533).

## author-node fixes (from 5-doc pilot)

- **Pronoun -> author remap**: LLM-extracted relations with bare first-person
  endpoints (ich/mir/wir/...) are remapped to the document author; self-loops
  dropped. Also the extraction prompt now names the narrator and forbids pronoun
  entities. (Pilot had `ich member_of NSDAP` as its own node.)
- **Authors protected**: LLM dedup never merges an `is_author` node away; quality
  review (rule + LLM) never drops authors. (Pilot lost an author to dedup/review.)
- **LetterID for multi-doc authors**: stamp uses the author's home doc
  (`author_doc`), so an author mentioned in other letters still gets their id.
- **Metadata name as label**: author node label set to the spreadsheet name
  (filename form kept as alias).

## metadata merge

- **Spreadsheet metadata merged onto author nodes** by letter_id
  (`domain/nazi_era/metadata.py`, `io.metadata_file`). All columns (birth date,
  place of birth, membership number, join date, ...) ride into `gephi_nodes.csv`
  / GEXF as `attr_*`. 579 rows load; openpyxl required.
- **ollama model fix**: nazi config now points at `qwen2.5:7b-instruct`; added
  `--ollama-model` and `--metadata` CLI overrides.

## provenance - LetterID + source snippets

- **documents.csv** manifest: `doc_id, letter_id, author, filename, source_path`.
  `letter_id` = trailing digits of the filename (Hoover/Abel id) - the join key
  to the metadata spreadsheet.
- **LetterID on every output**: author nodes get `attr_letter_id`; edges and
  timeline rows get a `letter_id` column (from the source doc).
- **Entity source snippets**: entities now carry `attr_evidence` (a source
  sentence) + `attr_evidence_doc`. Edges already had `evidence`; timeline has
  `description`. So every datapoint traces to text + document.
- Tokenizer deps declared (`sentencepiece`, `tiktoken`, `protobuf`) and GLiNER
  load now raises an actionable error (pip + cache-clear) instead of a traceback.

## abel prep - author nodes, batching, dedup quality, comment sweep

- **Author-from-filename**: Abel files are `<Author><hooverID>.rtf`; the narrator
  node is now the real author name (merges with the in-text mention, flagged
  `is_author`). Domain hook `narrator_name()` -> `german_nlp.author_from_filename`.
- **Filename silver gold**: `validate_run --inputs <dir>` reports author coverage
  (how many of the 540 named authors surface as PERSON / `is_author`). No manual
  annotation needed.
- **Cross-type resolution** (`dedup.resolve_cross_type`): collapse same-name
  entities split across types (e.g. "Soviet Union" as ORG + LOCATION) onto the
  dominant type; authors never folded.
- **Quality review batching** (`quality.review_batch_size`): LLM review now
  batches entities + their incident edges instead of truncating at 400 - needed
  for ~540-doc runs.
- **Isolate pruning** (`quality.drop_isolated_nodes`): drop degree-0 nodes for
  cleaner SNA graphs.
- **Enrichment after quality** (only enrich survivors); enrichment attributes now
  exported as `attr_*` node columns (visible in Gephi, not just entities.json).
- **Comment sweep**: collapsed all 64 module docstrings to one-line `#` headers
  and trimmed Google-style Args/Returns blocks to summaries. Confirmed: RTF/German
  extraction is clean (no OCR/encoding work needed).

## enrichment, LLM dedup, gold-free nazi validation

- **Enrichment stage** (`postprocess/enricher.py`): optional LLM pass over
  resolved entities -> `subtype` tag + attributes (rank/office/role/affiliation),
  text-grounded only. `enrichment.enabled` (api/ollama). Wired in `run_analyze`
  after dedup. Default off; on in `config_nazi_era.yaml`.
- **LLM-assisted dedup** (`postprocess/llm_dedup.py`): after rule dedup, the LLM
  proposes same-entity merges the rules missed; merges + remaps edges.
  `dedup.llm_assist` (api/ollama). Default off; on in nazi config.
- **Backends**: `enrich()` and `suggest_merges()` on api + ollama; no-op on base
  / python_only. Prompts in `prompts.py` (`ENRICHMENT_SYSTEM`, `MERGE_SYSTEM`).
- **Gold-free nazi validation** (`domain/nazi_era/validate_run.py`): no annotation
  needed. Checks anachronisms + rank/org consistency, alias application
  (alias and canonical must not be two nodes), known-entity coverage, and
  structural sanity (types, isolates, evidence %, edge sources, author/membership
  counts). `python -m domain.nazi_era.validate_run --run-dir <dir>`.

## dedup quality fixes (from benchmark audit)

- **ORG/LOCATION over-merge guard**: distinctive-token conflict block stops
  templated names collapsing ("University of Basel" vs "...Bonn", "South Africa"
  vs "South America", "SPD" vs "PSD"). Was fabricating false hub nodes.
- **PERSON under-merge fix**: family block no longer rejects first-name spelling/
  transliteration variants ("Angela"/"Angel", "Mahmoud"/"Mahmud") - matters for
  OCR'd / translated text.

## NER precision fix

- **Drop spaCy `NORP`** by default (`foundation.exclude_spacy_labels`).
  Nationalities/adjectives ("German", "American") were mapped to ORG: ~1,041 ORG
  false positives on 150 DWIE docs, only 24 ever gold. (No-op for German spaCy.)

## benchmarks + eval robustness

- **Benchmark adapters** (`benchmarks/`): Re-DocRED (`tonytan48/Re-DocRED`), DWIE
  (`DFKI-SLT/DWIE`) from HF; ACE2005 / TACRED from local LDC JSON. `run_benchmark`
  prepares inputs + gold + tuned config, optional `--run --eval`.
- **Scorer**: alias-aware, entity-linking-based matching (gold mention clusters);
  entity P/R/F1 (typed + type-agnostic + per type), relation P/R/F1 (typed +
  untyped). Edge-source tier filtering (conservative/moderate/full).
- **Runner flags**: `--types` (trim labels + gold), `--constrain-relations` (LLM
  emits the dataset's label set; makes typed RE comparable on DWIE),
  `--min-entity-confidence`, `--resume`. Variant-tagged run dirs/reports.
- **Off-target type leak fix**: foundation now restricts to configured types
  (`restrict_to_label_types`); also enforced at analyze. Killed 313 phantom DATE
  + 306 EVENT FPs on DWIE; DWIE entity F1 0.51 -> 0.77.
- **Exporter crash fix**: mixed/sparse columns (e.g. timeline `year` int+None)
  broke polars schema inference -> union keys + null-preserving write + stringify
  fallback; stdlib csv fallback when polars absent.
- **`--limit` consistency**: extract and analyze both score exactly the current
  input set, not the whole accumulated checkpoint. `write_inputs` clears stale
  files. `--resume` finishes interrupted runs without re-extracting.

## ingestion

- Inputs now accept files, directories, **http(s) URLs** (`--url`/`io.urls`/
  `--urls-file`, HTML+PDF fetch) and **raw text** (`--text`). No crawling; scanned
  PDFs need external OCR.

## nazi_era domain + foundation wiring

- **Multilingual GLiNER** (`urchade/gliner_multi-v2.1`) for German.
- **Coreference** (`core/coreference.py`): first-person narrator -> per-document
  author node (EN+DE pronouns), basis for authors-only membership; optional
  fastcoref third-person pass.
- **Mandatory NSDAP membership reframed** to `authors_only` (was every PERSON -
  wrongly tagged Marx/Lenin). `inference.mandatory_membership` =
  authors_only | all | off.
- **Relation ontology alignment** (`postprocess/ontology.py`): map raw relation
  types onto a canonical set (synonym + fuzzy); domain or config supplied; LLM
  constrained to the set when present.
- **German dates** in the date extractor (months + seasons), offsets preserved.
- Domain hooks wired into the foundation: GLiNER labels/label-map, spaCy
  EntityRuler patterns, extraction/quality prompts, temporal vocab, ontology.
- Domain content: 500+ aliases, SA/SS/Wehrmacht rank ladders, org hierarchy,
  4-tier evidence membership inference, german_nlp, historical_context, 24 GLiNER
  labels, EntityRuler patterns, Abel-tuned prompts.

## evaluation harness (initial)

- `evaluation/`: gold schema + loader, scorer, CLI; gold template + docs.

## initial implementation

- Generalized NER + SNA pipeline. Foundation (GLiNER + spaCy, always on) feeding
  three intelligence modes: `api` (Claude/OpenAI/Bedrock), `python_only` (rules +
  dependency parse + embeddings), `ollama` (local LLM).
- Stages: preprocess -> chunk -> foundation (NER/merge/dates) -> intelligence
  (relations) -> aggregate -> dedup (3-layer: alias/exact/fuzzy) -> quality ->
  inference (co-occurrence + canonical) -> tag -> graph (NetworkX metrics) ->
  export (CSV/Parquet/JSON/GEXF/JSONL). Append-only JSONL checkpoint, resume.
- Pluggable `domain/` (aliases, labels, patterns, inference); generic default.
- 5-level JSON repair for LLM output. Edge provenance (`edge_source`) for
  conservative/moderate/full sensitivity analysis.

---

## Backlog (queued)

1. **Document-level relation recall** - sentence-local extraction misses
   cross-sentence pairs. Whole-document LLM relation pass or ATLOP-style context
   pooling. Biggest remaining lever (less acute for Abel: short single-chunk docs).
2. **Dedup cross-initial bucketing** - buckets by (type, first initial); misses
   "JFK" vs "John F. Kennedy". LLM dedup partly covers this when enabled.
3. **ORG/LOCATION GPE gazetteer** - cross-type resolver + NORP drop handle most;
   a place/country gazetteer would catch the rest of the typed-vs-agnostic gap.
4. **Multilingual third-person coref** - fastcoref is EN-only; German er/sie/ihm
   unresolved (narrator heuristic already covers first-person).
5. **Biographical RE benchmark** (Plum et al., 2022) - on-target for Abel.
6. **Residual comment polish** - module docstrings + Args/Returns are swept;
   a few long inline prose comments remain.

Done from prior backlog: large-corpus LLM batching (quality review), isolate
pruning, comment sweep, ORG/LOC (cross-type + NORP). OCR normalization dropped -
the RTF/German source extracts cleanly.
