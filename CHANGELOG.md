# Changelog

Sequential record of what shipped. Newest first. Terse on purpose.

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
