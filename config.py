# YAML -> typed pydantic config. extra="forbid" on the root catches typos.

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# Sub-models
class CrawlConfig(BaseModel):
    """Whole-site crawl: expand seed URLs into their subpages before extraction.
    Off by default - the pipeline still takes explicit `urls`. Bounded + polite
    (robots.txt, per-host rate limit, page/depth/size caps). See core/crawler.py."""
    enabled: bool = False
    seeds: list[str] = Field(default_factory=list)    # start URLs to expand
    max_pages: int = 50                       # documents to fetch (hard cap)
    max_depth: int = 3                        # link hops from a seed
    stay_on_host: bool = True                 # don't leave the seed's host
    stay_under_path: bool = False             # also require the seed's dir prefix
    allow: list[str] = Field(default_factory=list)    # regex; URL must match one
    deny: list[str] = Field(default_factory=list)     # regex; URL dropped if matched
    delay: float = 1.0                        # min seconds between requests per host
    respect_robots: bool = True               # obey robots.txt + crawl-delay
    use_sitemap: bool = True                  # seed from sitemap.xml when present
    user_agent: str = ""                      # blank -> pipeline default UA
    timeout: int = 30
    max_bytes: int = 5_000_000                # per-page download ceiling
    # Render JavaScript with headless Chromium (Playwright) before reading the page.
    # Needed for SPA / JS-built sites that return an empty shell to a plain GET; a
    # no-op cost for server-rendered sites. Optional dep: `pip install playwright`
    # then `playwright install chromium`. robots.txt/sitemaps/PDFs bypass the browser.
    render_js: bool = False                   # CLI: --render-js


class IOConfig(BaseModel):
    input_path: str = ""                      # file, directory, or http(s) URL
    input_glob: str = "**/*"
    output_dir: str = "./output"
    encoding: str = "auto"
    urls: list[str] = Field(default_factory=list)   # web pages / PDFs to fetch
    urls_file: str = ""                       # path to a newline-delimited URL list
    # Portable corpus snapshot (the ingestion checkpoint). If set, the run LOADS
    # documents straight from this JSONL - no crawl, no fetch, no file walk - so a
    # scrape+preprocess done once on one machine can be shipped to another and run
    # in any --mode. Written automatically to <run_dir>/documents.jsonl on a normal
    # gather; `--stage fetch` writes it and stops. CLI: --ingest-from PATH.
    documents_file: str = ""
    crawl: CrawlConfig = Field(default_factory=CrawlConfig)   # whole-site expansion
    # Social-media sources: each entry is "platform:target" (reddit:datascience,
    # hackernews:top, mastodon:mastodon.social/tag/ai, twitter:from:nasa). The connector
    # (core/social) fetches posts AND the explicit reply/mention/posted_in structure.
    # CLI: --social. Twitter needs $TWITTER_BEARER_TOKEN; Facebook is not supported.
    social: list[str] = Field(default_factory=list)
    social_limit: int = 100                   # posts/items per source
    social_depth: int = 1                      # 1 = also pull comment/reply trees
    request_timeout: int = 30
    metadata_file: str = ""                   # xlsx of per-doc metadata, keyed by letter_id
    use_docling: bool = False                 # structure-aware ingestion (PDF tables/OCR); fail-soft


class ChunkingConfig(BaseModel):
    max_chars: int = 6000
    overlap_chars: int = 400
    respect_sentences: bool = True

    @model_validator(mode="after")
    def _overlap_below_max(self) -> "ChunkingConfig":
        # overlap >= max stalls the chunker's hard-split (step <= 0) and makes the
        # sentence-overlap back-up a no-op. Catch the misconfig at load, not mid-run.
        if self.overlap_chars >= self.max_chars:
            raise ValueError(
                f"chunking.overlap_chars ({self.overlap_chars}) must be < "
                f"max_chars ({self.max_chars})")
        if self.overlap_chars < 0 or self.max_chars <= 0:
            raise ValueError("chunking.max_chars must be > 0 and overlap_chars >= 0")
        return self


class CoreferenceConfig(BaseModel):
    enabled: bool = True
    # Resolve first-person narration to a per-document author/narrator node.
    # Essential for first-person sources (e.g. the Abel autobiographies) and the
    # basis for the "authors_only" mandatory-membership scope.
    narrator_resolution: bool = True
    # Optional fastcoref pass for third-person pronouns (English-oriented).
    pronoun_resolution: bool = False
    model: str = "biu-nlp/f-coref"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    max_narrator_mentions_per_chunk: int = 40
    # Coref microservice (services/coref_service.py). Empty = load fastcoref
    # in-process. Set to e.g. "http://127.0.0.1:8000" to offload coref to an
    # isolated env (fastcoref needs transformers <5, which conflicts with the
    # main env's GLiNER2). Unreachable service falls back to in-process/heuristic.
    service_url: str = ""
    service_timeout: int = 30
    # Languages whose first-person pronouns mark the narrator.
    languages: list[str] = Field(default_factory=lambda: ["en", "de"])


class FoundationConfig(BaseModel):
    spacy_model: str = "en_core_web_trf"
    spacy_disable: list[str] = Field(default_factory=list)
    gliner_model: str = "fastino/gliner2-multi-v1"   # multilingual, fits 8GB; large-v1 = English-only, heavier
    gliner_threshold: float = 0.45
    gliner_labels: list[str] = Field(
        default_factory=lambda: ["person", "organization", "location", "event"]
    )
    label_map: dict[str, str] = Field(
        default_factory=lambda: {
            "person": "PERSON",
            "organization": "ORG",
            "location": "LOCATION",
            "event": "EVENT",
        }
    )
    use_spacy_ner: bool = True
    # Drop merged mentions whose canonical type is not among label_map values.
    # Prevents spaCy's statistical NER from leaking off-target types (DATE/EVENT/
    # ORDINAL...) that were never requested - a major source of false positives.
    restrict_to_label_types: bool = True
    # spaCy NER labels to ignore entirely. NORP (nationalities / religious /
    # political adjectives: "American", "deutsche") maps to ORG and is the single
    # largest source of ORG false positives in benchmarks - excluded by default.
    exclude_spacy_labels: list[str] = Field(default_factory=lambda: ["NORP"])
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


class ApiConfig(BaseModel):
    provider: Literal["anthropic", "openai", "bedrock"] = "anthropic"
    model: str = "claude-opus-4-8"
    api_key_env: str = "ANTHROPIC_API_KEY"
    # OpenAI-compatible endpoint for any cheap host (DeepSeek, Together, Groq,
    # OpenRouter, local vLLM/llama.cpp). Set provider: openai and point base_url
    # at it. Empty = the provider's own default. For DeepSeek use deepseek-chat
    # (V3): cheap, fast, JSON-mode capable. deepseek-reasoner (R1) wastes tokens on
    # reasoning and its structured output is unreliable - same trap as qwen3.5.
    base_url: str = ""
    # response_format={"type":"json_object"} for openai-compatible servers that
    # honor it (OpenAI, DeepSeek-chat). Off for reasoner models - they reject it.
    json_mode: bool = False
    max_tokens: int = 4096
    temperature: float = 0.0
    max_retries: int = 4
    request_timeout: int = 120
    aws_region: str = "us-east-1"


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "qwen3.5:9b"
    temperature: float = 0.0
    request_timeout: int = 180
    num_ctx: int = 8192


class PythonOnlyConfig(BaseModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cooccurrence_window: Literal["sentence", "chunk"] = "sentence"
    min_relationship_confidence: float = 0.30
    embedding_similarity_threshold: float = 0.55


class LangExtractConfig(BaseModel):
    # LangExtract orchestrates an LLM (Ollama / Gemini / OpenAI) with few-shot
    # examples + char-level source grounding. An alternative to the ollama/api
    # backends - same underlying model, different extraction machinery. A/B it.
    provider: Literal["ollama", "gemini", "openai"] = "ollama"
    model_id: str = "qwen3.5:9b"
    model_url: str = "http://localhost:11434"   # Ollama server (ignored for cloud)
    api_key_env: str = ""                        # env var for gemini/openai
    temperature: float = 0.0
    extraction_passes: int = 1                   # >1 improves recall (re-runs + merges)
    max_workers: int = 4
    max_char_buffer: int = 6000


class IntelligenceConfig(BaseModel):
    api: ApiConfig = Field(default_factory=ApiConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    python_only: PythonOnlyConfig = Field(default_factory=PythonOnlyConfig)
    langextract: LangExtractConfig = Field(default_factory=LangExtractConfig)
    # Cost gate (LLM modes only): skip the relation call for a chunk too sparse to
    # contain a relation. A relation needs two entities co-occurring, so a chunk
    # with no two distinct entities inside one window can't yield one - skipping it
    # saves API tokens with no recall loss. Off by default (no behavior change).
    skip_sparse_chunks: bool = False
    sparse_window_words: int = 200
    sparse_min_entities: int = 2
    # Optional per-edge qualifier fields the LLM is asked to fill when present in
    # the text: a typed attribute on a relation, not a new relation. Generic - the
    # domain/config declares the names, the model fills them, and they ride through
    # to the Gephi/GEXF export as `qual_<name>` edge columns. Examples:
    # `monetary_value` (InfluenceWatch PAC->shell funding), `jurisdiction` (OREM
    # disaster coordination scope), `location`/`time` (any spatiotemporal record),
    # `weapon`/`setting` (a narrative/script). Empty = no qualifiers (default).
    edge_qualifiers: list[str] = Field(default_factory=list)
    # Show the model each constrained relation's argument types in the prompt
    # (born_in: person->place, employed_by: person->org, ...) so it forms fewer
    # type-violating edges at the source instead of being tagged after the fact
    # (postprocess.ontology.RELATION_TYPE_SIGNATURES). Only affects relations that
    # have a signature; loose stance/interaction types are unconstrained. Off by
    # default - flip on and A/B against type_violations_by_relation in the report.
    type_hints: bool = False
    # Schema-constrained generation for the extraction call: pass a JSON schema to
    # the backend so the model can only emit valid JSON of the right shape (ollama
    # format=<schema> grammar; OpenAI/Gemini response_format json_schema). Kills the
    # weak-model failure where reasoning leaks into the JSON and the whole document's
    # extraction is lost (the json_repair arms race). Enforces shape + the canonical
    # entity types, not the relation vocabulary (the aligner still maps surface
    # forms). Recommended for local models; off by default (some hosts/older ollama
    # builds ignore or mishandle format-schema). anthropic/bedrock ignore it.
    structured_output: bool = False
    # Recall pass: after chunk-by-chunk extraction, re-prompt over the WHOLE document
    # (entities + already-found relations in hand) for the ties the first pass missed
    # - chiefly relations whose endpoints fell in different chunks. The recall half of
    # the L3X loop; pair with quality.verify_relations for precision. LLM modes only,
    # one extra call per doc, so off by default. gemini_batch already sees whole docs,
    # so this mainly lifts api/ollama. A doc longer than recall_max_chars is skipped
    # (won't fit context); raise it for big-context models.
    recall_pass: bool = False
    recall_max_chars: int = 24000
    # Script/screenplay co-presence: when a document parses as a script (scene slugs +
    # speaker cues), add Newman-weighted co_present_in_scene edges between characters
    # who share a scene (core/script_parser). The standard character-network signal the
    # proximity window only approximates. Co-presence, not an asserted tie -> weakest
    # evidence tier (script_copresence), filterable by tie class. Off by default;
    # non-scripts are untouched (the heuristic returns nothing).
    parse_scripts: bool = False
    # gemini_batch: max characters of document text per emitted prompt file. A
    # corpus past this splits into numbered files. Lower it if the model truncates
    # its JSON reply (output length, not input context, is usually the limit).
    batch_char_budget: int = 4_000_000
    # gemini_batch: max DOCUMENTS per prompt file (0 = no cap, char budget only).
    # The reply's length scales with doc count, so this is the knob that actually
    # prevents truncation - 20-40 is a safe range for dense first-person sources.
    batch_max_docs: int = 0
    # gemini_batch --submit: auto-POST each prompt to the Gemini REST API and write
    # the reply files (no manual paste). Free AI Studio key in this env var. Forces
    # JSON output + a high output-token cap (the truncation fix the chat UI hides).
    batch_api_key_env: str = "GEMINI_API_KEY"
    batch_model: str = "gemini-2.5-flash"   # or gemini-2.5-pro (higher quality, lower free RPD)
    batch_base_url: str = ""                 # blank -> Google's generativelanguage endpoint
    batch_max_output_tokens: int = 65536
    # gemini_batch --submit: thinking-token budget. 2.5-flash thinks by DEFAULT and
    # those tokens count against maxOutputTokens, so they eat the JSON budget and
    # truncate the reply. 0 = thinking off (the right call for structured extraction;
    # reclaims the whole output budget). <0 = leave the API default on. 2.5-pro can't
    # go below 128, so 0 is bumped to 128 there.
    batch_thinking_budget: int = 0
    # gemini_batch extracts from the whole-document reply, so the post-extraction LLM
    # steps (dedup.llm_assist / quality.llm_review / enrichment) have no live backend
    # and are skipped. Flip this on to run them through Gemini's OpenAI-compatible
    # endpoint with the same --submit key (extra API calls at analyze time; off so a
    # benchmark/Abel run doesn't pay for them unasked). Needs the key set.
    batch_post_llm: bool = False
    batch_request_timeout: int = 600
    # gemini_batch only: after the whole-doc reply lands, re-run local GLiNER/spaCy
    # NER and fold the spans back onto the span-less LLM entities (the model gives
    # names, not offsets). Re-activates the within-doc proximity co-occurrence floor
    # + evidence grounding that gemini_batch otherwise loses. Post-hoc, not prompt
    # priming (priming anchors the model to GLiNER's noise). Loads the foundation
    # models at analyze time. Off by default; A/B it before committing a big run.
    reconcile_ner: bool = False
    # The recall net: also ADD GLiNER entities the LLM missed, tagged ner_only.
    # Measured FALSE by default: on the Abel pilot it added ~1850 GLiNER over-
    # extractions to 297 real entities (asserted tier collapsed 85% -> 2%), because a
    # thorough whole-doc extractor leaves GLiNER mostly noise. Turn on only for a
    # SPARSE corpus where the LLM genuinely under-extracts. Span transfer (the useful
    # half) does not need it.
    reconcile_add_missed: bool = False
    # Proactive client-side request cap for --submit (requests/minute). The free
    # gemini-2.5-flash tier is ~10 RPM; without pacing the loop POSTs back-to-back,
    # trips 429, and eats exponential backoff. Set to the tier limit to space
    # request starts evenly instead. No-op when batches are slower than the interval
    # anyway (whole-doc replies usually are). 0 = unthrottled (rely on 429 retry).
    batch_rpm: int = 0


class DedupConfig(BaseModel):
    fuzzy_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "PERSON": 0.85,
            "ORG": 0.72,
            "LOCATION": 0.82,
            "EVENT": 0.92,
        }
    )
    block_family_merges: bool = True
    block_year_mismatch_events: bool = True
    block_location_substring: bool = True
    # Fold demonyms into their place node ("American" -> "United States");
    # see core/demonyms.py. Domain aliases override the built-in table.
    fold_demonyms: bool = True
    # Collapse same-name entities that got different types (e.g. "Soviet Union"
    # as ORG and LOCATION) onto the dominant type.
    resolve_cross_type: bool = True
    # After rule dedup, ask the LLM to propose extra same-entity merges the rules
    # missed (api/ollama only). Off by default; deterministic without it.
    llm_assist: bool = False


class EnrichmentConfig(BaseModel):
    # LLM pass over resolved entities: subtype, rank/office, attributes - only
    # from text already extracted. api/ollama only; off by default.
    enabled: bool = False
    batch_size: int = 40


class QualityConfig(BaseModel):
    enabled: bool = True
    min_entity_mentions: int = 1
    min_edge_weight: int = 1
    # Drop entities whose best mention confidence is below this (0.0 = keep all).
    # A floor around 0.5 trims low-confidence single-source spans and lifts
    # precision on noisy types (e.g. ORG) at some cost to recall.
    min_entity_confidence: float = 0.0
    # Drop a PERSON entity that spaCy never tags as a proper noun across all
    # its mentions ("Monsieur", "der Vater") - a language-general common-noun
    # gate that needs no per-corpus stopword list. Borderline entities are
    # tagged suspect_common_noun and kept.
    pos_gate: bool = True
    llm_review: Any = "auto"        # "auto" | True | False
    review_batch_size: int = 150    # entities per LLM review call (large corpora)
    drop_isolated_nodes: bool = False   # drop degree-0 nodes from the final graph
    # LLM relation self-verification: re-check each LLM edge against its evidence
    # ("does this sentence assert that tie?") and tag verification=supported/unsupported
    # (filterable in Gephi). Catches the case verbatim grounding can't - evidence
    # present but not actually asserting the relation. LLM modes only; extra API calls,
    # so off by default. verify_drop deletes the unsupported instead of tagging.
    verify_relations: bool = False
    verify_drop: bool = False
    verify_max: int = 0             # cap edges verified (0 = all eligible); cost knob
    verify_batch_size: int = 20
    # Whether the verifier's unsupported flags are reliable enough to PRUNE the
    # SNA metric graph (brokerage/bridges/signed balance), not just tag edges in
    # the export. Strong verifiers (api/gemini, ~80% precision) earn the prune;
    # weak local self-verifiers (qwen over-rejects ~50%) do not - keep this False
    # for ollama or half the real ties vanish from the structure before it is
    # measured. Off => unsupported is tagged everywhere, dropped from metrics
    # nowhere. The export is unchanged either way.
    trust_verification: bool = False


class InferenceConfig(BaseModel):
    enable_cooccurrence_edges: bool = True
    cooccurrence_min_shared_docs: int = 2
    # Within-document proximity co-occurrence: link entities mentioned within
    # `proximity_window_chars` of each other (doc-absolute positions, so it spans
    # chunk boundaries the LLM never saw across). A windowed tie is the standard
    # character-network signal; far less noisy than the whole-doc complete graph,
    # and the only within-letter weak-tie layer in ollama/api mode. Co-occurrence
    # stays the weakest evidence tier (full only). 0 disables.
    enable_proximity_edges: bool = True
    proximity_window_chars: int = 600
    # Drop proximity pairs that co-occur fewer than this many times across the
    # corpus. 1 = keep every single windowed adjacency (default, tag-don't-filter).
    # Raise to 2-3 on dense/web corpora: a single accidental adjacency is the
    # weakest possible signal and on entity-dense pages it dominates the edge set
    # (a Wikipedia crawl yields ~60% weight-1 proximity edges). Typed/asserted
    # edges are never affected - this floors only the within-doc co-occurrence layer.
    proximity_min_count: int = 1
    # Disparity-filter backbone (Serrano et al. 2009) over the co-occurrence layer.
    # Every co_occurs_with edge is stamped with `disparity_alpha`; when this is > 0,
    # edges that aren't statistically significant for either endpoint at this level
    # are dropped, leaving the weighted backbone. 0 = off (tag only). 0.05-0.30 is
    # the useful band on dense/web corpora; smaller = sparser backbone. Typed edges
    # are never touched.
    cooccurrence_backbone_alpha: float = 0.0
    # Two-mode (affiliation) projection: people/agencies who share a formal group
    # (an org, institution, or event they're both tied to) get a `co_affiliated`
    # edge - the classic Breiger two-mode -> one-mode actor network. Newman 1/(k-1)
    # weighting (a 50-member org doesn't forge ties as strong as a 2-person board).
    # Off by default; a co-presence layer (full tier, not a direct asserted tie),
    # like co-occurrence but over shared affiliations instead of shared documents.
    # Built for affiliation-dense domains (boards/PACs, multi-agency disaster
    # response) where direct person-person ties are rare.
    enable_affiliation_projection: bool = False
    affiliation_min_shared: int = 1   # min shared groups to draw a co_affiliated edge
    # Which entity kinds are the actors vs the shared groups in the projection.
    # Default: people share orgs/events. Disaster response makes agencies the
    # actors sharing a response EVENT (actor_kinds: ["ORG","INSTITUTION"],
    # group_kinds: ["EVENT"]) so co_affiliated links agencies, not just people.
    affiliation_actor_kinds: list[str] = Field(default_factory=lambda: ["PERSON"])
    affiliation_group_kinds: list[str] = Field(
        default_factory=lambda: ["ORG", "INSTITUTION", "EVENT"])
    # Cross-document author anchoring: a corpus where many documents have a known
    # author (Abel's 537 letters) gives a closed name registry. A lone surname
    # mention in one letter that UNIQUELY identifies one author (no other person
    # shares it) is folded into that author node, forging the cross-letter edge that
    # makes the corpus one network instead of 537 ego-graphs. Zero-ambiguity only,
    # capped. Off by default (a merge, so conservative).
    link_known_authors: bool = False
    link_known_authors_min_len: int = 4   # ignore short/initial surnames
    enable_canonical_inference: bool = False
    # How the corpus-level mandatory-membership assumption is applied by domains
    # that implement one (e.g. nazi_era NSDAP). "authors_only" is the defensible
    # default; "all" over-connects; "off" disables it (evidence-based only).
    mandatory_membership: Literal["authors_only", "all", "off"] = "authors_only"
    # member_of/joined/served_in edges whose target isn't an ORG/INSTITUTION are
    # always TAGGED `suspect_membership` (filterable in Gephi). Set this True to
    # also delete them outright; default keeps them (tag, don't filter).
    drop_nonorg_membership: bool = False


class OntologyConfig(BaseModel):
    # Relation-ontology alignment: map raw extracted relation types onto a
    # canonical set (synonym + fuzzy), and (for LLM modes) constrain extraction
    # to that set. The ontology comes from `relations` here if set, else from
    # the active domain. When neither provides one, alignment is a no-op.
    enabled: bool = True
    fuzzy_threshold: float = 0.82
    drop_unmapped: bool = False        # drop relations that match nothing in the ontology
    # Type-signature consistency check (ASP-style): a relation whose endpoint
    # entity types contradict its signature ("led" pointing at a place) is
    # tagged type_violation, filterable in Gephi. Set true to drop instead.
    drop_type_violations: bool = False
    # Functional-property consistency (global, vs the per-edge type gate): a subject
    # with the same functional relation (born_in, birth_date, ...) pointing at two
    # different targets is a contradiction - the narrator-vs-relative birthplace
    # confound or a misread. Tag every edge in the conflict `functional_conflict`
    # (filterable). On by default (tag only; a column appears solely on real
    # conflicts). drop_functional_conflicts keeps only the best-supported target.
    check_functional_consistency: bool = True
    drop_functional_conflicts: bool = False
    relations: Any = None              # dict[str, list[str]] | list[str] | None
    # label -> one-line definition shown to the LLM next to the allowed types.
    # Make confusable labels contrastive ("associate: companions, NOT friends").
    # Local models default to intuitive labels; definitions pin the coding scheme.
    relation_guide: Any = None         # dict[str, str] | None


class ExpansionConfig(BaseModel):
    """Grow an existing network instead of building one from scratch. Point at a
    prior run dir (or its gephi_edges.csv / network.gexf) and the new documents
    are constrained to the schema already there - the relation vocabulary
    ("strict edge formatting") and the entity kinds - so the expansion stays
    consistent with what you already have. Off by default. See
    postprocess/expansion.py."""
    enabled: bool = False
    source: str = ""                  # prior run dir, its gephi_edges.csv, or a .gexf
    lock_relations: bool = True       # only keep relation types present in the source
    # Strict: a new edge whose type is not in the locked set is dropped, not kept
    # as an off-vocabulary tag. Set false to tag (ontology=unmapped) and keep.
    drop_unmapped_relations: bool = True
    lock_entity_types: bool = True    # keep only the entity kinds present in the source
    entity_types: list[str] = Field(default_factory=list)  # explicit override of kinds


class LinkingConfig(BaseModel):
    # Optional entity linking to Wikidata (adds wikidata_qid/url attributes).
    # Off by default - it makes network calls. Fail-soft and bounded.
    enabled: bool = False
    min_mentions: int = 2              # only link entities seen at least this often
    max_entities: int = 400           # hard cap on lookups per run
    types: list[str] = Field(default_factory=lambda: ["PERSON", "ORG", "LOCATION"])
    lang: str = "en"                  # Wikidata search language
    request_timeout: int = 8
    # Merge entities that resolved to the same QID (cross-doc identity). A shared
    # Wikidata id is stronger than any string match; folds variants dedup missed.
    consolidate_by_qid: bool = True


class DomainConfig(BaseModel):
    name: str = "generic"


class ExportConfig(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["csv", "json", "gexf"])
    gephi: bool = True
    # Precompute SNA metrics Gephi can't (Burt's constraint/effective_size,
    # bridges, articulation points) + a graph-health QA report. Fail-soft.
    graph_metrics: bool = False
    # Build the narrative-sequence network (Bearman & Stovel 2000): corpus-level
    # element->element transitions from the timeline. Writes narrative.gexf +
    # narrative_transitions.csv. Best on first-person life narratives. Fail-soft.
    narrative_network: bool = False
    # Element scheme for the narrative network: "life_course" (war/politics/family
    # life-course buckets - the Abel default) or "fiction" (plot beats: conflict ->
    # revelation -> resolution, for novels/scripts). A domain's narrative_rules()
    # overrides this. "auto" = life_course unless a domain supplies its own.
    narrative_scheme: Literal["auto", "life_course", "fiction"] = "auto"
    # Write codebook.xlsx into the run dir: variable definitions + this run's
    # type/tie-class/relation inventories, for readers new to the data. Fail-soft.
    codebook: bool = True
    # Free-text corpus caveat for the codebook overview (sampling frame, known
    # biases). Boundary specification includes how the data came to be.
    codebook_note: str = ""


class CheckpointConfig(BaseModel):
    enabled: bool = True
    flush_every: int = 1


# Root model
class Config(BaseModel):
    model_config = {"extra": "forbid"}

    run_name: str = "default_run"
    mode: Literal["api", "python_only", "ollama", "langextract", "gemini_batch"] = "python_only"
    io: IOConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    foundation: FoundationConfig = Field(default_factory=FoundationConfig)
    coreference: CoreferenceConfig = Field(default_factory=CoreferenceConfig)
    intelligence: IntelligenceConfig = Field(default_factory=IntelligenceConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    ontology: OntologyConfig = Field(default_factory=OntologyConfig)
    expansion: ExpansionConfig = Field(default_factory=ExpansionConfig)
    linking: LinkingConfig = Field(default_factory=LinkingConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @field_validator("export")
    @classmethod
    def _check_formats(cls, v: ExportConfig) -> ExportConfig:
        allowed = {"csv", "parquet", "json", "gexf", "jsonl"}
        bad = set(v.formats) - allowed
        if bad:
            raise ValueError(f"Unknown export formats: {sorted(bad)} (allowed: {sorted(allowed)})")
        return v

    # Convenience
    @property
    def run_output_dir(self) -> Path:
        return Path(self.io.output_dir) / self.run_name

    def canonical_label(self, gliner_label: str) -> str:
        """Map a GLiNER label to its canonical entity type."""
        return self.foundation.label_map.get(gliner_label.lower(), gliner_label.upper())


# Loader
def load_config(path: str | Path, overrides: Optional[dict[str, Any]] = None) -> Config:
    # overrides: top-level keys (e.g. {"mode": "api"}); None values ignored.
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**data)
