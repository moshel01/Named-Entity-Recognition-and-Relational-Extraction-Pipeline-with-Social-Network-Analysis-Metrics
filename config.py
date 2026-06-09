# YAML -> typed pydantic config. extra="forbid" on the root catches typos.

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# Sub-models
class IOConfig(BaseModel):
    input_path: str = ""                      # file, directory, or http(s) URL
    input_glob: str = "**/*"
    output_dir: str = "./output"
    encoding: str = "auto"
    urls: list[str] = Field(default_factory=list)   # web pages / PDFs to fetch
    urls_file: str = ""                       # path to a newline-delimited URL list
    request_timeout: int = 30
    metadata_file: str = ""                   # xlsx of per-doc metadata, keyed by letter_id
    use_docling: bool = False                 # structure-aware ingestion (PDF tables/OCR); fail-soft


class ChunkingConfig(BaseModel):
    max_chars: int = 6000
    overlap_chars: int = 400
    respect_sentences: bool = True


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
    # Languages whose first-person pronouns mark the narrator.
    languages: list[str] = Field(default_factory=lambda: ["en", "de"])


class FoundationConfig(BaseModel):
    spacy_model: str = "en_core_web_trf"
    spacy_disable: list[str] = Field(default_factory=list)
    gliner_model: str = "urchade/gliner_large-v2.1"
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
    max_tokens: int = 4096
    temperature: float = 0.0
    max_retries: int = 4
    request_timeout: int = 120
    aws_region: str = "us-east-1"


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
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
    model_id: str = "qwen2.5:7b-instruct"
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
    llm_review: Any = "auto"        # "auto" | True | False
    review_batch_size: int = 150    # entities per LLM review call (large corpora)
    drop_isolated_nodes: bool = False   # drop degree-0 nodes from the final graph


class InferenceConfig(BaseModel):
    enable_cooccurrence_edges: bool = True
    cooccurrence_min_shared_docs: int = 2
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
    relations: Any = None              # dict[str, list[str]] | list[str] | None


class LinkingConfig(BaseModel):
    # Optional entity linking to Wikidata (adds wikidata_qid/url attributes).
    # Off by default - it makes network calls. Fail-soft and bounded.
    enabled: bool = False
    min_mentions: int = 2              # only link entities seen at least this often
    max_entities: int = 400           # hard cap on lookups per run
    types: list[str] = Field(default_factory=lambda: ["PERSON", "ORG", "LOCATION"])
    lang: str = "en"                  # Wikidata search language
    request_timeout: int = 8


class DomainConfig(BaseModel):
    name: str = "generic"


class ExportConfig(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["csv", "json", "gexf"])
    gephi: bool = True
    # Precompute SNA metrics Gephi can't (Burt's constraint/effective_size,
    # bridges, articulation points) + a graph-health QA report. Fail-soft.
    graph_metrics: bool = False


class CheckpointConfig(BaseModel):
    enabled: bool = True
    flush_every: int = 1


# Root model
class Config(BaseModel):
    model_config = {"extra": "forbid"}

    run_name: str = "default_run"
    mode: Literal["api", "python_only", "ollama", "langextract"] = "python_only"
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
