# Dataclasses passed between stages, with to_dict/from_dict for JSON.

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# Documents & chunks
@dataclass
class Document:
    """A single source document after ingestion + text extraction."""

    doc_id: str
    source_path: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        return cls(**d)


@dataclass
class Chunk:
    """A contiguous slice of a document with absolute char offsets."""

    chunk_id: str
    doc_id: str
    index: int
    text: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**d)


# Entities
@dataclass
class EntityMention:
    """One occurrence of an entity, located in a specific chunk."""

    text: str
    label: str                     # Canonical type: PERSON / ORG / LOCATION / EVENT
    start_char: int                # Absolute offset within the parent document
    end_char: int
    chunk_id: str
    doc_id: str
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)  # {"gliner","spacy","llm","coref_narrator"}
    sentence: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def span_key(self) -> tuple[str, int, int]:
        return (self.doc_id, self.start_char, self.end_char)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EntityMention":
        return cls(**d)


@dataclass
class Entity:
    """A resolved (deduplicated) entity aggregated across the corpus."""

    entity_id: str
    canonical_name: str
    label: str
    aliases: list[str] = field(default_factory=list)
    mention_count: int = 0
    doc_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Entity":
        return cls(**d)


# Relationships / edges
@dataclass
class Relationship:
    """A directed-or-undirected relationship between two entity *surface forms*.

    Source/target are raw names at extraction time; the deduplicator later maps
    them onto resolved ``entity_id`` values.
    """

    source: str
    target: str
    rel_type: str
    doc_id: str
    chunk_id: str = ""
    evidence: str = ""             # Sentence / snippet supporting the relation
    confidence: float = 0.0
    directed: bool = False
    origin: str = "extracted"      # "extracted" | "inferred" | "canonical"
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Relationship":
        return cls(**d)


# Temporal events
@dataclass
class TimelineEvent:
    """A dated occurrence extracted from the text."""

    doc_id: str
    chunk_id: str
    date_text: str
    iso_date: Optional[str]        # Normalized YYYY[-MM[-DD]] if parseable
    year: Optional[int]
    description: str
    entities: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimelineEvent":
        return cls(**d)


# Per-chunk foundation output & per-document intelligence output
@dataclass
class FoundationResult:
    """Output of the foundation layer for one chunk."""

    chunk: Chunk
    mentions: list[EntityMention] = field(default_factory=list)
    dates: list[TimelineEvent] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "mentions": [m.to_dict() for m in self.mentions],
            "dates": [d.to_dict() for d in self.dates],
            "sentences": self.sentences,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FoundationResult":
        return cls(
            chunk=Chunk.from_dict(d["chunk"]),
            mentions=[EntityMention.from_dict(m) for m in d.get("mentions", [])],
            dates=[TimelineEvent.from_dict(x) for x in d.get("dates", [])],
            sentences=d.get("sentences", []),
        )


@dataclass
class DocumentExtraction:
    """The complete per-document result emitted by the intelligence tier.

    This is the unit that gets checkpointed (one JSONL line per document).
    """

    doc_id: str
    source_path: str
    mentions: list[EntityMention] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "mentions": [m.to_dict() for m in self.mentions],
            "relationships": [r.to_dict() for r in self.relationships],
            "timeline": [t.to_dict() for t in self.timeline],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DocumentExtraction":
        return cls(
            doc_id=d["doc_id"],
            source_path=d.get("source_path", ""),
            mentions=[EntityMention.from_dict(m) for m in d.get("mentions", [])],
            relationships=[Relationship.from_dict(r) for r in d.get("relationships", [])],
            timeline=[TimelineEvent.from_dict(t) for t in d.get("timeline", [])],
            meta=d.get("meta", {}),
        )


# ID helpers
def stable_id(*parts: Any, prefix: str = "", length: int = 12) -> str:
    """Deterministic short id from arbitrary parts (used for entity/edge ids)."""
    raw = "||".join(str(p) for p in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{digest}" if prefix else digest
