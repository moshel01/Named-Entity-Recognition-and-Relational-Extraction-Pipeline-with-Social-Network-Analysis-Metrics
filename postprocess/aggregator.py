# Aggregate per-document extractions into corpus-level tables.

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

from core.schema import (
    DocumentExtraction,
    Entity,
    EntityMention,
    Relationship,
    TimelineEvent,
    stable_id,
)

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s&.-]")


def normalize_name(name: str) -> str:
    """Cheap normalization for exact grouping (NOT fuzzy)."""
    n = name.strip()
    n = _PUNCT_RE.sub("", n)
    n = _WS_RE.sub(" ", n)
    return n.strip().lower()


@dataclass
class AggregateResult:
    """Bundle of corpus-level tables produced by aggregation."""

    entities: list[Entity] = field(default_factory=list)
    mentions: list[EntityMention] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)


def aggregate(extractions: list[DocumentExtraction]) -> AggregateResult:
    """Merge per-document extractions into corpus tables."""
    mentions: list[EntityMention] = []
    relationships: list[Relationship] = []
    timeline: list[TimelineEvent] = []

    # group key -> aggregation buffers
    by_key: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"surface": defaultdict(int), "docs": set(), "count": 0,
                 "conf": 0.0, "attrs": {}, "evidence": "", "evidence_doc": ""}
    )

    for ex in extractions:
        for m in ex.mentions:
            if not m.text.strip():
                continue
            mentions.append(m)
            key = (normalize_name(m.text), m.label)
            buf = by_key[key]
            buf["surface"][m.text.strip()] += 1
            buf["docs"].add(m.doc_id)
            buf["count"] += 1
            buf["conf"] = max(buf["conf"], m.confidence)
            # Keep one source sentence as provenance for the entity.
            if not buf["evidence"] and m.sentence.strip():
                buf["evidence"] = m.sentence.strip()[:300]
                buf["evidence_doc"] = m.doc_id
            # Carry forward salient mention attributes (e.g. narrator flag).
            for ak, av in (m.attributes or {}).items():
                if ak == "is_author":
                    buf["attrs"]["is_author"] = buf["attrs"].get("is_author", False) or bool(av)
                    if av:
                        buf["attrs"].setdefault("author_doc", m.doc_id)  # author's home letter
                else:
                    buf["attrs"].setdefault(ak, av)
        relationships.extend(ex.relationships)
        timeline.extend(ex.timeline)

    entities: list[Entity] = []
    for (norm, label), buf in by_key.items():
        # Canonical name = most frequent surface form (ties -> longest).
        surfaces = buf["surface"]
        canonical = max(surfaces, key=lambda s: (surfaces[s], len(s)))
        aliases = sorted(s for s in surfaces if s != canonical)
        attrs = dict(buf["attrs"])
        if buf["evidence"]:
            attrs["evidence"] = buf["evidence"]
            attrs["evidence_doc"] = buf["evidence_doc"]
        entities.append(
            Entity(
                entity_id=stable_id(norm, label, prefix="ent_", length=12),
                canonical_name=canonical,
                label=label,
                aliases=aliases,
                mention_count=buf["count"],
                doc_ids=sorted(buf["docs"]),
                confidence=round(buf["conf"], 3),
                attributes=attrs,
            )
        )

    logger.info(
        "Aggregated %d extractions -> %d raw entities, %d relationships, %d timeline events",
        len(extractions), len(entities), len(relationships), len(timeline),
    )
    return AggregateResult(
        entities=entities,
        mentions=mentions,
        relationships=relationships,
        timeline=timeline,
    )
