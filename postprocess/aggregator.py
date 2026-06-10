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

# Common UTF-8-as-MacRoman mojibake of German umlauts (e.g. an RTF whose codepage
# was misread): "Th√ºrling" -> "Thürling", "Stallup√∂nen" -> "Stallupönen".
_MOJIBAKE = {
    "√º": "ü", "√∂": "ö", "√§": "ä", "√ú": "Ü", "√ñ": "Ö", "√Ñ": "Ä", "√ü": "ß",
    "√©": "é", "√®": "è", "√°": "á", "√¥": "ô", "√¢": "â",
}
# Zero-width / formatting characters that corrupt names: soft hyphen (hyphenation
# artifact "Kaisers­lautern"), zero-width space, BOM.
_ZW_CHARS = ("­", "​", "‌", "‍", "﻿")


def _repair_text(text: str) -> str:
    """Fix umlaut mojibake and strip zero-width/soft-hyphen artifacts."""
    for bad, good in _MOJIBAKE.items():
        if bad in text:
            text = text.replace(bad, good)
    for ch in _ZW_CHARS:
        if ch in text:
            text = text.replace(ch, "")
    return text


def clean_surface(text: str) -> str:
    """Collapse internal whitespace/newlines + repair encoding artifacts.

    Entity spans can straddle a line break ("Robert\\nChen") or carry soft hyphens
    and umlaut mojibake from the source RTF; this yields the clean display name.
    ``normalize_name`` applies the same repair so grouping stays consistent.
    """
    return _WS_RE.sub(" ", _repair_text(text)).strip()


def normalize_name(name: str) -> str:
    """Cheap normalization for exact grouping (NOT fuzzy)."""
    n = _repair_text(name).strip()
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
            buf["surface"][clean_surface(m.text)] += 1
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
                elif ak == "propn_ratio":
                    # Averaged over mentions, not first-wins: one parse can
                    # mis-tag, the corpus-level mean is the signal.
                    buf["propn_sum"] = buf.get("propn_sum", 0.0) + float(av)
                    buf["propn_n"] = buf.get("propn_n", 0) + 1
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
        if buf.get("propn_n"):
            attrs["propn_ratio"] = round(buf["propn_sum"] / buf["propn_n"], 3)
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
