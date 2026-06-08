# Analytical tagging of entities and edges.

from __future__ import annotations

import logging
from collections import Counter

from core.schema import Entity, Relationship

from .aggregator import normalize_name

logger = logging.getLogger(__name__)

# Relation types that indicate ideological / affinity links rather than
# concrete transactional or organizational ties.
_IDEOLOGICAL_TYPES = {
    "supports", "support", "endorses", "endorsed", "opposes", "opposed",
    "allied_with", "sympathizes_with", "aligned_with", "promotes",
}


def _percentile_thresholds(values: list[float]) -> tuple[float, float]:
    """Return (p50, p85) thresholds for a list of values (0,0 if empty)."""
    if not values:
        return 0.0, 0.0
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return s[idx]

    return pct(0.50), pct(0.85)


class Tagger:
    """Attach scope / relevance / connection-quality tags."""

    def tag(
        self, entities: list[Entity], relationships: list[Relationship],
        reference_figures: set[str] | None = None,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Tag entities and edges in place and return them."""
        valid_ids = {e.entity_id for e in entities}

        # Degree per entity from the (resolved) relationship list.
        degree: Counter[str] = Counter()
        for r in relationships:
            if r.source in valid_ids and r.target in valid_ids:
                degree[r.source] += 1
                degree[r.target] += 1

        self._tag_entities(entities, degree)
        self._tag_reference_figures(entities, reference_figures or set())
        self._tag_edges(relationships)
        return entities, relationships

    # Public/historical figures: known list + cross-document recurrence.
    @staticmethod
    def _tag_reference_figures(entities: list[Entity], known: set[str]) -> None:
        persons = [e for e in entities if e.label == "PERSON"]
        doc_counts = sorted(len(e.doc_ids) for e in persons)
        # 95th percentile of person document spread, floored at 5 letters.
        cutoff = 5
        if doc_counts:
            cutoff = max(5, doc_counts[min(len(doc_counts) - 1, int(0.95 * (len(doc_counts) - 1)))])
        for e in persons:
            if e.attributes.get("is_author"):
                continue
            name = normalize_name(e.canonical_name)
            if name in known or len(e.doc_ids) >= cutoff:
                e.attributes["reference_figure"] = True
                e.tags["reference_figure"] = True

    # Entities
    def _tag_entities(self, entities: list[Entity], degree: Counter[str]) -> None:
        mentions = [float(e.mention_count) for e in entities]
        docs = [float(len(e.doc_ids)) for e in entities]
        degs = [float(degree.get(e.entity_id, 0)) for e in entities]

        m50, m85 = _percentile_thresholds(mentions)
        d50, d85 = _percentile_thresholds(degs)
        doc50, doc85 = _percentile_thresholds(docs)

        for e in entities:
            deg = float(degree.get(e.entity_id, 0))
            n_docs = float(len(e.doc_ids))

            # Scope: broad reach (many docs OR high degree) => macro.
            if deg >= d85 and d85 > 0 or n_docs >= doc85 and doc85 > 0:
                scope = "macro"
            else:
                scope = "specific"

            # Relevance: blended score across the three signals.
            score = 0.0
            score += 1.0 if e.mention_count >= m85 else (0.5 if e.mention_count >= m50 else 0.0)
            score += 1.0 if deg >= d85 else (0.5 if deg >= d50 else 0.0)
            score += 1.0 if n_docs >= doc85 else (0.5 if n_docs >= doc50 else 0.0)
            if score >= 2.0:
                tier = "core"
            elif score >= 1.0:
                tier = "secondary"
            else:
                tier = "peripheral"

            e.tags["entity_scope"] = scope
            e.tags["relevance_tier"] = tier
            e.tags["degree"] = int(deg)

    # Edges
    def _tag_edges(self, relationships: list[Relationship]) -> None:
        for r in relationships:
            if r.origin in ("inferred", "canonical"):
                quality = "structural"
            elif r.rel_type.lower() in _IDEOLOGICAL_TYPES:
                quality = "ideological"
            elif r.evidence:
                quality = "direct"
            else:
                quality = "structural"
            r.attributes["connection_quality"] = quality
