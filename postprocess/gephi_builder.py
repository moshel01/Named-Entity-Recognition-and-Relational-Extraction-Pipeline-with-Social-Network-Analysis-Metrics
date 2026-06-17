# Aggregate edges, classify ties, emit node/edge/timeline rows.
#
# We deliberately do NOT precompute centrality/community here - Gephi computes
# those in one click, and on the per-view graph you actually load. We only emit
# what Gephi can't derive: tie classes, corroboration counts, provenance, and
# semantic tags/attributes.

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from core.schema import Entity, Relationship, TimelineEvent, stable_id

from . import tie_classes
from .aggregator import _repair_text

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\b(1[89]\d{2})\b")
_TIE_CLASSES = ("interaction", "affiliation", "participation",
                "biographical", "stance", "cooccurrence", "other")


@dataclass
class GraphTables:
    """Final tabular representation ready for export."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)


def _aggregate_edges(
    relationships: list[Relationship],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Collapse parallel relationships into weighted edges keyed by (s,t,type).

    Undirected edges are normalized so (a,b) and (b,a) collapse together. We track
    distinct documents (corroboration) separately from the raw mention count.
    """
    agg: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in relationships:
        if r.directed:
            s, t = r.source, r.target
        else:
            s, t = tuple(sorted((r.source, r.target)))
        key = (s, t, r.rel_type)
        bucket = agg.get(key)
        if bucket is None:
            bucket = {
                "source": s, "target": t, "rel_type": r.rel_type,
                "n_mentions": 0, "directed": r.directed, "doc_ids": set(),
                "origins": set(), "edge_sources": set(), "confidence": 0.0,
                "evidence": r.evidence, "year": None, "suspect_membership": False,
                "evidence_unverified": False, "type_violation": False,
                "evidence_ungrounded": False, "cooccur_strength": None,
                "disparity_alpha": None, "affiliation_strength": None,
                "shared_groups": None,
            }
            agg[key] = bucket
        if r.attributes.get("suspect_membership"):
            bucket["suspect_membership"] = True
        if r.attributes.get("evidence_unverified"):
            bucket["evidence_unverified"] = True
        # Faithfulness flags set upstream (ontology type gate, anchor check). OR
        # across mentions: one bad supporting mention taints the aggregated edge.
        if r.attributes.get("type_violation"):
            bucket["type_violation"] = True
        if r.attributes.get("evidence_ungrounded"):
            bucket["evidence_ungrounded"] = True
        # Per-edge qualifiers (qual_*): domain-declared optional fields the LLM
        # filled (monetary_value, jurisdiction, ...). Generic passthrough - first
        # non-empty value per qualifier survives the merge.
        for k, v in r.attributes.items():
            if k.startswith("qual_") and v not in (None, "") and not bucket.get(k):
                bucket[k] = v
        # Co-occurrence backbone signals (None for typed edges). Keep the strongest
        # tie strength and the most-significant (smallest) disparity alpha.
        cs = r.attributes.get("cooccur_strength")
        if cs is not None:
            bucket["cooccur_strength"] = max(bucket["cooccur_strength"] or 0.0, cs)
        da = r.attributes.get("disparity_alpha")
        if da is not None:
            prev = bucket["disparity_alpha"]
            bucket["disparity_alpha"] = da if prev is None else min(prev, da)
        # Two-mode projection weight (co_affiliated edges). Keep the strongest.
        af = r.attributes.get("affiliation_strength")
        if af is not None:
            bucket["affiliation_strength"] = max(bucket["affiliation_strength"] or 0.0, af)
        sg = r.attributes.get("shared_groups")
        if sg is not None:
            bucket["shared_groups"] = max(bucket["shared_groups"] or 0, sg)
        bucket["n_mentions"] += 1
        for d in (r.doc_id or "").split(";"):
            if d:
                bucket["doc_ids"].add(d)
        bucket["origins"].add(r.origin)
        # Fine-grained edge_source for evidentiary sensitivity analysis;
        # default from origin when a creator did not stamp it.
        bucket["edge_sources"].add(
            r.attributes.get("edge_source") or f"{r.origin}_unspecified"
        )
        bucket["confidence"] = max(bucket["confidence"], r.confidence)
        if not bucket["evidence"] and r.evidence:
            bucket["evidence"] = r.evidence
        if bucket["year"] is None:
            m = _YEAR_RE.search(r.evidence or "")
            if m:
                bucket["year"] = int(m.group(1))
    return agg


class GephiBuilder:
    """Construct node/edge tables with tie classes and provenance."""

    def build(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        timeline: list[TimelineEvent],
        entity_id_to_name: dict[str, str] | None = None,
        manifest: dict[str, dict[str, str]] | None = None,
        period_fn: Callable[[int], str] | None = None,
    ) -> GraphTables:
        """Build node/edge/timeline tables."""
        id_to_name = entity_id_to_name or {e.entity_id: e.canonical_name for e in entities}
        id_to_label = {e.entity_id: e.label for e in entities}
        valid_ids = {e.entity_id for e in entities}
        manifest = manifest or {}

        def letter_of(doc_id: str) -> str:
            return manifest.get((doc_id or "").split(";")[0], {}).get("letter_id", "")

        agg = _aggregate_edges(
            [r for r in relationships if r.source in valid_ids and r.target in valid_ids]
        )

        # Classify ties + derive weight (distinct docs) and corroboration.
        directed_keys: set[tuple[str, str]] = set()
        for (s, t, rt), b in agg.items():
            b["tie_class"] = tie_classes.classify(rt, id_to_label.get(s, ""),
                                                  id_to_label.get(t, ""))
            b["connection_type"] = tie_classes.connection_type(
                rt, id_to_label.get(t, ""))
            b["weight"] = max(1, len(b["doc_ids"]))
            b["n_sources"] = len({letter_of(d) for d in b["doc_ids"] if letter_of(d)})
            b["period"] = period_fn(b["year"]) if (period_fn and b["year"]) else ""
            if b["directed"]:
                directed_keys.add((s, t))
        for (s, t, rt), b in agg.items():
            b["reciprocal"] = b["directed"] and (t, s) in directed_keys

        # Per-node degree split by tie class + datable year span (for temporal
        # views / dynamic graphs in Gephi).
        class_deg: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(_TIE_CLASSES, 0))
        node_years: dict[str, list[int]] = defaultdict(list)
        for (s, t, _rt), b in agg.items():
            class_deg[s][b["tie_class"]] += 1
            class_deg[t][b["tie_class"]] += 1
            if b["year"]:
                node_years[s].append(b["year"])
                node_years[t].append(b["year"])

        nodes = self._node_rows(entities, class_deg, node_years)
        edges = self._edge_rows(agg, id_to_name, letter_of)
        timeline_rows = self._timeline_rows(timeline, letter_of)

        n_int = sum(1 for b in agg.values() if b["tie_class"] == "interaction")
        logger.info("Built graph: %d nodes, %d edges (%d interaction), %d timeline rows",
                    len(nodes), len(edges), n_int, len(timeline_rows))
        return GraphTables(nodes=nodes, edges=edges, timeline=timeline_rows)

    # Row builders
    @staticmethod
    def _node_rows(entities, class_deg, node_years) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        for e in entities:
            cd = class_deg.get(e.entity_id, {})
            yrs = node_years.get(e.entity_id, [])
            nodes.append({
                "Id": e.entity_id,
                "Label": _repair_text(e.canonical_name),
                "type": e.label,
                "mention_count": e.mention_count,
                "doc_count": len(e.doc_ids),
                "aliases": "; ".join(_repair_text(a) for a in e.aliases),
                "first_year": min(yrs) if yrs else None,
                "last_year": max(yrs) if yrs else None,
                # Degree split by tie class (Gephi computes plain centrality itself).
                **{f"deg_{c}": cd.get(c, 0) for c in _TIE_CLASSES},
                "confidence": e.confidence,
                **{f"tag_{k}": v for k, v in e.tags.items()},
                # Surface primitive attributes (enrichment rank/office, metadata,
                # is_author/reference_figure, ...) as Gephi columns.
                **{f"attr_{k}": v for k, v in e.attributes.items()
                   if isinstance(v, (str, int, float, bool))},
            })
        return nodes

    @staticmethod
    def _edge_rows(agg, id_to_name, letter_of) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        for (s, t, rt), b in agg.items():
            doc0 = next(iter(b["doc_ids"]), "")
            edges.append({
                "Id": stable_id(s, t, rt, prefix="edge_", length=12),
                "Source": s,
                "Target": t,
                "Type": "Directed" if b["directed"] else "Undirected",
                "Label": rt,
                "rel_type": rt,
                "tie_class": b["tie_class"],
                "connection_type": b["connection_type"],
                "polarity": tie_classes.polarity(rt),
                "Weight": b["weight"],                 # distinct documents (corroboration)
                "n_mentions": b["n_mentions"],         # raw supporting mentions
                "n_sources": b["n_sources"],           # distinct letters
                "cooccur_strength": b.get("cooccur_strength"),  # Newman projection weight
                "disparity_alpha": b.get("disparity_alpha"),    # backbone significance
                "affiliation_strength": b.get("affiliation_strength"),  # two-mode projection weight
                "shared_groups": b.get("shared_groups"),        # n shared orgs/events
                "reciprocal": b["reciprocal"],
                "suspect_membership": b.get("suspect_membership", False),
                "evidence_unverified": b.get("evidence_unverified", False),
                "type_violation": b.get("type_violation", False),
                "evidence_ungrounded": b.get("evidence_ungrounded", False),
                "period": b["period"],
                "year": b["year"],
                "origin": ";".join(sorted(b["origins"])),
                "edge_source": ";".join(sorted(b["edge_sources"])),
                "confidence": round(b["confidence"], 3),
                "source_name": _repair_text(id_to_name.get(s, s)),
                "target_name": _repair_text(id_to_name.get(t, t)),
                "letter_id": letter_of(doc0),
                "evidence": (b["evidence"] or "")[:500],
                # Per-edge qualifiers (qual_*) ride through generically as columns.
                **{k: v for k, v in b.items() if k.startswith("qual_")},
            })
        return edges

    @staticmethod
    def _timeline_rows(timeline, letter_of) -> list[dict[str, Any]]:
        rows = [
            {
                "doc_id": t.doc_id,
                "letter_id": letter_of(t.doc_id),
                "date_text": t.date_text,
                "iso_date": t.iso_date or "",
                "year": t.year,                      # int or None (kept nullable)
                "description": t.description[:500],
                "entities": "; ".join(t.entities),
                "confidence": round(t.confidence, 3),
            }
            for t in timeline
        ]
        rows.sort(key=lambda r: (r["iso_date"] or "9999", r["doc_id"]))
        return rows
