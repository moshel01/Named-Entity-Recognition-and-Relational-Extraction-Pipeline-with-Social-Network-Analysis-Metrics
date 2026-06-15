# Evidence-based edge inference.

from __future__ import annotations

import itertools
import logging
from collections import defaultdict

from config import InferenceConfig
from core.schema import Entity, Relationship

from .aggregator import normalize_name

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Add inferred and canonical edges to the resolved graph."""

    def __init__(self, config: InferenceConfig, domain=None) -> None:
        self.config = config
        self.domain = domain

    def cooccurrence_edges(self, entities: list[Entity]) -> list[Relationship]:
        """Build co-occurrence edges from shared document membership."""
        if not self.config.enable_cooccurrence_edges:
            return []

        # Map doc_id -> set of entity_ids present in that document.
        doc_to_entities: dict[str, set[str]] = defaultdict(set)
        for e in entities:
            for d in e.doc_ids:
                doc_to_entities[d].add(e.entity_id)

        pair_docs: dict[frozenset[str], set[str]] = defaultdict(set)
        for doc_id, ent_ids in doc_to_entities.items():
            for a, b in itertools.combinations(sorted(ent_ids), 2):
                pair_docs[frozenset((a, b))].add(doc_id)

        edges: list[Relationship] = []
        for pair, docs in pair_docs.items():
            if len(docs) < self.config.cooccurrence_min_shared_docs:
                continue
            a, b = tuple(pair)
            edges.append(
                Relationship(
                    source=a,
                    target=b,
                    rel_type="co_occurs_with",
                    doc_id=";".join(sorted(docs)),
                    evidence=f"Co-occur in {len(docs)} documents",
                    confidence=min(1.0, 0.3 + 0.1 * len(docs)),
                    directed=False,
                    origin="inferred",
                    attributes={"shared_docs": len(docs), "edge_source": "rule_cooccurrence"},
                )
            )
        logger.info("Inferred %d co-occurrence edges", len(edges))
        return edges

    def proximity_edges(
        self, mentions: list, name_to_id: dict[str, str]
    ) -> list[Relationship]:
        """Window co-occurrence: link entities mentioned within
        `proximity_window_chars` of each other in a document.

        Mention positions are document-absolute (foundation offsets each chunk),
        so a windowed pair can straddle a chunk boundary the LLM never saw across
        - a partial floor under the cross-chunk recall ceiling. Far less noisy
        than whole-doc co-occurrence (which links every pair in a letter). Still
        the weakest evidence layer: co_occurs_with, full tier only.
        """
        if not self.config.enable_proximity_edges or not mentions or not name_to_id:
            return []
        window = self.config.proximity_window_chars
        if window <= 0:
            return []

        # doc_id -> sorted [(doc-absolute pos, surviving entity_id)].
        per_doc: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for m in mentions:
            eid = name_to_id.get(normalize_name(m.text))
            if eid is not None:
                per_doc[m.doc_id].append((m.start_char, eid))

        pair_stat: dict[frozenset[str], dict] = {}
        for doc_id, occ in per_doc.items():
            occ.sort()
            for i, (pos_i, eid_i) in enumerate(occ):
                for pos_j, eid_j in occ[i + 1:]:
                    if pos_j - pos_i > window:
                        break  # occ sorted: nothing further is in-window
                    if eid_i == eid_j:
                        continue
                    st = pair_stat.setdefault(
                        frozenset((eid_i, eid_j)),
                        {"docs": set(), "count": 0, "min_gap": window},
                    )
                    st["docs"].add(doc_id)
                    st["count"] += 1
                    st["min_gap"] = min(st["min_gap"], pos_j - pos_i)

        edges: list[Relationship] = []
        for pair, st in pair_stat.items():
            a, b = tuple(pair)
            conf = min(0.6, 0.3 + 0.05 * st["count"])  # closer/more often -> higher
            edges.append(
                Relationship(
                    source=a,
                    target=b,
                    rel_type="co_occurs_with",
                    doc_id=";".join(sorted(st["docs"])),
                    evidence=f"Within {window} chars, {st['count']}x in {len(st['docs'])} doc(s)",
                    confidence=round(conf, 3),
                    directed=False,
                    origin="inferred",
                    attributes={"edge_source": "rule_cooccurrence",
                                "cooccur_count": st["count"],
                                "min_gap_chars": st["min_gap"]},
                )
            )
        logger.info("Inferred %d proximity co-occurrence edges (window=%d)",
                    len(edges), window)
        return edges

    def canonical_edges(
        self, entities: list[Entity], edges: list[Relationship]
    ) -> list[Relationship]:
        """Delegate canonical inference to the active domain."""
        if not self.config.enable_canonical_inference or self.domain is None:
            return []
        options = {"mandatory_membership": self.config.mandatory_membership}
        try:
            extra = self.domain.infer_canonical_edges(entities, edges, options)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Domain canonical inference failed: %s", exc)
            return []
        logger.info("Domain produced %d canonical edges", len(extra))
        return extra

    def run(
        self, entities: list[Entity], edges: list[Relationship],
        mentions: list | None = None, name_to_id: dict[str, str] | None = None,
    ) -> list[Relationship]:
        """Return ``edges`` augmented with inferred + canonical edges.

        ``mentions`` + ``name_to_id`` (from aggregation + dedup) enable the
        within-document proximity layer; omit them to skip it.
        """
        augmented = list(edges)
        augmented.extend(self.cooccurrence_edges(entities))
        if mentions is not None and name_to_id is not None:
            augmented.extend(self.proximity_edges(mentions, name_to_id))
        augmented.extend(self.canonical_edges(entities, edges))
        return augmented
