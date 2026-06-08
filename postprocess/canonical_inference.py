# Evidence-based edge inference.

from __future__ import annotations

import itertools
import logging
from collections import defaultdict

from config import InferenceConfig
from core.schema import Entity, Relationship

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
                    attributes={"shared_docs": len(docs), "edge_source": "sna_inferred"},
                )
            )
        logger.info("Inferred %d co-occurrence edges", len(edges))
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
        self, entities: list[Entity], edges: list[Relationship]
    ) -> list[Relationship]:
        """Return ``edges`` augmented with inferred + canonical edges."""
        augmented = list(edges)
        augmented.extend(self.cooccurrence_edges(entities))
        augmented.extend(self.canonical_edges(entities, edges))
        return augmented
