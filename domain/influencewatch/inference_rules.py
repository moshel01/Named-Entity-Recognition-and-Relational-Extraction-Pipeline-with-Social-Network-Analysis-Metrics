# Canonical inference for InfluenceWatch. None for v1 - the affiliation->actor
# projection (inference.enable_affiliation_projection) and co-occurrence already
# do the structural work generically. Add domain rules here if a corpus needs them.

from __future__ import annotations

from core.schema import Entity, Relationship


def infer_edges(entities: list[Entity], edges: list[Relationship],
                options: dict | None = None) -> list[Relationship]:
    return []
