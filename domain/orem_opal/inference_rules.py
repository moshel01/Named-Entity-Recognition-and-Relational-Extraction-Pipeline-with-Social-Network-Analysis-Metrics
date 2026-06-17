# Canonical inference for OREM/OPAL. None for v1. The affiliation->actor projection
# and co-occurrence cover the structural layer; event reification (the hyperedge
# roadmap item) is where domain rules would go once grants/events are reified nodes.

from __future__ import annotations

from core.schema import Entity, Relationship


def infer_edges(entities: list[Entity], edges: list[Relationship],
                options: dict | None = None) -> list[Relationship]:
    return []
