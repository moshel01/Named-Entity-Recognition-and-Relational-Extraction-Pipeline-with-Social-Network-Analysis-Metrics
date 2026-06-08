# Canonical inference rules for the generic domain.

from __future__ import annotations

from core.schema import Entity, Relationship


def infer_edges(entities: list[Entity], edges: list[Relationship]) -> list[Relationship]:
    """Return canonical (domain-derived) edges. Generic domain: none."""
    return []
