# Domain inference entry point consumed by the pipeline's InferenceEngine.

from __future__ import annotations

from core.schema import Entity, Relationship

from .canonical_inference import MembershipInferenceEngine
from .org_hierarchy import PARENT_OF, classify_unit

_engine = MembershipInferenceEngine()


def _structural_unit_edges(
    entities: list[Entity], existing: set[tuple[str, str]]
) -> list[Relationship]:
    """Link unit/subdivision ORG nodes to their parent organization node."""
    org_index = {e.canonical_name.lower(): e.entity_id
                 for e in entities if e.label in ("ORG", "INSTITUTION")}
    for e in entities:
        for a in e.aliases:
            org_index.setdefault(a.lower(), e.entity_id)

    edges: list[Relationship] = []
    for e in entities:
        if e.label != "ORG":
            continue
        info = classify_unit(e.canonical_name)
        parent_name = None
        if info is not None:
            parent_name = info["parent"]
        else:
            # Direct keyword roll-up (e.g. a bare "Gau" or "Ortsgruppe" node).
            low = e.canonical_name.lower()
            for term, parent in PARENT_OF.items():
                if term in low:
                    parent_name = parent
                    break
        if not parent_name:
            continue
        parent_id = org_index.get(parent_name.lower())
        if not parent_id or parent_id == e.entity_id:
            continue
        if (e.entity_id, parent_id) in existing:
            continue
        existing.add((e.entity_id, parent_id))
        edges.append(
            Relationship(
                source=e.entity_id,
                target=parent_id,
                rel_type="subordinate_to",
                doc_id=";".join(e.doc_ids),
                evidence=f"{e.canonical_name} is a subunit of {parent_name}",
                confidence=0.80,
                directed=True,
                origin="canonical",
                attributes={"edge_source": "canonical_inferred",
                            "structural": True},
            )
        )
    return edges


def infer_edges(
    entities: list[Entity],
    edges: list[Relationship],
    options: dict | None = None,
) -> list[Relationship]:
    """Return all domain-derived canonical edges for the resolved graph.

    ``options`` may carry ``mandatory_membership`` ("authors_only" | "all" |
    "off"); defaults to the principled "authors_only".
    """
    scope = (options or {}).get("mandatory_membership", "authors_only")
    membership = _engine.infer(entities, edges, mandatory_scope=scope)
    seen = {(r.source, r.target) for r in edges} | {(r.source, r.target) for r in membership}
    structural = _structural_unit_edges(entities, seen)
    return membership + structural
