# LLM-assisted dedup: merge same-entity nodes the rules missed.

from __future__ import annotations

import logging

from core.schema import Entity, Relationship

from .aggregator import normalize_name
from .deduplicator import Deduplicator, _acronym_form

logger = logging.getLogger(__name__)


def apply_llm_merges(
    entities: list[Entity],
    relationships: list[Relationship],
    backend,
    max_per_type: int = 300,
) -> tuple[list[Entity], list[Relationship]]:
    if backend is None or not entities:
        return entities, relationships

    by_type: dict[str, list[Entity]] = {}
    for e in entities:
        by_type.setdefault(e.label, []).append(e)

    id_remap: dict[str, str] = {}
    removed: set[str] = set()

    for label, group in by_type.items():
        if len(group) < 2:
            continue
        ranked = sorted(group, key=lambda e: e.mention_count, reverse=True)[:max_per_type]
        name_to_ent = {normalize_name(e.canonical_name): e for e in ranked}
        try:
            suggestions = backend.suggest_merges(label, [e.canonical_name for e in ranked])
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM merge (%s) failed: %s", label, exc)
            continue

        for grp in suggestions:
            canon = name_to_ent.get(normalize_name(grp.get("canonical", "")))
            if canon is None or canon.entity_id in removed:
                continue
            for alias in grp.get("aliases", []):
                other = name_to_ent.get(normalize_name(alias))
                if other is None or other is canon or other.entity_id in removed:
                    continue
                # Never merge an author node away (it carries the metadata join).
                if other.attributes.get("is_author"):
                    continue
                # Distinct acronyms are distinct orgs (NSV vs NSDAP) even if the
                # LLM groups them - veto here as the rules do.
                ca, oa = _acronym_form(canon.canonical_name), _acronym_form(other.canonical_name)
                if ca and oa and ca != oa:
                    continue
                Deduplicator._merge_into(canon, other)
                id_remap[other.entity_id] = canon.entity_id
                removed.add(other.entity_id)

    if not removed:
        return entities, relationships

    kept = [e for e in entities if e.entity_id not in removed]
    out_rels: list[Relationship] = []
    for r in relationships:
        s = id_remap.get(r.source, r.source)
        t = id_remap.get(r.target, r.target)
        if s == t:
            continue
        r.source, r.target = s, t
        out_rels.append(r)

    logger.info("LLM dedup: merged %d extra nodes -> %d entities.", len(removed), len(kept))
    return kept, out_rels
