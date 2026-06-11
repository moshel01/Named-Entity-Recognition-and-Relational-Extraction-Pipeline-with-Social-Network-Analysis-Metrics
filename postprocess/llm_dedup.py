# LLM-assisted dedup: merge same-entity nodes the rules missed.

from __future__ import annotations

import logging
from collections import Counter
from difflib import SequenceMatcher

from core.schema import Entity, Relationship

from .aggregator import normalize_name
from .deduplicator import (Deduplicator, _acronym_form, _content_tokens,
                           _function_word_name)

logger = logging.getLogger(__name__)

# Guards against a weak LLM proposing a catastrophic merge group (e.g. qwen
# dumping dozens of unrelated orgs/places/dates into "NSDAP"). A real entity has
# only a handful of surface variants the rules missed.
_MAX_ALIASES_PER_CANON = 8     # total LLM merges accepted into one node per run
_MAX_GROUP = 16                # a single suggestion bigger than this is a hallucination


def _is_numeric_alias(name: str) -> bool:
    """True for date-/number-like names that are never a real merge alias."""
    s = name.replace(".", "").replace(" ", "").replace("-", "").replace("/", "")
    if not s:
        return True
    digits = sum(c.isdigit() for c in s)
    return digits >= 3 or digits / len(s) > 0.4


def _plausible_alias(canon: str, alias: str) -> bool:
    """A light string check so the LLM can't merge unrelated entities (e.g.
    'Deutsches Reich' the state into 'NSDAP' the party). Accepts a shared content
    word, an acronym relationship, or strong fuzzy similarity. Legit but
    string-dissimilar variants (acronym<->full name) are left to the alias dict."""
    # A conjunction/preposition name is a bad NER span, not a merge canon
    # ("Bofur and Bombur" must not absorb "Bofur").
    if _function_word_name(canon) or _function_word_name(alias):
        return False
    ca, aa = _acronym_form(canon), _acronym_form(alias)
    if ca and aa:                      # two acronyms: only the same one
        return ca == aa
    ct, at = set(_content_tokens(canon)), set(_content_tokens(alias))
    if ct & at:                        # share a meaningful word
        return True
    if ca and ca.lower() in at:        # acronym appears in the other's expansion
        return True
    if aa and aa.lower() in ct:
        return True
    # 0.75: distinct short names score deceptively high on SequenceMatcher
    # (Beorn~Bear 0.67, Thror~Thorin 0.73 - different characters).
    return SequenceMatcher(None, normalize_name(canon),
                           normalize_name(alias)).ratio() >= 0.75


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
    merged_per_canon: Counter = Counter()

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
            aliases = grp.get("aliases", []) or []
            # A suggestion proposing a huge alias set is a hallucinated mega-merge
            # (e.g. "everything is NSDAP") - drop the whole group.
            if len(aliases) > _MAX_GROUP:
                logger.warning("LLM dedup: dropping oversized merge group for '%s' "
                               "(%d aliases).", canon.canonical_name, len(aliases))
                continue
            for alias in aliases:
                if merged_per_canon[canon.entity_id] >= _MAX_ALIASES_PER_CANON:
                    break
                if _is_numeric_alias(alias):
                    continue  # dates/numbers are never a real alias
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
                # Reject semantically-wrong merges of dissimilar names.
                if not _plausible_alias(canon.canonical_name, other.canonical_name):
                    continue
                Deduplicator._merge_into(canon, other)
                id_remap[other.entity_id] = canon.entity_id
                removed.add(other.entity_id)
                merged_per_canon[canon.entity_id] += 1

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
