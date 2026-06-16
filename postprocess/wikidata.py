# Optional entity linking to Wikidata. Adds wikidata_qid / wikidata_url /
# wikidata_label to high-signal entities. Opt-in, bounded, and fail-soft: any
# network error just leaves the entity unlinked - it never breaks a run.

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from core.schema import Entity

logger = logging.getLogger(__name__)

_API = "https://www.wikidata.org/w/api.php"
_UA = "SNA-Extraction/1.0 (research entity linking)"
# Reject a hit when its Wikidata description clearly belongs to a different macro
# type (a person named "Florence" should not link to the city). Negative check:
# Wikidata descriptions give occupations, not "person", so a positive "human"
# requirement would wrongly drop most people.
_PLACE_WORDS = ("city", "town", "village", "country", "municipality", "capital",
                "river", "mountain", "region", "province", "district", "commune")
_ORG_WORDS = ("party", "organization", "company", "band", "club", "association")
_PERSON_WORDS = ("politician", "writer", "painter", "actor", "general", "officer",
                 "philosopher", "composer", "poet", "physician", "soldier",
                 "born", "–19", "–18", "human")
_TYPE_REJECT = {
    "PERSON": _PLACE_WORDS + ("film", "album", "song", "genus", "species"),
    "ORG": _PLACE_WORDS + _PERSON_WORDS,
    "LOCATION": _ORG_WORDS + _PERSON_WORDS,
}


def _search(name: str, lang: str, timeout: int) -> dict | None:
    """Return the best Wikidata hit for a name, or None."""
    params = urllib.parse.urlencode({
        "action": "wbsearchentities", "search": name, "language": lang,
        "uselang": lang, "format": "json", "limit": 5, "type": "item",
    })
    req = urllib.request.Request(f"{_API}?{params}", headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    hits = data.get("search") or []
    return hits[0] if hits else None


def link_entities(entities: list[Entity], config) -> list[Entity]:
    """Attach Wikidata ids to the highest-signal entities. Best-effort."""
    if not getattr(config, "enabled", False):
        return entities
    types = set(config.types)
    candidates = [e for e in entities
                  if e.label in types and e.mention_count >= config.min_mentions]
    candidates.sort(key=lambda e: e.mention_count, reverse=True)
    candidates = candidates[:config.max_entities]

    cache: dict[str, dict | None] = {}
    n_linked = n_fail = 0
    for e in candidates:
        key = e.canonical_name.lower()
        try:
            if key not in cache:
                cache[key] = _search(e.canonical_name, config.lang, config.request_timeout)
            hit = cache[key]
        except Exception:  # noqa: BLE001 - network/parse error: skip this one
            n_fail += 1
            if n_fail >= 10 and n_linked == 0:
                logger.warning("Wikidata linking: repeated failures, stopping early.")
                break
            continue
        if not hit:
            continue
        # Reject hits whose description belongs to a different macro type.
        desc = (hit.get("description") or "").lower()
        if desc and any(w in desc for w in _TYPE_REJECT.get(e.label, ())):
            continue
        e.attributes["wikidata_qid"] = hit.get("id", "")
        e.attributes["wikidata_url"] = hit.get("concepturi", "")
        e.attributes["wikidata_label"] = hit.get("label", "")
        n_linked += 1
    logger.info("Wikidata linking: %d linked, %d lookups, %d failures.",
                n_linked, len(candidates), n_fail)
    return entities


def consolidate_by_qid(entities, relationships, name_to_id=None):
    """Merge entities that resolved to the same Wikidata QID into one node.

    A high-precision cross-document identity signal that string dedup can miss:
    "Goebbels", "Joseph Goebbels", "Dr. Goebbels" all link to Q2622 and are one
    person. Folds the smaller mention nodes into the most-mentioned, remaps the
    relationship endpoints (entity ids at this stage), and refreshes name_to_id.
    Author/narrator nodes are per-document and never folded. Returns
    (entities, relationships, name_to_id)."""
    from collections import defaultdict

    from core.schema import stable_id

    from .aggregator import normalize_name
    from .deduplicator import Deduplicator

    groups: dict[str, list] = defaultdict(list)
    for e in entities:
        a = e.attributes or {}
        qid = a.get("wikidata_qid")
        if qid and not a.get("is_author") and not a.get("narrator"):
            groups[qid].append(e)

    id_remap: dict[str, str] = {}
    drop: set[int] = set()
    merged = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda e: e.mention_count, reverse=True)
        primary = group[0]
        for other in group[1:]:
            Deduplicator._merge_into(primary, other)
            drop.add(id(other))
        new_id = stable_id(normalize_name(primary.canonical_name), primary.label,
                           prefix="ent_", length=12)
        for e in group:
            id_remap[e.entity_id] = new_id
        primary.entity_id = new_id
        merged += 1

    if not drop:
        return entities, relationships, name_to_id

    new_entities = [e for e in entities if id(e) not in drop]
    new_rels = []
    for r in relationships:
        r.source = id_remap.get(r.source, r.source)
        r.target = id_remap.get(r.target, r.target)
        if r.source == r.target:        # self-loop after merge
            continue
        new_rels.append(r)
    if name_to_id is not None:
        for e in new_entities:
            for nm in {e.canonical_name, *e.aliases}:
                name_to_id[normalize_name(nm)] = e.entity_id

    logger.info("Wikidata QID consolidation: merged %d group(s), %d -> %d entities.",
                merged, len(entities), len(new_entities))
    return new_entities, new_rels, name_to_id
