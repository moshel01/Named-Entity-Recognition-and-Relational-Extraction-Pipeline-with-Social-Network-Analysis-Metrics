# Resolve first-person narrator/author placeholder nodes ("Narrator [doc]") into
# the real person the document identifies them as. Without metadata (the generic
# path), the author otherwise appears twice - once as the narrator (carrying
# first-person ties) and once as their named self (third-person mentions) - joined
# by a junk "is"/"self_reference" edge. We consume those identity edges to MERGE
# the two and then drop the edges. Runs before dedup; relationships still carry
# names (not ids) at this stage.

from __future__ import annotations

import logging

from core.schema import Entity, Relationship

from .aggregator import normalize_name

logger = logging.getLogger(__name__)

# Relation types that assert "A and B are the same person".
_IDENTITY_RELS = {
    "is", "am", "are", "was", "self_reference", "self-reference", "selfreference",
    "aka", "also_known_as", "named", "identified_as", "real_name", "same_as",
    "identity", "alias_of",
}


def _is_narrator(e: Entity) -> bool:
    a = e.attributes or {}
    return bool(a.get("narrator") or a.get("is_author")) or \
        normalize_name(e.canonical_name).startswith("narrator")


def link_known_authors(
    entities: list[Entity], relationships: list[Relationship],
    min_len: int = 4, max_merges: int | None = None,
) -> tuple[list[Entity], list[Relationship]]:
    """Fold a lone surname mention into the author it UNIQUELY names, across letters.

    Runs after dedup (endpoints are entity ids). The known authors are a closed name
    registry; a single-token PERSON whose surname matches exactly one author AND no
    other person in the corpus is the cross-letter reference generic fuzzy dedup can't
    catch (surname-vs-fullname ratio is too low to merge safely on its own). Strict
    zero-ambiguity: if any other person shares the surname, skip - that's the whole
    point of the guard. Full-name variants already merge in dedup, so this only acts
    on bare surnames."""
    authors = [e for e in entities if e.label == "PERSON" and (e.attributes or {}).get("is_author")]
    if not authors:
        return entities, relationships

    def surname(name: str) -> str:
        toks = normalize_name(name).split()
        return toks[-1] if toks else ""

    # Surname -> the authors carrying it; and the full set of persons per surname, so
    # a surname shared by more than one person (author or not) is never auto-linked.
    auth_by_surname: dict[str, list[Entity]] = {}
    persons_by_surname: dict[str, int] = {}
    for e in entities:
        if e.label != "PERSON":
            continue
        sn = surname(e.canonical_name)
        if not sn:
            continue
        persons_by_surname[sn] = persons_by_surname.get(sn, 0) + 1
        if (e.attributes or {}).get("is_author"):
            auth_by_surname.setdefault(sn, []).append(e)

    remap: dict[str, str] = {}  # mention entity_id -> author entity_id
    drop_ids: set[str] = set()
    merges = 0
    for e in entities:
        if e.label != "PERSON" or (e.attributes or {}).get("is_author"):
            continue
        norm = normalize_name(e.canonical_name)
        toks = norm.split()
        if len(toks) != 1 or len(norm) < min_len:   # bare surname only
            continue
        sn = toks[0]
        cands = auth_by_surname.get(sn, [])
        # Unique author AND this surname belongs to no one else (the mention itself
        # counts once, so a unique author surname yields exactly 2: author + mention).
        if len(cands) != 1 or persons_by_surname.get(sn, 0) > 2:
            continue
        author = cands[0]
        if author.entity_id == e.entity_id:
            continue
        if max_merges is not None and merges >= max_merges:
            break
        if e.canonical_name not in author.aliases:
            author.aliases.append(e.canonical_name)
        author.mention_count += e.mention_count
        author.doc_ids = sorted(set(author.doc_ids) | set(e.doc_ids))
        remap[e.entity_id] = author.entity_id
        drop_ids.add(e.entity_id)
        merges += 1

    if not remap:
        return entities, relationships
    new_entities = [e for e in entities if e.entity_id not in drop_ids]
    new_rels: list[Relationship] = []
    for r in relationships:
        r.source = remap.get(r.source, r.source)
        r.target = remap.get(r.target, r.target)
        if r.source == r.target:
            continue
        new_rels.append(r)
    logger.info("Author anchoring: linked %d surname mention(s) to their author.", merges)
    return new_entities, new_rels


def resolve_narrator_identities(
    entities: list[Entity], relationships: list[Relationship]
) -> tuple[list[Entity], list[Relationship]]:
    """Fold narrator placeholders into their named person; drop identity edges."""
    by_norm: dict[str, Entity] = {}
    for e in entities:
        by_norm.setdefault(normalize_name(e.canonical_name), e)

    drop_ids: set[int] = set()
    rename: dict[str, str] = {}  # normalized narrator name -> real canonical name
    seen: set[tuple[int, int]] = set()

    for r in relationships:
        if (r.rel_type or "").strip().lower() not in _IDENTITY_RELS:
            continue
        s = by_norm.get(normalize_name(r.source))
        t = by_norm.get(normalize_name(r.target))
        if not s or not t or s is t:
            continue
        s_n, t_n = _is_narrator(s), _is_narrator(t)
        # Exactly one side is a narrator placeholder and the other a real PERSON.
        if s_n and not t_n and t.label == "PERSON":
            narr, real = s, t
        elif t_n and not s_n and s.label == "PERSON":
            narr, real = t, s
        else:
            continue  # both narrators (hallucinated link) or non-person -> skip
        key = (id(narr), id(real))
        if key in seen or id(narr) in drop_ids:
            continue
        seen.add(key)
        # Fold the narrator into the named person; the real name stays canonical.
        if narr.canonical_name not in real.aliases:
            real.aliases.append(narr.canonical_name)
        real.mention_count += narr.mention_count
        real.doc_ids = sorted(set(real.doc_ids) | set(narr.doc_ids))
        real.confidence = max(real.confidence, narr.confidence)
        for k, v in (narr.attributes or {}).items():
            real.attributes.setdefault(k, v)
        if narr.attributes.get("is_author"):
            real.attributes["is_author"] = True
        if narr.attributes.get("narrator"):
            real.attributes["narrator"] = True
        rename[normalize_name(narr.canonical_name)] = real.canonical_name
        drop_ids.add(id(narr))

    new_entities = [e for e in entities if id(e) not in drop_ids]

    # Remap endpoints onto the real name and drop the identity edges themselves.
    new_rels: list[Relationship] = []
    for r in relationships:
        if (r.rel_type or "").strip().lower() in _IDENTITY_RELS:
            continue
        ns, nt = normalize_name(r.source), normalize_name(r.target)
        if ns in rename:
            r.source = rename[ns]
        if nt in rename:
            r.target = rename[nt]
        if normalize_name(r.source) == normalize_name(r.target):
            continue  # self-loop after merge
        new_rels.append(r)

    if drop_ids:
        logger.info("Narrator identity: merged %d placeholder author node(s) into "
                    "their named person.", len(drop_ids))
    return new_entities, new_rels
