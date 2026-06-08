# 3-layer dedup: aliases -> exact -> bucketed fuzzy, with block rules.

from __future__ import annotations

import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher

from config import DedupConfig
from core.schema import Entity, Relationship, stable_id

from .aggregator import normalize_name

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"(?:18|19|20)\d{2}")
_TOKEN_RE = re.compile(r"\w+")


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _tokens(name: str) -> list[str]:
    return _TOKEN_RE.findall(name.lower())


# Generic connector/legal tokens that don't distinguish two organizations.
_GENERIC_ORG_TOKENS = {
    "the", "of", "a", "an", "and", "de", "der", "die", "das", "la", "le", "el",
    "du", "von", "van", "den", "inc", "ltd", "llc", "gmbh", "ag", "co", "corp",
    "plc", "sa", "se",
}


def _content_tokens(name: str) -> list[str]:
    return [t for t in _tokens(name) if t not in _GENERIC_ORG_TOKENS and len(t) >= 2]


def _acronym_form(name: str) -> str:
    # "N.S.D.A.P." / "NSDAP" -> "NSDAP"; non-acronyms -> "". Used to keep distinct
    # acronyms (DVP vs DNVP) from fuzzy-merging while still merging NSDAP/NSDAP.
    s = name.replace(".", "").replace(" ", "").strip()
    return s.upper() if s.isalpha() and s.isupper() and 2 <= len(s) <= 6 else ""


def _distinctive_conflict(a: str, b: str, token_thr: float = 0.8) -> bool:
    """True if each name carries a distinctive (content) token the other lacks.

    Prevents over-merging templated names that share a long common skeleton but
    name different entities - e.g. "University of Basel" vs "University of Bonn",
    "South Africa" vs "South America", "Social Democratic Party (SPD)" vs "(PSD)".
    Tokens are matched fuzzily so spelling variants ("democrats"~"democratic")
    don't count as distinctive.
    """
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return False

    def unmatched(src: list[str], dst: list[str]) -> bool:
        return any(
            not any(SequenceMatcher(None, t, u).ratio() >= token_thr for u in dst)
            for t in src
        )

    return unmatched(ta, tb) and unmatched(tb, ta)


def _years(name: str) -> set[str]:
    return set(_YEAR_RE.findall(name))


class Deduplicator:
    """Resolve raw entities into canonical entities with alias collapsing."""

    def __init__(self, config: DedupConfig, domain_aliases: dict[str, str] | None = None) -> None:
        self.config = config
        self.thresholds = config.fuzzy_thresholds
        # Key aliases by the same normalization used at lookup time so keys
        # containing punctuation/parentheses still match (e.g. "SS-Nr.").
        self.aliases = {normalize_name(k): v for k, v in (domain_aliases or {}).items()}

    # Blocking rules
    def _blocked(self, a: Entity, b: Entity) -> bool:
        """Return True if ``a`` and ``b`` must NOT be merged."""
        label = a.label

        # Narrator/author nodes represent distinct document authors. They are
        # near-identical by name ("Narrator [doc_01]" vs "Narrator [doc_02]") and
        # would otherwise fuzzy-merge into a single person. Exact-name merges are
        # already handled by the exact layer; block all fuzzy merges here.
        if a.attributes.get("narrator") or b.attributes.get("narrator") or \
           a.attributes.get("is_author") or b.attributes.get("is_author"):
            return True

        # Distinct acronyms must not fuzzy-merge (DVP vs DNVP, SPD vs SED).
        aa, bb = _acronym_form(a.canonical_name), _acronym_form(b.canonical_name)
        if aa and bb and aa != bb:
            return True

        if self.config.block_year_mismatch_events and label == "EVENT":
            ya, yb = _years(a.canonical_name), _years(b.canonical_name)
            if ya and yb and ya != yb:
                return True

        if self.config.block_location_substring and label == "LOCATION":
            na, nb = normalize_name(a.canonical_name), normalize_name(b.canonical_name)
            if na != nb and (na in nb or nb in na):
                # One is a more-specific place than the other.
                return True

        # Over-merge guard for orgs/locations: distinct templated names that
        # differ on a distinctive token must not collapse (different universities,
        # parties, ministries, "South Africa" vs "South America", ...).
        if label in ("ORG", "LOCATION", "INSTITUTION"):
            if _distinctive_conflict(a.canonical_name, b.canonical_name):
                return True

        if self.config.block_family_merges and label == "PERSON":
            ta, tb = _tokens(a.canonical_name), _tokens(b.canonical_name)
            if len(ta) >= 2 and len(tb) >= 2:
                # Shared surname (last token) but different given name -> kin.
                # BUT allow first-name spelling/transliteration variants of the
                # same person (e.g. "Angela"/"Angel", "Mahmoud"/"Mahmud") - common
                # in OCR'd / translated historical text - by not blocking when the
                # given names are fuzzily similar.
                if ta[-1] == tb[-1] and ta[0] != tb[0] and _ratio(ta[0], tb[0]) < 0.8:
                    return True
        return False

    # Layer 1: aliases
    def _apply_aliases(self, entities: list[Entity]) -> list[Entity]:
        if not self.aliases:
            return entities
        for e in entities:
            n = normalize_name(e.canonical_name)
            canon = self.aliases.get(n)
            # German genitive: "Hitlers"/"Führers"/"Deutschlands" -> known alias.
            if canon is None and len(n) > 4 and n.endswith("s"):
                canon = self.aliases.get(n[:-1])
            if canon and canon != e.canonical_name:
                if e.canonical_name not in e.aliases:
                    e.aliases.append(e.canonical_name)
                e.canonical_name = canon
        return entities

    # Merge helper
    @staticmethod
    def _merge_into(primary: Entity, other: Entity, keep_primary_name: bool = False) -> None:
        names = {primary.canonical_name, *primary.aliases,
                 other.canonical_name, *other.aliases}
        # Keep the higher-mention name as canonical, unless the caller pins it
        # (e.g. folding a bare first name into a full name keeps the full name).
        if not keep_primary_name and other.mention_count > primary.mention_count:
            primary.canonical_name = other.canonical_name
        names.discard(primary.canonical_name)
        primary.aliases = sorted(names)
        primary.mention_count += other.mention_count
        primary.doc_ids = sorted(set(primary.doc_ids) | set(other.doc_ids))
        primary.confidence = max(primary.confidence, other.confidence)
        author_flag = primary.attributes.get("is_author") or other.attributes.get("is_author")
        primary.attributes.update(other.attributes)
        if author_flag:
            primary.attributes["is_author"] = True

    # Prefer one type per name when GLiNER/spaCy disagree (GPE/ORG confusion).
    # Tie-break order favors concrete types over ORG.
    _TYPE_PREF = {"PERSON": 4, "LOCATION": 3, "EVENT": 2, "INSTITUTION": 1, "ORG": 0}

    def _resolve_cross_type(self, entities: list[Entity]) -> list[Entity]:
        by_name: dict[str, list[Entity]] = defaultdict(list)
        for e in entities:
            by_name[normalize_name(e.canonical_name)].append(e)
        out: list[Entity] = []
        for group in by_name.values():
            if len(group) == 1:
                out.append(group[0])
                continue
            # Author/narrator nodes are never folded into another type.
            if any(e.attributes.get("is_author") for e in group):
                out.extend(group)
                continue
            winner = max(group, key=lambda e: (e.mention_count,
                                               self._TYPE_PREF.get(e.label, -1)))
            for e in group:
                if e is not winner:
                    self._merge_into(winner, e)
            out.append(winner)
        return out

    # Fold bare first/last names into a unique full name.
    def _fold_partial_persons(self, entities: list[Entity]) -> list[Entity]:
        """Merge a single-token PERSON into the one full name it belongs to.

        Surname-initial bucketing keeps "Eleanor" and "Eleanor Vance" in different
        buckets, so they never fuzzy-merge. Here we fold each bare single-token
        person into a multi-token person whose FIRST or LAST token matches it -
        but ONLY when that target is unambiguous (exactly one candidate). An
        ambiguous bare name ("Robert" with both "Robert Chen" and "Robert Downey")
        is left as its own node. Author/narrator nodes are never folded.
        """
        persons = [e for e in entities if e.label == "PERSON"]
        if len(persons) < 2:
            return entities
        multi = [e for e in persons
                 if len(normalize_name(e.canonical_name).split()) >= 2]
        singles = [e for e in persons
                   if len(normalize_name(e.canonical_name).split()) == 1]
        if not multi or not singles:
            return entities

        token_index: dict[str, list[Entity]] = defaultdict(list)
        for e in multi:
            toks = normalize_name(e.canonical_name).split()
            for t in {toks[0], toks[-1]}:
                token_index[t].append(e)

        alive = {id(e) for e in multi}
        folded: set[int] = set()
        for s in singles:
            if s.attributes.get("is_author") or s.attributes.get("narrator"):
                continue
            cands = [c for c in token_index.get(normalize_name(s.canonical_name), [])
                     if id(c) in alive and not self._blocked(c, s)]
            if len(cands) == 1:
                self._merge_into(cands[0], s, keep_primary_name=True)
                folded.add(id(s))

        if not folded:
            return entities
        return [e for e in entities if id(e) not in folded]

    # Main resolve
    def resolve(
        self, entities: list[Entity], relationships: list[Relationship]
    ) -> tuple[list[Entity], list[Relationship], dict[str, str]]:
        """Deduplicate entities and remap relationship endpoints."""
        entities = self._apply_aliases(list(entities))

        # Layer 2: exact normalized name + label. Author nodes also key on their
        # home document: two letters by different people who share a first-name-only
        # filename ("Emil") must stay distinct so each keeps its own letter_id.
        exact: dict[tuple, Entity] = {}
        for e in entities:
            if e.attributes.get("is_author"):
                key = (normalize_name(e.canonical_name), e.label,
                       e.attributes.get("author_doc") or e.entity_id)
            else:
                key = (normalize_name(e.canonical_name), e.label)
            if key in exact:
                self._merge_into(exact[key], e)
            else:
                exact[key] = e
        merged = list(exact.values())

        # Layer 3: bucketed fuzzy within (label, initial). Persons bucket on the
        # surname (last token) so "Joseph Goebbels"/"Dr. Goebbels"/"Goebbels" and
        # K/C spelling variants ("Karl"/"Carl Liebknecht") land together.
        buckets: dict[tuple[str, str], list[Entity]] = defaultdict(list)
        for e in merged:
            norm = normalize_name(e.canonical_name)
            if e.label == "PERSON":
                toks = norm.split()
                initial = toks[-1][0] if toks and toks[-1] else "#"
            else:
                initial = norm[0] if norm else "#"
            buckets[(e.label, initial)].append(e)

        canonical: list[Entity] = []
        for (label, _initial), bucket in buckets.items():
            threshold = self.thresholds.get(label, 0.85)
            bucket.sort(key=lambda x: x.mention_count, reverse=True)
            survivors: list[Entity] = []
            for cand in bucket:
                cnorm = normalize_name(cand.canonical_name)
                matched = None
                for surv in survivors:
                    if self._blocked(surv, cand):
                        continue
                    if _ratio(cnorm, normalize_name(surv.canonical_name)) >= threshold:
                        matched = surv
                        break
                if matched is not None:
                    self._merge_into(matched, cand)
                else:
                    survivors.append(cand)
            canonical.extend(survivors)

        if self.config.resolve_cross_type:
            canonical = self._resolve_cross_type(canonical)

        # Fold bare first/last names into their unique full name (after fuzzy,
        # which can't cross surname-initial buckets).
        canonical = self._fold_partial_persons(canonical)

        # Recompute stable ids and build name -> id index.
        name_to_id: dict[str, str] = {}
        for e in canonical:
            e.entity_id = stable_id(normalize_name(e.canonical_name), e.label,
                                    prefix="ent_", length=12)
            for nm in {e.canonical_name, *e.aliases}:
                name_to_id[normalize_name(nm)] = e.entity_id

        # Remap relationships onto entity ids (drop those we can't resolve both ends).
        remapped: list[Relationship] = []
        for r in relationships:
            sid = name_to_id.get(normalize_name(r.source))
            tid = name_to_id.get(normalize_name(r.target))
            if not sid or not tid or sid == tid:
                continue
            nr = Relationship(**{**r.to_dict()})
            nr.source = sid
            nr.target = tid
            remapped.append(nr)

        logger.info(
            "Dedup: %d raw -> %d canonical entities; %d/%d relationships remapped",
            len(entities), len(canonical), len(remapped), len(relationships),
        )
        return canonical, remapped, name_to_id
