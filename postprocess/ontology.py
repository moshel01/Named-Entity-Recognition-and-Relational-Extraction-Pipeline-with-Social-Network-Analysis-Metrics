# Relation ontology alignment.

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Optional

from core.schema import Relationship

logger = logging.getLogger(__name__)

_NORM = re.compile(r"[\s_\-]+")


def _norm(s: str) -> str:
    return _NORM.sub(" ", str(s).strip().lower())


# Default relation schema for the generic (non-domain) path. Without it the LLM
# invents a different verb phrase per edge (e.g.
# sent_letters_to_requesting_compliance_info_about_funding_of_affiliates_of) -
# unusable as an SNA edge vocabulary. Constraining the prompt to this closed set
# (and fuzzy-aligning the verbose tail) gives a stable vocabulary that lines up
# with the tie_classes maps. Direction matters: funded vs funded_by are separate
# so alignment never reverses an edge - list the common surface forms exactly
# (exact match wins before token-containment), and avoid bare generic tokens
# (from/at/with) that token-containment would over-match.
GENERIC_RELATION_ONTOLOGY: dict[str, list[str]] = {
    # interaction (person<->person)
    "met_with": ["met", "meeting with", "spoke with", "talked with", "contacted",
                 "corresponded with", "communicated with", "negotiated with",
                 "wrote to", "sent letters to", "in contact with"],
    "mentored": ["mentor of", "mentored", "advised", "tutored", "coached"],
    "recruited": ["recruited", "enlisted", "brought into"],
    "succeeded": ["succeeded", "took over from", "replaced as"],
    "family_of": ["relative of", "cousin of", "uncle of", "aunt of", "nephew of",
                  "niece of", "in-law of", "kin of"],
    "married_to": ["wife of", "husband of", "spouse of", "married to", "wed to"],
    "sibling_of": ["brother of", "sister of"],
    "friend_of": ["friend of", "befriended", "close friend of"],
    # affiliation (person->org / org->org)
    "employed_by": ["works for", "worked for", "employee of", "former employee of",
                    "hired by", "on the staff of", "employed at", "employed by"],
    "led": ["president of", "former president of", "chairman of", "chair of",
            "ceo of", "chief executive of", "executive director of", "director of",
            "head of", "leader of", "led", "managing director of"],
    "member_of": ["board member of", "trustee of", "member of", "belongs to",
                  "sits on the board of", "fellow of", "on the board of"],
    "affiliated_with": ["affiliated with", "associated with", "connected to",
                        "tied to", "linked to"],
    "founded": ["founder of", "co-founder of", "founded", "established", "created",
                "formed", "set up", "started"],
    "founded_by": ["founded by", "established by", "created by", "formed by",
                   "set up by"],
    "owns": ["owner of", "owns", "controls", "parent company of", "acquired"],
    "owned_by": ["owned by", "controlled by", "subsidiary of", "unit of",
                 "division of", "acquired by"],
    "funded": ["funded", "provided funding to", "granted to", "gave a grant to",
               "gave grant to", "provided grant to", "provided grants to",
               "awarded grant to", "donated to", "donates to", "financed",
               "bankrolled", "gave money to", "contributed to", "supports financially"],
    "funded_by": ["funded by", "financed by", "received funding from",
                  "grant from", "donation from", "backed by", "bankrolled by"],
    "partnered_with": ["partner of", "partnered with", "collaborated with",
                       "cooperated with", "joint venture with", "teamed with",
                       "joined forces with", "works with", "working with",
                       "worked with"],
    "studied_at": ["studied at", "graduated from", "alumnus of", "alumna of",
                   "educated at", "degree from", "schooled at"],
    # participation (person->event)
    "participated_in": ["took part in", "participated in", "involved in", "engaged in"],
    "attended_event": ["attended", "spoke at", "present at", "appeared at"],
    "fought_in": ["fought in", "served in", "combatant in", "deployed to"],
    # biographical (person->place)
    "located_in": ["located in", "based in", "headquartered in", "situated in",
                   "operates in"],
    "born_in": ["born in", "native of", "birthplace"],
    "lived_in": ["lived in", "resided in", "moved to", "settled in"],
    "died_in": ["died in", "passed away in"],
    # stance (attitude, not a social tie)
    "supported": ["supported", "endorsed", "backed", "advocated for", "championed",
                  "praised", "defended"],
    "opposed": ["opposed", "criticized", "condemned", "denounced", "fought against",
                "campaigned against", "sued", "investigated", "challenged"],
    "influenced_by": ["influenced by", "inspired by", "shaped by", "modeled on"],
    "allied_with": ["allied with", "aligned with", "ally of", "in alliance with"],
    # causal (one thing brings about another) - driver->impact, cause->effect.
    # Direction-sensitive: caused vs caused_by are separate so alignment never
    # reverses the arrow. For disaster storylines, news narratives, plot chains.
    "caused": ["caused", "led to", "resulted in", "triggered", "brought about",
               "gave rise to", "set off", "sparked"],
    "caused_by": ["caused by", "resulted from", "due to", "because of",
                  "stemmed from", "triggered by", "brought on by"],
    "contributed_to": ["contributed to", "fueled", "exacerbated", "drove",
                       "aggravated", "worsened"],
    "prevented": ["prevented", "averted", "stopped", "blocked", "forestalled",
                  "headed off"],
}

# One-line scenario notes for the direction-sensitive / confusable relations,
# shown next to the type in the prompt (the rest are self-evident).
GENERIC_RELATION_GUIDE: dict[str, str] = {
    "employed_by": "person works/worked for an org as staff (not its head).",
    "led": "person heads/headed an org (president, CEO, chair, director).",
    "member_of": "person belongs to an org or its board, not as staff or head.",
    "founded": "subject created the object org/group.",
    "founded_by": "object created the subject org/group (reverse of founded).",
    "owns": "subject owns/controls the object (parent over subsidiary).",
    "owned_by": "subject is owned/controlled by the object (reverse of owns).",
    "funded": "subject gives money/grants to the object.",
    "funded_by": "subject receives money/grants from the object (reverse of funded).",
    "partnered_with": "two parties collaborate as equals (symmetric).",
    "supported": "subject publicly backs the object's cause or position.",
    "opposed": "subject publicly works against the object.",
    "influenced_by": "subject's ideas were shaped by the object.",
    "succeeded": "subject took over the object's role or position.",
    "caused": "subject brought about the object (an event/outcome).",
    "caused_by": "object brought about the subject (reverse of caused).",
    "contributed_to": "subject was a partial cause/aggravator of the object.",
    "prevented": "subject stopped the object from happening.",
}


# Type signatures for the constraining generic relations: (source types, target
# types). A relation whose endpoint types contradict its signature is a likely
# misextraction - a local model emitting "led" between two places, or "born_in"
# pointing at an org. Inspired by the ASP consistency check in Tran et al. 2025
# (LLM + ASP for joint entity-relation extraction): encode each relation's
# argument types and reject the violations. Kept high-precision - only relations
# with a reliable signature are listed; loose stance/interaction relations
# (supported, opposed, met_with) are unconstrained on purpose. Membership
# (member_of/joined/served_in) already has its own target-is-org check in
# main.py (suspect_membership), so it stays out of here. A type outside
# CORE_TYPES (a domain label we didn't model) never triggers a violation - it is
# a wildcard, matching ASP's "no type_def for this slot -> accept".
_PERSON = frozenset({"PERSON"})
_ORG = frozenset({"ORG", "INSTITUTION"})
_PLACE = frozenset({"LOCATION", "GPE"})
_RANK = frozenset({"RANK"})
CORE_TYPES = _PERSON | _ORG | _PLACE | _RANK | frozenset({"EVENT"})
RELATION_TYPE_SIGNATURES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "employed_by": (_PERSON, _ORG),
    "led":         (_PERSON, _ORG),
    "studied_at":  (_PERSON, _ORG),
    "founded":     (_PERSON | _ORG, _ORG),
    "founded_by":  (_ORG, _PERSON | _ORG),
    "owns":        (_PERSON | _ORG, _ORG),
    "owned_by":    (_ORG, _PERSON | _ORG),
    "born_in":     (_PERSON, _PLACE),
    "lived_in":    (_PERSON, _PLACE),
    "resided_in":  (_PERSON, _PLACE),   # domain (nazi_era) vocab alongside lived_in
    "died_in":     (_PERSON, _PLACE),
    # located_in is permissive on the source (a person, org, or place can all be
    # "in" a place - the nazi_era domain maps "lived in/wohnte in/from" here and
    # tie_classes treats person->place as biographical). The real constraint is
    # the TARGET: located_in pointing at a person/org is the misextraction.
    "located_in":  (_PERSON | _ORG | _PLACE, _PLACE),
    "married_to":  (_PERSON, _PERSON),
    "sibling_of":  (_PERSON, _PERSON),
    "family_of":   (_PERSON, _PERSON),
    "promoted_to": (_PERSON, _RANK),   # a person rises to a rank, not an org/place
    # board/governance + corporate structure (InfluenceWatch). A board seat or
    # officer post is held BY a person; a subsidiary/parent tie is org-to-org.
    "board_member_of": (_PERSON, _ORG),
    "director_of":     (_PERSON, _ORG),
    "subsidiary_of":   (_ORG, _ORG),
    "fiscal_sponsor_of": (_ORG, _ORG),
    "project_of":      (_ORG, _ORG),
}


# Friendly label per core type, for rendering a signature into a prompt hint.
_TYPE_WORD = {"PERSON": "person", "ORG": "org", "INSTITUTION": "org",
              "LOCATION": "place", "GPE": "place", "RANK": "rank", "EVENT": "event"}
_WORD_ORDER = ["person", "org", "place", "rank", "event"]


def _render_slot(allowed: frozenset[str]) -> str:
    words = {_TYPE_WORD[a] for a in allowed if a in _TYPE_WORD}
    return "/".join(w for w in _WORD_ORDER if w in words)


def relation_signature_hints(relation_types: list[str]) -> dict[str, str]:
    """{rel_type -> "person->place"} for the listed types that have a signature.
    Feeds the extraction prompt so the model gets the argument types up front
    (structure-aware extraction) instead of only being tagged after the fact."""
    out: dict[str, str] = {}
    for rt in relation_types:
        sig = RELATION_TYPE_SIGNATURES.get(rt)
        if sig:
            out[rt] = f"{_render_slot(sig[0])}->{_render_slot(sig[1])}"
    return out


def _slot_violation(actual: Optional[str], allowed: frozenset[str]) -> bool:
    """A slot is violated only when the endpoint's type is a core type we model
    AND it's not in the allowed set. Unknown/exotic types pass (wildcard)."""
    return actual in CORE_TYPES and actual not in allowed


def check_relation_types(relationships: list[Relationship],
                         type_of: dict[str, str],
                         drop: bool = False) -> tuple[list[Relationship], int]:
    """Tag (or drop) relations whose endpoint types contradict their signature.
    `type_of` maps entity_id -> canonical label. Returns the (possibly filtered)
    list and the count flagged. Relations with no signature are never touched."""
    out: list[Relationship] = []
    flagged = 0
    for r in relationships:
        sig = RELATION_TYPE_SIGNATURES.get(r.rel_type)
        if sig and (_slot_violation(type_of.get(r.source), sig[0])
                    or _slot_violation(type_of.get(r.target), sig[1])):
            flagged += 1
            if drop:
                continue
            r.attributes["type_violation"] = True
        out.append(r)
    return out, flagged


# Functional properties: at most one true value per subject. A person has one
# birthplace/birth date/death place; conflicting targets are a contradiction (the
# narrator-vs-relative birthplace confound, or a misread). Knowledge-alignment noise
# detection (Hofer et al. 2024) - the global-consistency complement to the per-edge
# type-signature gate. resided_in/member_of are NOT functional (many residences/orgs).
FUNCTIONAL_RELATIONS: frozenset[str] = frozenset({
    "born_in", "birth_date", "date_of_birth", "died_in", "place_of_death",
    "date_of_death",
})


def check_functional_consistency(
    relationships: list[Relationship], functional: frozenset[str] | None = None,
    drop: bool = False,
) -> tuple[list[Relationship], int]:
    """Tag (or fuse) functional-property contradictions: a subject with the same
    functional relation pointing at two different targets. Tags every edge in the
    conflict `functional_conflict`; with ``drop`` keeps only the best-supported
    target (most edges, then highest confidence) and drops the rest. Returns the
    (possibly filtered) list and the count flagged."""
    from collections import Counter, defaultdict
    fset = functional or FUNCTIONAL_RELATIONS
    groups: dict[tuple[str, str], list[Relationship]] = defaultdict(list)
    for r in relationships:
        if r.rel_type in fset and r.source and r.target:
            groups[(r.source, r.rel_type)].append(r)

    flagged = 0
    drop_ids: set[int] = set()
    for rels in groups.values():
        targets = {r.target for r in rels}
        if len(targets) < 2:
            continue  # consistent (one target, however many supporting mentions)
        tc = Counter(r.target for r in rels)
        best = max(targets, key=lambda t: (tc[t], max(r.confidence for r in rels
                                                       if r.target == t)))
        for r in rels:
            r.attributes["functional_conflict"] = True
            flagged += 1
            if drop and r.target != best:
                drop_ids.add(id(r))
    out = ([r for r in relationships if id(r) not in drop_ids]
           if drop and drop_ids else relationships)
    return out, flagged


def resolve_relation_ontology(config, domain=None) -> dict[str, list[str]]:
    """Active relation ontology: config, else domain, else the generic default
    (unless ontology is disabled, which means free-form relations)."""
    rel = getattr(getattr(config, "ontology", None), "relations", None)
    if rel:
        if isinstance(rel, dict):
            return {k: list(v or []) for k, v in rel.items()}
        if isinstance(rel, (list, tuple)):
            return {str(c): [] for c in rel}
    if domain is not None:
        try:
            dom = domain.relation_ontology()
            if dom:
                return {k: list(v or []) for k, v in dom.items()}
        except Exception:  # noqa: BLE001
            pass
    if getattr(getattr(config, "ontology", None), "enabled", True):
        return {k: list(v) for k, v in GENERIC_RELATION_ONTOLOGY.items()}
    return {}


def resolve_relation_guide(config, domain=None) -> dict[str, str]:
    """label -> definition, from config.ontology.relation_guide, else domain."""
    guide = getattr(getattr(config, "ontology", None), "relation_guide", None)
    if isinstance(guide, dict) and guide:
        return {str(k): str(v) for k, v in guide.items() if v}
    if domain is not None:
        try:
            dom = domain.relation_guide()
            if isinstance(dom, dict) and dom:
                return {str(k): str(v) for k, v in dom.items() if v}
        except Exception:  # noqa: BLE001
            pass
    if getattr(getattr(config, "ontology", None), "enabled", True):
        return dict(GENERIC_RELATION_GUIDE)
    return {}


class OntologyAligner:
    """Align relation-type strings onto a canonical ontology."""

    def __init__(self, ontology: Optional[dict[str, list[str]]],
                 fuzzy_threshold: float = 0.82, drop_unmapped: bool = False) -> None:
        self.ontology = ontology or {}
        self.threshold = fuzzy_threshold
        self.drop_unmapped = drop_unmapped
        # Normalized synonym/canonical -> canonical.
        self.index: dict[str, str] = {}
        for canon, syns in self.ontology.items():
            self.index[_norm(canon)] = canon
            for s in syns:
                self.index[_norm(s)] = canon

    @property
    def active(self) -> bool:
        return bool(self.ontology)

    @property
    def canonical_types(self) -> list[str]:
        return sorted(self.ontology.keys())

    def align(self, rel_type: str) -> Optional[str]:
        """Return the canonical type for ``rel_type`` or None if unmatched."""
        if not self.active:
            return rel_type
        n = _norm(rel_type)
        if not n:
            return None
        if n in self.index:                                  # 1. exact
            return self.index[n]
        nt = set(n.split())                                  # 2. whole-token
        for key, canon in self.index.items():                #    containment
            kt = set(key.split())                            #    (avoids led->fled)
            if kt and (kt <= nt or nt <= kt):
                return canon
        best, best_r = None, 0.0                             # 3. fuzzy
        for key, canon in self.index.items():
            # Skip very short keys: fuzzy ratio on 3-4 char strings is unreliable
            # ("fled"~"led" = 0.86). Short canonicals must match exactly/by synonym.
            if len(key) < 5:
                continue
            r = SequenceMatcher(None, n, key).ratio()
            if r > best_r:
                best, best_r = canon, r
        return best if best_r >= self.threshold else None

    def apply(self, relationships: list[Relationship]) -> list[Relationship]:
        """Remap relation types in place; optionally drop unmapped relations."""
        if not self.active:
            return relationships
        out: list[Relationship] = []
        n_aligned = n_unmapped = n_dropped = 0
        for r in relationships:
            mapped = self.align(r.rel_type)
            if mapped is None:
                if self.drop_unmapped:
                    n_dropped += 1
                    continue
                r.attributes["ontology"] = "unmapped"
                n_unmapped += 1
            else:
                if mapped != r.rel_type:
                    n_aligned += 1
                r.rel_type = mapped
                r.attributes["ontology"] = "aligned"
            out.append(r)
        logger.info("Ontology alignment: %d remapped, %d unmapped, %d dropped.",
                    n_aligned, n_unmapped, n_dropped)
        return out
