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
}


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
