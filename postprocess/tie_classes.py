# Classify each relation into a social-tie class, so the SNA graph can separate
# real interpersonal ties from affiliations, stance, and bare co-occurrence.
#
# Tie classes:
#   interaction   - person<->person social tie actually narrated (the real SNA)
#   affiliation   - person->org/institution (two-mode)
#   participation - person->event (two-mode)
#   biographical  - person->place / person->rank (attribute-like)
#   stance        - attitude/opinion, not a social tie (discourse layer)
#   cooccurrence  - mere co-presence, not a tie (weakest layer)
#   other         - unclassifiable / dedup artifacts

from __future__ import annotations

# Relation type -> tie class (canonical relation vocabulary).
_REL_CLASS: dict[str, str] = {
    # interaction (person<->person)
    "met_with": "interaction", "visited": "interaction", "recruited": "interaction",
    "subordinate_to": "interaction", "commanded": "interaction",
    "appointed_by": "interaction", "imprisoned_by": "interaction",
    "family_of": "interaction", "married_to": "interaction", "friend_of": "interaction",
    "sibling_of": "interaction", "related_to": "interaction", "knew": "interaction",
    "mentored": "interaction", "served_with": "interaction", "succeeded": "interaction",
    # affiliation (person->org)
    "member_of": "affiliation", "joined": "affiliation", "served_in": "affiliation",
    "led": "affiliation", "founded": "affiliation", "employed_by": "affiliation",
    "studied_at": "affiliation", "expelled_from": "affiliation",
    "propagandized_for": "affiliation", "worked_for": "affiliation",
    # participation (person->event)
    "participated_in": "participation", "fought_in": "participation",
    "wounded_at": "participation", "attended_event": "participation",
    # biographical (person->place / rank)
    "born_in": "biographical", "resided_in": "biographical", "located_in": "biographical",
    "lived_in": "biographical", "promoted_to": "biographical", "died_in": "biographical",
    # stance (attitude, NOT a social tie) - even between two people
    "supported": "stance", "opposed": "stance", "influenced_by": "stance",
    "fought_against": "stance", "admired": "stance", "read": "stance",
    "believed_in": "stance", "allied_with": "stance", "sympathized_with": "stance",
    # not ties
    "co_occurs_with": "cooccurrence", "alias_of": "other",
}

# Reciprocal ties -> undirected; everything else is directed.
SYMMETRIC: set[str] = {
    "met_with", "family_of", "married_to", "friend_of", "sibling_of", "related_to",
    "knew", "served_with", "allied_with", "co_occurs_with",
}

# Fallback by target entity type when the relation type is unknown.
_LABEL_CLASS: dict[str, str] = {
    "ORG": "affiliation", "INSTITUTION": "affiliation",
    "EVENT": "participation", "LOCATION": "biographical", "RANK": "biographical",
}


def classify(rel_type: str, src_label: str = "", tgt_label: str = "") -> str:
    """Tie class from the relation type, with an endpoint-type fallback."""
    rt = (rel_type or "").strip().lower()
    cls = _REL_CLASS.get(rt)
    if cls:
        # Interaction is strictly person<->person. A hierarchical verb pointing at
        # an org ("recruited by the SA", "subordinate_to NSDAP") is an affiliation.
        if cls == "interaction" and src_label and tgt_label \
                and not (src_label == "PERSON" and tgt_label == "PERSON"):
            return _LABEL_CLASS.get(tgt_label, "affiliation")
        return cls
    # Unknown relation: infer from the target, else person<->person interaction.
    by_tgt = _LABEL_CLASS.get(tgt_label)
    if by_tgt:
        return by_tgt
    if src_label == "PERSON" and tgt_label == "PERSON":
        return "interaction"
    return "other"


def is_symmetric(rel_type: str) -> bool:
    return (rel_type or "").strip().lower() in SYMMETRIC


# Edge sign for signed-network analysis (balance theory etc.).
_POSITIVE = {"supported", "admired", "allied_with", "sympathized_with",
             "friend_of", "mentored", "recruited"}
_NEGATIVE = {"opposed", "fought_against", "imprisoned_by", "expelled_from"}


def polarity(rel_type: str) -> str:
    """Sign of an affective/antagonistic tie: positive / negative / neutral."""
    rt = (rel_type or "").strip().lower()
    if rt in _POSITIVE:
        return "positive"
    if rt in _NEGATIVE:
        return "negative"
    return "neutral"


# Edge classes that count as the headline social network.
SOCIAL = {"interaction"}
# Classes that are two-mode but still structural (membership/biography).
STRUCTURAL = {"affiliation", "participation", "biographical"}
# Classes excluded from interpersonal centrality.
NON_SOCIAL = {"stance", "cooccurrence", "other"}
