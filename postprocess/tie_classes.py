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


# Substring markers so free-form LLM relations (no controlled vocabulary, e.g.
# the generic domain) still classify correctly - the curated dicts above only
# cover the domain-normalized vocabulary.
_OPPOSITION = ("oppos", "against", "rival", "enem", "resist", "denounce",
               "boycott", "defied", "defy", "fought_against")

# Curated relations that are genuinely interpersonal when BOTH endpoints are
# people. Kept deliberately narrow: relations whose semantics imply a non-person
# target (promoted_to -> rank, located_in -> place, studied_at -> org) are NOT
# here, so a place/role/org mis-tagged PERSON does not get promoted into the
# headline interaction layer. Free-form person<->person verbs are handled by the
# unknown-relation fallback below.
_PERSON_TO_PERSON = {"led", "commanded", "recruited", "mentored",
                     "appointed_by", "subordinate_to", "succeeded"}


def classify(rel_type: str, src_label: str = "", tgt_label: str = "") -> str:
    """Tie class from the relation type, with endpoint-aware corrections."""
    rt = (rel_type or "").strip().lower()
    # Opposition toward an org/group is a discourse stance, NOT membership - so a
    # person who "opposes the NSDAP" is not tied to it as an affiliate.
    if tgt_label in ("ORG", "INSTITUTION") and any(s in rt for s in _OPPOSITION):
        return "stance"
    both_person = src_label == "PERSON" and tgt_label == "PERSON"
    cls = _REL_CLASS.get(rt)
    if cls:
        # Interaction is strictly person<->person. A hierarchical verb pointing at
        # an org ("recruited by the SA", "subordinate_to NSDAP") is an affiliation.
        if cls == "interaction" and src_label and tgt_label and not both_person:
            return _LABEL_CLASS.get(tgt_label, "affiliation")
        # A genuinely interpersonal verb between two PEOPLE ("X led/commanded Y")
        # is interaction, not affiliation - but only for relations whose meaning
        # implies a person target (not located_in/promoted_to, where a PERSON
        # endpoint signals a mis-typed place/rank).
        if both_person and rt in _PERSON_TO_PERSON:
            return "interaction"
        return cls
    # Unknown relation: infer from the target, else person<->person interaction.
    by_tgt = _LABEL_CLASS.get(tgt_label)
    if by_tgt:
        return by_tgt
    if both_person:
        return "interaction"
    return "other"


# Reciprocal markers: free-form relations that denote a mutual (undirected) tie.
_SYM_SUBSTR = ("_with", "married", "spouse", "partner", "colleague", "friend",
               "sibling", "relative", "related_to", "associate", "allied",
               "co_founded", "cofounded", "co-founded", "negotiat", "acquainted")


def is_symmetric(rel_type: str) -> bool:
    rt = (rel_type or "").strip().lower()
    if rt in SYMMETRIC:
        return True
    return any(s in rt for s in _SYM_SUBSTR)


# Edge sign for signed-network analysis (balance theory etc.).
_POSITIVE = {"supported", "admired", "allied_with", "sympathized_with",
             "friend_of", "mentored", "recruited"}
_NEGATIVE = {"opposed", "fought_against", "imprisoned_by", "expelled_from"}
_POS_SUBSTR = ("friend", "ally", "allied", "support", "admir", "recruit",
               "mentor", "trust", "loyal", "marri", "partner", "praise",
               "thank", "respect", "befriend", "sympath", "favor")
_NEG_SUBSTR = ("oppos", "against", "dislik", "undermin", "clash", "disagree",
               "rival", "enem", "betray", "attack", "fought", "conflict",
               "denounce", "hostile", "threat", "distrust", "hate", "feud",
               "boycott", "persecut", "imprison", "expel", "arrest")


def polarity(rel_type: str) -> str:
    """Sign of an affective/antagonistic tie: positive / negative / neutral."""
    rt = (rel_type or "").strip().lower()
    if rt in _POSITIVE:
        return "positive"
    if rt in _NEGATIVE:
        return "negative"
    # Free-form fallback: check antagonism before affinity.
    if any(s in rt for s in _NEG_SUBSTR):
        return "negative"
    if any(s in rt for s in _POS_SUBSTR):
        return "positive"
    return "neutral"


# Edge classes that count as the headline social network.
SOCIAL = {"interaction"}
# Classes that are two-mode but still structural (membership/biography).
STRUCTURAL = {"affiliation", "participation", "biographical"}
# Classes excluded from interpersonal centrality.
NON_SOCIAL = {"stance", "cooccurrence", "other"}
