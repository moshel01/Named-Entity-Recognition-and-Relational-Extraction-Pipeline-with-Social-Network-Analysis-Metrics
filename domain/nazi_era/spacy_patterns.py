# Custom spaCy EntityRuler patterns for Nazi-era text.

from __future__ import annotations

from typing import Any

from .rank_systems import SA_RANKS, SS_RANKS, WEHRMACHT_RANKS


# Rank phrase patterns (generated from the ladders)
def _rank_phrase_patterns() -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ladder in (SA_RANKS, SS_RANKS, WEHRMACHT_RANKS):
        for rank in ladder:
            for form in (rank["canonical"], *rank.get("variants", [])):
                key = form.lower()
                if key in seen or len(form) < 3:
                    continue
                seen.add(key)
                patterns.append({"label": "RANK", "pattern": form})
    return patterns


# Unit-designation token patterns (handle open-ended numbering)
_SA_UNIT_TOKENS: list[dict[str, Any]] = [
    {
        "label": "ORG",
        "pattern": [
            {"LOWER": "sa"},
            {"TEXT": "-", "OP": "?"},
            {"LOWER": {"IN": ["sturm", "sturmbann", "standarte", "brigade",
                              "gruppe", "schar", "trupp"]}},
            {"LIKE_NUM": True, "OP": "?"},
            {"TEXT": "/", "OP": "?"},
            {"LIKE_NUM": True, "OP": "?"},
        ],
    },
]
_SS_UNIT_TOKENS: list[dict[str, Any]] = [
    {
        "label": "ORG",
        "pattern": [
            {"LOWER": "ss"},
            {"TEXT": "-", "OP": "?"},
            {"LOWER": {"IN": ["sturm", "sturmbann", "standarte", "abschnitt",
                              "oberabschnitt", "schar", "trupp"]}},
            {"TEXT": {"REGEX": "^[IVXLC]+$"}, "OP": "?"},
            {"LIKE_NUM": True, "OP": "?"},
            {"TEXT": "/", "OP": "?"},
            {"LIKE_NUM": True, "OP": "?"},
        ],
    },
]

# -- NSDAP territorial subdivisions (keyword + optional following proper noun) -
_NSDAP_SUBDIV_TOKENS: list[dict[str, Any]] = [
    {
        "label": "ORG",
        "pattern": [
            {"LOWER": {"IN": ["gau", "kreisleitung", "ortsgruppe", "zelle",
                              "block", "reichsleitung", "kreis"]}},
            {"IS_TITLE": True, "OP": "?"},
        ],
    },
]

# Named historical events (phrase patterns)
_EVENT_PHRASES: list[str] = [
    "Machtergreifung", "Machtübernahme", "Machtuebernahme",
    "Hitlerputsch", "Hitler-Putsch", "Hitler Putsch",
    "Münchener Putsch", "Muenchener Putsch", "Munich Putsch",
    "Novemberrevolution", "November Revolution",
    "Kapp-Putsch", "Kapp Putsch",
    "Röhm-Putsch", "Roehm-Putsch", "Röhm Putsch",
    "Reichstagsbrand", "Reichstag Fire",
    "Kristallnacht", "Reichspogromnacht",
    "Ruhrbesetzung", "Ruhrkampf",
    "Versailler Vertrag", "Treaty of Versailles",
    "Weltwirtschaftskrise",
    "Beer Hall Putsch", "Bürgerbräu Putsch",
]
_EVENT_PATTERNS: list[dict[str, Any]] = [
    {"label": "EVENT", "pattern": phrase} for phrase in _EVENT_PHRASES
]

# The merged EntityRuler pattern set the foundation loads
PATTERNS: list[dict[str, Any]] = (
    _rank_phrase_patterns()
    + _SA_UNIT_TOKENS
    + _SS_UNIT_TOKENS
    + _NSDAP_SUBDIV_TOKENS
    + _EVENT_PATTERNS
)

# Membership-number regexes (evidence only; NOT entity nodes)
# Used by canonical_inference.text_patterns, not by the EntityRuler.
MEMBERSHIP_NUMBER_PATTERNS: dict[str, str] = {
    "NSDAP": r"\b(?:NSDAP|Partei)[-\s]?(?:Nr\.?|Nummer|No\.?)\s*\d{2,7}\b",
    "SS (Schutzstaffel)": r"\bSS[-\s]?(?:Nr\.?|Nummer)\s*\d{1,7}\b",
}
