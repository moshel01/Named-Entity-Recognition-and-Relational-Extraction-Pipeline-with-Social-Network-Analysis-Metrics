# NSDAP organizational structure and SA/SS/party unit naming conventions.

from __future__ import annotations

import re
from typing import Optional

# NSDAP territorial hierarchy (largest -> smallest)
NSDAP_TERRITORIAL: list[dict] = [
    {"echelon": "Reichsleitung", "english": "Reich leadership", "leader_title": "Reichsleiter"},
    {"echelon": "Gau", "english": "regional district", "leader_title": "Gauleiter"},
    {"echelon": "Kreis", "english": "county", "leader_title": "Kreisleiter"},
    {"echelon": "Ortsgruppe", "english": "local group", "leader_title": "Ortsgruppenleiter"},
    {"echelon": "Zelle", "english": "cell", "leader_title": "Zellenleiter"},
    {"echelon": "Block", "english": "block", "leader_title": "Blockleiter"},
]

# SA unit echelons (smallest -> largest)
SA_UNITS: list[dict] = [
    {"unit": "Schar", "english": "section", "approx_strength": "4-12"},
    {"unit": "Trupp", "english": "troop", "approx_strength": "12-30"},
    {"unit": "Sturm", "english": "company", "approx_strength": "70-200"},
    {"unit": "Sturmbann", "english": "battalion", "approx_strength": "250-600"},
    {"unit": "Standarte", "english": "regiment", "approx_strength": "1000-3000"},
    {"unit": "Brigade", "english": "brigade", "approx_strength": "multiple Standarten"},
    {"unit": "Gruppe", "english": "group", "approx_strength": "multiple brigades"},
    {"unit": "Obergruppe", "english": "senior group", "approx_strength": "regional command"},
]

# SS unit echelons (smallest -> largest)
SS_UNITS: list[dict] = [
    {"unit": "Schar", "english": "section", "approx_strength": "8"},
    {"unit": "Trupp", "english": "troop", "approx_strength": "20-60"},
    {"unit": "Sturm", "english": "company", "approx_strength": "70-120"},
    {"unit": "Sturmbann", "english": "battalion", "approx_strength": "250-600"},
    {"unit": "Standarte", "english": "regiment", "approx_strength": "1000-3000"},
    {"unit": "Abschnitt", "english": "sub-district", "approx_strength": "several Standarten"},
    {"unit": "Oberabschnitt", "english": "main district", "approx_strength": "regional command"},
]

# Parent-of map for structural edges (unit/echelon term -> parent org)
PARENT_OF: dict[str, str] = {
    # SA/SS units roll up to their parent organization.
    "sturm": "SA (Sturmabteilung)",
    "sturmbann": "SA (Sturmabteilung)",
    "standarte": "SA (Sturmabteilung)",
    "ss-sturm": "SS (Schutzstaffel)",
    "ss-sturmbann": "SS (Schutzstaffel)",
    "ss-standarte": "SS (Schutzstaffel)",
    # NSDAP territorial echelons roll up to the party.
    "gau": "NSDAP",
    "kreis": "NSDAP",
    "ortsgruppe": "NSDAP",
    "zelle": "NSDAP",
    "block": "NSDAP",
    "reichsleitung": "NSDAP",
}

# Detection patterns
_SA_UNIT_RE = re.compile(
    r"\bSA[-\s]?(Sturm|Sturmbann|Standarte|Brigade|Gruppe|Schar|Trupp)\b"
    r"(?:\s*(?:Nr\.?\s*)?[IVXLC]+\s*/?\s*\d+|\s*\d+(?:/\d+)?)?",
    re.IGNORECASE,
)
_SS_UNIT_RE = re.compile(
    r"\bSS[-\s]?(Sturm|Sturmbann|Standarte|Abschnitt|Oberabschnitt|Schar|Trupp)\b"
    r"(?:\s*(?:Nr\.?\s*)?[IVXLC]+\s*/?\s*\d+|\s*\d+(?:/\d+)?)?",
    re.IGNORECASE,
)
_NSDAP_SUBDIV_RE = re.compile(
    r"\b(Gau|Kreisleitung|Kreis|Ortsgruppe|Zelle|Block|Reichsleitung)\b",
    re.IGNORECASE,
)


def classify_unit(text: str) -> Optional[dict]:
    """Classify a unit designation found in ``text``.

    Returns a dict ``{"org", "echelon", "match", "parent"}`` or ``None``.
    """
    m = _SA_UNIT_RE.search(text)
    if m:
        echelon = m.group(1).capitalize()
        return {"org": "SA (Sturmabteilung)", "echelon": echelon,
                "match": m.group(0).strip(), "parent": "SA (Sturmabteilung)"}
    m = _SS_UNIT_RE.search(text)
    if m:
        echelon = m.group(1).capitalize()
        return {"org": "SS (Schutzstaffel)", "echelon": echelon,
                "match": m.group(0).strip(), "parent": "SS (Schutzstaffel)"}
    m = _NSDAP_SUBDIV_RE.search(text)
    if m:
        echelon = m.group(1).capitalize()
        return {"org": "NSDAP", "echelon": echelon,
                "match": m.group(0).strip(), "parent": "NSDAP"}
    return None
