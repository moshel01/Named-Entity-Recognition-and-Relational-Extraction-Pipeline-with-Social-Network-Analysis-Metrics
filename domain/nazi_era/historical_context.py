# Historical date validation, anachronism detection, and German date parsing.

from __future__ import annotations

import re
from typing import Optional

# -- Organization existence windows (founding_year, dissolution_year_or_None) -
# A value of None for the end means "still existed at the end of the period
# under study (1945)". Years are first-order approximations for anachronism
# checks, not precise legal dates.
ORGANIZATION_EXISTENCE: dict[str, tuple[int, Optional[int]]] = {
    "NSDAP": (1920, 1945),                       # DAP 1919; renamed NSDAP 1920
    "DAP": (1919, 1920),
    "SA (Sturmabteilung)": (1921, 1945),
    "SS (Schutzstaffel)": (1925, 1945),
    "Hitler Youth (Hitlerjugend)": (1926, 1945),
    "League of German Girls (BDM)": (1930, 1945),
    "NSKK": (1931, 1945),
    "NSFK": (1937, 1945),
    "German Labor Front (DAF)": (1933, 1945),
    "NSV": (1933, 1945),
    "Gestapo": (1933, 1945),
    "SD (Sicherheitsdienst)": (1931, 1945),
    "Stahlhelm": (1918, 1935),                   # absorbed into SA, 1933-35
    "Freikorps": (1918, 1923),
    "Reichsbanner Schwarz-Rot-Gold": (1924, 1933),
    "Rotfrontkämpferbund": (1924, 1929),         # banned 1929
    "Reichswehr": (1919, 1935),                  # became Wehrmacht 1935
    "Wehrmacht": (1935, 1945),
    "Weimar Republic": (1919, 1933),
    "German Empire": (1871, 1918),
    "Third Reich": (1933, 1945),
    "SPD": (1875, 1945),
    "KPD": (1919, 1945),
    "DNVP": (1918, 1933),
    "Center Party (Zentrum)": (1870, 1933),
    "DDP": (1918, 1930),
    "DVP": (1918, 1933),
}

# German month names -> numeric
GERMAN_MONTHS: dict[str, int] = {
    "januar": 1, "jänner": 1, "februar": 2, "märz": 3, "maerz": 3,
    "april": 4, "mai": 5, "juni": 6, "juli": 7, "august": 8,
    "september": 9, "oktober": 10, "november": 11, "dezember": 12,
}

# German seasons -> representative month
GERMAN_SEASONS: dict[str, int] = {
    "frühling": 3, "fruehling": 3, "frühjahr": 3, "fruehjahr": 3,
    "sommer": 6,
    "herbst": 9,
    "winter": 12,
}

_YEAR_RE = re.compile(r"\b(1[89]\d{2})\b")
PERIOD_START = 1900
PERIOD_END = 1946


def temporal_period(year: int) -> str:
    """Map a year to a German-history period for temporal edge slicing."""
    if not year:
        return ""
    if year < 1919:
        return "imperial_ww1"      # Empire + WWI
    if year <= 1932:
        return "weimar"            # Weimar Republic
    return "nazi_rule"             # 1933+


def german_month_to_number(text: str) -> Optional[int]:
    """Return the month number for a German month name, or None."""
    return GERMAN_MONTHS.get(text.strip().lower().rstrip("."))


def season_to_month(text: str) -> Optional[int]:
    """Map a German season word to a representative month (or None)."""
    return GERMAN_SEASONS.get(text.strip().lower())


def sanitize_german_date(text: str) -> str:
    """Replace German season words with an approximate 'Month Year' phrase.

    e.g. ``"Herbst 1923"`` -> ``"September 1923"`` so the generic date parser
    can normalize it. Leaves text without a recognized season unchanged.
    """
    out = text
    for season, month in GERMAN_SEASONS.items():
        if season in out.lower():
            # Insert an English month name the downstream dateutil parser knows.
            month_name = ["", "January", "February", "March", "April", "May",
                          "June", "July", "August", "September", "October",
                          "November", "December"][month]
            out = re.sub(season, month_name, out, flags=re.IGNORECASE)
    return out


def validate_date_range(
    year: Optional[int], organization: Optional[str] = None
) -> tuple[bool, str]:
    """Validate a year against the study period and (optionally) an org window.

    Returns ``(is_valid, reason)``. ``reason`` is empty when valid; otherwise it
    explains the anachronism so the caller can flag the claim rather than drop it.
    """
    if year is None:
        return True, ""
    if not (PERIOD_START <= year <= PERIOD_END):
        return False, f"Year {year} is outside the study period {PERIOD_START}-{PERIOD_END}."
    if organization:
        window = ORGANIZATION_EXISTENCE.get(organization)
        if window:
            start, end = window
            if year < start:
                return False, (f"{organization} did not exist until {start}; "
                               f"claim references {year}.")
            if end is not None and year > end:
                return False, (f"{organization} ceased to exist by {end}; "
                               f"claim references {year}.")
    return True, ""


def extract_year(text: str) -> Optional[int]:
    """Extract the first plausible 1800s/1900s year from a string."""
    m = _YEAR_RE.search(text)
    return int(m.group(1)) if m else None
