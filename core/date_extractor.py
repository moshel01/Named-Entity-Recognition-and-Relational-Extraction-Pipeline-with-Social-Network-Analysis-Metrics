# Temporal expression extraction.

from __future__ import annotations

import re
from typing import Optional

from dateutil import parser as dtparser

from .schema import EntityMention, TimelineEvent

_EN_MONTHS = (
    r"January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)

_EN_MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def _build_patterns(month_words: dict[str, int], season_words: dict[str, int]) -> list[re.Pattern]:
    """Build the ordered (most specific first) date patterns, incl. domain words."""
    months_alt = _EN_MONTHS
    if month_words:
        months_alt = months_alt + "|" + "|".join(re.escape(w) for w in sorted(month_words, key=len, reverse=True))
    patterns = [
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                              # 2021-03-04
        re.compile(rf"\b\d{{1,2}}\.?\s+(?:{months_alt})\.?\s+\d{{4}}\b", re.I),  # 4 March 1921 / 4. März 1921
        re.compile(rf"\b(?:{months_alt})\.?\s+\d{{1,2}},?\s+\d{{4}}\b", re.I),   # March 4, 1921
        re.compile(rf"\b(?:{months_alt})\.?\s+\d{{4}}\b", re.I),                 # March 1921 / März 1921
        re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),                        # 03/04/2021
    ]
    if season_words:
        seasons_alt = "|".join(re.escape(w) for w in sorted(season_words, key=len, reverse=True))
        patterns.append(re.compile(rf"\b(?:{seasons_alt})\s+\d{{4}}\b", re.I))   # Herbst 1923
    patterns.append(re.compile(r"\b(?:19|20)\d{2}\b"))                     # 2021 (least specific)
    return patterns


def _translate_foreign(date_text: str, month_words: dict[str, int],
                       season_words: dict[str, int]) -> str:
    """Replace foreign month/season words in a date string with English ones."""
    out = date_text
    for word, month in {**season_words, **month_words}.items():
        if word.lower() in out.lower():
            out = re.sub(re.escape(word), _EN_MONTH_NAMES[month], out, flags=re.IGNORECASE)
    return out


_FULL_YEAR_RE = re.compile(r"(1[89]\d{2}|20\d{2})")
_TRAIL_2DIGIT_RE = re.compile(r"(?<!\d)(\d{2})\s*$")


def normalize_date(text: str, month_words: dict[str, int] | None = None,
                   season_words: dict[str, int] | None = None,
                   pivot_max: int | None = None) -> tuple[Optional[str], Optional[int]]:
    """(iso, year) for a date string, or (None, None) if no plausible year.

    Resolves the year first (4-digit, else 2-digit pivoted into 18xx/19xx for a
    historical study period), then forces dateutil to that year. Strings with no
    year (e.g. "6 Jahre alt") are rejected so they don't become timeline events.
    """
    t = _translate_foreign(text, month_words or {}, season_words or {})
    m4 = _FULL_YEAR_RE.search(t)
    year = int(m4.group(1)) if m4 else None
    if year is None and pivot_max:
        m2 = _TRAIL_2DIGIT_RE.search(t.strip())
        # only treat a trailing 2-digit as a year if the string looks date-ish
        if m2 and re.search(r"[A-Za-z]|[./]", t):
            yy = int(m2.group(1))
            year = 1900 + yy if (1900 + yy) <= pivot_max else 1800 + yy
    if year is None:
        return None, None
    try:
        dt = dtparser.parse(t, default=dtparser.parse(f"{year}-01-01"), fuzzy=True)
        dt = dt.replace(year=year)
        return dt.date().isoformat(), year
    except (ValueError, OverflowError):
        return f"{year}", year


def extract_dates(
    text: str,
    chunk_id: str,
    doc_id: str,
    offset: int = 0,
    sentences: Optional[list[tuple[int, int, str]]] = None,
    mentions: Optional[list[EntityMention]] = None,
    month_words: Optional[dict[str, int]] = None,
    season_words: Optional[dict[str, int]] = None,
    pivot_max: Optional[int] = None,
) -> list[TimelineEvent]:
    """Extract dated events from chunk ``text``."""
    sentences = sentences or []
    mentions = mentions or []
    month_words = {k.lower(): v for k, v in (month_words or {}).items()}
    season_words = {k.lower(): v for k, v in (season_words or {}).items()}
    patterns = _build_patterns(month_words, season_words)
    found: dict[tuple[int, int], TimelineEvent] = {}

    for pattern in patterns:
        for m in pattern.finditer(text):
            abs_start = offset + m.start()
            abs_end = offset + m.end()
            if any(s <= abs_start < e for (s, e) in found):
                continue
            iso, year = normalize_date(m.group(), month_words, season_words, pivot_max)

            description = m.group()
            related: list[str] = []
            for s_start, s_end, s_text in sentences:
                if s_start <= abs_start < s_end:
                    description = s_text
                    related = [
                        ent.text for ent in mentions
                        if s_start <= ent.start_char < s_end
                    ]
                    break

            found[(abs_start, abs_end)] = TimelineEvent(
                doc_id=doc_id,
                chunk_id=chunk_id,
                date_text=m.group(),
                iso_date=iso,
                year=year,
                description=description.strip(),
                entities=sorted(set(related)),
                confidence=0.7 if iso else 0.4,
            )
    return list(found.values())
