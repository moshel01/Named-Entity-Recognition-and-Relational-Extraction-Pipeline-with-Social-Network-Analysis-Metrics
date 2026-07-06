# German-language NLP utilities for name and term normalization.

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Noble / nobiliary particles
NOBLE_PARTICLES: set[str] = {
    "von", "zu", "vom", "zum", "zur", "van", "de", "del", "della", "di",
    "von und zu", "von der", "von dem", "auf",
}
# Multi-word particles checked first (longest match).
_MULTIWORD_PARTICLES: list[str] = ["von und zu", "von der", "von dem"]

# Honorific titles (stripped from the front of a name)
TITLES: set[str] = {
    "dr", "dr.", "prof", "prof.", "professor", "herr", "frau", "fräulein",
    "general", "oberst", "major", "hauptmann", "leutnant", "graf", "gräfin",
    "freiherr", "baron", "fürst", "ritter", "pfarrer", "pastor",
}

# Generational / academic suffixes
SUFFIXES: set[str] = {"sen.", "sr.", "jun.", "jr.", "der ältere", "der jüngere", "i", "ii", "iii"}

# Umlaut / ß transliteration
_UMLAUT_MAP = {
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
}


def normalize_umlauts(text: str) -> str:
    """Transliterate umlauts and ß (ä->ae, ö->oe, ü->ue, ß->ss)."""
    for src, dst in _UMLAUT_MAP.items():
        text = text.replace(src, dst)
    return text


# Abbreviation expansion
ABBREVIATIONS: dict[str, str] = {
    "str.": "Straße",
    "str": "Straße",
    "pl.": "Platz",
    "hbf.": "Hauptbahnhof",
    "hbf": "Hauptbahnhof",
    "bhf.": "Bahnhof",
    "geb.": "geboren",          # born
    "gest.": "gestorben",       # died
    "verh.": "verheiratet",     # married
    "ev.": "evangelisch",
    "kath.": "katholisch",
    "jg.": "Jahrgang",
    "nr.": "Nummer",
    "abt.": "Abteilung",
    "rgt.": "Regiment",
    "btl.": "Bataillon",
    "kp.": "Kompanie",
    "div.": "Division",
    "a.d.": "an der",
    "a.m.": "am Main",
    "b.": "bei",
}

_TOKEN_SPLIT = re.compile(r"\s+")


def expand_german_abbreviation(token: str) -> str:
    """Expand a single abbreviation token (case-insensitive); else return as-is."""
    return ABBREVIATIONS.get(token.lower().strip(), token)


def expand_abbreviations(text: str) -> str:
    """Expand all known abbreviations in a string, token by token."""
    return " ".join(expand_german_abbreviation(t) for t in _TOKEN_SPLIT.split(text.strip()) if t)


# Full-name parsing
@dataclass
class ParsedName:
    """Structured decomposition of a German personal name."""

    raw: str
    titles: list[str] = field(default_factory=list)
    first: str = ""
    middle: list[str] = field(default_factory=list)
    particle: str = ""
    last: str = ""
    suffix: str = ""

    @property
    def canonical(self) -> str:
        """Reconstruct ``First [particle] Last`` (no titles), best effort."""
        parts: list[str] = []
        if self.first:
            parts.append(self.first)
        parts.extend(self.middle)
        if self.particle:
            parts.append(self.particle)
        if self.last:
            parts.append(self.last)
        return " ".join(parts).strip() or self.raw

    @property
    def surname_key(self) -> str:
        """Particle + last name, for grouping/blocking (e.g. 'von hindenburg')."""
        key = f"{self.particle} {self.last}".strip()
        return normalize_umlauts(key).lower()


def author_from_filename(filename: str) -> str | None:
    # Abel files are "<Author Name><hoover_id>.rtf" e.g. "August Spanku239694.rtf".
    # Strip extension + trailing id digits -> author name.
    # Anonymous letters are filed "unknown<id>.rtf" - that's a label, not a
    # name. Returning it would hand six different anonymous authors the same
    # "person" and fuse them into one fake hub; None falls back to the unique
    # per-doc "Narrator [stem]" placeholder.
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    stem = re.sub(r"\d+$", "", stem).strip()
    if stem.lower() in {"unknown", "unbekannt", "anonym", "anonymous", "nn", "n.n"}:
        return None
    return stem or None


def parse_german_full_name(name: str) -> ParsedName:
    """Parse a German full name into structured components.

    Recognizes leading honorific titles, multi-word noble particles, and
    trailing generational suffixes. Robust to ``"Surname, Given"`` ordering.
    """
    raw = name.strip()
    parsed = ParsedName(raw=raw)
    if not raw:
        return parsed

    # Handle "Last, First" ordering by flipping.
    if "," in raw:
        left, _, right = raw.partition(",")
        right = right.strip()
        # If right side looks like a suffix, keep as suffix.
        if right.lower().strip(". ") in {s.strip(". ") for s in SUFFIXES}:
            parsed.suffix = right
            raw = left.strip()
        else:
            raw = f"{right} {left}".strip()

    tokens = [t for t in _TOKEN_SPLIT.split(raw) if t]

    # Strip leading titles.
    while tokens and tokens[0].lower().strip(".") in {t.strip(".") for t in TITLES}:
        parsed.titles.append(tokens.pop(0))

    # Strip trailing suffix.
    if tokens and tokens[-1].lower() in SUFFIXES:
        parsed.suffix = tokens.pop()

    if not tokens:
        return parsed

    # Detect a noble particle: find the longest multiword particle, else single.
    lowered = [t.lower() for t in tokens]
    particle_idx = -1
    particle_len = 0
    joined = " ".join(lowered)
    for mw in _MULTIWORD_PARTICLES:
        pos = joined.find(mw)
        if pos != -1:
            # token index where mw begins
            prefix_tokens = joined[:pos].split()
            particle_idx = len(prefix_tokens)
            particle_len = len(mw.split())
            break
    if particle_idx == -1:
        for i, tok in enumerate(lowered):
            if tok in NOBLE_PARTICLES and i > 0:  # particle precedes surname
                particle_idx = i
                particle_len = 1
                break

    if particle_idx != -1:
        parsed.first = tokens[0]
        parsed.middle = tokens[1:particle_idx]
        parsed.particle = " ".join(tokens[particle_idx:particle_idx + particle_len])
        parsed.last = " ".join(tokens[particle_idx + particle_len:])
    else:
        parsed.first = tokens[0]
        parsed.last = tokens[-1] if len(tokens) > 1 else ""
        parsed.middle = tokens[1:-1] if len(tokens) > 2 else []

    return parsed
