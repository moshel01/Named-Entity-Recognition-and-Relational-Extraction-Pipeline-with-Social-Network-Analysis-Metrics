# Merge GLiNER and spaCy entity mentions, resolving span overlaps.

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from .schema import EntityMention

# Confidence boost applied when both NER systems agree on (roughly) a span.
_AGREEMENT_BOOST = 0.15

# Closed-class words that are never part of a name when they lead or trail a
# span ("in Fili", "Bofur and"). Articles are NOT stripped - they are often
# part of real names ("The Shire", "Der Stahlhelm") - and name particles
# (von, van, de) stay untouched.
_FUNCTION_WORDS = {
    "in", "on", "at", "to", "by", "with", "for", "from", "as",
    "and", "or",
    "und", "oder", "im", "am", "beim", "bei", "mit", "zu", "zur", "zum",
}
# Conjunctions that split a two-person span ("Bofur and Bombur").
_CONJ_SPLIT = re.compile(r"\s+(?:and|und|&)\s+")
# Generic words that signal the conjunct is not a person ("Thorin and Company").
_NON_PERSON_CONJUNCTS = {"company", "co", "co.", "sons", "söhne", "others", "family"}


def _looks_like_name(part: str) -> bool:
    toks = part.split()
    return (0 < len(toks) <= 3
            and all(t[:1].isupper() for t in toks)
            and toks[-1].lower() not in _NON_PERSON_CONJUNCTS)


def repair_spans(mentions: list[EntityMention]) -> list[EntityMention]:
    """Fix common NER span errors before they become entities.

    - Strip leading/trailing function words ("in Fili" -> "Fili", "Bofur and"
      -> "Bofur"), adjusting char offsets.
    - Split a PERSON span joining two names with a conjunction
      ("Bofur and Bombur" -> "Bofur" + "Bombur"). Skipped when a conjunct is a
      generic group word ("Thorin and Company") - that is one entity.
    Junk spans like these otherwise become dedup attractors that swallow the
    individual names as aliases.
    """
    out: list[EntityMention] = []
    for m in mentions:
        text = m.text
        start, end = m.start_char, m.end_char

        # Leading function words.
        changed = True
        while changed:
            changed = False
            toks = text.split()
            if len(toks) >= 2 and toks[0].lower() in _FUNCTION_WORDS:
                cut = len(toks[0])
                rest = text[cut:]
                start += cut + (len(rest) - len(rest.lstrip()))
                text = rest.lstrip()
                changed = True
            elif len(toks) >= 2 and toks[-1].lower() in _FUNCTION_WORDS:
                cut = len(toks[-1])
                kept = text[: len(text) - cut]
                end -= cut + (len(kept) - len(kept.rstrip()))
                text = kept.rstrip()
                changed = True
        if not text:
            continue

        # Conjunction split (PERSON only).
        if m.label == "PERSON":
            parts = _CONJ_SPLIT.split(text)
            if len(parts) == 2 and all(_looks_like_name(p) for p in parts):
                pos = start
                for part in parts:
                    p_start = m.start_char + m.text.find(part) if part in m.text else pos
                    out.append(replace(m, text=part, start_char=p_start,
                                       end_char=p_start + len(part),
                                       sources=list(m.sources),
                                       attributes=dict(m.attributes or {})))
                    pos = p_start + len(part)
                continue

        if text != m.text:
            out.append(replace(m, text=text, start_char=start, end_char=end))
        else:
            out.append(m)
    return out


def _overlaps(a: EntityMention, b: EntityMention) -> bool:
    return a.start_char < b.end_char and b.start_char < a.end_char


def _label_compatible(a: str, b: str) -> bool:
    """Whether two labels are close enough to merge across systems."""
    return a == b


def _dominant(group: list[EntityMention]) -> EntityMention:
    """Choose the representative mention for an overlap group.

    Preference: higher confidence, then wider span, then GLiNER over spaCy.
    """
    def key(m: EntityMention) -> tuple:
        width = m.end_char - m.start_char
        gliner = 1 if "gliner" in m.sources else 0
        return (round(m.confidence, 3), width, gliner)

    return max(group, key=key)


def merge_mentions(
    mentions: Iterable[EntityMention],
    sentence_lookup: Iterable[tuple[int, int, str]] | None = None,
) -> list[EntityMention]:
    """Merge overlapping mentions from multiple NER sources."""
    items = sorted(mentions, key=lambda m: (m.start_char, m.end_char))
    if not items:
        return []

    # Build overlap groups via a simple sweep.
    groups: list[list[EntityMention]] = []
    current: list[EntityMention] = [items[0]]
    current_end = items[0].end_char
    for m in items[1:]:
        if m.start_char < current_end and any(
            _label_compatible(m.label, g.label) for g in current
        ):
            current.append(m)
            current_end = max(current_end, m.end_char)
        else:
            groups.append(current)
            current = [m]
            current_end = m.end_char
    groups.append(current)

    merged: list[EntityMention] = []
    sentences = list(sentence_lookup) if sentence_lookup else []

    for group in groups:
        rep = _dominant(group)
        all_sources: set[str] = set()
        for g in group:
            all_sources.update(g.sources)
        rep.sources = sorted(all_sources)

        if "gliner" in all_sources and "spacy" in all_sources:
            rep.confidence = min(1.0, rep.confidence + _AGREEMENT_BOOST)

        if not rep.sentence and sentences:
            for s_start, s_end, s_text in sentences:
                if s_start <= rep.start_char < s_end:
                    rep.sentence = s_text
                    break
        merged.append(rep)

    return merged
