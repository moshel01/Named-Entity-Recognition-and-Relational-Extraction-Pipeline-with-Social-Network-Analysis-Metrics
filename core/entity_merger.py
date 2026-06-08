# Merge GLiNER and spaCy entity mentions, resolving span overlaps.

from __future__ import annotations

from typing import Iterable

from .schema import EntityMention

# Confidence boost applied when both NER systems agree on (roughly) a span.
_AGREEMENT_BOOST = 0.15


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
