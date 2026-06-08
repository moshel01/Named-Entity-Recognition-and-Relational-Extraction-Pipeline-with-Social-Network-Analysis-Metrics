# Dependency-parse relationship patterns for the python-only backend (Mode 2).

from __future__ import annotations

from typing import Optional

from core.schema import EntityMention, Relationship

# Verb lemma -> normalized relation type. Extend per domain as needed.
_VERB_LEXICON: dict[str, tuple[str, bool]] = {
    # lemma: (rel_type, directed)
    "work": ("works_for", True),
    "employ": ("employs", True),
    "lead": ("leads", True),
    "head": ("leads", True),
    "found": ("founded", True),
    "establish": ("founded", True),
    "join": ("member_of", True),
    "belong": ("member_of", True),
    "own": ("owns", True),
    "acquire": ("acquired", True),
    "buy": ("acquired", True),
    "meet": ("met_with", False),
    "marry": ("married_to", False),
    "visit": ("visited", True),
    "attend": ("attended", True),
    "represent": ("represents", True),
    "support": ("supports", True),
    "fund": ("funded", True),
    "finance": ("funded", True),
    "appoint": ("appointed", True),
    "report": ("reports_to", True),
    "live": ("located_in", True),
    "locate": ("located_in", True),
}

_SUBJECT_DEPS = {"nsubj", "nsubjpass", "agent"}
_OBJECT_DEPS = {"dobj", "obj", "pobj", "dative", "attr", "obl"}


def _find_entity(token_start: int, token_end: int,
                 mentions_by_span: list[tuple[int, int, EntityMention]]) -> Optional[EntityMention]:
    """Return a mention whose span overlaps [token_start, token_end)."""
    for s, e, m in mentions_by_span:
        if s < token_end and token_start < e:
            return m
    return None


def extract_dependency_relations(
    spacy_doc,
    mentions: list[EntityMention],
    chunk_id: str,
    doc_id: str,
    offset: int = 0,
    base_confidence: float = 0.55,
) -> list[Relationship]:
    """Extract SVO relations linking entity mentions via the dependency parse."""
    if not spacy_doc.has_annotation("DEP"):
        return []

    # Index mentions by their char span relative to the chunk text.
    spans: list[tuple[int, int, EntityMention]] = [
        (m.start_char - offset, m.end_char - offset, m) for m in mentions
    ]

    relations: list[Relationship] = []
    seen: set[tuple[str, str, str]] = set()

    for token in spacy_doc:
        if token.pos_ != "VERB":
            continue
        lemma = token.lemma_.lower()
        subjects: list[EntityMention] = []
        objects: list[EntityMention] = []

        for child in token.children:
            ent = _find_entity(child.left_edge.idx,
                               child.right_edge.idx + len(child.right_edge),
                               spans)
            if ent is None:
                continue
            if child.dep_ in _SUBJECT_DEPS:
                subjects.append(ent)
            elif child.dep_ in _OBJECT_DEPS:
                objects.append(ent)
            elif child.dep_ == "prep":
                for gchild in child.children:
                    gent = _find_entity(
                        gchild.left_edge.idx,
                        gchild.right_edge.idx + len(gchild.right_edge),
                        spans,
                    )
                    if gent is not None and gchild.dep_ in _OBJECT_DEPS:
                        objects.append(gent)

        if not subjects or not objects:
            continue

        rel_type, directed = _VERB_LEXICON.get(lemma, (lemma, True))
        sentence = token.sent.text.strip()
        for s in subjects:
            for o in objects:
                if s.text.strip().lower() == o.text.strip().lower():
                    continue
                key = (s.text.lower(), rel_type, o.text.lower())
                if key in seen:
                    continue
                seen.add(key)
                relations.append(
                    Relationship(
                        source=s.text.strip(),
                        target=o.text.strip(),
                        rel_type=rel_type,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        evidence=sentence,
                        confidence=base_confidence,
                        directed=directed,
                        origin="extracted",
                        attributes={"verb": lemma, "edge_source": "rule_extracted"},
                    )
                )
    return relations
