# Recall pass: a second extraction over the WHOLE document, told which entities were
# found and which relations already exist, asked only for the ties the first pass
# MISSED. Chunk-by-chunk extraction can't see a relation whose endpoints fall in
# different chunks; this re-prompt over the assembled doc recovers them - the recall
# half of the L3X generate-then-scrutinize loop (verification is the precision half).
# Duck-types backend._complete; reuses the same _map_extraction + ontology/qualifier
# schema as the first pass, so new edges are indistinguishable downstream except for
# a recall_pass tag. A doc that fits the budget is one pass; a longer one (a book, a
# long transcript) is split into overlapping windows so length no longer forfeits
# recall. gemini_batch already sees whole docs, so this mainly lifts api/ollama.

from __future__ import annotations

import logging
from typing import Callable

from core.schema import EntityMention, Relationship

from .api_backend import _map_extraction
from .json_repair import repair_json
from .prompts import (
    qualifier_constraint_block,
    relation_constraint_block,
    relationship_schema_str,
)

logger = logging.getLogger(__name__)

_RECALL_SYSTEM = (
    "You find relations a first extraction pass MISSED. You receive the full document, "
    "the ENTITIES already identified in it, and the relations ALREADY extracted. Output "
    "only relations that (a) are stated in the document, (b) hold between two of the "
    "listed entities, and (c) are NOT already in the extracted list. Do not invent new "
    "entities. If you find none, return an empty relationships list."
)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def recall_relations(
    doc_text: str,
    mentions: list[EntityMention],
    existing: list[Relationship],
    complete: Callable[[str, str], str],
    *,
    label_types: list[str],
    relation_types: list[str],
    relation_guide: dict,
    edge_qualifiers: list[str],
    type_signatures: dict,
    doc_id: str,
    date_vocab=({}, {}, None),
    max_chars: int = 24000,
    overlap: int = 2000,
    max_windows: int = 60,
) -> list[Relationship]:
    """Re-prompt for missed relations among the doc's entities. Returns the NEW ones
    (tagged recall_pass), endpoints constrained to the known entity set, deduped
    against ``existing``. A doc that fits ``max_chars`` is a single pass; a longer one
    (a book, a long transcript) is split into overlapping windows, each prompted with
    its window-local entities - so length no longer means no recall (the old behavior
    was to skip). Cross-window ties beyond ``overlap`` are still out of reach, but
    those are rare in real text; the within-window cross-chunk ties are the recall."""
    if not doc_text:
        return []
    have: set = {(_norm(r.source), _norm(r.target), (r.rel_type or "").lower())
                 for r in existing}

    if len(doc_text) <= max_chars:
        windows = [(0, len(doc_text))]
    else:
        step = max(1, max_chars - max(0, overlap))
        starts = range(0, len(doc_text), step)
        windows = [(s, min(s + max_chars, len(doc_text))) for s in starts][:max_windows]

    out: list[Relationship] = []
    months, seasons, pivot = date_vocab
    for ws, we in windows:
        wtext = doc_text[ws:we]
        # Window-local entities by doc-absolute offset (whole-doc set when single pass).
        if len(windows) == 1:
            wmentions = mentions
        else:
            wmentions = [m for m in mentions
                         if ws <= getattr(m, "start_char", 0) < we]
        out.extend(_recall_window(
            wtext, wmentions, existing, complete, have,
            label_types=label_types, relation_types=relation_types,
            relation_guide=relation_guide, edge_qualifiers=edge_qualifiers,
            type_signatures=type_signatures, doc_id=doc_id,
            date_vocab=(months, seasons, pivot)))
    if out:
        logger.info("Recall pass: +%d missed relation(s) in %s (%d window(s)).",
                    len(out), doc_id, len(windows))
    return out


def _recall_window(
    doc_text, mentions, existing, complete, have, *,
    label_types, relation_types, relation_guide, edge_qualifiers,
    type_signatures, doc_id, date_vocab,
) -> list[Relationship]:
    """One recall prompt over one text window. Mutates ``have`` to dedup across windows."""
    names: dict[str, str] = {}   # normalized -> display, the closed entity set
    for m in mentions:
        nm = (m.text or "").strip()
        if nm and m.label in label_types:
            names.setdefault(_norm(nm), nm)
    if len(names) < 2:
        return []

    ent_list = ", ".join(f"{d} [{lbl}]" for d, lbl in
                         ((names[k], _label_of(k, mentions)) for k in names))
    rel_lines = "\n".join(
        f"- {r.source} {r.rel_type} {r.target}" for r in existing[:200]) or "(none)"
    schema = relationship_schema_str(edge_qualifiers)
    rel_block = relation_constraint_block(relation_types, relation_guide, type_signatures)
    qual_block = qualifier_constraint_block(edge_qualifiers)
    user = (
        f"ENTITIES (use only these): {ent_list}\n\n"
        f"ALREADY EXTRACTED (do not repeat):\n{rel_lines}\n\n"
        f"DOCUMENT:\n{doc_text}\n\n"
        f"Return ONLY missed relations as JSON: {schema}{rel_block}{qual_block}"
    )
    try:
        raw = complete(_RECALL_SYSTEM, user)
    except Exception as exc:  # noqa: BLE001 - recall is best-effort, never fatal
        logger.warning("recall pass failed for %s: %s", doc_id, exc)
        return []
    data = repair_json(raw)
    if not isinstance(data, dict):
        return []
    months, seasons, pivot = date_vocab
    _, rels, _ = _map_extraction(
        data, [], doc_id, doc_id, label_types, months, seasons, pivot,
        chunk_text=doc_text, qualifiers=edge_qualifiers or None)
    out: list[Relationship] = []
    for r in rels:
        ks, kt = _norm(r.source), _norm(r.target)
        if ks not in names or kt not in names:
            continue  # endpoint isn't a known entity - a hallucinated node
        key = (ks, kt, (r.rel_type or "").lower())
        if key in have:
            continue  # already had it (this doc or an earlier window)
        have.add(key)
        r.attributes["recall_pass"] = True
        out.append(r)
    return out


def _label_of(norm_name: str, mentions: list[EntityMention]) -> str:
    for m in mentions:
        if _norm(m.text) == norm_name:
            return m.label
    return "ENTITY"
