# Reconcile span-less LLM entities (gemini_batch) against local GLiNER/spaCy NER.
#
# gemini_batch entities arrive with (0,0) spans - the model returns names, not
# offsets - so the within-document proximity co-occurrence floor (canonical_inference
# skips span-less mentions) and verbatim evidence grounding are dead. This re-runs
# the local NER per document AFTER extraction and folds the spans back in:
#
#   - a GLiNER mention whose normalized name matches an LLM entity is RELABELED to
#     the LLM's type and added (span transfer: it aggregates into the same node -
#     same (normalize_name, label) key - which now carries a real position).
#   - an unmatched GLiNER mention is added as a recall net (gated by add_missed,
#     tagged ner_only so it stays auditable/filterable in Gephi).
#
# The LLM stays the PRIMARY extractor; GLiNER is a post-hoc second opinion, never
# priming the prompt. Priming a strong model with GLiNER candidates anchors it to
# GLiNER's noise (measured: ollama entity F1 == GLiNER's standalone F1 - the LLM
# rubber-stamped the candidate list), so reconciliation is deliberately post-hoc.

from __future__ import annotations

import logging
from typing import Callable

from core.schema import DocumentExtraction, EntityMention

from .aggregator import normalize_name

logger = logging.getLogger(__name__)

# A callable that returns span-grounded mentions (doc-absolute offsets) for a doc.
NerFn = Callable[[str, str], list[EntityMention]]


def _respan(m: EntityMention, label: str, source: str) -> EntityMention:
    """A copy of a GLiNER mention relabeled + provenance-tagged for reconciliation."""
    return EntityMention(
        text=m.text, label=label,
        start_char=m.start_char, end_char=m.end_char,
        chunk_id=m.chunk_id, doc_id=m.doc_id,
        confidence=m.confidence, sources=[source],
        sentence=m.sentence,
        attributes={**(m.attributes or {}), "reconciled": source},
    )


def reconcile_spans(
    extractions: list[DocumentExtraction],
    doc_texts: dict[str, str],
    ner_fn: NerFn,
    *,
    add_missed: bool = True,
) -> dict[str, int]:
    """Fold local NER spans into span-less LLM extractions. Mutates in place.

    For each extraction, runs ``ner_fn(doc_id, text)`` to get span-grounded
    mentions, then adds: span-transfer mentions (GLiNER name matches an LLM entity,
    relabeled to the LLM type) and - when ``add_missed`` - GLiNER-only mentions as a
    recall net. Returns counts {transferred, added, docs}.
    """
    transferred = added = touched = 0
    for ex in extractions:
        text = doc_texts.get(ex.doc_id, "")
        if not text:
            continue
        ner = ner_fn(ex.doc_id, text)
        if not ner:
            continue
        # LLM entity normalized-name -> its type (first mention wins).
        llm_label: dict[str, str] = {}
        for m in ex.mentions:
            llm_label.setdefault(normalize_name(m.text), m.label)
        extra: list[EntityMention] = []
        for g in ner:
            if g.end_char <= g.start_char:
                continue  # need a real span to be worth adding
            nn = normalize_name(g.text)
            if not nn:
                continue
            if nn in llm_label:
                extra.append(_respan(g, llm_label[nn], "ner_reconciled"))
                transferred += 1
            elif add_missed:
                extra.append(_respan(g, g.label, "ner_only"))
                added += 1
        if extra:
            ex.mentions.extend(extra)
            touched += 1
    logger.info(
        "Span reconciliation: %d spans transferred to LLM entities, %d NER-only "
        "mentions added (recall net), across %d documents.",
        transferred, added, touched,
    )
    return {"transferred": transferred, "added": added, "docs": touched}
