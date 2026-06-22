# Smart document chunking with sentence-aligned overlap.

from __future__ import annotations

import re
from typing import Optional

from .schema import Chunk, Document, stable_id

# Fallback sentence splitter: split on ., !, ? followed by whitespace + capital.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def _fallback_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char spans for sentences using regex."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _SENT_RE.finditer(text):
        end = m.start()
        if end > pos:
            spans.append((pos, end))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans or [(0, len(text))]


def _spacy_sentences(text: str, nlp) -> list[tuple[int, int]]:
    """Return sentence char spans using a spaCy pipeline."""
    doc = nlp(text)
    spans = [(s.start_char, s.end_char) for s in doc.sents]
    return spans or [(0, len(text))]


def chunk_document(
    document: Document,
    max_chars: int = 6000,
    overlap_chars: int = 400,
    respect_sentences: bool = True,
    nlp: Optional[object] = None,
) -> list[Chunk]:
    """Split a document into overlapping, sentence-aligned chunks."""
    text = document.text
    if len(text) <= max_chars:
        return [
            Chunk(
                chunk_id=stable_id(document.doc_id, 0, prefix="ck_", length=10),
                doc_id=document.doc_id,
                index=0,
                text=text,
                start_char=0,
                end_char=len(text),
            )
        ]

    if respect_sentences:
        use_spacy = nlp is not None and len(text) < getattr(nlp, "max_length", 1_000_000)
        sent_spans = _spacy_sentences(text, nlp) if use_spacy else _fallback_sentences(text)
    else:
        # Treat the whole doc as one "sentence stream" split on hard boundaries.
        sent_spans = _fallback_sentences(text)

    # Pathological inputs (scraped pages, OCR dumps) can yield a "sentence" far
    # longer than max_chars with no boundary at all - hard-split those so no
    # chunk ever exceeds the cap (an oversize chunk silently overflows the LLM
    # context window downstream).
    # step must be > 0 or the hard-split spins forever; clamp in case a config
    # slips through with overlap_chars >= max_chars (the validator should catch it).
    step = max(1, max_chars - overlap_chars)
    bounded: list[tuple[int, int]] = []
    for s_start, s_end in sent_spans:
        while s_end - s_start > max_chars:
            bounded.append((s_start, s_start + max_chars))
            s_start += step
        bounded.append((s_start, s_end))
    sent_spans = bounded

    chunks: list[Chunk] = []
    cur_start = sent_spans[0][0]
    cur_end = cur_start
    idx = 0

    def _emit(start: int, end: int) -> None:
        nonlocal idx
        if end <= start:
            return
        chunks.append(
            Chunk(
                chunk_id=stable_id(document.doc_id, idx, prefix="ck_", length=10),
                doc_id=document.doc_id,
                index=idx,
                text=text[start:end],
                start_char=start,
                end_char=end,
            )
        )
        idx += 1

    for s_start, s_end in sent_spans:
        # If adding this sentence would overflow, emit current chunk first.
        if s_end - cur_start > max_chars and cur_end > cur_start:
            _emit(cur_start, cur_end)
            # Begin next chunk with overlap: back up ~overlap_chars but not past 0.
            overlap_start = max(cur_start, cur_end - overlap_chars)
            cur_start = overlap_start
        cur_end = s_end

    _emit(cur_start, cur_end)
    return chunks


def chunk_documents(
    documents: list[Document],
    max_chars: int = 6000,
    overlap_chars: int = 400,
    respect_sentences: bool = True,
    nlp: Optional[object] = None,
) -> dict[str, list[Chunk]]:
    """Chunk many documents; returns ``{doc_id: [chunks]}``."""
    return {
        d.doc_id: chunk_document(d, max_chars, overlap_chars, respect_sentences, nlp)
        for d in documents
    }
