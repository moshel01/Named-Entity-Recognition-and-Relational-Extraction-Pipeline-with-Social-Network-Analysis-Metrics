# Always-on base pass: spaCy + GLiNER -> merge -> coref -> dates.

from __future__ import annotations

import logging
import re

from config import Config

from .chunker import chunk_document
from .coreference import CoreferenceResolver
from .date_extractor import extract_dates
from .entity_merger import merge_mentions, repair_spans
from .gliner_engine import GlinerEngine
from .schema import Chunk, Document, FoundationResult
from .spacy_engine import SpacyEngine

logger = logging.getLogger(__name__)

# Detect the author's own name from a first-person document's opening, so the
# narrator node becomes a real person ("Johann Alff") instead of a generic
# "Narrator [doc]" placeholder. Name capture is case-sensitive (must be
# Titlecase); only the lead-in phrase is case-insensitive.
# Inter-token spacing is [ \t]+ (never newlines, so a title doesn't bleed into the
# body); no "." in the class so "Brandt." doesn't glue onto the next word.
_NAME_RE = r"([A-ZÄÖÜ][A-Za-zäöüßÄÖÜ'\-]+(?:[ \t]+[A-ZÄÖÜ][A-Za-zäöüßÄÖÜ'\-]+){0,2})"
_AUTHOR_PATTERNS = [
    re.compile(r"(?i:\b(?:memoir|diary|diaries|autobiography|recollections?|"
               r"testimony|account)\s+of\s+)" + _NAME_RE),
    re.compile(r"(?i:\bI\s+am\s+)" + _NAME_RE),
    re.compile(r"(?i:\bI,\s*)" + _NAME_RE + r"\s*,"),
    re.compile(r"(?i:\bmy\s+name\s+is\s+)" + _NAME_RE),
    re.compile(r"(?i:\bich\s+bin\s+)" + _NAME_RE),
    re.compile(r"(?i:\bmein\s+name\s+ist\s+)" + _NAME_RE),
    re.compile(r"(?i:\bich,\s*)" + _NAME_RE + r"\s*,"),
]
_NAME_STOP = {"the", "memoir", "diary", "i", "my", "name", "is", "am", "of",
              "ich", "bin", "mein", "ist", "a", "an", "we"}


def _detect_author_from_text(text: str) -> str:
    """Best-effort author name from a document's opening; '' if none confident."""
    head = (text or "")[:400]
    for pat in _AUTHOR_PATTERNS:
        m = pat.search(head)
        if not m:
            continue
        name = " ".join(m.group(1).split()).strip(" ,.;:")
        toks = name.split()
        if not (1 <= len(toks) <= 3):
            continue
        if any(t.lower() in _NAME_STOP for t in toks):
            continue
        if len(toks) == 1 and len(toks[0]) < 3:
            continue
        return name
    return ""


def _annotate_propn_ratio(spacy_doc, mentions, offset: int) -> None:
    """Tag each mention with the share of its tokens spaCy tags PROPN.

    Language-general common-noun signal: an entity that is never a proper noun
    anywhere in the corpus ("Monsieur", "der Vater") is a category word, not a
    name - regardless of capitalization conventions (German capitalizes all
    nouns). Averaged per entity in aggregation; consumed by the quality POS
    gate. No-op for pipelines without a POS tagger (blank fallback models).
    """
    if not spacy_doc.has_annotation("POS"):
        return
    for m in mentions:
        span = spacy_doc.char_span(
            m.start_char - offset, m.end_char - offset, alignment_mode="expand"
        )
        if span is None or not len(span):
            continue
        propn = sum(1 for t in span if t.pos_ == "PROPN")
        m.attributes["propn_ratio"] = round(propn / len(span), 3)


class FoundationLayer:
    """Always-on NER + linguistic analysis foundation.

    A ``domain`` plugin may override the GLiNER labels / label map and inject
    spaCy EntityRuler patterns, so domain knowledge influences extraction at the
    foundation level rather than only in post-processing.
    """

    def __init__(self, config: Config, domain=None) -> None:
        self.config = config
        self.domain = domain
        fc = config.foundation
        prefer_gpu = fc.device in ("auto", "cuda", "mps")

        # Domain may override the zero-shot labels and the label->type map.
        labels = fc.gliner_labels
        label_map = dict(fc.label_map)
        if domain is not None:
            dom_labels = domain.gliner_labels()
            if dom_labels:
                labels = dom_labels
                logger.info("Domain '%s' supplied %d GLiNER labels.",
                            getattr(domain, "name", "?"), len(dom_labels))
            label_map.update(domain.gliner_label_map())

        self.spacy = SpacyEngine(
            model=fc.spacy_model,
            disable=fc.spacy_disable,
            prefer_gpu=prefer_gpu,
        )
        if domain is not None:
            self._install_entity_ruler(domain.spacy_patterns())

        self.gliner = GlinerEngine(
            model_name=fc.gliner_model,
            labels=labels,
            threshold=fc.gliner_threshold,
            device=fc.device,
            label_map=label_map,
        )
        self.use_spacy_ner = fc.use_spacy_ner and self.spacy.has_ner
        # Canonical types the pipeline was asked to produce. spaCy's statistical
        # NER and the EntityRuler can emit types beyond these; filter them out.
        self.allowed_types: set[str] | None = (
            set(label_map.values()) if fc.restrict_to_label_types else None
        )
        self._exclude_spacy: set[str] = set(fc.exclude_spacy_labels)

        # Coreference resolver (narrator + optional pronoun resolution).
        self.coref = CoreferenceResolver(config, domain=domain) \
            if config.coreference.enabled else None

        # Domain-supplied date vocabulary (e.g. German months/seasons).
        vocab = domain.temporal_vocab() if domain is not None else {"months": {}, "seasons": {}}
        self._month_words = vocab.get("months", {})
        self._season_words = vocab.get("seasons", {})
        self._pivot_max = vocab.get("pivot_max")

    def _install_entity_ruler(self, patterns: list[dict]) -> None:
        """Add a domain EntityRuler before the statistical NER component."""
        if not patterns:
            return
        nlp = self.spacy.nlp
        try:
            if "entity_ruler" in nlp.pipe_names:
                nlp.remove_pipe("entity_ruler")
            before = "ner" if nlp.has_pipe("ner") else None
            ruler = nlp.add_pipe(
                "entity_ruler",
                before=before,
                config={"overwrite_ents": False, "validate": True,
                        "phrase_matcher_attr": "LOWER"},
            )
            ruler.add_patterns(patterns)
            logger.info("Installed %d domain spaCy EntityRuler patterns.", len(patterns))
        except Exception as exc:  # noqa: BLE001 - never let patterns abort the run
            logger.warning("Could not install domain EntityRuler patterns: %s", exc)

    # Chunking
    def chunk(self, document: Document) -> list[Chunk]:
        cc = self.config.chunking
        return chunk_document(
            document,
            max_chars=cc.max_chars,
            overlap_chars=cc.overlap_chars,
            respect_sentences=cc.respect_sentences,
            nlp=self.spacy.nlp,
        )

    # Per-chunk processing
    def process_chunk(self, chunk: Chunk, narrator_name: str = "") -> FoundationResult:
        """Run spaCy + GLiNER + coref + merge + dates on a single chunk."""
        text = chunk.text
        offset = chunk.start_char

        # Sentence spans (document-absolute) for evidence + association.
        spacy_doc = self.spacy(text)
        sent_spans: list[tuple[int, int, str]] = [
            (offset + s.start_char, offset + s.end_char, s.text.strip())
            for s in spacy_doc.sents
            if s.text.strip()
        ]
        sentences = [s[2] for s in sent_spans]

        # GLiNER mentions.
        mentions = self.gliner.extract(text, chunk.chunk_id, chunk.doc_id, offset)

        # spaCy statistical NER (optional).
        if self.use_spacy_ner:
            mentions += self.spacy.spacy_entities(
                text, chunk.chunk_id, chunk.doc_id, offset,
                exclude_labels=self._exclude_spacy,
            )

        merged = merge_mentions(mentions, sentence_lookup=sent_spans)
        merged = repair_spans(merged)

        # Restrict to requested canonical types (drops spaCy's off-target NER
        # such as DATE/EVENT/ORDINAL when those weren't configured).
        if self.allowed_types is not None:
            merged = [m for m in merged if m.label in self.allowed_types]

        _annotate_propn_ratio(spacy_doc, merged, offset)

        # Coreference: narrator + (optional) pronoun resolution. Appended after
        # merge so resolved spans (e.g. first-person pronouns) become candidates
        # for relationship extraction and co-occurrence. Pronoun resolution must
        # run even without a narrator (third-person books have no narrator).
        if self.coref is not None:
            merged += self.coref.resolve(
                text, merged, chunk.doc_id, chunk.chunk_id, offset, narrator_name
            )

        dates = extract_dates(
            text,
            chunk.chunk_id,
            chunk.doc_id,
            offset=offset,
            sentences=sent_spans,
            mentions=merged,
            month_words=self._month_words,
            season_words=self._season_words,
            pivot_max=self._pivot_max,
        )

        return FoundationResult(
            chunk=chunk,
            mentions=merged,
            dates=dates,
            sentences=sentences,
        )

    # Per-document processing
    def _narrator_name(self, document: Document) -> str:
        # Domain can name the author from the filename (Abel: real author name).
        fn = document.meta.get("filename") or document.doc_id
        if self.domain is not None:
            name = self.domain.narrator_name(fn, document.doc_id)
            if name:
                return name
        # No domain/filename author name (generic path): read it from the text.
        detected = _detect_author_from_text(document.text)
        if detected:
            return detected
        stem = fn.rsplit(".", 1)[0] if "." in fn else fn
        return f"Narrator [{stem}]"

    def process_document(self, document: Document) -> list[FoundationResult]:
        """Chunk a document and run the foundation layer on each chunk."""
        chunks = self.chunk(document)
        narrator_name = self._narrator_name(document)
        results = [self.process_chunk(c, narrator_name=narrator_name) for c in chunks]
        n_ent = sum(len(r.mentions) for r in results)
        logger.debug(
            "Foundation: %s -> %d chunks, %d mentions",
            document.doc_id, len(chunks), n_ent,
        )
        return results
