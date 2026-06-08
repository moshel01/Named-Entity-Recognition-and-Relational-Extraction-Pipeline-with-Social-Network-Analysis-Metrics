# Always-on base pass: spaCy + GLiNER -> merge -> coref -> dates.

from __future__ import annotations

import logging

from config import Config

from .chunker import chunk_document
from .coreference import CoreferenceResolver
from .date_extractor import extract_dates
from .entity_merger import merge_mentions
from .gliner_engine import GlinerEngine
from .schema import Chunk, Document, FoundationResult
from .spacy_engine import SpacyEngine

logger = logging.getLogger(__name__)


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

        # Restrict to requested canonical types (drops spaCy's off-target NER
        # such as DATE/EVENT/ORDINAL when those weren't configured).
        if self.allowed_types is not None:
            merged = [m for m in merged if m.label in self.allowed_types]

        # Coreference: narrator + (optional) pronoun resolution. Appended after
        # merge so resolved spans (e.g. first-person pronouns) become candidates
        # for relationship extraction and co-occurrence.
        if self.coref is not None and narrator_name:
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
