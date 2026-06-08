# spaCy wrapper: sentences + statistical NER. Falls back trf -> sm -> blank.

from __future__ import annotations

import logging
from typing import Optional

from .schema import EntityMention

logger = logging.getLogger(__name__)

# spaCy NER label -> canonical type. Used when use_spacy_ner is enabled.
_SPACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORG": "ORG",
    "NORP": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "EVENT": "EVENT",
}

# Labels a domain EntityRuler may emit that are already canonical pipeline types
# and should pass through unmapped.
_CANONICAL_PASSTHROUGH = {
    "PERSON", "ORG", "LOCATION", "EVENT", "RANK", "DATE", "INSTITUTION",
}


class SpacyEngine:
    """Thin wrapper around a loaded spaCy ``Language`` pipeline."""

    def __init__(
        self,
        model: str = "en_core_web_trf",
        disable: Optional[list[str]] = None,
        prefer_gpu: bool = False,
        max_length: int = 2_000_000,
    ) -> None:
        self.model_name = model
        self.disable = disable or []
        self.nlp = self._load(model, self.disable, prefer_gpu)
        self.nlp.max_length = max_length

    # Loading
    @staticmethod
    def _load(model: str, disable: list[str], prefer_gpu: bool):
        import spacy

        if prefer_gpu:
            try:
                spacy.prefer_gpu()
            except Exception:  # noqa: BLE001
                logger.debug("spaCy GPU unavailable; using CPU.")

        for candidate in (model, "en_core_web_sm"):
            try:
                nlp = spacy.load(candidate, disable=disable)
                if candidate != model:
                    logger.warning(
                        "spaCy model '%s' unavailable; fell back to '%s'.",
                        model, candidate,
                    )
                return nlp
            except Exception:  # noqa: BLE001
                continue

        logger.warning(
            "No spaCy model installed; using blank 'en' pipeline with sentencizer. "
            "Install with: python -m spacy download en_core_web_sm"
        )
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp

    @property
    def max_length(self) -> int:
        return self.nlp.max_length

    @property
    def has_ner(self) -> bool:
        return self.nlp.has_pipe("ner")

    @property
    def has_parser(self) -> bool:
        return self.nlp.has_pipe("parser") or self.nlp.has_pipe("senter")

    # Core API
    def __call__(self, text: str):
        """Run the full pipeline; returns a spaCy ``Doc``."""
        return self.nlp(text)

    def sentences(self, text: str) -> list[str]:
        """Return sentence strings for ``text``."""
        doc = self.nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]

    def spacy_entities(
        self,
        text: str,
        chunk_id: str,
        doc_id: str,
        offset: int = 0,
        exclude_labels: Optional[set[str]] = None,
    ) -> list[EntityMention]:
        """Extract statistical NER mentions, mapped to canonical types."""
        if not self.has_ner:
            return []
        exclude_labels = exclude_labels or set()
        doc = self.nlp(text)
        mentions: list[EntityMention] = []
        for ent in doc.ents:
            if ent.label_ in exclude_labels:
                continue
            if ent.label_ in _CANONICAL_PASSTHROUGH:
                canon = ent.label_
            else:
                canon = _SPACY_LABEL_MAP.get(ent.label_)
            if canon is None:
                continue
            mentions.append(
                EntityMention(
                    text=ent.text,
                    label=canon,
                    start_char=offset + ent.start_char,
                    end_char=offset + ent.end_char,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    confidence=0.60,          # spaCy gives no per-span score
                    sources=["spacy"],
                    sentence=ent.sent.text.strip() if ent.sent else "",
                )
            )
        return mentions
