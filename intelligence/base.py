# Backend base: per-chunk extract + shared doc assembly; optional LLM hooks.

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from config import Config
from core.schema import (
    DocumentExtraction,
    EntityMention,
    FoundationResult,
    Relationship,
    TimelineEvent,
)
from .prompts import EXTRACTION_SYSTEM as _DEFAULT_EXTRACTION_SYSTEM

logger = logging.getLogger(__name__)


class BackendUnavailable(RuntimeError):
    """Server/endpoint is down. Must abort the run, not degrade per chunk -
    extract_document's catch-all would otherwise eat the circuit breaker."""


# First-person pronouns -> remapped to the author. Third-person -> unresolvable.
_FIRST_PERSON = {
    "ich", "mich", "mir", "mein", "meine", "meinen", "meinem", "meiner",
    "wir", "uns", "unser", "unsere", "i", "me", "my", "myself", "we", "us", "our",
}
_THIRD_PERSON = {
    "er", "sie", "ihn", "ihm", "ihr", "ihre", "es", "sein", "seine", "man",
    "he", "him", "his", "she", "her", "they", "them", "their", "it",
}
_PRONOUNS = _FIRST_PERSON | _THIRD_PERSON


def _remap_pronoun_endpoints(rels: list[Relationship], author: str) -> list[Relationship]:
    # First-person -> author; third-person endpoints are unresolvable -> drop.
    out = []
    for r in rels:
        s, t = r.source.strip().lower(), r.target.strip().lower()
        if s in _THIRD_PERSON or t in _THIRD_PERSON:
            continue
        if s in _FIRST_PERSON:
            r.source = author
        if t in _FIRST_PERSON:
            r.target = author
        if r.source.strip().lower() != r.target.strip().lower():
            out.append(r)
    return out


class IntelligenceBackend(ABC):
    """Base class for relationship/entity extraction backends."""

    name: str = "base"

    def __init__(self, config: Config, domain=None) -> None:
        self.config = config
        self.domain = domain
        # Canonical types: union of config + any domain label-map overrides.
        types = set(config.foundation.label_map.values())
        if domain is not None:
            types.update(domain.gliner_label_map().values())
        self.label_types = sorted(types)
        # Domain may override the extraction / quality-review system prompts.
        self.extraction_system: str = (
            (domain.extraction_system_prompt() if domain else None)
            or _DEFAULT_EXTRACTION_SYSTEM
        )
        self.quality_system: str | None = (
            domain.quality_review_system_prompt() if domain else None
        )
        # Resolved relation ontology -> constrain LLM relation extraction.
        from postprocess.ontology import resolve_relation_guide, resolve_relation_ontology
        onto = resolve_relation_ontology(config, domain)
        self.relation_types: list[str] = sorted(onto.keys())
        # Optional per-label definitions rendered next to the allowed types.
        self.relation_guide: dict[str, str] = resolve_relation_guide(config, domain)
        # (month_words, season_words, pivot_max) for normalizing LLM timeline dates.
        v = domain.temporal_vocab() if domain is not None else {}
        self._date_vocab = (v.get("months", {}), v.get("seasons", {}), v.get("pivot_max"))
        # Set by extract_chunk when it degrades to foundation-only output
        # (timeout, unrepairable JSON). extract_document records the failure in
        # checkpoint meta so resume can tell a failed doc from an empty one.
        self._chunk_failed = False

    # Subclass hook
    @abstractmethod
    def extract_chunk(
        self,
        chunk_text: str,
        candidates: list[EntityMention],
        sentences: list[str],
        chunk_id: str,
        doc_id: str,
        chunk_start: int = 0,
        author_name: str = "",
    ) -> tuple[list[EntityMention], list[Relationship], list[TimelineEvent]]:
        """Extract refined entities, relationships, and timeline for one chunk."""
        raise NotImplementedError

    # Optional hooks (LLM backends override)
    def review(
        self, entities_summary: str, edges_summary: str
    ) -> Optional[dict[str, Any]]:
        """Optional LLM/rule quality review. Default: no review available."""
        return None

    def enrich(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """name -> {subtype, attributes}. Default: nothing."""
        return {}

    def suggest_merges(self, entity_type: str, names: list[str]) -> list[dict[str, Any]]:
        """Propose [{canonical, aliases}] merge groups. Default: none."""
        return []

    # Shared document assembly
    def extract_document(
        self,
        doc_id: str,
        source_path: str,
        foundation_results: list[FoundationResult],
    ) -> DocumentExtraction:
        """Run the backend over every chunk and assemble a document result."""
        all_mentions: list[EntityMention] = []
        all_rels: list[Relationship] = []
        all_timeline: list[TimelineEvent] = []

        # The narrator/author name (from coref) so the LLM attributes first-person
        # statements to it instead of emitting bare pronouns.
        author_name = ""
        for fr in foundation_results:
            for m in fr.mentions:
                if m.attributes.get("is_author"):
                    author_name = m.text
                    break
            if author_name:
                break

        failed_chunks: list[str] = []
        for fr in foundation_results:
            # Foundation dates always carry through (cheap, deterministic).
            all_timeline.extend(fr.dates)
            self._chunk_failed = False
            try:
                mentions, rels, timeline = self.extract_chunk(
                    fr.chunk.text,
                    fr.mentions,
                    fr.sentences,
                    fr.chunk.chunk_id,
                    fr.chunk.doc_id,
                    fr.chunk.start_char,
                    author_name,
                )
            except BackendUnavailable:
                raise  # circuit breaker - abort the run, don't degrade
            except Exception as exc:  # noqa: BLE001 - never let one chunk kill a doc
                logger.warning(
                    "Backend '%s' failed on chunk %s: %s; "
                    "falling back to foundation mentions only.",
                    self.name, fr.chunk.chunk_id, exc,
                )
                mentions, rels, timeline = list(fr.mentions), [], []
                self._chunk_failed = True
            if self._chunk_failed:
                failed_chunks.append(fr.chunk.chunk_id)
            all_mentions.extend(mentions)
            all_rels.extend(rels)
            all_timeline.extend(timeline)

        if author_name:
            all_rels = _remap_pronoun_endpoints(all_rels, author_name)
        # Drop bare pronoun entities (never useful nodes).
        all_mentions = [m for m in all_mentions
                        if m.text.strip().lower() not in _PRONOUNS]

        meta: dict[str, Any] = {"backend": self.name,
                                "n_chunks": len(foundation_results),
                                "chunks_failed": len(failed_chunks)}
        if failed_chunks:
            meta["failed_chunks"] = failed_chunks[:20]
        return DocumentExtraction(
            doc_id=doc_id,
            source_path=source_path,
            mentions=all_mentions,
            relationships=all_rels,
            timeline=all_timeline,
            meta=meta,
        )
