# Backend base: per-chunk extract + shared doc assembly; optional LLM hooks.

from __future__ import annotations

import logging
import re
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


# Tokens too generic to count as grounding a mention (EN + DE articles/particles).
_GROUNDING_STOP = frozenset({
    "the", "and", "of", "for", "von", "der", "die", "das", "und", "den", "dem",
})


def _name_in_evidence(name: str, ev_lower: str) -> bool:
    """True if `name` is grounded in the evidence text: the whole surface appears,
    or any significant token of it does. Token match (not exact) so "Goebbels"
    grounds "Joseph Goebbels" and "NSDAP" grounds "the NSDAP" - and it survives
    the alias/abbreviation variation a verbatim check would miss."""
    n = name.strip().lower()
    if not n:
        return False
    if n in ev_lower:
        return True
    for tok in re.split(r"[\s,.;:'\"()-]+", n):
        if len(tok) >= 3 and tok not in _GROUNDING_STOP and tok in ev_lower:
            return True
    return False


def _tag_ungrounded_evidence(rels: list[Relationship], author: str) -> int:
    """AEVS-style anchor check (Yang et al. 2026): a typed relation whose evidence
    span names NEITHER endpoint is likely misattributed - the model picked a real
    sentence (passes evidence_unverified) that doesn't actually mention the pair.
    Tag it `evidence_ungrounded`; never drop (coref-resolved first-person evidence
    legitimately uses pronouns, so this stays a filterable signal, not a filter).
    Only text-asserted extractions with evidence are checked. Returns the count."""
    a = author.strip().lower()
    flagged = 0
    for r in rels:
        if r.origin != "extracted":
            continue
        ev = (r.evidence or "").strip().lower()
        if not ev:
            continue
        # Author endpoint = first-person evidence ("I joined..."), grounded by coref.
        src_ok = (a and r.source.strip().lower() == a) or _name_in_evidence(r.source, ev)
        tgt_ok = (a and r.target.strip().lower() == a) or _name_in_evidence(r.target, ev)
        if not (src_ok or tgt_ok):
            r.attributes["evidence_ungrounded"] = "true"
            flagged += 1
    return flagged


def _dense_enough(mentions, chunk_text: str, chunk_start: int,
                  window_words: int, min_entities: int) -> bool:
    """True if some window of ``window_words`` holds >= ``min_entities`` DISTINCT
    entities. A relation needs two entities co-occurring, so a chunk that fails
    this can't yield one - skip its LLM call. Word index is approximated by the
    space count before each mention (cheap, good enough for a gate)."""
    pts: list[tuple[int, str]] = []
    for m in mentions:
        name = (m.text or "").strip().lower()
        if not name or name in _PRONOUNS:
            continue
        rel = max(0, m.start_char - chunk_start)
        pts.append((chunk_text.count(" ", 0, rel), name))
    if len({n for _, n in pts}) < min_entities:
        return False
    pts.sort()
    from collections import deque
    win: deque[tuple[int, str]] = deque()
    counts: dict[str, int] = {}
    for wpos, name in pts:
        win.append((wpos, name))
        counts[name] = counts.get(name, 0) + 1
        while win and wpos - win[0][0] > window_words:
            _, old = win.popleft()
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]
        if len(counts) >= min_entities:
            return True
    return False


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

        # Cost gate: skip the LLM relation call for chunks too sparse to hold a
        # relation (LLM modes only; python_only's local rules are free and supply
        # the co-occurrence floor, so never gate it).
        ic = self.config.intelligence
        gate = ic.skip_sparse_chunks and self.name != "python_only"

        failed_chunks: list[str] = []
        skipped = 0
        for fr in foundation_results:
            # Foundation dates always carry through (cheap, deterministic).
            all_timeline.extend(fr.dates)
            if gate and not _dense_enough(fr.mentions, fr.chunk.text, fr.chunk.start_char,
                                          ic.sparse_window_words, ic.sparse_min_entities):
                all_mentions.extend(fr.mentions)   # entities kept; no LLM relations
                skipped += 1
                continue
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
        # Anchor check: tag relations whose evidence names neither endpoint.
        ungrounded = _tag_ungrounded_evidence(all_rels, author_name)
        # Drop bare pronoun entities (never useful nodes).
        all_mentions = [m for m in all_mentions
                        if m.text.strip().lower() not in _PRONOUNS]

        if skipped:
            logger.info("Cost gate: skipped %d/%d sparse chunks (no LLM call) for %s.",
                        skipped, len(foundation_results), doc_id)
        if ungrounded:
            logger.info("Anchor check: %d relations with evidence naming neither "
                        "endpoint (tagged evidence_ungrounded) in %s.", ungrounded, doc_id)
        meta: dict[str, Any] = {"backend": self.name,
                                "n_chunks": len(foundation_results),
                                "chunks_skipped": skipped,
                                "chunks_failed": len(failed_chunks),
                                "evidence_ungrounded": ungrounded}
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
