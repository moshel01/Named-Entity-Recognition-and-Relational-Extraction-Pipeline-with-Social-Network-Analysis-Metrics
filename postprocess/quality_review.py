# Quality review: drop spurious entities/edges via rules and (optionally) an LLM.

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from config import QualityConfig
from core.schema import Entity, Relationship

from .aggregator import normalize_name

logger = logging.getLogger(__name__)

# Guards on the LLM reviewer's drop list (a weak model can hallucinate a huge
# or wrong drop set). Salient entities are never dropped on the LLM's say-so, and
# a batch asking to drop too much is ignored wholesale.
_LLM_DROP_BATCH_FRACTION = 0.5    # ignore a batch's drops if it exceeds this share
_LLM_PROTECT_MENTIONS = 5         # protect entities mentioned at least this often
_LLM_PROTECT_DOCS = 3             # protect entities spanning at least this many docs

# Generic terms that are almost never useful standalone entities.
_STOP_NAMES = {
    "the", "a", "an", "he", "she", "they", "it", "we", "i", "you",
    "this", "that", "these", "those", "today", "yesterday", "tomorrow",
    "mr", "mrs", "ms", "dr", "sir", "madam",
}

# Leading articles/possessives stripped before stopword matching, so "mein Vater"
# / "der Soldat" hit the bare "vater" / "soldat" stopword.
_DETERMINERS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "einer", "eines", "mein", "meine", "meinen", "meinem", "meiner", "unser",
    "unsere", "unserem", "dein", "sein", "seine", "ihr", "ihre", "the", "a",
    "an", "my", "our", "his", "her",
}


def _strip_determiner(norm: str) -> str:
    toks = norm.split()
    if len(toks) > 1 and toks[0] in _DETERMINERS:
        return " ".join(toks[1:])
    return norm


def _bad_entity_name(name: str) -> bool:
    n = name.strip()
    if not n or len(n) < 2:
        return True
    if normalize_name(n) in _STOP_NAMES:
        return True
    if n.isdigit():
        return True
    return False


def _junk_person(name: str) -> bool:
    """OCR fragments / abbreviations mislabeled as people ('lch', 'Nie', 'Pg.')."""
    n = normalize_name(name)
    toks = n.split()
    if len(toks) == 1 and len(toks[0]) <= 3:        # 'nie', 'kgl', 'pg', 'w'
        return True
    if not any(c.isupper() for c in name):          # all-lowercase = fragment
        return True
    return False


class QualityReviewer:
    """Filter entities and edges by rules and optional LLM judgement."""

    def __init__(self, config: QualityConfig, stopwords: set[str] | None = None) -> None:
        self.config = config
        self.stopwords = stopwords or set()

    # Rule layer
    def rule_filter(
        self, entities: list[Entity], edges: list[Relationship]
    ) -> tuple[list[Entity], list[Relationship]]:
        kept_entities = [
            e for e in entities
            if e.attributes.get("is_author")          # authors are always kept
            or (e.mention_count >= self.config.min_entity_mentions
                and e.confidence >= self.config.min_entity_confidence
                and not _bad_entity_name(e.canonical_name)
                and not (e.label == "PERSON" and _junk_person(e.canonical_name))
                and normalize_name(e.canonical_name) not in self.stopwords
                and _strip_determiner(normalize_name(e.canonical_name)) not in self.stopwords)
        ]
        kept_ids = {e.entity_id for e in kept_entities}

        # Edge weight = number of supporting relationships for an unordered pair+type.
        weights: Counter[tuple[str, str, str]] = Counter()
        for r in edges:
            weights[(r.source, r.target, r.rel_type)] += 1

        kept_edges = [
            r for r in edges
            if r.source in kept_ids and r.target in kept_ids
            and weights[(r.source, r.target, r.rel_type)] >= self.config.min_edge_weight
        ]
        logger.info(
            "Rule filter: entities %d -> %d, edges %d -> %d",
            len(entities), len(kept_entities), len(edges), len(kept_edges),
        )
        return kept_entities, kept_edges

    # LLM layer
    def llm_filter(
        self,
        entities: list[Entity],
        edges: list[Relationship],
        backend,
        batch_size: int = 150,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Apply an LLM reviewer's drop/merge suggestions, if available.

        Batched so large corpora are fully reviewed (not truncated). Each batch
        gets its entities plus the edges among them.
        """
        if backend is None or not hasattr(backend, "review"):
            return entities, edges

        id_to_name = {e.entity_id: e.canonical_name for e in entities}
        drop_ent: set[str] = set()
        drop_edge: set[str] = set()

        for i in range(0, len(entities), max(1, batch_size)):
            batch = entities[i:i + batch_size]
            batch_ids = {e.entity_id for e in batch}
            ent_summary = "\n".join(
                f"{e.canonical_name} | {e.label} | {e.mention_count}" for e in batch
            )
            edge_summary = "\n".join(
                f"{id_to_name.get(r.source, r.source)}||{r.rel_type}||"
                f"{id_to_name.get(r.target, r.target)}"
                for r in edges if r.source in batch_ids and r.target in batch_ids
            )
            result = backend.review(ent_summary, edge_summary)
            if not result:
                continue
            batch_drop = {normalize_name(n) for n in result.get("drop_entities", [])}
            # A batch that wants to drop most of its entities is hallucinating.
            if len(batch_drop) > _LLM_DROP_BATCH_FRACTION * max(1, len(batch)):
                logger.warning("LLM review: ignoring oversized drop list (%d/%d) for a batch.",
                               len(batch_drop), len(batch))
                batch_drop = set()
            drop_ent |= batch_drop
            drop_edge |= set(result.get("drop_edges", []))

        if not drop_ent and not drop_edge:
            return entities, edges

        # Never let the LLM drop a salient entity (author / reference figure /
        # well-attested). Junk it should drop is low-mention by definition.
        protected = {
            normalize_name(e.canonical_name) for e in entities
            if e.attributes.get("is_author") or e.attributes.get("reference_figure")
            or e.mention_count >= _LLM_PROTECT_MENTIONS or len(e.doc_ids) >= _LLM_PROTECT_DOCS
        }
        kept_entities = [
            e for e in entities
            if e.attributes.get("is_author")
            or normalize_name(e.canonical_name) not in drop_ent
            or normalize_name(e.canonical_name) in protected
        ]
        kept_ids = {e.entity_id for e in kept_entities}

        kept_edges = []
        for r in edges:
            if r.source not in kept_ids or r.target not in kept_ids:
                continue
            key = f"{id_to_name.get(r.source, r.source)}||{r.rel_type}||{id_to_name.get(r.target, r.target)}"
            if key in drop_edge:
                continue
            kept_edges.append(r)

        logger.info(
            "LLM review dropped %d entities, %d edges",
            len(entities) - len(kept_entities), len(edges) - len(kept_edges),
        )
        return kept_entities, kept_edges

    # Orchestration
    def review(
        self,
        entities: list[Entity],
        edges: list[Relationship],
        mode: str,
        backend=None,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Run rule filtering, then LLM review when enabled and supported."""
        if not self.config.enabled:
            return entities, edges
        entities, edges = self.rule_filter(entities, edges)

        use_llm = self._should_use_llm(mode)
        if use_llm:
            entities, edges = self.llm_filter(entities, edges, backend,
                                              batch_size=self.config.review_batch_size)
        return entities, edges

    def _should_use_llm(self, mode: str) -> bool:
        setting = self.config.llm_review
        if setting is True:
            return True
        if setting is False:
            return False
        # "auto": use LLM only in LLM-capable modes.
        return mode in ("api", "ollama")
