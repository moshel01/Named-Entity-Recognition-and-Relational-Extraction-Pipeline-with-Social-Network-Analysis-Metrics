# Mode 2: dependency-parse SVO + co-occurrence + embeddings. No network.

from __future__ import annotations

import itertools
import logging

from config import Config
from core.schema import EntityMention, Relationship, TimelineEvent

from .base import IntelligenceBackend
from .embedding_utils import get_embedding_model
from .relationship_patterns import extract_dependency_relations

logger = logging.getLogger(__name__)


class PythonBackend(IntelligenceBackend):
    """Rules + embeddings extraction backend (fully local, no LLM)."""

    name = "python_only"

    def __init__(self, config: Config, spacy_engine=None, domain=None) -> None:
        super().__init__(config, domain=domain)
        self.cfg = config.intelligence.python_only
        self.spacy = spacy_engine          # Injected by the runner to reuse the model
        self.embedder = get_embedding_model(self.cfg.embedding_model)
        self.min_conf = self.cfg.min_relationship_confidence
        self.sim_threshold = self.cfg.embedding_similarity_threshold

    def _nlp(self, text: str):
        if self.spacy is None:
            # Lazy fallback: build a parser-only pipeline.
            from core.spacy_engine import SpacyEngine
            self.spacy = SpacyEngine(self.config.foundation.spacy_model)
        return self.spacy(text)

    def _mentions_by_sentence(
        self, spacy_doc, mentions: list[EntityMention], offset: int
    ) -> list[tuple[str, list[EntityMention]]]:
        """Group mentions by the sentence (relative spans) that contains them."""
        groups: list[tuple[str, list[EntityMention]]] = []
        for sent in spacy_doc.sents:
            s_start = sent.start_char
            s_end = sent.end_char
            in_sent = [
                m for m in mentions
                if s_start <= (m.start_char - offset) < s_end
            ]
            if len(in_sent) >= 1:
                groups.append((sent.text.strip(), in_sent))
        return groups

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
        # Candidate mention spans are document-absolute; chunk_start lets us map
        # them onto the chunk-relative spaCy parse below.
        offset = chunk_start

        spacy_doc = self._nlp(chunk_text)

        rels: list[Relationship] = []

        # Signal 1: dependency SVO relations.
        rels.extend(
            extract_dependency_relations(
                spacy_doc, candidates, chunk_id, doc_id, offset=offset,
                base_confidence=max(self.min_conf, 0.55),
            )
        )

        # Signal 2 + 3: co-occurrence with embedding-modulated confidence.
        existing_pairs = {
            frozenset((r.source.lower(), r.target.lower())) for r in rels
        }
        groups = self._mentions_by_sentence(spacy_doc, candidates, offset)
        for sent_text, in_sent in groups:
            # Unique entities by surface form within the sentence.
            uniq: dict[str, EntityMention] = {}
            for m in in_sent:
                uniq.setdefault(m.text.strip().lower(), m)
            members = list(uniq.values())
            for a, b in itertools.combinations(members, 2):
                pair = frozenset((a.text.lower(), b.text.lower()))
                if pair in existing_pairs:
                    continue
                sim = self.embedder.similarity(a.text, b.text)
                # Co-occurrence base confidence, nudged by semantic similarity.
                conf = 0.35 + 0.30 * (1.0 if sim >= self.sim_threshold else sim)
                if conf < self.min_conf:
                    continue
                existing_pairs.add(pair)
                rels.append(
                    Relationship(
                        source=a.text.strip(),
                        target=b.text.strip(),
                        rel_type="co_occurs_with",
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        evidence=sent_text,
                        confidence=round(conf, 3),
                        directed=False,
                        origin="extracted",
                        attributes={"embedding_similarity": round(sim, 3),
                                    "edge_source": "rule_cooccurrence"},
                    )
                )

        # Entities and timeline pass through from the foundation layer.
        return list(candidates), rels, []
