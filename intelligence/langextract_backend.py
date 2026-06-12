# Mode: langextract. Google LangExtract orchestrates an LLM (Ollama / Gemini /
# OpenAI) with few-shot examples and char-level source grounding. It is an
# ALTERNATIVE to the ollama/api backends - same underlying model, different
# extraction machinery - intended to be A/B'd against them. Fail-soft: a missing
# package or a failed call falls back to foundation mentions only.

from __future__ import annotations

import logging
import os
from typing import Any

from config import Config
from core.schema import EntityMention, Relationship, TimelineEvent

from .base import IntelligenceBackend

logger = logging.getLogger(__name__)

# LangExtract extraction_class (lowercase) -> canonical pipeline type.
_CLASS_TO_TYPE = {
    "person": "PERSON", "people": "PERSON",
    "organization": "ORG", "org": "ORG", "organisation": "ORG",
    "location": "LOCATION", "place": "LOCATION", "gpe": "LOCATION",
    "event": "EVENT", "institution": "INSTITUTION", "rank": "RANK",
    "date": "DATE", "time": "DATE",
}


class LangExtractBackend(IntelligenceBackend):
    """Relationship/entity extraction via Google's LangExtract."""

    name = "langextract"

    def __init__(self, config: Config, domain=None) -> None:
        super().__init__(config, domain=domain)
        self.cfg = config.intelligence.langextract
        self._lx = self._import_lx()
        self._examples = self._build_examples()
        self._prompt = self._build_prompt()
        self._api_key = os.environ.get(self.cfg.api_key_env, "") if self.cfg.api_key_env else ""

    @staticmethod
    def _import_lx():
        try:
            import langextract as lx
            return lx
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "mode 'langextract' requires the 'langextract' package. "
                "Fix: pip install langextract"
            ) from exc

    def _build_prompt(self) -> str:
        types = ", ".join(t.lower() for t in self.label_types)
        rels = (", ".join(self.relation_types) if self.relation_types
                else "use a short verb-like snake_case relation (e.g. met_with, member_of)")
        return (
            "Extract named entities and the relationships between them from the text. "
            f"Entity classes: {types}. "
            "Also extract relationships as class 'relationship' with attributes "
            "'source', 'target', and 'relation'. "
            f"Allowed relation values: {rels}. "
            "Use exact text spans from the source; do not paraphrase. "
            "Attribute every first-person statement to the document's author."
        )

    def _build_examples(self) -> list:
        lx = self._lx
        # Domain-neutral example demonstrating entities + two relationships.
        text = ("In 1998, Eleanor Vance founded Meridian Logistics in Boston with "
                "Marcus Reyes, who later became her business partner.")
        return [
            lx.data.ExampleData(
                text=text,
                extractions=[
                    lx.data.Extraction(extraction_class="person",
                                       extraction_text="Eleanor Vance"),
                    lx.data.Extraction(extraction_class="organization",
                                       extraction_text="Meridian Logistics"),
                    lx.data.Extraction(extraction_class="location",
                                       extraction_text="Boston"),
                    lx.data.Extraction(extraction_class="person",
                                       extraction_text="Marcus Reyes"),
                    lx.data.Extraction(
                        extraction_class="relationship", extraction_text="founded",
                        attributes={"source": "Eleanor Vance",
                                    "target": "Meridian Logistics", "relation": "founded"}),
                    lx.data.Extraction(
                        extraction_class="relationship",
                        extraction_text="business partner",
                        attributes={"source": "Eleanor Vance",
                                    "target": "Marcus Reyes", "relation": "partner_of"}),
                ],
            )
        ]

    def _extract(self, text: str):
        kwargs: dict[str, Any] = dict(
            text_or_documents=text,
            prompt_description=self._prompt,
            examples=self._examples,
            model_id=self.cfg.model_id,
            extraction_passes=self.cfg.extraction_passes,
            max_workers=self.cfg.max_workers,
            max_char_buffer=self.cfg.max_char_buffer,
        )
        if self.cfg.provider == "ollama":
            kwargs["model_url"] = self.cfg.model_url
        elif self._api_key:
            kwargs["api_key"] = self._api_key
        return self._lx.extract(**kwargs)

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
        try:
            result = self._extract(chunk_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LangExtract failed for chunk %s: %s", chunk_id, exc)
            self._chunk_failed = True
            return list(candidates), [], []

        # Foundation candidates are always kept; LangExtract adds grounded
        # entities + relationships on top. Dedup reconciles overlaps later.
        mentions: list[EntityMention] = list(candidates)
        rels: list[Relationship] = []
        for e in getattr(result, "extractions", []) or []:
            ci = getattr(e, "char_interval", None)
            # LangExtract's CharInterval exposes start_pos/end_pos (older builds
            # used start/end); accept either. May be None when the span can't be
            # aligned verbatim (common for relationship trigger phrases).
            cstart = getattr(ci, "start_pos", getattr(ci, "start", None)) if ci else None
            cend = getattr(ci, "end_pos", getattr(ci, "end", None)) if ci else None
            cls = (e.extraction_class or "").lower()
            attrs = dict(e.attributes or {})
            if cls == "relationship":
                # A relationship is defined by its (grounded) endpoints, so keep it
                # even when the trigger phrase itself can't be span-aligned. Record
                # the offsets when available.
                src = (attrs.get("source") or "").strip()
                tgt = (attrs.get("target") or "").strip()
                if not src or not tgt or src.lower() == tgt.lower():
                    continue
                rel_attrs: dict = {"edge_source": "langextract_extracted"}
                if cstart is not None and cend is not None:
                    rel_attrs["char_start"] = chunk_start + int(cstart)
                    rel_attrs["char_end"] = chunk_start + int(cend)
                rels.append(Relationship(
                    source=src, target=tgt,
                    rel_type=(attrs.get("relation") or "related_to").strip(),
                    doc_id=doc_id, chunk_id=chunk_id,
                    evidence=e.extraction_text or "",
                    confidence=0.7, origin="extracted", attributes=rel_attrs,
                ))
            else:
                # Entities must be grounded - char offsets are the whole point.
                if cstart is None or cend is None:
                    continue
                label = _CLASS_TO_TYPE.get(cls, cls.upper())
                if self.label_types and label not in self.label_types:
                    continue
                mentions.append(EntityMention(
                    text=e.extraction_text or "", label=label,
                    start_char=chunk_start + int(cstart), end_char=chunk_start + int(cend),
                    chunk_id=chunk_id, doc_id=doc_id,
                    confidence=0.7, sources=["langextract"],
                ))
        return mentions, rels, []
