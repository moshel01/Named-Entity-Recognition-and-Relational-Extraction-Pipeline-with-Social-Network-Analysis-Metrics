# Mode 1: cloud LLM (Anthropic/OpenAI/Bedrock).

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from config import Config
from core.schema import EntityMention, Relationship, TimelineEvent
from core.date_extractor import normalize_date

from .base import IntelligenceBackend
from .json_repair import repair_json
from .prompts import (
    ENRICHMENT_SYSTEM,
    EXTRACTION_SYSTEM,
    MERGE_SYSTEM,
    build_enrichment_prompt,
    build_extraction_prompt,
    build_merge_prompt,
    build_quality_review_prompt,
    coerce_extraction,
)

logger = logging.getLogger(__name__)


class ApiBackend(IntelligenceBackend):
    """LLM-backed extraction via a hosted provider."""

    name = "api"

    def __init__(self, config: Config, domain=None) -> None:
        super().__init__(config, domain=domain)
        self.cfg = config.intelligence.api
        self.provider = self.cfg.provider
        self._client = self._build_client()

    # Client construction
    def _build_client(self) -> Any:
        key = os.environ.get(self.cfg.api_key_env, "")
        if self.provider == "anthropic":
            import anthropic
            if not key:
                raise RuntimeError(f"Env var {self.cfg.api_key_env} is not set.")
            return anthropic.Anthropic(api_key=key, timeout=self.cfg.request_timeout)
        if self.provider == "openai":
            from openai import OpenAI
            if not key:
                raise RuntimeError(f"Env var {self.cfg.api_key_env} is not set.")
            return OpenAI(api_key=key, timeout=self.cfg.request_timeout)
        if self.provider == "bedrock":
            import boto3
            return boto3.client("bedrock-runtime", region_name=self.cfg.aws_region)
        raise ValueError(f"Unknown provider: {self.provider}")

    # Unified completion
    def _complete(self, system: str, user: str) -> str:
        """Send a (system, user) prompt and return the raw text response."""
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                return self._complete_once(system, user)
            except Exception as exc:  # noqa: BLE001 - provider-specific transient errors
                last_exc = exc
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "API call failed (attempt %d/%d): %s; retrying in %ds",
                    attempt + 1, self.cfg.max_retries, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"API completion failed after retries: {last_exc}")

    def _complete_once(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        if self.provider == "bedrock":
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            resp = self._client.invoke_model(
                modelId=self.cfg.model, body=json.dumps(body)
            )
            payload = json.loads(resp["body"].read())
            content = payload.get("content", [])
            return "".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
        raise ValueError(f"Unknown provider: {self.provider}")

    # Extraction
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
        prompt = build_extraction_prompt(chunk_text, candidates, self.label_types,
                                         relation_types=self.relation_types or None,
                                         author_name=author_name,
                                         relation_guide=self.relation_guide or None)
        raw = self._complete(self.extraction_system, prompt)
        obj = repair_json(raw)
        if obj is None:
            logger.warning("Unparseable LLM output for chunk %s; using candidates.", chunk_id)
            self._chunk_failed = True
            return list(candidates), [], []
        data = coerce_extraction(obj)
        return _map_extraction(data, candidates, chunk_id, doc_id, self.label_types,
                               *self._date_vocab, chunk_text=chunk_text)

    # Quality review
    def review(self, entities_summary: str, edges_summary: str) -> dict[str, Any] | None:
        system, user = build_quality_review_prompt(entities_summary, edges_summary)
        try:
            raw = self._complete(system, user)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM quality review failed: %s", exc)
            return None
        obj = repair_json(raw)
        return obj if isinstance(obj, dict) else None

    # Enrichment
    def enrich(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        try:
            raw = self._complete(ENRICHMENT_SYSTEM, build_enrichment_prompt(rows))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM enrichment failed: %s", exc)
            return {}
        return _parse_enrichment(repair_json(raw))

    # LLM-assisted dedup
    def suggest_merges(self, entity_type: str, names: list[str]) -> list[dict[str, Any]]:
        try:
            raw = self._complete(MERGE_SYSTEM, build_merge_prompt(entity_type, names))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM merge suggestion failed: %s", exc)
            return []
        return _parse_merges(repair_json(raw))


# Shared parse helpers (also used by ollama_backend)
def _parse_enrichment(obj: Any) -> dict[str, dict[str, Any]]:
    """LLM enrichment JSON -> {name: {subtype, attributes}}."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(obj, dict):
        return out
    for e in obj.get("entities", []) or []:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        attrs = {k: v for k, v in (e.get("attributes") or {}).items() if v}
        rec: dict[str, Any] = {}
        if e.get("subtype"):
            rec["subtype"] = str(e["subtype"]).strip()
        if attrs:
            rec["attributes"] = attrs
        if rec:
            out[name] = rec
    return out


def _parse_merges(obj: Any) -> list[dict[str, Any]]:
    """LLM merge JSON -> [{canonical, aliases}]."""
    out: list[dict[str, Any]] = []
    if not isinstance(obj, dict):
        return out
    for g in obj.get("groups", []) or []:
        if not isinstance(g, dict):
            continue
        canon = str(g.get("canonical", "")).strip()
        aliases = [str(a).strip() for a in (g.get("aliases") or []) if str(a).strip()]
        if canon and aliases:
            out.append({"canonical": canon, "aliases": aliases})
    return out


# Shared mapping helper (also used by ollama_backend)
# Fold unicode punctuation variants the LLM normalizes when quoting (curly
# quotes, dashes, ellipsis) so verbatim evidence isn't flagged as paraphrase.
_PUNCT_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "−": "-", "­": None,
    "…": "...", " ": " ",
})


def _normalize_ws(s: str) -> str:
    return " ".join(s.translate(_PUNCT_FOLD).split()).casefold()


def _evidence_verbatim(evidence: str, norm_chunk: str) -> bool:
    """True when the evidence is quoted from the chunk. Models legitimately
    stitch multiple spans with '...' - each substantial segment must appear."""
    segments = [s.strip(" .\"'") for s in _normalize_ws(evidence).split("...")]
    segments = [s for s in segments if len(s) >= 8]
    if not segments:
        return True
    return all(s in norm_chunk for s in segments)


def _map_extraction(
    data: dict[str, list],
    candidates: list[EntityMention],
    chunk_id: str,
    doc_id: str,
    label_types: list[str],
    month_words: dict[str, int] | None = None,
    season_words: dict[str, int] | None = None,
    pivot_max: int | None = None,
    chunk_text: str = "",
) -> tuple[list[EntityMention], list[Relationship], list[TimelineEvent]]:
    """Map a parsed LLM extraction object onto pipeline dataclasses.

    Entity spans from the LLM are approximate, so mentions are emitted with a
    chunk-level span of (0,0); the foundation candidates supply precise spans
    and are retained alongside the LLM's confirmations.
    """
    valid_types = set(label_types)

    # Entities - merge LLM confirmations with foundation candidates by name.
    mentions: list[EntityMention] = list(candidates)
    known_names = {m.text.strip().lower() for m in candidates}
    for e in data.get("entities", []):
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        if not name or name.lower() in known_names:
            continue
        etype = str(e.get("type", "")).upper()
        if etype not in valid_types:
            continue
        mentions.append(
            EntityMention(
                text=name,
                label=etype,
                start_char=0,
                end_char=0,
                chunk_id=chunk_id,
                doc_id=doc_id,
                confidence=float(e.get("confidence", 0.6) or 0.6),
                sources=["llm"],
            )
        )
        known_names.add(name.lower())

    # Relationships. Evidence the model paraphrased (not a verbatim span of
    # the chunk) is tagged, not dropped, so Gephi can filter on it.
    norm_chunk = _normalize_ws(chunk_text)
    rels: list[Relationship] = []
    for r in data.get("relationships", []):
        if not isinstance(r, dict):
            continue
        src = str(r.get("source", "")).strip()
        tgt = str(r.get("target", "")).strip()
        rtype = str(r.get("type", "related_to")).strip() or "related_to"
        if not src or not tgt or src.lower() == tgt.lower():
            continue
        evidence = str(r.get("evidence", "")).strip()
        attrs = {"edge_source": "llm_extracted"}
        if evidence and norm_chunk and not _evidence_verbatim(evidence, norm_chunk):
            attrs["evidence_unverified"] = "true"
        rels.append(
            Relationship(
                source=src,
                target=tgt,
                rel_type=rtype,
                doc_id=doc_id,
                chunk_id=chunk_id,
                evidence=evidence,
                confidence=float(r.get("confidence", 0.6) or 0.6),
                directed=bool(r.get("directed", False)),
                origin="extracted",
                attributes=attrs,
            )
        )

    # Timeline. Use the shared normalizer; drop entries with no resolvable year
    # (the LLM sometimes returns non-dates like "6 Jahre alt").
    timeline: list[TimelineEvent] = []
    for t in data.get("timeline", []):
        if not isinstance(t, dict):
            continue
        date_text = str(t.get("date", "")).strip()
        if not date_text:
            continue
        iso, year = normalize_date(date_text, month_words, season_words, pivot_max)
        if year is None:
            continue
        ents = t.get("entities") or []
        timeline.append(
            TimelineEvent(
                doc_id=doc_id,
                chunk_id=chunk_id,
                date_text=date_text,
                iso_date=iso,
                year=year,
                description=str(t.get("description", "")).strip(),
                entities=[str(x).strip() for x in ents if str(x).strip()],
                confidence=float(t.get("confidence", 0.6) or 0.6),
            )
        )

    return mentions, rels, timeline
