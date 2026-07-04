# Mode 3: local LLM via Ollama. Same prompts/mapping as the API backend.

from __future__ import annotations

import logging
from typing import Any

import requests

from config import Config
from core.schema import EntityMention, Relationship, TimelineEvent

from .api_backend import _map_extraction, _parse_enrichment, _parse_merges
from .base import BackendUnavailable, IntelligenceBackend
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


class OllamaBackend(IntelligenceBackend):
    """LLM-backed extraction via a locally hosted Ollama server."""

    name = "ollama"

    # Abort after this many consecutive failed LLM calls: a downed server would
    # otherwise produce a silently degraded run (mentions but zero relations).
    _MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self, config: Config, domain=None) -> None:
        super().__init__(config, domain=domain)
        self.cfg = config.intelligence.ollama
        self._endpoint = self.cfg.host.rstrip("/") + "/api/chat"
        # One TCP connection for the whole run: tens of thousands of chunk calls,
        # and the Tailscale-remote host pays connect+slow-start on every new one.
        self._session = requests.Session()
        self._consecutive_failures = 0
        self._ctx_warned = False
        self._verify_server()

    def _verify_server(self) -> None:
        try:
            resp = requests.get(self.cfg.host.rstrip("/") + "/api/tags", timeout=5)
            resp.raise_for_status()
            models = {m.get("name", "") for m in resp.json().get("models", [])}
            if self.cfg.model not in models and models:
                logger.warning(
                    "Ollama model '%s' not found in installed models %s. "
                    "Pull it with: ollama pull %s",
                    self.cfg.model, sorted(models), self.cfg.model,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not reach Ollama at %s (%s). Is `ollama serve` running?",
                self.cfg.host, exc,
            )

    def _complete(self, system: str, user: str, schema: dict | None = None) -> str:
        # Ollama silently truncates a prompt past num_ctx FROM THE TOP - the system
        # prompt and candidate list go first, and quality craters with no error.
        # ~3 chars/token is conservative for German; warn once per run.
        est = (len(system) + len(user)) // 3
        if est > self.cfg.num_ctx and not self._ctx_warned:
            self._ctx_warned = True
            logger.warning(
                "Prompt ~%d tokens exceeds num_ctx=%d - ollama silently drops the "
                "top of the prompt (system + candidates). Raise intelligence."
                "ollama.num_ctx or lower chunking.max_chars.",
                est, self.cfg.num_ctx,
            )
        options = {
            "temperature": self.cfg.temperature,
            "num_ctx": self.cfg.num_ctx,
        }
        if self.cfg.num_predict > 0:
            options["num_predict"] = self.cfg.num_predict
        payload = {
            "model": self.cfg.model,
            "stream": False,
            # Schema-constrained grammar when given (structured_output): the model
            # can only emit valid JSON of that shape, so reasoning can't leak into
            # the array slots. Plain "json" mode otherwise.
            "format": schema if schema else "json",
            # Disable "thinking" so reasoning models (Qwen3.5, etc.) emit clean JSON
            # instead of <think> traces that break parsing. Ignored by models that
            # don't support it.
            "think": False,
            "options": options,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        resp = self._session.post(
            self._endpoint, json=payload, timeout=self.cfg.request_timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

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
                                         relation_guide=self.relation_guide or None,
                                         edge_qualifiers=self.edge_qualifiers or None,
                                         type_signatures=self.type_signatures or None)
        import time
        t0 = time.monotonic()
        try:
            raw = self._complete(self.extraction_system, prompt, self._extraction_schema)
            self._consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001
            self._consecutive_failures += 1
            logger.warning("Ollama call failed for chunk %s: %s", chunk_id, exc)
            if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                raise BackendUnavailable(
                    f"{self._consecutive_failures} consecutive Ollama failures - "
                    f"server at {self.cfg.host} looks down. Aborting instead of "
                    "writing an extraction checkpoint with no relationships. "
                    "Restart ollama and re-run with --resume."
                ) from exc
            self._chunk_failed = True
            return list(candidates), [], []
        obj = repair_json(raw)
        if obj is None:
            self._chunk_failed = True
            return list(candidates), [], []
        data = coerce_extraction(obj)
        mentions, rels, timeline = _map_extraction(
            data, candidates, chunk_id, doc_id, self.label_types,
            *self._date_vocab, chunk_text=chunk_text,
            qualifiers=self.edge_qualifiers or None)
        # Heartbeat: chunks take minutes each and the doc-level progress bar
        # sits still for a whole chapter - show that work is happening.
        logger.info("chunk %s: %d relationships, %d mentions (%.0fs)",
                    chunk_id, len(rels), len(mentions), time.monotonic() - t0)
        return mentions, rels, timeline

    def review(self, entities_summary: str, edges_summary: str) -> dict[str, Any] | None:
        system, user = build_quality_review_prompt(entities_summary, edges_summary)
        try:
            raw = self._complete(system, user)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama quality review failed: %s", exc)
            return None
        obj = repair_json(raw)
        return obj if isinstance(obj, dict) else None

    def enrich(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        try:
            raw = self._complete(ENRICHMENT_SYSTEM, build_enrichment_prompt(rows))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama enrichment failed: %s", exc)
            return {}
        return _parse_enrichment(repair_json(raw))

    def suggest_merges(self, entity_type: str, names: list[str]) -> list[dict[str, Any]]:
        try:
            raw = self._complete(MERGE_SYSTEM, build_merge_prompt(entity_type, names))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama merge suggestion failed: %s", exc)
            return []
        return _parse_merges(repair_json(raw))
