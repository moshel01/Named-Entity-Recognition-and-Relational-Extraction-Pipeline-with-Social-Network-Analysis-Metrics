# Mode 4: manual batch / "gemini_batch". Instead of chunking and calling an LLM
# per chunk, emit ONE self-contained prompt file holding whole documents + the
# JSON schema, paste it into a long-context model (Gemini 2M, Claude, etc.), and
# feed the returned JSON back in. No chunk-boundary recall loss, no API key, and
# the model sees each document whole. Two halves:
#   build_batch_prompt(...) -> prompt file(s)   (the extract stage writes these)
#   parse_batch_response(...) -> [DocumentExtraction]  (the analyze stage imports)
# The parser reuses the same _map_extraction the api/ollama backends use, so the
# downstream pipeline can't tell the difference.

from __future__ import annotations

import json
import logging
from typing import Any

from core.schema import DocumentExtraction

from .api_backend import _map_extraction
from .base import _PRONOUNS, _remap_pronoun_endpoints, _tag_ungrounded_evidence
from .json_repair import repair_json
from .prompts import (
    EXTRACTION_SYSTEM,
    qualifier_constraint_block,
    relation_constraint_block,
    relationship_schema_str,
)

logger = logging.getLogger(__name__)

# Per-doc text budget per prompt file. Gemini 2.x holds ~2M tokens (~8M chars
# English), but leave headroom for the scaffolding and the (large) JSON output, so
# default well under that. A corpus past the budget splits into numbered files.
DEFAULT_CHAR_BUDGET = 4_000_000

_DOC_OPEN = '<doc id="{doc_id}">'
_DOC_CLOSE = "</doc>"


def extraction_spec(config, domain=None) -> dict:
    """Label types, relation ontology/guide, qualifiers, type hints, and date vocab
    the prompt + parser need - the same derivation the live backends do in their
    __init__, but standalone (gemini_batch never instantiates a backend)."""
    from postprocess.ontology import (relation_signature_hints, resolve_relation_guide,
                                       resolve_relation_ontology)
    types = set(config.foundation.label_map.values())
    if domain is not None:
        types.update(domain.gliner_label_map().values())
    relation_types = sorted(resolve_relation_ontology(config, domain).keys())
    type_sigs = (relation_signature_hints(relation_types)
                 if config.intelligence.type_hints else {})
    v = domain.temporal_vocab() if domain is not None else {}
    return {
        "label_types": sorted(types),
        "relation_types": relation_types,
        "relation_guide": resolve_relation_guide(config, domain),
        "edge_qualifiers": list(config.intelligence.edge_qualifiers or []),
        "type_signatures": type_sigs,
        "date_vocab": (v.get("months", {}), v.get("seasons", {}), v.get("pivot_max")),
    }


_NARRATOR_INSTR = (
    "\nFIRST-PERSON: when a <doc> tag carries author=\"NAME\", that document is a "
    "first-person account by NAME - use NAME as the entity for every first-person "
    "reference (I, my, we; ich, mir, mein, wir) and never output a pronoun as an "
    "entity. Third parties named in it stay themselves.\n"
)


def _header(label_types: list[str], relation_types: list[str] | None,
            relation_guide: dict[str, str] | None, edge_qualifiers: list[str] | None,
            type_signatures: dict[str, str] | None, has_authors: bool = False) -> str:
    """The shared instruction block - identical across split files."""
    rel = relation_constraint_block(relation_types, relation_guide, type_signatures)
    qual = qualifier_constraint_block(edge_qualifiers)
    inner = relationship_schema_str(edge_qualifiers)
    return (
        EXTRACTION_SYSTEM
        + "\n\nBATCH MODE: you receive MULTIPLE documents, each wrapped in "
        '<doc id="...">...</doc> tags. Process EACH document on its own (use the '
        "whole document as context) and return a SINGLE JSON object whose keys are "
        "the exact doc ids, each mapping to that document's extraction:\n"
        '{\n  "<doc id>": ' + inner.replace("\n", "\n  ") + "\n}\n"
        "Emit one key per document, nothing outside the JSON object.\n"
        f"\nENTITY TYPES IN USE: {', '.join(label_types)}\n"
        + (_NARRATOR_INSTR if has_authors else "")
        + rel + qual
    )


def _doc_tag(doc_id: str, author: str) -> str:
    if author:
        return f'<doc id="{doc_id}" author="{author.replace(chr(34), chr(39))}">'
    return _DOC_OPEN.format(doc_id=doc_id)


def build_batch_prompt(
    documents: list[tuple[str, str]],
    label_types: list[str],
    relation_types: list[str] | None = None,
    relation_guide: dict[str, str] | None = None,
    edge_qualifiers: list[str] | None = None,
    type_signatures: dict[str, str] | None = None,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    max_docs: int = 0,
    authors: dict[str, str] | None = None,
) -> list[str]:
    """Build one or more prompt strings from (doc_id, text) pairs.

    Splits into multiple self-contained files when a batch hits ``max_docs``
    documents OR ``char_budget`` characters - whichever comes first. Doc count is
    usually the real limit: the model's JSON REPLY length scales with the number of
    documents, not input size, so a too-large batch gets truncated mid-reply. A
    single document larger than the budget is still emitted whole. ``authors``
    (doc_id -> name) stamps a first-person narrator into its <doc> tag."""
    authors = authors or {}
    header = _header(label_types, relation_types, relation_guide,
                     edge_qualifiers, type_signatures, has_authors=bool(authors))
    prompts: list[str] = []
    batch: list[str] = []
    size = 0
    for doc_id, text in documents:
        block = f"{_doc_tag(doc_id, authors.get(doc_id, ''))}\n{text}\n{_DOC_CLOSE}\n"
        full = (max_docs and len(batch) >= max_docs) or (size + len(block) > char_budget)
        if batch and full:
            prompts.append(_assemble(header, batch))
            batch, size = [], 0
        batch.append(block)
        size += len(block)
    if batch:
        prompts.append(_assemble(header, batch))
    return prompts


def _assemble(header: str, blocks: list[str]) -> str:
    return header + "\nDOCUMENTS:\n" + "\n".join(blocks)


def _retry_after_seconds(resp) -> "float | None":
    """The server-requested wait for a throttled response, or None.

    Honors the `Retry-After` header, else Gemini's RetryInfo.retryDelay (e.g.
    "27s") in the error body. Lets a 429 back off exactly to the quota refresh
    instead of guessing - the per-minute limit refreshes in seconds, so a fixed
    exponential climb both over-waits (early) and under-waits (re-trips the limit)."""
    ra = getattr(resp, "headers", {}).get("Retry-After") if hasattr(resp, "headers") else None
    if ra:
        try:
            return float(ra)
        except (TypeError, ValueError):
            pass
    try:
        for d in (resp.json().get("error", {}).get("details", []) or []):
            rd = d.get("retryDelay")
            if isinstance(rd, str) and rd.endswith("s"):
                return float(rd[:-1])
    except Exception:  # noqa: BLE001 - body not JSON / unexpected shape -> no hint
        pass
    return None


def submit_to_gemini(prompt: str, api_key: str, model: str = "gemini-2.5-flash",
                     base_url: str = "", max_output_tokens: int = 65536,
                     thinking_budget: int = 0, timeout: int = 600,
                     max_retries: int = 5) -> str:
    """POST one batch prompt to the Gemini REST API; return the raw text reply.

    Forces JSON output and a high output-token cap - the truncation fix the chat
    UI won't let you set. thinking_budget 0 turns off the default reasoning pass so
    the whole output budget goes to JSON, not thinking tokens (which otherwise count
    against maxOutputTokens and truncate the reply); <0 keeps the API default.
    Retries 429/5xx with exponential backoff. Raises on a non-recoverable error so
    the caller can skip that batch and keep going."""
    import time

    import requests
    base = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
    url = f"{base}/v1beta/models/{model}:generateContent"
    gen_config = {"temperature": 0, "maxOutputTokens": max_output_tokens,
                  "responseMimeType": "application/json"}
    # thinkingConfig is a Gemini 2.5+ feature; Gemma (and the open models) reject it,
    # so never send it for those - lets `--batch-model gemma-...` work unchanged.
    supports_thinking = "gemma" not in model.lower()
    if thinking_budget >= 0 and supports_thinking:
        tb = thinking_budget
        if tb == 0 and "pro" in model.lower():
            tb = 128  # 2.5-pro can't fully disable thinking; 128 is its floor
        gen_config["thinkingConfig"] = {"thinkingBudget": tb}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    backoff = 5.0
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            # Read timeout / dropped connection = the server stalled before sending a
            # byte, not us. The POST blocks waiting on the response; nothing is parsed
            # until it returns (line below), so this is never our ingest speed - it is
            # the model taking >timeout to generate. The free gemma endpoint does this
            # under load; retry like a 5xx instead of failing the batch. Last attempt
            # re-raises so the caller skips it and --resume re-queues it later.
            if attempt < max_retries - 1:
                logger.warning("Gemini network error (attempt %d/%d): %s; waiting %.0fs",
                               attempt + 1, max_retries, type(exc).__name__, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                continue
            raise
        if resp.status_code in (429, 500, 503) and attempt < max_retries - 1:
            server = _retry_after_seconds(resp)
            # Server hint wins (capped so a day-quota delay fails fast -> --resume
            # later rather than hang for hours); else exponential backoff.
            wait = min(server, 120.0) if server is not None else backoff
            logger.warning("Gemini %s (attempt %d/%d); waiting %.0fs%s",
                           resp.status_code, attempt + 1, max_retries, wait,
                           " (server-requested)" if server is not None else "")
            time.sleep(wait)
            backoff = min(backoff * 2, 120)
            continue
        resp.raise_for_status()
        data = resp.json()
        cands = data.get("candidates") or []
        if not cands:
            raise RuntimeError(f"Gemini returned no candidates: {str(data)[:300]}")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        finish = cands[0].get("finishReason", "")
        if finish == "MAX_TOKENS":
            logger.warning("Gemini hit MAX_TOKENS - this batch is truncated; use a "
                           "smaller --batch-docs.")
        elif finish and finish != "STOP":
            logger.warning("Gemini finishReason=%s for a batch.", finish)
        return text
    raise RuntimeError("Gemini submit exhausted retries")


def _coerce_doc_map(obj: Any) -> dict[str, dict]:
    """Normalize the model's reply to {doc_id: {entities, relationships, timeline}}.

    Accepts the keyed object we ask for, a list of {doc_id|id: ..., ...} records,
    or a bare single-document object (one unkeyed extraction)."""
    if isinstance(obj, str):
        obj = repair_json(obj)
    if isinstance(obj, dict):
        # Already keyed by doc id, OR a single bare extraction.
        if any(k in obj for k in ("entities", "relationships", "timeline")):
            return {"": obj}
        return {str(k): v for k, v in obj.items() if isinstance(v, dict)}
    if isinstance(obj, list):
        out: dict[str, dict] = {}
        for rec in obj:
            if isinstance(rec, dict):
                did = str(rec.get("doc_id") or rec.get("id") or len(out))
                out[did] = rec
        return out
    return {}


def _flag_narrator(mentions: list, author: str) -> None:
    """Mark the document's first-person narrator mention is_author, so its entity
    carries is_author downstream - the batch analog of coref's narrator flag. The
    letter_id stamp (main._apply_metadata) and the whole metadata join key off it,
    so without this the German metadata merges onto zero authors. Exact normalized
    name wins; else the closest spelling variant >= 0.9, since the model sometimes
    'corrects' the filename's spelling from the text (Vilwak -> Villwak)."""
    if not author:
        return
    import difflib

    from postprocess.aggregator import normalize_name
    want = normalize_name(author)
    if not want:
        return
    persons = [m for m in mentions if m.label == "PERSON"]
    exact = [m for m in persons if normalize_name(m.text) == want]
    if exact:
        for m in exact:
            m.attributes["is_author"] = True
        return
    best, best_r = None, 0.0
    for m in persons:
        r = difflib.SequenceMatcher(None, normalize_name(m.text), want).ratio()
        if r > best_r:
            best, best_r = m, r
    if best is not None and best_r >= 0.9:
        best.attributes["is_author"] = True


def parse_batch_response(
    response: Any,
    doc_meta: dict[str, dict[str, str]],
    label_types: list[str],
    edge_qualifiers: list[str] | None = None,
    date_vocab: tuple[dict, dict, Any] = ({}, {}, None),
) -> list[DocumentExtraction]:
    """Map a model's batch reply onto DocumentExtraction records (one per doc).

    ``doc_meta`` maps doc_id -> {"text", "source_path", "author"}; the text drives
    the evidence-verbatim check and the author the pronoun remap, exactly as the
    live backends do. doc ids the reply doesn't cover are skipped (logged)."""
    months, seasons, pivot = date_vocab
    doc_map = _coerce_doc_map(response)
    out: list[DocumentExtraction] = []
    for doc_id, data in doc_map.items():
        meta = doc_meta.get(doc_id, {})
        text = meta.get("text", "")
        author = meta.get("author", "")
        mentions, rels, timeline = _map_extraction(
            data if isinstance(data, dict) else {}, [], doc_id, doc_id, label_types,
            months, seasons, pivot, chunk_text=text,
            qualifiers=edge_qualifiers or None,
        )
        if author:
            rels = _remap_pronoun_endpoints(rels, author)
            _flag_narrator(mentions, author)
        ungrounded = _tag_ungrounded_evidence(rels, author)
        mentions = [m for m in mentions if m.text.strip().lower() not in _PRONOUNS]
        out.append(DocumentExtraction(
            doc_id=doc_id, source_path=meta.get("source_path", ""),
            mentions=mentions, relationships=rels, timeline=timeline,
            meta={"backend": "gemini_batch", "evidence_ungrounded": ungrounded},
        ))
    # No missing-doc warning here: with split files this parser sees one batch at a
    # time, so the caller (which knows the full set) does the coverage check.
    return out
