# Crash-resilient checkpoint manager.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from core.schema import DocumentExtraction

logger = logging.getLogger(__name__)


def _failure_score(rec: DocumentExtraction) -> int:
    """0 = clean, 1 = degraded, 2 = full failure artifact.

    Backends record n_chunks/chunks_failed in meta. A failed LLM chunk still
    passes foundation mentions through, so 'has mentions' proves nothing -
    only the meta accounting separates a failed doc from an empty one.

    Legacy records (no accounting): empty = full failure; mentions with zero
    relationships = the known degraded-LLM signature, ranked below a record
    that did produce relationships but still good enough to count as done."""
    meta = rec.meta or {}
    n, failed = meta.get("n_chunks"), meta.get("chunks_failed")
    if isinstance(n, int) and isinstance(failed, int) and n > 0:
        return 2 if failed >= n else (1 if failed else 0)
    if not rec.mentions and not rec.relationships:
        return 2
    if not rec.relationships:
        # Single-chunk LLM doc with mentions only: the one chunk failed, so
        # nothing is lost by retrying. python_only emits no relationships for
        # some docs legitimately - never treat those as failures.
        if meta.get("n_chunks") == 1 and meta.get("backend") != "python_only":
            return 2
        return 1
    return 0


class CheckpointManager:
    """Append-only JSONL checkpoint with resume support."""

    def __init__(self, checkpoint_dir: str | Path, run_name: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.dir = Path(checkpoint_dir)
        self.run_name = run_name
        self.path = self.dir / f"{run_name}.extractions.jsonl"
        self._done: set[str] = set()
        self._fh = None
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._load_done()

    # Resume
    def _load_done(self) -> None:
        if not self.path.exists():
            return
        failed = 0
        for rec in self._iter_records(skip_bad=True):
            # Full failures stay out of the done set so --resume retries them.
            # Partial failures count as done: the good chunks are not redone.
            if _failure_score(rec) >= 2:
                failed += 1
                continue
            self._done.add(rec.doc_id)
        logger.info("Checkpoint: %d completed documents on disk (use --resume to skip them).",
                    len(self._done))
        if failed:
            logger.info("Checkpoint: %d failed-extraction entries will be retried on --resume.", failed)

    def _iter_records(self, skip_bad: bool = True) -> Iterator[DocumentExtraction]:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    yield DocumentExtraction.from_dict(obj)
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    if skip_bad:
                        logger.debug("Skipping corrupt checkpoint line: %s", exc)
                        continue
                    raise

    # Query
    def is_done(self, doc_id: str) -> bool:
        return doc_id in self._done

    @property
    def completed_ids(self) -> set[str]:
        return set(self._done)

    # Write
    def _ensure_open(self) -> None:
        if self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8")

    def save(self, extraction: DocumentExtraction, flush: bool = True) -> None:
        """Append a document extraction to the checkpoint."""
        if not self.enabled:
            return
        self._ensure_open()
        self._fh.write(json.dumps(extraction.to_dict(), ensure_ascii=False) + "\n")
        self._done.add(extraction.doc_id)
        if flush:
            self._fh.flush()

    # Load all
    def load_all(self) -> list[DocumentExtraction]:
        """Return all checkpointed extractions, deduplicated by doc_id.

        Cleanest record wins, last among equals: a re-run that hit an API
        error mid-doc appends a degraded extraction, which must not shadow a
        clean earlier pass. A failure artifact survives only when the doc
        never produced anything better.
        """
        if not self.path.exists():
            return []
        latest: dict[str, DocumentExtraction] = {}
        for rec in self._iter_records(skip_bad=True):
            prev = latest.get(rec.doc_id)
            if prev is not None and _failure_score(rec) > _failure_score(prev):
                continue
            latest[rec.doc_id] = rec
        return list(latest.values())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "CheckpointManager":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
