# Crash-resilient checkpoint manager.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from core.schema import DocumentExtraction

logger = logging.getLogger(__name__)


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
        good_lines = 0
        for rec in self._iter_records(skip_bad=True):
            self._done.add(rec.doc_id)
            good_lines += 1
        logger.info("Checkpoint: %d completed documents on disk (use --resume to skip them).", good_lines)

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
        """Return all checkpointed extractions (deduplicated by doc_id, last wins)."""
        if not self.path.exists():
            return []
        latest: dict[str, DocumentExtraction] = {}
        for rec in self._iter_records(skip_bad=True):
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
