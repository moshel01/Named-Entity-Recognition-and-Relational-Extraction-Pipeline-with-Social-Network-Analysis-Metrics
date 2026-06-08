# GLiNER zero-shot NER. Windows long chunks; returns doc-absolute spans.

from __future__ import annotations

import logging
from typing import Optional

from .schema import EntityMention

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class GlinerEngine:
    """Wrapper around a GLiNER model for zero-shot entity extraction."""

    def __init__(
        self,
        model_name: str = "urchade/gliner_large-v2.1",
        labels: Optional[list[str]] = None,
        threshold: float = 0.45,
        device: str = "auto",
        label_map: Optional[dict[str, str]] = None,
        window_chars: int = 2000,
        window_overlap: int = 200,
    ) -> None:
        self.model_name = model_name
        self.labels = labels or ["person", "organization", "location", "event"]
        self.threshold = threshold
        self.label_map = label_map or {}
        self.window_chars = window_chars
        self.window_overlap = window_overlap
        self.device = _resolve_device(device)
        self.model = self._load(model_name, self.device)

    @staticmethod
    def _load(model_name: str, device: str):
        from gliner import GLiNER
        logger.info("Loading GLiNER model '%s' on %s ...", model_name, device)
        try:
            model = GLiNER.from_pretrained(model_name)
        except Exception as exc:  # noqa: BLE001
            # mdeberta/deberta GLiNER models need a SentencePiece tokenizer; a
            # truncated spm.model download trips the tiktoken fallback.
            raise RuntimeError(
                f"Failed to load GLiNER '{model_name}': {exc}\n"
                "Fix: pip install sentencepiece tiktoken protobuf ; then clear a "
                "partial download:\n  rmdir /s "
                f"%USERPROFILE%\\.cache\\huggingface\\hub\\models--{model_name.replace('/', '--')}\n"
                "and re-run."
            ) from exc
        try:
            model = model.to(device)
        except Exception:  # noqa: BLE001
            logger.debug("Could not move GLiNER to %s; staying on CPU.", device)
        return model

    def _canon(self, gliner_label: str) -> str:
        return self.label_map.get(gliner_label.lower(), gliner_label.upper())

    def _windows(self, text: str) -> list[tuple[int, str]]:
        """Yield (offset, window_text) covering ``text`` with overlap."""
        if len(text) <= self.window_chars:
            return [(0, text)]
        windows: list[tuple[int, str]] = []
        step = max(1, self.window_chars - self.window_overlap)
        pos = 0
        while pos < len(text):
            windows.append((pos, text[pos:pos + self.window_chars]))
            pos += step
        return windows

    def extract(
        self,
        text: str,
        chunk_id: str,
        doc_id: str,
        offset: int = 0,
    ) -> list[EntityMention]:
        """Extract entity mentions from chunk ``text``."""
        seen: dict[tuple[int, int], EntityMention] = {}
        for win_offset, win_text in self._windows(text):
            try:
                preds = self.model.predict_entities(
                    win_text, self.labels, threshold=self.threshold
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("GLiNER prediction failed on a window: %s", exc)
                continue
            for p in preds:
                abs_start = offset + win_offset + p["start"]
                abs_end = offset + win_offset + p["end"]
                key = (abs_start, abs_end)
                conf = float(p.get("score", 0.0))
                existing = seen.get(key)
                if existing is None or conf > existing.confidence:
                    seen[key] = EntityMention(
                        text=p["text"],
                        label=self._canon(p["label"]),
                        start_char=abs_start,
                        end_char=abs_end,
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        confidence=conf,
                        sources=["gliner"],
                    )
        return list(seen.values())
