# GLiNER2 zero-shot NER (fastino/* models). Windows long chunks; returns
# doc-absolute spans. The original GLiNER (urchade/*) is deprecated - GLiNER2 is
# the only supported NER backend.

from __future__ import annotations

import contextlib
import io
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
    """Wrapper around a GLiNER2 model for zero-shot entity extraction."""

    def __init__(
        self,
        model_name: str = "fastino/gliner2-multi-v1",
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
        try:
            from gliner2 import GLiNER2
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "GLiNER2 is required for NER but the 'gliner2' package is not "
                "installed. Fix: pip install gliner2"
            ) from exc
        logger.info("Loading GLiNER2 model '%s' on %s ...", model_name, device)
        # GLiNER2 prints a banner with emoji during init; on a cp1252 Windows
        # console that raises UnicodeEncodeError. Swallow stdout during load.
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model = GLiNER2.from_pretrained(model_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load GLiNER2 '{model_name}': {exc}\n"
                "Fix: pip install gliner2 ; then clear a partial download:\n"
                f"  rmdir /s %USERPROFILE%\\.cache\\huggingface\\hub\\models--"
                f"{model_name.replace('/', '--')}\nand re-run."
            ) from exc
        for obj in (model, getattr(model, "model", None)):
            try:
                if obj is not None and hasattr(obj, "to"):
                    obj.to(device)
                    break
            except Exception:  # noqa: BLE001
                logger.debug("Could not move GLiNER2 to %s; staying on CPU.", device)
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

    def _parse_result(self, res: dict) -> list[dict]:
        """Normalize one GLiNER2 result to [{text,label,start,end,score}]."""
        out: list[dict] = []
        for label, items in (res.get("entities") or {}).items():
            for it in items or []:
                if not isinstance(it, dict):
                    continue
                start, end = it.get("start"), it.get("end")
                if not isinstance(start, int) or not isinstance(end, int):
                    continue
                score = float(it.get("confidence", 0.0))
                if score < self.threshold:
                    continue
                out.append({"text": it.get("text", ""), "label": label,
                            "start": start, "end": end, "score": score})
        return out

    def _predict(self, text: str) -> list[dict]:
        """Run GLiNER2 and normalize to a list of {text,label,start,end,score}."""
        with contextlib.redirect_stdout(io.StringIO()):
            res = self.model.extract_entities(
                text, self.labels, include_spans=True, include_confidence=True
            )
        return self._parse_result(res)

    def _predict_batch(self, texts: list[str]) -> list[list[dict]]:
        """One GPU call for many windows. Caller falls back to per-window on error."""
        with contextlib.redirect_stdout(io.StringIO()):
            results = self.model.batch_extract_entities(
                texts, self.labels, include_spans=True, include_confidence=True
            )
        return [self._parse_result(r) for r in results]

    def extract(
        self,
        text: str,
        chunk_id: str,
        doc_id: str,
        offset: int = 0,
    ) -> list[EntityMention]:
        """Extract entity mentions from chunk ``text``."""
        seen: dict[tuple[int, int], EntityMention] = {}
        windows = self._windows(text)
        # Batch a multi-window chunk in one GPU call; fall back to per-window on any
        # error so a batch-API mismatch can never break NER (it only speeds it up).
        batched: list[list[dict]] | None = None
        if len(windows) > 1:
            try:
                batched = self._predict_batch([w for _, w in windows])
            except Exception as exc:  # noqa: BLE001
                logger.warning("GLiNER2 batch predict failed (%s); per-window.", exc)
                batched = None
        for i, (win_offset, win_text) in enumerate(windows):
            if batched is not None:
                preds = batched[i]
            else:
                try:
                    preds = self._predict(win_text)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLiNER2 prediction failed on a window: %s", exc)
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
