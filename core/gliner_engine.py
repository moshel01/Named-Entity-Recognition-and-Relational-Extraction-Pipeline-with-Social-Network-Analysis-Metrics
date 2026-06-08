# GLiNER zero-shot NER. Supports both the original GLiNER (urchade/*) and
# GLiNER2 (fastino/*). Windows long chunks; returns doc-absolute spans.

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


def _pick_backend(backend: str, model_name: str) -> str:
    """Decide whether a model name is GLiNER (v1) or GLiNER2."""
    if backend in ("gliner", "gliner2"):
        return backend
    name = model_name.lower()
    if "gliner2" in name or name.startswith("fastino/"):
        return "gliner2"
    return "gliner"


class GlinerEngine:
    """Wrapper around a GLiNER / GLiNER2 model for zero-shot entity extraction."""

    def __init__(
        self,
        model_name: str = "urchade/gliner_large-v2.1",
        labels: Optional[list[str]] = None,
        threshold: float = 0.45,
        device: str = "auto",
        label_map: Optional[dict[str, str]] = None,
        window_chars: int = 2000,
        window_overlap: int = 200,
        backend: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.labels = labels or ["person", "organization", "location", "event"]
        self.threshold = threshold
        self.label_map = label_map or {}
        self.window_chars = window_chars
        self.window_overlap = window_overlap
        self.device = _resolve_device(device)
        self.backend = _pick_backend(backend, model_name)
        self.model = self._load(model_name, self.device, self.backend)

    # Model loading
    @staticmethod
    def _load(model_name: str, device: str, backend: str):
        if backend == "gliner2":
            return GlinerEngine._load_v2(model_name, device)
        return GlinerEngine._load_v1(model_name, device)

    @staticmethod
    def _load_v1(model_name: str, device: str):
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

    @staticmethod
    def _load_v2(model_name: str, device: str):
        try:
            from gliner2 import GLiNER2
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                f"GLiNER2 model '{model_name}' requested but the 'gliner2' package "
                "is not installed. Fix: pip install gliner2"
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

    # Prediction (normalized to a list of {text,label,start,end,score} dicts)
    def _predict(self, text: str) -> list[dict]:
        if self.backend == "gliner2":
            return self._predict_v2(text)
        return self.model.predict_entities(text, self.labels, threshold=self.threshold)

    def _predict_v2(self, text: str) -> list[dict]:
        with contextlib.redirect_stdout(io.StringIO()):
            res = self.model.extract_entities(
                text, self.labels, include_spans=True, include_confidence=True
            )
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
                preds = self._predict(win_text)
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
