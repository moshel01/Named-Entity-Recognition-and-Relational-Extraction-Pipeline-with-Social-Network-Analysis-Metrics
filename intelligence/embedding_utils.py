# Sentence-transformer utilities for the python-only backend (Mode 2).

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Lazy wrapper around a sentence-transformers model with cosine helpers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._available = True

    def _ensure(self) -> None:
        if self._model is not None or not self._available:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sentence-transformers unavailable (%s); embedding scores "
                "will default to 0.5.", exc,
            )
            self._available = False

    @property
    def available(self) -> bool:
        self._ensure()
        return self._available

    def encode(self, texts: list[str]):
        """Return L2-normalized embeddings for ``texts`` (or None if unavailable)."""
        self._ensure()
        if not self._available:
            return None
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two strings in [0,1]; 0.5 if unavailable."""
        self._ensure()
        if not self._available or not a or not b:
            return 0.5
        import numpy as np
        embs = self.encode([a, b])
        if embs is None:
            return 0.5
        sim = float(np.dot(embs[0], embs[1]))
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

    def pairwise_to_anchor(self, anchor: str, candidates: list[str]) -> list[float]:
        """Cosine similarities of each candidate to the anchor string."""
        self._ensure()
        if not self._available or not candidates:
            return [0.5] * len(candidates)
        import numpy as np
        embs = self.encode([anchor] + candidates)
        if embs is None:
            return [0.5] * len(candidates)
        anchor_vec = embs[0]
        sims = []
        for vec in embs[1:]:
            sim = float(np.dot(anchor_vec, vec))
            sims.append(max(0.0, min(1.0, (sim + 1.0) / 2.0)))
        return sims


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str) -> EmbeddingModel:
    """Cached accessor so the model is loaded at most once per name."""
    return EmbeddingModel(model_name)
