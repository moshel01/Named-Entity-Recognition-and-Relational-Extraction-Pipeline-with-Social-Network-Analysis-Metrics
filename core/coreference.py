# Coref: first-person narrator node (EN+DE) + optional fastcoref pronouns.

from __future__ import annotations

import logging
import re
from typing import Optional

from .schema import EntityMention

logger = logging.getLogger(__name__)

# First-person pronouns by language (lowercased, whole-word matched).
_FIRST_PERSON = {
    "en": {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"},
    "de": {"ich", "mich", "mir", "mein", "meine", "meinen", "meinem", "meiner",
           "meines", "wir", "uns", "unser", "unsere", "unserem", "unseren", "unserer"},
}

# Tokens treated as pronouns when re-attaching fastcoref clusters.
_PRONOUN_LIKE = (
    _FIRST_PERSON["en"] | _FIRST_PERSON["de"]
    | {"he", "him", "his", "she", "her", "hers", "they", "them", "their",
       "er", "ihn", "ihm", "sein", "seine", "sie", "ihr", "ihre"}
)


class CoreferenceResolver:
    """Resolve narration and (optionally) pronouns to entity mentions."""

    def __init__(self, config, domain=None) -> None:
        cc = config.coreference
        self.enabled = cc.enabled
        self.narrator_resolution = cc.narrator_resolution
        self.pronoun_resolution = cc.pronoun_resolution
        self.model_name = cc.model
        self.device = cc.device
        self.max_narrator = cc.max_narrator_mentions_per_chunk
        langs = [l for l in cc.languages if l in _FIRST_PERSON] or ["en"]
        words = set().union(*(_FIRST_PERSON[l] for l in langs))
        self._first_person_re = re.compile(
            r"\b(" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")\b",
            re.IGNORECASE,
        )
        self._fcoref = None
        self._fcoref_failed = False

    # Narrator (always available)
    def narrator_mentions(
        self, text: str, doc_id: str, chunk_id: str, offset: int, narrator_name: str
    ) -> list[EntityMention]:
        """Emit narrator mentions at first-person pronoun spans in ``text``."""
        if not (self.enabled and self.narrator_resolution):
            return []
        mentions: list[EntityMention] = []
        for m in self._first_person_re.finditer(text):
            if len(mentions) >= self.max_narrator:
                break
            mentions.append(
                EntityMention(
                    text=narrator_name,
                    label="PERSON",
                    start_char=offset + m.start(),
                    end_char=offset + m.end(),
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    confidence=0.5,
                    sources=["coref_narrator"],
                    attributes={"is_author": True, "narrator": True,
                                "surface_pronoun": m.group(0)},
                )
            )
        return mentions

    # fastcoref (optional)
    def _ensure_fcoref(self) -> None:
        if self._fcoref is not None or self._fcoref_failed:
            return
        try:
            from fastcoref import FCoref
            dev = self.device if self.device != "auto" else "cpu"
            self._fcoref = FCoref(model_name_or_path=self.model_name, device=dev)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "fastcoref unavailable (%s); pronoun resolution disabled.", exc
            )
            self._fcoref_failed = True

    def pronoun_mentions(
        self, text: str, mentions: list[EntityMention], doc_id: str,
        chunk_id: str, offset: int,
    ) -> list[EntityMention]:
        """Re-emit named identities at third-person pronoun spans via fastcoref."""
        if not (self.enabled and self.pronoun_resolution):
            return []
        self._ensure_fcoref()
        if self._fcoref is None:
            return []
        try:
            preds = self._fcoref.predict(texts=[text])
            clusters = preds[0].get_clusters(as_strings=False)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fastcoref prediction failed: %s", exc)
            return []

        # Index named mentions by their chunk-relative span for cluster matching.
        rel_spans = [(mm.start_char - offset, mm.end_char - offset, mm) for mm in mentions]
        extra: list[EntityMention] = []
        for cluster in clusters:
            # Find the named identity in this cluster: a real detected mention
            # whose span overlaps one of the cluster's spans.
            identity: Optional[EntityMention] = None
            for (c_start, c_end) in cluster:
                for s, e, mm in rel_spans:
                    if s < c_end and c_start < e:
                        identity = mm
                        break
                if identity is not None:
                    break
            if identity is None:
                continue
            for (c_start, c_end) in cluster:
                span_text = text[c_start:c_end].strip().lower()
                if span_text not in _PRONOUN_LIKE:
                    continue
                extra.append(
                    EntityMention(
                        text=identity.text,
                        label=identity.label,
                        start_char=offset + c_start,
                        end_char=offset + c_end,
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        confidence=0.45,
                        sources=["coref"],
                        attributes={"resolved_from": span_text},
                    )
                )
        return extra

    # Combined
    def resolve(
        self, text: str, mentions: list[EntityMention], doc_id: str,
        chunk_id: str, offset: int, narrator_name: str,
    ) -> list[EntityMention]:
        """Return all coref-derived extra mentions for a chunk."""
        extra = self.narrator_mentions(text, doc_id, chunk_id, offset, narrator_name)
        extra += self.pronoun_mentions(text, mentions, doc_id, chunk_id, offset)
        return extra
