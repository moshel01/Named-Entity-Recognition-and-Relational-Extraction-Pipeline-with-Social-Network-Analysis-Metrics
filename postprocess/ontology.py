# Relation ontology alignment.

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Optional

from core.schema import Relationship

logger = logging.getLogger(__name__)

_NORM = re.compile(r"[\s_\-]+")


def _norm(s: str) -> str:
    return _NORM.sub(" ", str(s).strip().lower())


def resolve_relation_ontology(config, domain=None) -> dict[str, list[str]]:
    """Resolve the active relation ontology from config, else domain, else {}."""
    rel = getattr(getattr(config, "ontology", None), "relations", None)
    if rel:
        if isinstance(rel, dict):
            return {k: list(v or []) for k, v in rel.items()}
        if isinstance(rel, (list, tuple)):
            return {str(c): [] for c in rel}
    if domain is not None:
        try:
            dom = domain.relation_ontology()
            if dom:
                return {k: list(v or []) for k, v in dom.items()}
        except Exception:  # noqa: BLE001
            pass
    return {}


class OntologyAligner:
    """Align relation-type strings onto a canonical ontology."""

    def __init__(self, ontology: Optional[dict[str, list[str]]],
                 fuzzy_threshold: float = 0.82, drop_unmapped: bool = False) -> None:
        self.ontology = ontology or {}
        self.threshold = fuzzy_threshold
        self.drop_unmapped = drop_unmapped
        # Normalized synonym/canonical -> canonical.
        self.index: dict[str, str] = {}
        for canon, syns in self.ontology.items():
            self.index[_norm(canon)] = canon
            for s in syns:
                self.index[_norm(s)] = canon

    @property
    def active(self) -> bool:
        return bool(self.ontology)

    @property
    def canonical_types(self) -> list[str]:
        return sorted(self.ontology.keys())

    def align(self, rel_type: str) -> Optional[str]:
        """Return the canonical type for ``rel_type`` or None if unmatched."""
        if not self.active:
            return rel_type
        n = _norm(rel_type)
        if not n:
            return None
        if n in self.index:                                  # 1. exact
            return self.index[n]
        for key, canon in self.index.items():                # 2. substring
            if key and (key in n or n in key):
                return canon
        best, best_r = None, 0.0                             # 3. fuzzy
        for key, canon in self.index.items():
            r = SequenceMatcher(None, n, key).ratio()
            if r > best_r:
                best, best_r = canon, r
        return best if best_r >= self.threshold else None

    def apply(self, relationships: list[Relationship]) -> list[Relationship]:
        """Remap relation types in place; optionally drop unmapped relations."""
        if not self.active:
            return relationships
        out: list[Relationship] = []
        n_aligned = n_unmapped = n_dropped = 0
        for r in relationships:
            mapped = self.align(r.rel_type)
            if mapped is None:
                if self.drop_unmapped:
                    n_dropped += 1
                    continue
                r.attributes["ontology"] = "unmapped"
                n_unmapped += 1
            else:
                if mapped != r.rel_type:
                    n_aligned += 1
                r.rel_type = mapped
                r.attributes["ontology"] = "aligned"
            out.append(r)
        logger.info("Ontology alignment: %d remapped, %d unmapped, %d dropped.",
                    n_aligned, n_unmapped, n_dropped)
        return out
