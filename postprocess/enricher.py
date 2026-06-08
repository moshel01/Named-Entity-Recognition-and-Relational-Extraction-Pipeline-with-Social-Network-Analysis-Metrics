# LLM enrichment of resolved entities: subtype + attributes.

from __future__ import annotations

import logging

from core.schema import Entity

from .aggregator import normalize_name

logger = logging.getLogger(__name__)


# Junk subtypes weak models echo back instead of a real category.
_BAD_SUBTYPES = {"string", "none", "unknown", "n/a", "na", "other", ""}


class Enricher:
    def __init__(self, batch_size: int = 40,
                 subtypes: dict[str, list[str]] | None = None) -> None:
        self.batch_size = max(1, batch_size)
        self.subtypes = subtypes or {}

    def run(self, entities: list[Entity], backend) -> list[Entity]:
        if backend is None:
            return entities
        by_name = {normalize_name(e.canonical_name): e for e in entities}
        n_enriched = 0
        for i in range(0, len(entities), self.batch_size):
            batch = entities[i:i + self.batch_size]
            rows = [{"name": e.canonical_name, "type": e.label,
                     "allowed": self.subtypes.get(e.label, [])} for e in batch]
            try:
                result = backend.enrich(rows)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Enrichment batch failed: %s", exc)
                continue
            for name, rec in (result or {}).items():
                ent = by_name.get(normalize_name(name))
                if ent is None:
                    continue
                st = str(rec.get("subtype") or "").strip().lower()
                # Reject echoes of the type, placeholder junk, and subtypes that
                # belong to a different type's vocabulary (PERSON != nazi_organization).
                allowed = {a.lower() for a in self.subtypes.get(ent.label, [])}
                if st and st not in _BAD_SUBTYPES and st != ent.label.lower() \
                        and (not allowed or st in allowed):
                    ent.tags["subtype"] = st
                for k, v in (rec.get("attributes") or {}).items():
                    if v not in (None, "", "string"):
                        ent.attributes.setdefault(k, v)
                n_enriched += 1
        logger.info("Enrichment: updated %d/%d entities.", n_enriched, len(entities))
        return entities
