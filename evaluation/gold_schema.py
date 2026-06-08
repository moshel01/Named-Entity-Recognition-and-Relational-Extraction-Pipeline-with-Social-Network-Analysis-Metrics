# Gold-annotation schema and loader for the evaluation harness.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GoldEntity:
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)

    @property
    def surface_forms(self) -> list[str]:
        """All acceptable surface forms (canonical name + aliases)."""
        return [self.name, *self.aliases]


@dataclass
class GoldRelation:
    source: str
    target: str
    type: str = ""          # empty => untyped


@dataclass
class GoldDocument:
    doc_id: str
    entities: list[GoldEntity] = field(default_factory=list)
    relations: list[GoldRelation] = field(default_factory=list)


@dataclass
class GoldSet:
    documents: list[GoldDocument] = field(default_factory=list)

    @property
    def entities(self) -> list[GoldEntity]:
        return [e for d in self.documents for e in d.entities]

    @property
    def relations(self) -> list[GoldRelation]:
        return [r for d in self.documents for r in d.relations]


def load_gold(path: str | Path) -> GoldSet:
    """Load and validate a gold-annotation JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Gold file not found: {p}")
    raw: Any = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("documents", [])
    if not isinstance(raw, list):
        raise ValueError("Gold file must be a list of documents or {'documents': [...]}.")

    docs: list[GoldDocument] = []
    for i, d in enumerate(raw):
        if not isinstance(d, dict):
            raise ValueError(f"Document {i} is not an object.")
        doc_id = str(d.get("doc_id") or d.get("source") or f"doc_{i}")
        ents = [
            GoldEntity(
                name=str(e["name"]).strip(),
                type=str(e.get("type", "")).upper().strip(),
                aliases=[str(a).strip() for a in e.get("aliases", []) if str(a).strip()],
            )
            for e in d.get("entities", []) if str(e.get("name", "")).strip()
        ]
        rels = []
        for r in d.get("relations", []):
            src = str(r.get("source", "")).strip()
            tgt = str(r.get("target", "")).strip()
            if not src or not tgt:
                continue
            rels.append(GoldRelation(source=src, target=tgt,
                                     type=str(r.get("type", "")).lower().strip()))
        docs.append(GoldDocument(doc_id=doc_id, entities=ents, relations=rels))
    return GoldSet(documents=docs)
