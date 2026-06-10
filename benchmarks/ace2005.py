# ACE 2005 adapter - LOCAL, license-gated (not available on Hugging Face).

from __future__ import annotations

import json
from pathlib import Path

from .common import BenchDoc, BenchEntity, BenchRelation

DEFAULT_TARGET_TYPES = ["PERSON", "ORG", "LOCATION"]

TYPE_MAP = {
    "PER": "PERSON", "PERSON": "PERSON",
    "ORG": "ORG", "ORGANIZATION": "ORG",
    "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION", "LOCATION": "LOCATION",
    "VEH": "MISC", "WEA": "MISC",
}
GLINER_LABELS = ["person", "organization", "location", "facility", "vehicle", "weapon"]
LABEL_MAP = {"person": "PERSON", "organization": "ORG", "location": "LOCATION",
             "facility": "LOCATION", "vehicle": "MISC", "weapon": "MISC"}


def _detok(tokens: list[str]) -> str:
    return " ".join(tokens)


def _parse_line(obj: dict, idx: int) -> BenchDoc:
    doc_id = str(obj.get("doc_id") or obj.get("sent_id") or obj.get("id") or f"ace_{idx}")
    tokens = obj.get("tokens") or obj.get("sentence") or []
    text = _detok(tokens) if isinstance(tokens, list) else str(tokens)

    raw_ents = obj.get("entity_mentions") or obj.get("ner") or []
    ents: list[BenchEntity] = []
    id_to_name: dict[str, str] = {}
    for j, e in enumerate(raw_ents):
        if isinstance(e, dict):
            name = (e.get("text") or " ".join(tokens[e.get("start", 0):e.get("end", 0)])).strip()
            etype = TYPE_MAP.get(str(e.get("entity_type") or e.get("type") or "").upper(), "MISC")
            eid = str(e.get("id") or e.get("entity_id") or j)
        else:  # DyGIE list form [start, end, type]
            s, en, typ = e[0], e[1], e[2]
            name = " ".join(tokens[s:en + 1]).strip()
            etype = TYPE_MAP.get(str(typ).upper(), "MISC")
            eid = f"{s}_{en}"
        if name:
            id_to_name[eid] = name
            ents.append(BenchEntity(name=name, type=etype))

    rels: list[BenchRelation] = []
    for r in obj.get("relation_mentions") or obj.get("relations") or []:
        if isinstance(r, dict) and "arguments" in r:
            args = r["arguments"]
            if len(args) >= 2:
                a = id_to_name.get(str(args[0].get("entity_id")))
                b = id_to_name.get(str(args[1].get("entity_id")))
                if a and b:
                    rels.append(BenchRelation(a, b, str(r.get("relation_type") or r.get("type") or "").lower()))
        elif isinstance(r, (list, tuple)) and len(r) >= 5:
            hs, he, ts, te, typ = r[0], r[1], r[2], r[3], r[4]
            a = " ".join(tokens[hs:he + 1]).strip()
            b = " ".join(tokens[ts:te + 1]).strip()
            if a and b:
                rels.append(BenchRelation(a, b, str(typ).lower()))
    return BenchDoc(doc_id=doc_id, text=text, entities=ents, relations=rels)


def load(path: str, limit: int = 0, **_) -> list[BenchDoc]:
    """Load ACE2005 from a local OneIE/DyGIE JSONL file at ``path``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"ACE2005 file not found: {p}. ACE2005 is LDC-licensed; preprocess it "
            f"to OneIE/DyGIE JSONL and pass --path."
        )
    docs: list[BenchDoc] = []
    with p.open("r", encoding="utf-8") as fh:
        # Support both JSONL and a single JSON array.
        head = fh.read(1); fh.seek(0)
        records = json.load(fh) if head == "[" else (json.loads(l) for l in fh if l.strip())
        for i, obj in enumerate(records):
            if limit and i >= limit:
                break
            docs.append(_parse_line(obj, i))
    return docs
