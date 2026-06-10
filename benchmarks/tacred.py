# TACRED / TAC-KBP adapter - LOCAL, license-gated.

from __future__ import annotations

import json
from pathlib import Path

from .common import BenchDoc, BenchEntity, BenchRelation

DEFAULT_TARGET_TYPES = ["PERSON", "ORG", "LOCATION", "DATE"]

TYPE_MAP = {
    "PERSON": "PERSON", "ORGANIZATION": "ORG", "LOCATION": "LOCATION",
    "CITY": "LOCATION", "COUNTRY": "LOCATION", "STATE_OR_PROVINCE": "LOCATION",
    "DATE": "DATE", "NATIONALITY": "MISC", "TITLE": "MISC", "MISC": "MISC",
    "NUMBER": "NUM", "DURATION": "DATE", "CAUSE_OF_DEATH": "MISC",
    "RELIGION": "MISC", "IDEOLOGY": "MISC", "URL": "MISC", "CRIMINAL_CHARGE": "MISC",
}
GLINER_LABELS = ["person", "organization", "location", "date", "number", "miscellaneous"]
LABEL_MAP = {"person": "PERSON", "organization": "ORG", "location": "LOCATION",
             "date": "DATE", "number": "NUM", "miscellaneous": "MISC"}


def load(path: str, limit: int = 0, **_) -> list[BenchDoc]:
    """Load TACRED from a local JSON file (list of examples)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"TACRED file not found: {p}. TACRED is LDC-licensed; pass the JSON via --path."
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    docs: list[BenchDoc] = []
    for i, ex in enumerate(data):
        if limit and i >= limit:
            break
        tokens = ex["token"]
        text = " ".join(tokens)
        subj = " ".join(tokens[ex["subj_start"]:ex["subj_end"] + 1]).strip()
        obj = " ".join(tokens[ex["obj_start"]:ex["obj_end"] + 1]).strip()
        ents = [
            BenchEntity(subj, TYPE_MAP.get(str(ex.get("subj_type", "")).upper(), "MISC")),
            BenchEntity(obj, TYPE_MAP.get(str(ex.get("obj_type", "")).upper(), "MISC")),
        ]
        rels = []
        rel = str(ex.get("relation", "no_relation"))
        if rel and rel != "no_relation" and subj and obj:
            rels.append(BenchRelation(subj, obj, rel.lower()))
        docs.append(BenchDoc(doc_id=str(ex.get("id", f"tacred_{i}")), text=text,
                             entities=ents, relations=rels))
    return docs
