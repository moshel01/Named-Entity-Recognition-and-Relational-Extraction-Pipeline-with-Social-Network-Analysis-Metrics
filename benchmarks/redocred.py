# Re-DocRED adapter (tonytan48/Re-DocRED).

from __future__ import annotations

import re
from collections import Counter

from .common import BenchDoc, BenchEntity, BenchRelation

HF_ID = "tonytan48/Re-DocRED"

# The substantive SNA types to score by default (NUM/MISC/DATE are noisy in
# DocRED and GLiNER is weak on them - they drag the headline down). Override
# with --types on the runner.
DEFAULT_TARGET_TYPES = ["PERSON", "ORG", "LOCATION"]

# Re-DocRED entity type -> canonical pipeline type.
TYPE_MAP = {
    "PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION",
    "TIME": "DATE", "NUM": "NUM", "MISC": "MISC",
}

# GLiNER labels + label_map the pipeline should use for this dataset.
GLINER_LABELS = ["person", "organization", "location", "date", "number", "miscellaneous"]
LABEL_MAP = {
    "person": "PERSON", "organization": "ORG", "location": "LOCATION",
    "date": "DATE", "number": "NUM", "miscellaneous": "MISC",
}

_NO_SPACE_BEFORE = re.compile(r"^[.,;:!?)\]}'\"%]")
_NO_SPACE_AFTER = {"(", "[", "{", "$", "``"}


def _detokenize(tokens: list[str]) -> str:
    out = ""
    prev = ""
    for i, tok in enumerate(tokens):
        if i > 0 and not _NO_SPACE_BEFORE.match(tok) and prev not in _NO_SPACE_AFTER:
            out += " "
        out += tok
        prev = tok
    return out


def _entity_repr(cluster: list[dict]) -> tuple[str, str, list[str]]:
    """Return (representative_name, canonical_type, aliases) for a cluster."""
    names = [m["name"].strip() for m in cluster if m.get("name", "").strip()]
    types = [m.get("type", "MISC") for m in cluster]
    canon = TYPE_MAP.get(Counter(types).most_common(1)[0][0], "MISC")
    if names:
        # Representative = most frequent surface; tie-break to longest.
        freq = Counter(names)
        rep = max(names, key=lambda n: (freq[n], len(n)))
    else:
        rep = "UNKNOWN"
    aliases = sorted({n for n in names if n != rep})
    return rep, canon, aliases


def load(split: str = "test", limit: int = 0) -> list[BenchDoc]:
    """Load Re-DocRED into BenchDocs. ``split`` in {train,validation,test}."""
    from datasets import load_dataset
    ds = load_dataset(HF_ID, split=split)
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    docs: list[BenchDoc] = []
    for i, ex in enumerate(ds):
        text = "\n".join(_detokenize(s) for s in ex["sents"])
        vset = ex["vertexSet"]
        entities: list[BenchEntity] = []
        reps: list[str] = []
        for cluster in vset:
            rep, canon, aliases = _entity_repr(cluster)
            reps.append(rep)
            entities.append(BenchEntity(name=rep, type=canon, aliases=aliases))

        relations: list[BenchRelation] = []
        for lab in ex.get("labels", []):
            h, t = lab.get("h"), lab.get("t")
            if h is None or t is None or h >= len(reps) or t >= len(reps):
                continue
            relations.append(BenchRelation(source=reps[h], target=reps[t],
                                           type=str(lab.get("r", "")).lower()))

        title = ex.get("title") or f"redocred_{split}_{i}"
        docs.append(BenchDoc(doc_id=f"{i:05d}_{title}", text=text,
                             entities=entities, relations=relations))
    return docs
