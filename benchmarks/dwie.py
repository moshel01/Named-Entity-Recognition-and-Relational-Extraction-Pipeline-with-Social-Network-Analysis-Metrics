# DWIE adapter (DFKI-SLT/DWIE, config 'Task_1').

from __future__ import annotations

from collections import defaultdict

from .common import BenchDoc, BenchEntity, BenchRelation

HF_ID = "DFKI-SLT/DWIE"
HF_CONFIG = "Task_1"

# EVENT dropped from defaults: DWIE gold EVENT is tiny (~51) while GLiNER/spaCy
# over-produce it (P~0.13), dragging the headline. Add it back with
# --types PERSON,ORG,LOCATION,EVENT if you specifically want event nodes.
DEFAULT_TARGET_TYPES = ["PERSON", "ORG", "LOCATION"]

# DWIE fine-grained "type::X" tag -> canonical pipeline type.
TYPE_MAP = {
    "person": "PERSON",
    "organization": "ORG", "company": "ORG", "igo": "ORG", "ngo": "ORG",
    "media": "ORG", "political_party": "ORG", "broadcaster": "ORG",
    "location": "LOCATION", "gpe": "LOCATION", "facility": "LOCATION",
    "event": "EVENT",
}

GLINER_LABELS = ["person", "organization", "location", "event"]
LABEL_MAP = {"person": "PERSON", "organization": "ORG",
             "location": "LOCATION", "event": "EVENT"}


def _concept_type(tags: list[str]) -> str | None:
    """Resolve a concept's canonical type from its 'type::X' tags."""
    raw_types = [t.split("::", 1)[1] for t in tags if t.startswith("type::")]
    for rt in raw_types:
        if rt in TYPE_MAP:
            return TYPE_MAP[rt]
    # Has a type tag but unmapped -> MISC; no type tag -> not an entity.
    return "MISC" if raw_types else None


def load(split: str = "train", limit: int = 0, **_) -> list[BenchDoc]:
    """Load DWIE into BenchDocs (HF version exposes a single 'train' split)."""
    from datasets import load_dataset
    ds = load_dataset(HF_ID, HF_CONFIG, split=split)
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    docs: list[BenchDoc] = []
    for ex in ds:
        text = ex["content"] or ""

        # Mentions per concept -> aliases.
        alias_by_concept: dict[int, set[str]] = defaultdict(set)
        for m in ex.get("mentions", []):
            t = (m.get("text") or "").strip()
            if t:
                alias_by_concept[m.get("concept")].add(t)

        entities: list[BenchEntity] = []
        rep_by_id: dict[int, str] = {}
        type_by_id: dict[int, str] = {}
        for c in ex.get("concepts", []):
            ctype = _concept_type(c.get("tags", []) or [])
            if ctype is None:
                continue
            cid = c.get("concept")
            rep = (c.get("text") or "").strip()
            aliases = sorted(a for a in alias_by_concept.get(cid, set()) if a != rep)
            if not rep and aliases:
                rep, aliases = aliases[0], aliases[1:]
            if not rep:
                continue
            rep_by_id[cid] = rep
            type_by_id[cid] = ctype
            entities.append(BenchEntity(name=rep, type=ctype, aliases=aliases))

        relations: list[BenchRelation] = []
        for r in ex.get("relations", []):
            s, o = r.get("s"), r.get("o")
            if s in rep_by_id and o in rep_by_id:
                relations.append(BenchRelation(source=rep_by_id[s], target=rep_by_id[o],
                                               type=str(r.get("p", "")).lower()))

        docs.append(BenchDoc(doc_id=str(ex.get("id") or f"dwie_{len(docs)}"),
                             text=text, entities=entities, relations=relations))
    return docs
