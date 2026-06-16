# Shared helpers for benchmark adapters: data classes, writers, config builder.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Intermediate representation produced by every adapter
@dataclass
class BenchEntity:
    name: str                       # representative surface form
    type: str                       # canonical type (PERSON/ORG/LOCATION/...)
    aliases: list[str] = field(default_factory=list)


@dataclass
class BenchRelation:
    source: str                     # representative name of head entity
    target: str                     # representative name of tail entity
    type: str = ""                  # relation label (may be a code / readable)


@dataclass
class BenchDoc:
    doc_id: str
    text: str
    entities: list[BenchEntity] = field(default_factory=list)
    relations: list[BenchRelation] = field(default_factory=list)


# Writers
# Canonical pipeline type -> the GLiNER zero-shot label used to elicit it.
CANON_TO_GLINER = {
    "PERSON": "person", "ORG": "organization", "LOCATION": "location",
    "EVENT": "event", "DATE": "date", "NUM": "number", "MISC": "miscellaneous",
    "INSTITUTION": "institution", "RANK": "rank",
}


def labels_for_types(types: list[str]) -> tuple[list[str], dict[str, str]]:
    """Build (gliner_labels, label_map) for a target canonical-type list."""
    labels: list[str] = []
    label_map: dict[str, str] = {}
    for t in types:
        g = CANON_TO_GLINER.get(t, t.lower())
        labels.append(g)
        label_map[g] = t
    return labels, label_map


def filter_docs_to_types(docs: list["BenchDoc"], types: list[str]) -> list["BenchDoc"]:
    """Keep only gold entities of the target types (and relations among them).

    Makes the benchmark apples-to-apples: the pipeline is only asked to find
    these types, and the gold is scored only on these types.
    """
    allowed = set(types)
    for d in docs:
        d.entities = [e for e in d.entities if e.type in allowed]
        names = {e.name for e in d.entities}
        d.relations = [r for r in d.relations if r.source in names and r.target in names]
    return docs


# BIO token-classification helpers (CoNLL / OntoNotes / WNUT / UNER / GermEval)
def classlabel_names(feature: Any) -> list[str] | None:
    """ClassLabel names for a (possibly Sequence-wrapped) HF feature, else None.

    None means the column stores raw ints with no embedded label list (tner
    parquet mirrors do this) - the adapter must supply its own id->tag map.
    """
    names = getattr(getattr(feature, "feature", None), "names", None)
    if names is None:
        names = getattr(feature, "names", None)      # non-sequence ClassLabel
    return list(names) if names else None


def decode_bio(tokens: list[str], tags: list[str],
               type_map: dict[str, str]) -> list[tuple[str, str]]:
    """IOB2/BIO string tags -> [(surface, canonical_type)].

    A span's base tag (after the B-/I- prefix) is looked up in type_map; spans
    whose base is absent are dropped, as are orphan I- runs with no B-.
    """
    spans: list[tuple[str, str]] = []
    cur: list[str] = []
    cur_base = ""

    def flush() -> None:
        nonlocal cur, cur_base
        if cur and cur_base in type_map:
            spans.append((" ".join(cur), type_map[cur_base]))
        cur, cur_base = [], ""

    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            flush()
            cur_base = tag[2:]
            cur = [tok]
        elif tag.startswith("I-") and cur and tag[2:] == cur_base:
            cur.append(tok)
        else:
            flush()
    flush()
    return spans


def build_ner_docs(sentences: list[tuple[list[str], list[str]]],
                   type_map: dict[str, str], *, dataset: str, split: str,
                   sents_per_doc: int = 25, limit: int = 0) -> list["BenchDoc"]:
    """Group (tokens, string-tags) sentences into pseudo-docs with deduped gold.

    Token-classification corpora are sentence-level; grouping gives corpus-level
    scoring document units. ``limit`` counts pseudo-docs (matches the adapters).
    """
    docs: list[BenchDoc] = []
    n_docs = (len(sentences) + sents_per_doc - 1) // sents_per_doc
    if limit and limit > 0:
        n_docs = min(n_docs, limit)
    for d in range(n_docs):
        chunk = sentences[d * sents_per_doc:(d + 1) * sents_per_doc]
        sents_text: list[str] = []
        entities: dict[tuple[str, str], BenchEntity] = {}
        for toks, tags in chunk:
            sents_text.append(" ".join(toks))
            for surface, etype in decode_bio(toks, tags, type_map):
                entities.setdefault((surface.lower(), etype),
                                    BenchEntity(name=surface, type=etype))
        docs.append(BenchDoc(doc_id=f"{d:05d}_{dataset}_{split}",
                             text="\n".join(sents_text),
                             entities=list(entities.values()), relations=[]))
    return docs


def load_token_dataset(repo: str, split: str, config: str | None = None):
    """Load a token-classification HF dataset under datasets 4.x.

    datasets 4.x dropped script loading, so the canonical NER datasets
    (conll2003, tner/*, ...) no longer load by id. Fall back to the
    auto-converted parquet on the refs/convert/parquet branch, which every public
    dataset gets. ClassLabel names survive the conversion when the source
    declared them (conll2003); tner stores raw ints (see hf_iob_label_map).
    """
    import datasets

    try:
        if config:
            return datasets.load_dataset(repo, config, split=split)
        return datasets.load_dataset(repo, split=split)
    except RuntimeError as e:
        if "no longer supported" not in str(e).lower():
            raise
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"datasets/{repo}@refs/convert/parquet"
    files = (fs.glob(f"{base}/**/{split}/*.parquet")
             or fs.glob(f"{base}/**/*{split}*.parquet"))
    if config:
        files = [f for f in files if f"/{config}/" in f] or files
    if not files:
        raise RuntimeError(f"{repo}: no parquet for split={split}"
                           + (f" config={config}" if config else "")
                           + " (no refs/convert/parquet branch?)")
    return datasets.load_dataset(
        "parquet", data_files={split: [f"hf://{f}" for f in files]}, split=split)


def hf_iob_label_map(repo: str) -> dict[int, str]:
    """id -> IOB tag, scraped from a tner-style dataset README's label2id block.

    tner mirrors store tags as raw ints with no embedded ClassLabel; the
    authoritative label2id lives in the README. Returns {} if not found (the
    adapter then uses its hardcoded fallback). Grabs every "TAG": int pair, so
    it does not depend on the JSON being well-formed (trailing commas etc.).
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo, "README.md", repo_type="dataset")
    except Exception:  # noqa: BLE001
        return {}
    txt = Path(path).read_text(encoding="utf-8")
    pairs = re.findall(r'"((?:B-|I-)[\w./-]+|O)"\s*:\s*(\d+)', txt)
    out = {int(i): tag for tag, i in pairs}
    return out if "O" in out.values() else {}


def parse_iob2(path: str) -> list[tuple[list[str], list[str]]]:
    """Two-column IOB2 (token<ws>tag) into [(tokens, tags)] sentences.

    Blank line = sentence break, # comment lines skipped. Tolerates CoNLL-U-ish
    rows (first column = token, last column = tag). Used for UNER, whose HF repo
    is script-only - download a treebank .iob2 and pass it via --path.
    """
    sents: list[tuple[list[str], list[str]]] = []
    toks: list[str] = []
    tags: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            if toks:
                sents.append((toks, tags))
                toks, tags = [], []
            continue
        if raw.startswith("#"):
            continue
        parts = raw.split("\t") if "\t" in raw else raw.split()
        if len(parts) < 2:
            continue
        toks.append(parts[0])
        tags.append(parts[-1])
    if toks:
        sents.append((toks, tags))
    return sents


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(s: str, fallback: str) -> str:
    s = _SAFE.sub("_", s).strip("_")
    return s[:80] or fallback


def write_inputs(docs: list[BenchDoc], input_dir: Path) -> int:
    """Write each doc's text to ``input_dir/<doc_id>.txt``. Returns count.

    Clears stale ``*.txt`` first so the directory exactly matches the requested
    document set (otherwise a smaller --limit would leave old files behind and
    the pipeline would ingest more documents than the gold covers).
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    for old in input_dir.glob("*.txt"):
        old.unlink()
    for i, d in enumerate(docs):
        fname = _safe_name(d.doc_id, f"doc_{i:05d}") + ".txt"
        (input_dir / fname).write_text(d.text, encoding="utf-8")
    return len(docs)


def write_gold(docs: list[BenchDoc], gold_path: Path) -> None:
    """Write the gold.json (evaluation-harness format) for a list of docs."""
    payload = {"documents": [
        {
            "doc_id": d.doc_id,
            "entities": [
                {"name": e.name, "type": e.type, "aliases": e.aliases}
                for e in d.entities
            ],
            "relations": [
                {"source": r.source, "target": r.target, "type": r.type}
                for r in d.relations
            ],
        }
        for d in docs
    ]}
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_config(
    *,
    run_name: str,
    input_dir: Path,
    output_dir: Path,
    gliner_labels: list[str],
    label_map: dict[str, str],
    mode: str = "python_only",
    spacy_model: str = "en_core_web_sm",
    gliner_model: str = "fastino/gliner2-large-v1",
    config_path: Path,
    ollama_model: str = "qwen3.5:9b",
    ontology_relations: list[str] | None = None,
    min_entity_confidence: float = 0.0,
) -> Path:
    """Write a pipeline config tuned for a benchmark run.

    Benchmarks use the GENERIC domain with coreference, canonical inference, and
    mandatory membership all OFF - those are corpus-specific helpers (Abel) that
    would add non-gold nodes/edges and depress precision on benchmark data.
    """
    cfg: dict[str, Any] = {
        "run_name": run_name,
        "mode": mode,
        "io": {
            "input_path": str(input_dir).replace("\\", "/"),
            "input_glob": "**/*.txt",
            "output_dir": str(output_dir).replace("\\", "/"),
            "encoding": "utf-8",
        },
        "chunking": {"max_chars": 8000, "overlap_chars": 400, "respect_sentences": True},
        "foundation": {
            "spacy_model": spacy_model,
            "gliner_model": gliner_model,
            "gliner_threshold": 0.40,
            "gliner_labels": gliner_labels,
            "label_map": label_map,
            "use_spacy_ner": True,
            "device": "auto",
        },
        "coreference": {"enabled": False, "narrator_resolution": False,
                        "pronoun_resolution": False},
        "intelligence": {
            "ollama": {"model": ollama_model, "request_timeout": 600},
            "python_only": {"cooccurrence_window": "sentence",
                            "min_relationship_confidence": 0.30},
        },
        "inference": {
            "enable_cooccurrence_edges": True,
            "cooccurrence_min_shared_docs": 2,
            "enable_canonical_inference": False,
            "mandatory_membership": "off",
        },
        "quality": {"enabled": True, "min_entity_mentions": 1, "min_edge_weight": 1,
                    "min_entity_confidence": min_entity_confidence, "llm_review": False},
        "domain": {"name": "generic"},
        "export": {"formats": ["csv", "json"], "gephi": True},
        "checkpoint": {"enabled": True, "flush_every": 5},
    }
    if ontology_relations:
        # Constrain LLM relation extraction + align output to this label set,
        # making TYPED relation F1 comparable against the dataset's inventory.
        cfg["ontology"] = {"enabled": True, "drop_unmapped": False,
                           "relations": list(ontology_relations)}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                           encoding="utf-8")
    return config_path
