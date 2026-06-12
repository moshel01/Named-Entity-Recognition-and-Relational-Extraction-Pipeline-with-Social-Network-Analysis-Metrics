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
    ollama_model: str = "qwen3:8b",
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
