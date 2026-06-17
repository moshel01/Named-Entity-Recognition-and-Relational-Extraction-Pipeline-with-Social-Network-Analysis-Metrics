# Network expansion: read the schema of an existing graph so a new run can grow
# it under the same vocabulary. We only need three things from the old network -
# the relation types it uses, the entity kinds it contains, and (for reference)
# the canonical names already in it. Source can be a prior run directory, its
# gephi_edges.csv, or a network.gexf. Fail-soft: a missing/partial source yields
# an empty schema and the caller treats locks as no-ops.

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class NetworkSchema:
    relation_types: set[str] = field(default_factory=set)
    entity_types: set[str] = field(default_factory=set)
    entity_names: dict[str, str] = field(default_factory=dict)  # canonical_name -> type

    @property
    def empty(self) -> bool:
        return not (self.relation_types or self.entity_types)


def _from_entities_json(p: Path, schema: NetworkSchema) -> None:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("expansion: could not read %s: %s", p, exc)
        return
    ents = data if isinstance(data, list) else data.get("entities", [])
    for e in ents:
        lab = e.get("label")
        name = e.get("canonical_name")
        if lab:
            schema.entity_types.add(lab)
        if name and lab:
            schema.entity_names[name] = lab


def _from_edges_csv(p: Path, schema: NetworkSchema) -> None:
    try:
        with p.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rt = row.get("rel_type") or row.get("Label")
                # co_occurs_with is a structural layer, never a typed relation -
                # locking onto it would let the LLM emit nothing useful.
                if rt and rt != "co_occurs_with":
                    schema.relation_types.add(rt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("expansion: could not read %s: %s", p, exc)


def _from_gexf(p: Path, schema: NetworkSchema) -> None:
    try:
        import networkx as nx
    except Exception:  # noqa: BLE001 - networkx optional
        logger.warning("expansion: networkx not installed, cannot read %s", p)
        return
    try:
        G = nx.read_gexf(p)
    except Exception as exc:  # noqa: BLE001
        logger.warning("expansion: could not read %s: %s", p, exc)
        return
    for _, d in G.nodes(data=True):
        t = d.get("type")
        n = d.get("label") or d.get("Label")
        if t:
            schema.entity_types.add(t)
        if n and t:
            schema.entity_names[n] = t
    for _, _, d in G.edges(data=True):
        for rt in str(d.get("rel_type", "")).split(";"):
            if rt and rt != "co_occurs_with":
                schema.relation_types.add(rt)


def load_network_schema(source: str) -> NetworkSchema:
    """Read relation types + entity kinds from an existing network. `source` is a
    run directory (uses entities.json + gephi_edges.csv), a *.csv edge table, or a
    *.gexf. Returns an empty schema (locks become no-ops) when nothing is found."""
    schema = NetworkSchema()
    if not source:
        return schema
    p = Path(source)
    if not p.exists():
        logger.warning("expansion: source %s does not exist", source)
        return schema

    if p.is_dir():
        ej = p / "entities.json"
        if ej.exists():
            _from_entities_json(ej, schema)
        ec = p / "gephi_edges.csv"
        if ec.exists():
            _from_edges_csv(ec, schema)
        if schema.empty:  # fall back to a gexf in the dir
            for g in p.glob("*.gexf"):
                _from_gexf(g, schema)
                if not schema.empty:
                    break
    elif p.suffix.lower() == ".gexf":
        _from_gexf(p, schema)
    elif p.suffix.lower() == ".csv":
        _from_edges_csv(p, schema)
        sib = p.parent / "entities.json"
        if sib.exists():
            _from_entities_json(sib, schema)
    else:
        logger.warning("expansion: unrecognized source %s", source)

    logger.info("expansion: loaded %d relation types, %d entity kinds from %s",
                len(schema.relation_types), len(schema.entity_types), source)
    return schema
