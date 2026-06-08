# Write CSV/Parquet/JSON/GEXF/JSONL. Polars fast path, stdlib csv fallback.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from core.schema import DocumentExtraction, Entity

from .gephi_builder import GraphTables

logger = logging.getLogger(__name__)


def _union_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Stable union of keys across heterogeneous rows (preserves first-seen order)."""
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = _union_keys(rows)
    # Normalize so every row has every column; keep None as None (NOT "") so a
    # numeric column with some missing values stays a single nullable type
    # instead of becoming a mixed int/str column that breaks schema inference.
    norm = [{k: r.get(k) for k in keys} for r in rows]
    try:
        import polars as pl
        try:
            # Scan all rows for a common supertype rather than just the first 100.
            pl.DataFrame(norm, infer_schema_length=None).write_csv(path)
        except Exception:  # noqa: BLE001 - robust fallback: stringify everything
            str_rows = [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in norm]
            pl.DataFrame(str_rows).write_csv(path)
    except ModuleNotFoundError:
        # Fail-soft: stdlib csv keeps CSV/Gephi export working without polars.
        import csv
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore", restval="")
            writer.writeheader()
            for r in norm:
                writer.writerow({k: ("" if v is None else v) for k, v in r.items()})


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import polars as pl
    if not rows:
        return
    keys = _union_keys(rows)
    norm = [{k: r.get(k) for k in keys} for r in rows]
    try:
        pl.DataFrame(norm, infer_schema_length=None).write_parquet(path)
    except Exception:  # noqa: BLE001
        str_rows = [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in norm]
        pl.DataFrame(str_rows).write_parquet(path)


class Exporter:
    """Write all configured export formats for a run."""

    def __init__(self, output_dir: str | Path, formats: list[str], gephi: bool = True) -> None:
        self.out = Path(output_dir)
        self.formats = set(formats)
        self.gephi = gephi
        self.out.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        tables: GraphTables,
        entities: list[Entity],
        extractions: list[DocumentExtraction] | None = None,
        manifest: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, str]:
        """Write all artifacts; return ``{artifact_name: path}``."""
        written: dict[str, str] = {}

        # Document manifest: doc_id -> letter_id, author, filename (join key).
        if manifest:
            rows = [{"doc_id": k, **v} for k, v in manifest.items()]
            p_manifest = self.out / "documents.csv"
            _write_csv(rows, p_manifest)
            written["documents"] = str(p_manifest)

        if "csv" in self.formats or self.gephi:
            p_nodes = self.out / "gephi_nodes.csv"
            p_edges = self.out / "gephi_edges.csv"
            p_time = self.out / "timeline.csv"
            _write_csv(tables.nodes, p_nodes)
            _write_csv(tables.edges, p_edges)
            _write_csv(tables.timeline, p_time)
            written["gephi_nodes"] = str(p_nodes)
            written["gephi_edges"] = str(p_edges)
            written["timeline"] = str(p_time)

        if "parquet" in self.formats:
            _write_parquet(tables.nodes, self.out / "gephi_nodes.parquet")
            _write_parquet(tables.edges, self.out / "gephi_edges.parquet")
            _write_parquet(tables.timeline, self.out / "timeline.parquet")
            written["parquet"] = str(self.out)

        if "json" in self.formats:
            p_ent = self.out / "entities.json"
            with p_ent.open("w", encoding="utf-8") as fh:
                json.dump([e.to_dict() for e in entities], fh, ensure_ascii=False, indent=2)
            written["entities"] = str(p_ent)

        if "gexf" in self.formats or self.gephi:
            p_gexf = self.out / "network.gexf"
            self._write_gexf(tables, p_gexf)
            written["gexf"] = str(p_gexf)
            # Tie-class views: the interpersonal SNA, the affiliation (two-mode)
            # network, and the discourse (stance + co-occurrence) layer.
            views = {
                "graph_interaction": {"interaction"},
                "graph_affiliation": {"affiliation", "participation", "biographical"},
                "graph_discourse": {"stance", "cooccurrence"},
            }
            for name, classes in views.items():
                p = self.out / f"{name}.gexf"
                self._write_gexf(tables, p, tie_classes=classes)
                written[name] = str(p)
            # Dynamic graph (edges/nodes carry start years) for Gephi's timeline.
            # Best-effort: only if some edges are datable, and never fatal.
            if any(e.get("year") for e in tables.edges):
                try:
                    p_dyn = self.out / "network_dynamic.gexf"
                    self._write_gexf(tables, p_dyn, dynamic=True)
                    written["network_dynamic"] = str(p_dyn)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Dynamic GEXF skipped: %s", exc)

        if "jsonl" in self.formats and extractions is not None:
            p_raw = self.out / "raw_extractions.jsonl"
            with p_raw.open("w", encoding="utf-8") as fh:
                for ex in extractions:
                    fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            written["raw_extractions"] = str(p_raw)

        logger.info("Exported %d artifacts to %s", len(written), self.out)
        return written

    # GEXF
    @staticmethod
    def _write_gexf(tables: GraphTables, path: Path,
                    tie_classes: set[str] | None = None,
                    dynamic: bool = False) -> None:
        """Write a self-contained GEXF graph with node/edge attributes.

        ``tie_classes`` restricts edges to those classes (and to the nodes they
        touch) for the per-view exports; None writes the full graph. ``dynamic``
        stamps a ``start`` year on datable nodes/edges for Gephi's timeline.
        """
        sel = [e for e in tables.edges
               if tie_classes is None or e.get("tie_class") in tie_classes]
        keep_nodes = None
        if tie_classes is not None:
            keep_nodes = {str(e["Source"]) for e in sel} | {str(e["Target"]) for e in sel}

        directed = any(e.get("Type") == "Directed" for e in sel)
        G: nx.Graph = nx.DiGraph() if directed else nx.Graph()
        if dynamic:
            G.graph["mode"] = "dynamic"
            G.graph["timeformat"] = "long"

        for node in tables.nodes:
            nid = str(node["Id"])
            if keep_nodes is not None and nid not in keep_nodes:
                continue
            attrs = {k: v for k, v in node.items()
                     if k != "Id" and v is not None and not (dynamic and k == "first_year")}
            # GEXF needs primitive attribute values.
            attrs = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                     for k, v in attrs.items()}
            if dynamic and node.get("first_year"):
                attrs["start"] = int(node["first_year"])
            G.add_node(nid, **attrs)

        for edge in sel:
            attrs = {
                "label": edge.get("Label", ""),
                "weight": edge.get("Weight", 1),
                "rel_type": edge.get("rel_type", ""),
                "tie_class": edge.get("tie_class", ""),
                "polarity": edge.get("polarity", ""),
                "n_sources": edge.get("n_sources", 0),
                "period": edge.get("period", ""),
                "origin": edge.get("origin", ""),
                "edge_source": edge.get("edge_source", ""),
                "confidence": edge.get("confidence", 0.0),
            }
            if dynamic and edge.get("year"):
                attrs["start"] = int(edge["year"])
            G.add_edge(str(edge["Source"]), str(edge["Target"]), **attrs)

        nx.write_gexf(G, path)
