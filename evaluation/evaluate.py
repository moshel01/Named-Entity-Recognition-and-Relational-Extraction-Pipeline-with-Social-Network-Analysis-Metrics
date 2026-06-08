# Evaluation CLI: score a pipeline run against gold annotations.

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .gold_schema import load_gold
from .scorer import score_all

_TIERS = {
    "conservative": {"llm_extracted", "gliner_extracted", "rule_extracted"},
    "moderate": {"llm_extracted", "gliner_extracted", "rule_extracted",
                 "sna_inferred", "rule_cooccurrence"},
}


def _load_entities(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_edges(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _filter_edges(edges: list[dict], tier: str) -> list[dict]:
    if tier in ("full", "all"):
        return edges
    allowed = _TIERS.get(tier)
    if allowed is None:
        return edges
    out = []
    for e in edges:
        sources = set((e.get("edge_source") or "").split(";"))
        if sources & allowed:
            out.append(e)
    return out


def _print_report(report: dict[str, Any], tier: str) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        console.rule(f"[bold]Evaluation (edge tier: {tier})")
        for section in ("entities", "entities_type_agnostic",
                        "relations_typed", "relations_untyped"):
            o = report[section]["overall"]
            t = Table(title=section, show_header=True, header_style="bold cyan")
            for col in ("precision", "recall", "f1", "tp", "fp", "fn"):
                t.add_column(col, justify="right")
            t.add_row(*(str(o[c]) for c in ("precision", "recall", "f1", "tp", "fp", "fn")))
            console.print(t)
        # Per-type entity table.
        pt = report["entities"]["per_type"]
        if pt:
            t = Table(title="entities by type", header_style="bold magenta")
            t.add_column("type"); t.add_column("P", justify="right")
            t.add_column("R", justify="right"); t.add_column("F1", justify="right")
            t.add_column("tp/fp/fn", justify="right")
            for typ, m in pt.items():
                t.add_row(typ, f'{m["precision"]:.3f}', f'{m["recall"]:.3f}',
                          f'{m["f1"]:.3f}', f'{m["tp"]}/{m["fp"]}/{m["fn"]}')
            console.print(t)
    except Exception:  # noqa: BLE001 - fall back to plain text
        print(f"=== Evaluation (edge tier: {tier}) ===")
        for section in ("entities", "entities_type_agnostic",
                        "relations_typed", "relations_untyped"):
            o = report[section]["overall"]
            print(f"\n[{section}] P={o['precision']:.3f} R={o['recall']:.3f} "
                  f"F1={o['f1']:.3f} (tp={o['tp']} fp={o['fp']} fn={o['fn']})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score pipeline output against gold.")
    ap.add_argument("--gold", required=True, help="Gold annotation JSON file.")
    ap.add_argument("--run-dir", help="Pipeline run dir (entities.json + gephi_edges.csv).")
    ap.add_argument("--entities", help="Path to entities.json (overrides --run-dir).")
    ap.add_argument("--edges", help="Path to gephi_edges.csv (overrides --run-dir).")
    ap.add_argument("--edge-sources", default="full",
                    choices=["conservative", "moderate", "full", "all"],
                    help="Restrict predicted edges to an evidentiary tier.")
    ap.add_argument("--out", help="Write the full JSON report to this path.")
    args = ap.parse_args(argv)

    if args.run_dir:
        run = Path(args.run_dir)
        ent_path = Path(args.entities) if args.entities else run / "entities.json"
        edge_path = Path(args.edges) if args.edges else run / "gephi_edges.csv"
    else:
        if not args.entities:
            ap.error("Provide --run-dir or --entities (and optionally --edges).")
        ent_path = Path(args.entities)
        edge_path = Path(args.edges) if args.edges else Path("/nonexistent")

    if not ent_path.exists():
        ap.error(f"entities file not found: {ent_path}")

    gold = load_gold(args.gold)
    pred_entities = _load_entities(ent_path)
    pred_edges = _filter_edges(_load_edges(edge_path), args.edge_sources)

    report = score_all(gold, pred_entities, pred_edges)
    report["meta"] = {
        "gold": str(args.gold), "entities": str(ent_path), "edges": str(edge_path),
        "edge_tier": args.edge_sources, "n_pred_edges_after_filter": len(pred_edges),
        "n_gold_documents": len(gold.documents),
    }

    _print_report(report, args.edge_sources)
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nFull report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
