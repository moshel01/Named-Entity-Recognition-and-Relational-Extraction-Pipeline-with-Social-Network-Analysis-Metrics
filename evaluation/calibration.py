# Confidence calibration report: do the pipeline's confidence scores mean
# anything? Bins predicted entities/edges by confidence and reports empirical
# precision per bin (reliability diagram data) plus expected calibration error.
# A well-calibrated 0.9 bin should be ~90% correct against gold.
#
#   python -m evaluation.calibration --gold data/bench/redocred.gold.json \
#       --run-dir data/bench/redocred/output/redocred_test_ollama

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from postprocess.aggregator import normalize_name

from .gold_schema import load_gold
from .scorer import _build_gold_nodes, _build_pred_nodes, _overlap

_BINS = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def _bin_of(c: float) -> int:
    for i, (lo, hi) in enumerate(_BINS):
        if lo <= c < hi:
            return i
    return len(_BINS) - 1


def entity_reliability(gold, entities: list[dict]) -> list[dict]:
    gnodes = _build_gold_nodes(gold)
    pred = _build_pred_nodes(entities)
    conf_by_name = {normalize_name(e.get("canonical_name") or e.get("name") or ""):
                    float(e.get("confidence", 0.0)) for e in entities}
    rows = [{"lo": lo, "hi": hi, "n": 0, "correct": 0} for lo, hi in _BINS]
    for p in pred:
        ok = any(p.type == g.type and _overlap(p.surfaces, g.surfaces) for g in gnodes)
        b = rows[_bin_of(conf_by_name.get(p.canonical_norm, 0.0))]
        b["n"] += 1
        b["correct"] += int(ok)
    return rows


def edge_reliability(gold, entities: list[dict], edges: list[dict]) -> list[dict]:
    # Untyped pair matching, mirroring scorer.score_relations.
    from .scorer import score_relations
    # Cheap approach: score once to get the matched pair set via tp keys.
    gnodes = _build_gold_nodes(gold)
    g_surfaces = {s for g in gnodes for s in g.surfaces}
    gold_pairs = set()
    rep_to_node = {}
    for i, g in enumerate(gnodes):
        for s in g.surfaces:
            rep_to_node.setdefault(s, i)
    for r in gold.relations:
        a = rep_to_node.get(normalize_name(r.source))
        b = rep_to_node.get(normalize_name(r.target))
        if a is not None and b is not None and a != b:
            gold_pairs.add(frozenset((a, b)))

    rows = [{"lo": lo, "hi": hi, "n": 0, "correct": 0} for lo, hi in _BINS]
    for e in edges:
        s = normalize_name(e.get("source_name") or e.get("Source") or "")
        t = normalize_name(e.get("target_name") or e.get("Target") or "")
        a, b = rep_to_node.get(s), rep_to_node.get(t)
        if a is None or b is None:
            # Endpoint not even a gold entity: count as incorrect prediction.
            ok = False
        elif a == b:
            continue
        else:
            ok = frozenset((a, b)) in gold_pairs
        try:
            conf = float(e.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        brow = rows[_bin_of(conf)]
        brow["n"] += 1
        brow["correct"] += int(ok)
    return rows


def _finish(rows: list[dict]) -> tuple[list[dict], float]:
    """Add empirical precision per bin and compute ECE (weighted |conf-acc|)."""
    total = sum(r["n"] for r in rows) or 1
    ece = 0.0
    for r in rows:
        mid = (r["lo"] + min(r["hi"], 1.0)) / 2
        acc = r["correct"] / r["n"] if r["n"] else None
        r["precision"] = round(acc, 3) if acc is not None else None
        r["bin_mid"] = round(mid, 2)
        if acc is not None:
            ece += (r["n"] / total) * abs(mid - acc)
    return rows, round(ece, 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Confidence calibration report.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="", help="Optional JSON output path.")
    args = ap.parse_args(argv)

    gold = load_gold(args.gold)
    run = Path(args.run_dir)
    entities = json.loads((run / "entities.json").read_text(encoding="utf-8"))
    edges_path = run / "gephi_edges.csv"
    with edges_path.open("r", encoding="utf-8", newline="") as fh:
        edges = list(csv.DictReader(fh))

    ent_rows, ent_ece = _finish(entity_reliability(gold, entities))
    edge_rows, edge_ece = _finish(edge_reliability(gold, entities, edges))

    def show(name, rows, ece):
        print(f"\n{name} reliability (ECE = {ece}):")
        print("  conf bin   n     empirical precision")
        for r in rows:
            p = "-" if r["precision"] is None else f"{r['precision']:.3f}"
            print(f"  {r['lo']:.1f}-{min(r['hi'],1.0):.1f}  {r['n']:5d}   {p}")

    show("Entities", ent_rows, ent_ece)
    show("Edges (untyped pairs)", edge_rows, edge_ece)
    print("\nECE = expected calibration error (0 = perfectly calibrated).")
    print("Caveat: edge 'incorrect' includes pairs absent from a possibly "
          "non-exhaustive gold; treat edge bins as lower bounds.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"entities": ent_rows, "entities_ece": ent_ece,
             "edges": edge_rows, "edges_ece": edge_ece}, indent=2),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
