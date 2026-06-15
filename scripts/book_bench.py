# Benchmark the pipeline on a full book against gold annotations.
#
#   python scripts/book_bench.py --book data/hobbit.txt --gold data/hobbit.gold.json
#   python scripts/book_bench.py --book data/hobbit.txt --gold data/hobbit.gold.json \
#       --mode ollama --ollama-model qwen3:8b --limit 10
#
# Splits the book into chapter documents (tolerant heading regex; falls back to
# one document if no chapter structure is found), runs the pipeline with the
# fiction-appropriate switches, then scores entities + relations against the
# gold at all three evidentiary tiers.
#
# Gold format (evaluation/gold_schema.py): {"documents": [{"doc_id": ...,
#   "entities": [{"name", "type", "aliases": [...]}],
#   "relations": [{"source", "target", "type"}]}]}
# Scoring is corpus-level, so a single document holding every annotation works:
#   {"documents": [{"doc_id": "book", "entities": [...], "relations": [...]}]}

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

_CHAPTER_RE = re.compile(
    r"(?mi)^\s*(chapter|kapitel|chapitre)\s+([ivxlc]+|\d+)\b[^\n]*$"
)


def split_chapters(text: str) -> list[str]:
    """Split on chapter headings; [] if the book has no usable structure."""
    starts = [m.start() for m in _CHAPTER_RE.finditer(text)]
    if len(starts) < 3:
        return []
    parts = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        body = text[s:end].strip()
        if len(body) > 500:
            parts.append(body)
    return parts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run + score the pipeline on a book.")
    ap.add_argument("--book", required=True, help="Plain-text book file (UTF-8).")
    ap.add_argument("--gold", required=True, help="Gold JSON (evaluation schema).")
    ap.add_argument("--name", default="", help="Run tag (default: book filename stem).")
    ap.add_argument("--mode", default="python_only", choices=["python_only", "ollama"])
    ap.add_argument("--ollama-model", default="qwen3:8b")
    ap.add_argument("--limit", type=int, default=0, help="Max chapters (0 = all).")
    ap.add_argument("--resume", action="store_true", help="Continue an interrupted run.")
    ap.add_argument("--constrain-relations", action="store_true",
                    help="Constrain the LLM to the gold's relation labels so "
                         "TYPED relation F1 is meaningful (ollama mode).")
    ap.add_argument("--relation-guide", default="",
                    help="JSON file of {label: definition} shown to the LLM next "
                         "to the allowed types (A/B the typing accuracy). Implies "
                         "--constrain-relations; tags the run _guide.")
    ap.add_argument("--coref", action="store_true",
                    help="Enable fastcoref pronoun resolution (A/B vs default off).")
    args = ap.parse_args(argv)

    book = Path(args.book)
    if not book.exists():
        ap.error(f"book not found: {book}")
    # A guide only constrains if the labels are constrained too.
    if args.relation_guide:
        args.constrain_relations = True
    name = args.name or re.sub(r"[^a-z0-9]+", "_", book.stem.lower()).strip("_")
    run_name = f"{name}_{args.mode}"
    if args.constrain_relations:
        run_name += "_constr"
    if args.relation_guide:
        run_name += "_guide"
    if args.coref:
        run_name += "_coref"

    # Validate the gold up front so a format problem fails in seconds, not after
    # the pipeline run.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation.gold_schema import load_gold
    gold = load_gold(args.gold)
    n_ge = len(gold.entities)
    n_gr = len(gold.relations)
    print(f"Gold: {len(gold.documents)} document(s), {n_ge} entities, {n_gr} relations")

    # 1. Chapter inputs.
    text = book.read_text(encoding="utf-8-sig", errors="replace")
    chapters = split_chapters(text)
    input_dir = Path("data") / f"book_{name}"
    input_dir.mkdir(parents=True, exist_ok=True)
    for old in input_dir.glob("ch_*.txt"):
        old.unlink()
    if chapters:
        if args.limit:
            chapters = chapters[: args.limit]
        for i, ch in enumerate(chapters, 1):
            (input_dir / f"ch_{i:04d}.txt").write_text(ch, encoding="utf-8")
        print(f"Inputs: {len(chapters)} chapters -> {input_dir}")
    else:
        (input_dir / "ch_0001.txt").write_text(text, encoding="utf-8")
        print(f"Inputs: no chapter structure detected; single document -> {input_dir}")

    # 2. Config: generic domain with the fiction switches.
    cfg = {
        "run_name": run_name,
        "mode": args.mode,
        "io": {"input_path": str(input_dir).replace("\\", "/"), "input_glob": "ch_*.txt",
               "output_dir": "./output", "encoding": "utf-8"},
        "coreference": {"narrator_resolution": False,
                        "pronoun_resolution": args.coref},
        "dedup": {"llm_assist": args.mode == "ollama"},
        "intelligence": {"ollama": {"model": args.ollama_model,
                                    "request_timeout": 600}},
        "inference": {"enable_cooccurrence_edges": True,
                      "cooccurrence_min_shared_docs": 2},
        "export": {"formats": ["csv", "json"], "gephi": True, "graph_metrics": True},
    }
    if args.constrain_relations:
        rel_labels = sorted({r.type for r in gold.relations if r.type})
        if rel_labels:
            onto = {"enabled": True, "drop_unmapped": False, "relations": rel_labels}
            if args.relation_guide:
                import json as _json
                full = _json.loads(Path(args.relation_guide).read_text(encoding="utf-8"))
                # Only ship definitions for labels actually in the gold inventory.
                guide = {k: full[k] for k in rel_labels if k in full}
                onto["relation_guide"] = guide
                print(f"Relation guide: definitions for {len(guide)}/{len(rel_labels)} labels")
            cfg["ontology"] = onto
            print(f"Constraining LLM to {len(rel_labels)} relation labels: {rel_labels}")
    config_path = input_dir / f"config_{run_name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 3. Run (inherits the terminal, so the live progress bars show).
    cmd = [sys.executable, "main.py", "--config", str(config_path)]
    if args.resume:
        cmd.append("--resume")
    print("RUN :", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"Pipeline exited {rc}; skipping evaluation.")
        return rc

    # 4. Score at every evidentiary tier.
    run_dir = Path("output") / run_name
    for tier in ("conservative", "moderate", "full"):
        subprocess.call([
            sys.executable, "-m", "evaluation.evaluate",
            "--gold", args.gold, "--run-dir", str(run_dir),
            "--edge-sources", tier,
            "--out", str(run_dir / f"eval_report.{tier}.json"),
        ])
    print(f"\nReports: {run_dir}\\eval_report.<tier>.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
