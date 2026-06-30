#!/usr/bin/env python
# Convert the LittleSis bulk dump into a pipeline snapshot (documents.jsonl), with
# filters, so the full CC BY-SA graph (or a slice of it) merges into a run.
#
# 1. Download the dump (one-time, from https://littlesis.org/bulk_data):
#      relationships.json.gz   (self-contained edges - this is all you need)
#      entities.json.gz        (optional; not required here)
# 2. Convert (filter to the slice you want):
#      # only the entities already in your scraped network, enriched with LittleSis ties:
#      python -m scripts.littlesis_bulk relationships.json.gz \
#          --names-from output/influencewatch/entities.json --out littlesis_bulk.jsonl
#      # or a category/amount slice of the whole graph:
#      python -m scripts.littlesis_bulk relationships.json.gz \
#          --categories 1,5,10 --min-amount 100000 --out littlesis_bulk.jsonl
# 3. Merge + extract. Either ingest it alone, or append it to your crawl snapshot first:
#      python -m scripts.littlesis_bulk relationships.json.gz --names-from ... \
#          --append output/influencewatch/documents.jsonl       # adds to the crawl corpus
#      python main.py --config domain/influencewatch/config_influencewatch.yaml \
#          --ingest-from output/influencewatch/documents.jsonl --mode python_only
#
# The LittleSis edges import as asserted ties (edge_source=littlesis) and dedup folds
# their nodes into your scraped nodes by name. LICENSE: CC BY-SA 4.0 - attribute
# "LittleSis / Public Accountability Initiative" in any published network.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.littlesis import load_bulk  # noqa: E402
from core.preprocessor import write_documents_snapshot  # noqa: E402


def _load_names(path: str) -> set[str]:
    """Names to keep, from a prior run's entities.json or a plain-text list (one/line)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    names: set[str] = set()
    if p.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        rows = data.get("entities") if isinstance(data, dict) else data
        for r in (rows or []):
            if isinstance(r, dict):
                nm = r.get("canonical_name") or r.get("name") or r.get("label") or ""
                if nm:
                    names.add(nm)
                for a in (r.get("aliases") or []):
                    if a:
                        names.add(a)
    if not names:  # plain text fallback (also covers a json that wasn't entity-shaped)
        names = {ln.strip() for ln in text.splitlines() if ln.strip()}
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description="Import the LittleSis bulk dump as a pipeline snapshot.")
    ap.add_argument("relationships", help="Path to relationships.json.gz (or .json).")
    ap.add_argument("--entities", default="", help="Path to entities.json(.gz) - enriches each "
                    "node with blurb/types/website/aliases (the 'add the entities' part).")
    ap.add_argument("--induced", action="store_true", help="Keep only edges where BOTH endpoints "
                    "match --names-from/--ids (induced subgraph). Default keeps either (1-hop).")
    ap.add_argument("--include-isolated", action="store_true", help="With --entities, also emit "
                    "entities that have no kept edge (only sensible for the whole-dump variant).")
    ap.add_argument("--out", default="littlesis_bulk.jsonl", help="Snapshot to write.")
    ap.add_argument("--append", default="", help="Append to this existing snapshot instead of --out "
                    "(e.g. your crawl's documents.jsonl, to merge in one corpus).")
    ap.add_argument("--names-from", default="", help="Keep only edges touching a name in this file "
                    "(a run's entities.json, or one name per line) - the 'enrich my entities' filter.")
    ap.add_argument("--ids", default="", help="Comma-separated LittleSis entity ids to keep.")
    ap.add_argument("--categories", default="", help="Comma-separated category_ids to keep "
                    "(1=position 2=education 3=membership 4=family 5=donation 6=transaction "
                    "7=lobbying 8=social 9=professional 10=ownership 11=hierarchy 12=generic).")
    ap.add_argument("--min-amount", type=float, default=None, help="Keep only money edges >= this.")
    ap.add_argument("--max-edges", type=int, default=0, help="Hard cap on edges (0 = unlimited).")
    args = ap.parse_args()

    names = _load_names(args.names_from) if args.names_from else None
    ids = [s.strip() for s in args.ids.split(",") if s.strip()] or None
    cats = [int(s) for s in args.categories.split(",") if s.strip()] or None
    if names:
        print(f"Filtering to {len(names)} name(s) from {args.names_from}.")
    if not (names or ids or cats or args.min_amount or args.max_edges):
        print("WARNING: no filter set - importing the ENTIRE LittleSis graph "
              "(millions of edges). Ctrl-C and add --names-from/--categories to slice it.")

    docs = load_bulk(args.relationships, entities_path=(args.entities or None),
                     ids=ids, names=names, categories=cats, min_amount=args.min_amount,
                     max_edges=args.max_edges, both_endpoints=args.induced,
                     include_isolated=args.include_isolated)
    n_edges = sum(len((d.meta or {}).get("ls_edges") or []) for d in docs)

    if args.append:
        target = Path(args.append)
        with target.open("a", encoding="utf-8") as fh:
            for d in docs:
                fh.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
        print(f"Appended {len(docs)} LittleSis entities ({n_edges} edges) to {target}.")
        out = target
    else:
        write_documents_snapshot(docs, args.out)
        print(f"Wrote {len(docs)} LittleSis entities ({n_edges} edges) to {args.out}.")
        out = Path(args.out)

    print("LittleSis data is CC BY-SA 4.0 - attribute 'LittleSis / Public Accountability "
          "Initiative' in any published network.")
    print(f"Next: python main.py --config <cfg> --ingest-from {out} --mode <python_only|ollama>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
