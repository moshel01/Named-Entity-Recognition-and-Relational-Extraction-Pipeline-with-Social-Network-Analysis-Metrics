# Derive a German relation gold from the Abel metadata, for the authors in a
# finished run. Zero hand-annotation: the spreadsheet's verified birth /
# residence / membership facts become gold relations, and we measure how many
# the TEXT extraction recovered on its own.
#
# The metadata is already merged onto each author node at run time, so we read
# the run's entities.json - no xlsx path needed (pass --metadata only for older
# runs whose authors predate the merge).
#
# Usage:
#   python scripts/metadata_gold.py <run_dir> [--out gold.json] [--metadata x.xlsx]
#   python -m evaluation.evaluate --gold gold.json --run-dir <run_dir> \
#       --edge-sources conservative --exclude-edge-source metadata
#
# Read RECALL on relations_untyped as the headline: the text uses its own labels
# (joined vs member_of, located_in vs born_in), so endpoints-only is the honest
# match. Precision is NOT meaningful - the prose asserts many true ties the four
# spreadsheet fields never list. Note NSDAP membership is on every author, so it
# dominates the gold; the per-label typed table separates born_in / resided_in.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domain.nazi_era.metadata import load_metadata, metadata_edges  # noqa: E402

# Author carries these only if a spreadsheet row actually matched (letter_id is
# stamped from the filename regardless, so it alone does not prove a match).
_MATCH_FIELDS = ("place_of_birth", "place_of_residence", "prior_party",
                 "membership_number", "join_date", "meta_name")


def build(run_dir: str, metadata_file: str = "") -> tuple[dict, int, int]:
    entities = json.loads((Path(run_dir) / "entities.json").read_text(encoding="utf-8"))
    meta = load_metadata(metadata_file) if metadata_file else {}

    docs: list[dict] = []
    n_authors = n_rel = 0
    for e in entities:
        attrs = e.get("attributes") or {}
        if not attrs.get("is_author") or e.get("label") != "PERSON":
            continue
        author = (e.get("canonical_name") or "").strip()
        if not author:
            continue
        # Prefer the spreadsheet row (full field set); fall back to the fields
        # merged onto the node at run time.
        lid = str(attrs.get("letter_id") or "")
        row = meta.get(lid) if meta else None
        if row is None:
            if not any(attrs.get(f) for f in _MATCH_FIELDS):
                continue  # no metadata matched this author; nothing verified
            row = attrs

        ents = [{"name": author, "type": "PERSON"}]
        rels = []
        for spec in metadata_edges(row):
            ents.append({"name": spec["target"], "type": spec.get("type", "ORG")})
            rels.append({"source": author, "target": spec["target"], "type": spec["rel"]})
        docs.append({"doc_id": f"meta_{lid or len(docs)}", "entities": ents, "relations": rels})
        n_authors += 1
        n_rel += len(rels)
    return {"documents": docs}, n_authors, n_rel


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="Finished run dir (reads entities.json).")
    ap.add_argument("--out", default="", help="Gold path (default <run_dir>/metadata_gold.json).")
    ap.add_argument("--metadata", default="",
                    help="Abel xlsx; only needed if the run predates the metadata merge.")
    args = ap.parse_args(argv)

    gold, n_authors, n_rel = build(args.run_dir, args.metadata)
    if not n_authors:
        print("No metadata-matched authors in this run; nothing to write. "
              "Was --metadata configured for the run? Pass --metadata here for old runs.")
        return 1
    out = Path(args.out) if args.out else Path(args.run_dir) / "metadata_gold.json"
    out.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}: {n_authors} authors, {n_rel} gold relations")
    print("score the text's recall of it with:")
    print(f"  python -m evaluation.evaluate --gold {out} --run-dir {args.run_dir} "
          f"--edge-sources conservative --exclude-edge-source metadata")
    return 0


if __name__ == "__main__":
    sys.exit(main())
