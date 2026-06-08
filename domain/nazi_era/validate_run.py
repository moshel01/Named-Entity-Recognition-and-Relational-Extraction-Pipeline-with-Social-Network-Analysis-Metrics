# Gold-free validation for a nazi_era run.

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from core.schema import Entity, Relationship, TimelineEvent
from postprocess.aggregator import normalize_name

from . import aliases as alias_mod
from .german_nlp import author_from_filename
from .validation import validate_all


def _load_entities(p: Path) -> list[Entity]:
    return [Entity.from_dict(d) for d in json.loads(p.read_text(encoding="utf-8"))]


def _load_edges(p: Path) -> list[Relationship]:
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Relationship(
                source=row.get("Source", ""), target=row.get("Target", ""),
                rel_type=row.get("rel_type", ""), doc_id="",
                evidence=row.get("evidence", ""),
                attributes={"edge_source": row.get("edge_source", "")},
            ))
    return out


def _load_timeline(p: Path) -> list[TimelineEvent]:
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            year = row.get("year") or ""
            out.append(TimelineEvent(
                doc_id=row.get("doc_id", ""), chunk_id="",
                date_text=row.get("date_text", ""), iso_date=row.get("iso_date") or None,
                year=int(year) if str(year).isdigit() else None,
                description=row.get("description", ""), entities=[],
            ))
    return out


def _alias_consistency(entities: list[Entity]) -> dict:
    """A known alias and its canonical appearing as two distinct nodes = miss."""
    present = {normalize_name(e.canonical_name): e.entity_id for e in entities}
    misses = []
    for surface, canonical in alias_mod.ALIASES.items():
        ns, nc = normalize_name(surface), normalize_name(canonical)
        if ns == nc:
            continue
        if ns in present and nc in present and present[ns] != present[nc]:
            misses.append((surface, canonical))
    return {"unmerged_alias_pairs": len(misses), "samples": misses[:15]}


def _coverage(entities: list[Entity]) -> dict:
    present = set()
    for e in entities:
        present.add(normalize_name(e.canonical_name))
        for a in e.aliases:
            present.add(normalize_name(a))
    canon = {normalize_name(v) for v in alias_mod.ALIASES.values()}
    found = sorted(v for v in canon if v in present)
    return {"known_canonicals": len(canon), "found": len(found),
            "coverage_pct": round(100 * len(found) / max(1, len(canon)), 1)}


def _author_coverage(entities: list[Entity], run_dir: Path, inputs_dir: str) -> dict:
    # Expected authors = the docs actually processed (documents.csv), not the whole
    # inputs folder. Match against node canonical name OR aliases (metadata may
    # have renamed "B Huth" -> "B Heiht").
    docs_csv = run_dir / "documents.csv"
    if docs_csv.exists():
        with docs_csv.open(encoding="utf-8", newline="") as fh:
            authors = {normalize_name(r["author"]) for r in csv.DictReader(fh) if r.get("author")}
    else:
        authors = {normalize_name(author_from_filename(p.name) or "")
                   for p in Path(inputs_dir).glob("*") if p.is_file()}
    authors.discard("")

    surface_to_person = {}
    for e in entities:
        if e.label != "PERSON":
            continue
        for nm in [e.canonical_name, *e.aliases]:
            surface_to_person.setdefault(normalize_name(nm), e)
    found = [a for a in authors if a in surface_to_person]
    flagged = [a for a in found if surface_to_person[a].attributes.get("is_author")]
    missing = sorted(a for a in authors if a not in surface_to_person)
    return {"authors": len(authors), "found_as_person": len(found),
            "found_pct": round(100 * len(found) / max(1, len(authors)), 1),
            "flagged_is_author": len(flagged), "missing_samples": missing[:15]}


def validate(run_dir: str, inputs_dir: str = "") -> dict:
    d = Path(run_dir)
    if not (d / "entities.json").exists():
        alt = d / d.name  # tolerate a doubled path like output/abel_papers/abel_papers
        if (alt / "entities.json").exists():
            d = alt
        else:
            have = [p.name for p in d.glob("*")] if d.exists() else "missing directory"
            raise SystemExit(f"entities.json not found under {run_dir}. Found: {have}")
    entities = _load_entities(d / "entities.json")
    edges = _load_edges(d / "gephi_edges.csv")
    timeline = _load_timeline(d / "timeline.csv")

    deg = Counter()
    for r in edges:
        deg[r.source] += 1
        deg[r.target] += 1
    isolated = sum(1 for e in entities if deg[e.entity_id] == 0)

    types = Counter(e.label for e in entities)
    with_alias = sum(1 for e in entities if e.aliases)
    authors = sum(1 for e in entities if e.attributes.get("is_author"))
    edge_src = Counter(r.attributes.get("edge_source", "?") for r in edges)
    with_ev = sum(1 for r in edges if r.evidence.strip())
    member_edges = sum(1 for r in edges if r.rel_type == "member_of")

    hist = validate_all(entities, edges, timeline)

    return {
        "counts": {"entities": len(entities), "edges": len(edges),
                   "timeline": len(timeline), "isolated_nodes": isolated},
        "entity_types": dict(types),
        "pct_entities_with_alias": round(100 * with_alias / max(1, len(entities)), 1),
        "pct_edges_with_evidence": round(100 * with_ev / max(1, len(edges)), 1),
        "edge_sources": dict(edge_src),
        "author_nodes": authors,
        "member_of_edges": member_edges,
        "alias_consistency": _alias_consistency(entities),
        "known_entity_coverage": _coverage(entities),
        "author_coverage": _author_coverage(entities, d, inputs_dir) if inputs_dir else None,
        "historical": {"warnings": hist.n_warnings, "errors": hist.n_errors,
                       "samples": hist.to_rows()[:15]},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Gold-free nazi_era run validation.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--inputs", default="", help="Source dir; checks author coverage from filenames.")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    rep = validate(args.run_dir, args.inputs)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
