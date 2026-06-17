# Scoring logic: entity and relation precision / recall / F1.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from postprocess.aggregator import normalize_name

from .gold_schema import GoldSet


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"precision": round(self.precision, 4), "recall": round(self.recall, 4),
                "f1": round(self.f1, 4), "tp": self.tp, "fp": self.fp, "fn": self.fn}


@dataclass
class PredNode:
    canonical_norm: str
    type: str
    surfaces: set[str] = field(default_factory=set)


@dataclass
class GoldNode:
    rep_norm: str
    type: str
    surfaces: set[str] = field(default_factory=set)


def _build_pred_nodes(pred_entities: list[dict]) -> list[PredNode]:
    nodes: list[PredNode] = []
    for e in pred_entities:
        name = e.get("canonical_name") or e.get("Label") or e.get("name") or ""
        etype = (e.get("label") or e.get("type") or "").upper()
        surfaces = {normalize_name(name)}
        for a in e.get("aliases", []) or []:
            if isinstance(a, str):
                surfaces.update(normalize_name(x) for x in a.split(";"))
        surfaces.discard("")
        if surfaces:
            nodes.append(PredNode(normalize_name(name), etype, surfaces))
    return nodes


def _build_gold_nodes(gold: GoldSet) -> list[GoldNode]:
    """Collapse gold entities (corpus-level) keyed by (rep_norm, type).

    Duplicate gold entities across documents that share a representative name +
    type are merged, unioning their surface forms.
    """
    by_key: dict[tuple[str, str], GoldNode] = {}
    for e in gold.entities:
        rep = normalize_name(e.name)
        if not rep:
            continue
        key = (rep, e.type)
        node = by_key.get(key)
        if node is None:
            node = GoldNode(rep, e.type, set())
            by_key[key] = node
        for s in e.surface_forms:
            ns = normalize_name(s)
            if ns:
                node.surfaces.add(ns)
    return list(by_key.values())


def _overlap(a: set[str], b: set[str]) -> bool:
    return not a.isdisjoint(b)


def _greedy_match(pred: list, gold: list, tmatch) -> tuple[int, set[int], set[int]]:
    """Greedy 1:1 alignment: each pred consumes one still-free gold whose type
    matches and surfaces overlap. Returns (tp, matched_pred_idx, used_gold_idx)."""
    used_gold: set[int] = set()
    matched_pred: set[int] = set()
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if j in used_gold:
                continue
            if tmatch(p.type, g.type) and _overlap(p.surfaces, g.surfaces):
                used_gold.add(j)
                matched_pred.add(i)
                break
    return len(matched_pred), matched_pred, used_gold


# Entity scoring
def score_entities(
    gold: GoldSet, pred_entities: list[dict], type_sensitive: bool = True,
) -> dict[str, Any]:
    pred = _build_pred_nodes(pred_entities)
    gnodes = _build_gold_nodes(gold)

    def tmatch(a: str, b: str) -> bool:
        return (a == b) if type_sensitive else True

    # Greedy 1:1 matching: each predicted node consumes at most one gold node and
    # vice versa. The old "any overlap" double-counted - 3 pred nodes all hitting
    # one gold scored 3 TP and didn't penalize over-segmentation; a single bipartite
    # match makes tp consistent for precision and recall and charges the splits.
    tp, matched_pred, used_gold = _greedy_match(pred, gnodes, tmatch)
    fp = [{"name": p.canonical_norm, "type": p.type}
          for i, p in enumerate(pred) if i not in matched_pred]
    fn = [{"name": g.rep_norm, "type": g.type}
          for j, g in enumerate(gnodes) if j not in used_gold]
    prf = PRF(tp=tp, fp=len(pred) - tp, fn=len(gnodes) - tp)

    # Per-type (type-sensitive) breakdown, same 1:1 matching within each type.
    per_type: dict[str, dict] = {}
    types = {g.type for g in gnodes} | {p.type for p in pred}
    for t in sorted(types):
        gt = [g for g in gnodes if g.type == t]
        pt = [p for p in pred if p.type == t]
        tp_t, _, _ = _greedy_match(pt, gt, lambda a, b: True)
        per_type[t] = PRF(tp=tp_t, fp=len(pt) - tp_t, fn=len(gt) - tp_t).as_dict()

    return {"overall": prf.as_dict(), "per_type": per_type,
            "false_negatives": fn[:50], "false_positives": fp[:50],
            "n_gold": len(gnodes), "n_pred": len(pred)}


# Relation scoring (entity-linking based)
def _surface_to_gold_id(gnodes: list[GoldNode]) -> dict[str, int]:
    """Map each gold surface form -> a gold node index (first wins)."""
    m: dict[str, int] = {}
    for i, g in enumerate(gnodes):
        for s in g.surfaces:
            m.setdefault(s, i)
    return m


def _name_to_pred_surfaces(pred: list[PredNode]) -> dict[str, set[str]]:
    """Map a normalized name -> the surface set of the pred node owning it."""
    m: dict[str, set[str]] = {}
    for p in pred:
        for s in p.surfaces:
            m.setdefault(s, p.surfaces)
    return m


def score_relations(
    gold: GoldSet, pred_entities: list[dict], pred_edges: list[dict],
    type_sensitive: bool = True, directed: bool = True, family: bool = False,
) -> dict[str, Any]:
    # family=True scores at the tie-class level: a predicted `located_in` matches
    # a gold `born_in` (both biographical) but not a gold `member_of` (affiliation).
    # The honest middle between strict-typed (label must match exactly) and untyped
    # (endpoints only) - the text uses its own labels for the same kind of tie.
    from postprocess import tie_classes
    pred = _build_pred_nodes(pred_entities)
    gnodes = _build_gold_nodes(gold)
    g_surface_id = _surface_to_gold_id(gnodes)
    pred_name_surfaces = _name_to_pred_surfaces(pred)

    def endpoint_id(name: str) -> str:
        """Resolve an edge-endpoint name to a shared id: a linked gold id, else
        a unique pseudo-id from the predicted node's surfaces."""
        n = normalize_name(name)
        # Direct surface match to a gold entity.
        if n in g_surface_id:
            return f"g{g_surface_id[n]}"
        # Try via the predicted node's other surfaces (aliases).
        for s in pred_name_surfaces.get(n, {n}):
            if s in g_surface_id:
                return f"g{g_surface_id[s]}"
        return f"p::{n}"

    def rel_key_from_ids(a: str, b: str, t: str) -> tuple:
        # Direction matters for asymmetric relations (X recruited Y != Y recruited
        # X); symmetric ties (married_to, met_with) collapse either order. Ignoring
        # direction inflates F1 vs the official directed scorers - default directed.
        if directed and t and not tie_classes.is_symmetric(t):
            pair = (a, b)
        else:
            pair = tuple(sorted((a, b)))
        if not type_sensitive:
            slot = ""
        elif family:
            # tie-class slot. "other" is tie_classes' catch-all for relations it
            # doesn't model (e.g. a benchmark's foreign Wikidata vocabulary); its
            # class carries no signal and would let two unrelated unknown types
            # collide, so fall back to the exact label there - family then
            # degrades to typed for unmodeled vocab, never above it.
            cls = tie_classes.classify(t) if t else ""
            slot = t if cls == "other" else cls
        else:
            slot = t
        return (pair, slot)

    pred_set: set[tuple] = set()
    for e in pred_edges:
        s = e.get("source_name") or e.get("Source") or e.get("source") or ""
        t = e.get("target_name") or e.get("Target") or e.get("target") or ""
        rtype = (e.get("rel_type") or e.get("Label") or "").lower()
        if s and t:
            ka, kb = endpoint_id(s), endpoint_id(t)
            if ka != kb:
                pred_set.add(rel_key_from_ids(ka, kb, rtype))

    # Gold relations use gold ids directly. Map any gold surface -> gold node id.
    rep_to_id: dict[str, int] = {}
    for i, g in enumerate(gnodes):
        rep_to_id[g.rep_norm] = i
        for s in g.surfaces:
            rep_to_id.setdefault(s, i)

    gold_set: set[tuple] = set()
    for r in gold.relations:
        a = rep_to_id.get(normalize_name(r.source))
        b = rep_to_id.get(normalize_name(r.target))
        if a is None or b is None or a == b:
            continue
        gold_set.add(rel_key_from_ids(f"g{a}", f"g{b}", r.type.lower()))

    tp = len(pred_set & gold_set)
    prf = PRF(tp=tp, fp=len(pred_set - gold_set), fn=len(gold_set - pred_set))

    # Per-label breakdown (typed scoring only): shows which relation labels
    # actually land, not just the aggregate. Sorted by gold support.
    per_type: dict[str, dict] = {}
    if type_sensitive:
        labels = {t for _, t in pred_set} | {t for _, t in gold_set}
        rows = []
        for t in labels:
            ps = {k for k in pred_set if k[1] == t}
            gs = {k for k in gold_set if k[1] == t}
            rows.append((t, PRF(tp=len(ps & gs), fp=len(ps - gs), fn=len(gs - ps))))
        rows.sort(key=lambda r: (-(r[1].tp + r[1].fn), r[0]))
        per_type = {t: p.as_dict() for t, p in rows}

    def readable(eid: str) -> str:
        """Resolve an internal endpoint id back to a display name."""
        if eid.startswith("g") and eid[1:].isdigit():
            return gnodes[int(eid[1:])].rep_norm
        return eid.removeprefix("p::")

    def fmt(keys: Iterable[tuple]) -> list[dict]:
        out = []
        for (pair, t) in list(keys)[:50]:
            out.append({"a": readable(pair[0]), "b": readable(pair[1]), "type": t})
        return out

    return {"overall": prf.as_dict(), "per_type": per_type,
            "false_positives": fmt(pred_set - gold_set),
            "false_negatives": fmt(gold_set - pred_set),
            "n_gold": len(gold_set), "n_pred": len(pred_set)}


def score_all(
    gold: GoldSet, pred_entities: list[dict], pred_edges: list[dict],
    directed: bool = True,
) -> dict[str, Any]:
    """Run the full evaluation and return a structured report."""
    return {
        "entities": score_entities(gold, pred_entities, type_sensitive=True),
        "entities_type_agnostic": score_entities(gold, pred_entities, type_sensitive=False),
        "relations_typed": score_relations(gold, pred_entities, pred_edges,
                                            type_sensitive=True, directed=directed),
        "relations_family": score_relations(gold, pred_entities, pred_edges,
                                            type_sensitive=True, directed=directed,
                                            family=True),
        "relations_untyped": score_relations(gold, pred_entities, pred_edges,
                                              type_sensitive=False, directed=directed),
    }
