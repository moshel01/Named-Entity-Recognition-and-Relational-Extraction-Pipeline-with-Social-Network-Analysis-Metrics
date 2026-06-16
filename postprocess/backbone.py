# Disparity-filter backbone extraction for weighted co-occurrence layers.
#
# Co-occurrence graphs are dense by construction: every pair sharing a document (or
# a text window) gets an edge, so a global weight cutoff either keeps hubs' noise or
# cuts leaf nodes' only ties. The disparity filter (Serrano, Boguna, Vespignani,
# PNAS 2009) instead asks, per node, whether an incident edge carries more weight
# than a random allocation of that node's strength would predict. For a node of
# degree k, an edge with normalized weight p = w/strength is significant at level
# alpha when (1 - p)^(k-1) < alpha. An edge stays in the backbone if it is
# significant for EITHER endpoint (the standard, recall-friendly choice).
#
# We run it on the co_occurs_with layer only - typed/asserted edges are never
# touched. Every co-occurrence edge gets a `disparity_alpha` (its best p-value
# across the two endpoints); when a threshold is set, edges above it are dropped.

from __future__ import annotations

import logging
from collections import defaultdict

from core.schema import Relationship

logger = logging.getLogger(__name__)


def _pair_weight(r: Relationship) -> float:
    """Observation weight for a co-occurrence edge: Newman projection strength if
    present (cross-doc), else within-window count, else 1."""
    a = r.attributes or {}
    return float(a.get("cooccur_strength")
                 or a.get("cooccur_count")
                 or a.get("shared_docs") or 1)


def disparity_filter(
    edges: list[Relationship], alpha: float = 0.0
) -> tuple[list[Relationship], int]:
    """Tag co_occurrence edges with `disparity_alpha`; drop the non-backbone ones
    when ``alpha`` > 0. Non-co-occurrence edges pass through untouched.

    Returns (kept_edges, dropped_count). With alpha == 0 nothing is dropped - the
    alpha is still stamped so a Gephi user can filter on it by hand.
    """
    cooc = [r for r in edges if (r.attributes or {}).get("edge_source") == "rule_cooccurrence"]
    if not cooc:
        return edges, 0

    # Aggregate undirected weight + degree per node over the co-occurrence graph.
    strength: dict[str, float] = defaultdict(float)
    degree: dict[str, int] = defaultdict(int)
    pair_w: dict[frozenset, float] = defaultdict(float)
    for r in cooc:
        w = _pair_weight(r)
        pair_w[frozenset((r.source, r.target))] += w
    for pair, w in pair_w.items():
        a, b = tuple(pair) if len(pair) == 2 else (next(iter(pair)), next(iter(pair)))
        strength[a] += w
        strength[b] += w
        degree[a] += 1
        degree[b] += 1

    def edge_alpha(u: str, v: str, w: float) -> float:
        # Best (smallest) p across the two endpoints. k == 1 -> undefined; keep the
        # edge (a leaf's single tie can't be pruned), report alpha 0.0.
        best = 1.0
        for node in (u, v):
            k = degree[node]
            s = strength[node]
            if k > 1 and s > 0:
                p = w / s
                best = min(best, (1.0 - p) ** (k - 1))
            elif k == 1:
                return 0.0
        return best

    kept: list[Relationship] = []
    dropped = 0
    for r in cooc:
        w = pair_w[frozenset((r.source, r.target))]
        a = edge_alpha(r.source, r.target, w)
        r.attributes["disparity_alpha"] = round(a, 5)
        if alpha > 0 and a >= alpha:
            dropped += 1
            continue
        kept.append(r)

    others = [r for r in edges if (r.attributes or {}).get("edge_source") != "rule_cooccurrence"]
    if alpha > 0:
        logger.info("Disparity backbone (alpha=%.3f): kept %d/%d co-occurrence edges, dropped %d.",
                    alpha, len(kept), len(cooc), dropped)
    return others + kept, dropped
