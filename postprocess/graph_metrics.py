# SNA metrics that Gephi does NOT compute well (or at all), so they are worth
# precomputing here. Standard centrality/community (degree, betweenness,
# eigenvector, PageRank, modularity, closeness, clustering, k-core) are LEFT to
# Gephi - it does them in one click on whichever view you load.
#
# What we add:
#   - Burt's structural-hole brokerage: constraint + effective_size (the classic
#     "broker between otherwise-disconnected groups" measure; absent from Gephi).
#   - bridges / articulation points (whose removal fragments the network).
#   - a graph-health QA summary (components, isolates, density, giant component %)
#     so a malformed network is caught before it ever reaches Gephi.
#
# Brokerage/bridges are computed on the SUBSTANTIVE graph (real ties only - we
# drop the weak co-occurrence and stance layers). Everything is fail-soft: any
# error just leaves the columns unset and never breaks a run.

from __future__ import annotations

import logging
from typing import Any

from . import tie_classes

logger = logging.getLogger(__name__)

# Real ties for brokerage/bridges (exclude cooccurrence, stance, other).
_SUBSTANTIVE = tie_classes.SOCIAL | tie_classes.STRUCTURAL  # interaction + affiliation + participation + biographical


def _build_graph(node_ids, edges, keep_classes: set[str] | None):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for e in edges:
        if keep_classes is not None and e.get("tie_class") not in keep_classes:
            continue
        s, t = e.get("Source"), e.get("Target")
        if not s or not t or s == t:
            continue
        w = e.get("Weight") or 1
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)
    return G


def _qa(G) -> dict[str, Any]:
    import networkx as nx
    n = G.number_of_nodes()
    if n == 0:
        return {"nodes": 0, "edges": 0, "components": 0, "isolates": 0,
                "largest_cc_pct": 0.0, "density": 0.0}
    comps = list(nx.connected_components(G))
    largest = max((len(c) for c in comps), default=0)
    return {
        "nodes": n,
        "edges": G.number_of_edges(),
        "components": len(comps),
        "isolates": sum(1 for _ in nx.isolates(G)),
        "largest_cc_pct": round(100.0 * largest / n, 1),
        "density": round(nx.density(G), 5),
    }


def enrich(tables, *, max_constraint_nodes: int = 6000) -> dict[str, Any]:
    """Attach brokerage/bridge columns to ``tables`` and return a QA report.

    Mutates ``tables.nodes`` (adds ``sna_constraint``, ``sna_effective_size``,
    ``sna_is_articulation``) and ``tables.edges`` (adds ``is_bridge``). Returns a
    diagnostics dict for logging / a report file. Fail-soft throughout.
    """
    report: dict[str, Any] = {}
    try:
        import networkx as nx
    except Exception as exc:  # noqa: BLE001
        logger.warning("networkx unavailable; skipping graph metrics: %s", exc)
        return report

    try:
        node_ids = [n["Id"] for n in tables.nodes]
        g_full = _build_graph(node_ids, tables.edges, None)
        g_sub = _build_graph(node_ids, tables.edges, _SUBSTANTIVE)
        report["qa_full"] = _qa(g_full)
        report["qa_substantive"] = _qa(g_sub)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Graph QA failed: %s", exc)
        return report

    # Brokerage (structural holes) on the substantive graph. Constraint is
    # O(n*deg^2); guard very large graphs so a full corpus never stalls a run.
    constraint: dict[str, float] = {}
    effective: dict[str, float] = {}
    sub_nodes = [v for v in g_sub.nodes if g_sub.degree(v) > 0]
    if 0 < len(sub_nodes) <= max_constraint_nodes:
        active = g_sub.subgraph(sub_nodes)
        try:
            for v, c in nx.constraint(active).items():
                if c == c:  # filter NaN
                    constraint[v] = round(float(c), 4)
        except Exception as exc:  # noqa: BLE001
            logger.debug("constraint failed: %s", exc)
        try:
            for v, s in nx.effective_size(active).items():
                if s == s:
                    effective[v] = round(float(s), 3)
        except Exception as exc:  # noqa: BLE001
            logger.debug("effective_size failed: %s", exc)
    elif len(sub_nodes) > max_constraint_nodes:
        logger.info("Skipping brokerage: substantive graph too large (%d nodes > %d).",
                    len(sub_nodes), max_constraint_nodes)

    # Bridges + articulation points on the substantive graph.
    bridges: set[frozenset[str]] = set()
    articulation: set[str] = set()
    try:
        bridges = {frozenset((u, v)) for u, v in nx.bridges(g_sub)}
    except Exception as exc:  # noqa: BLE001
        logger.debug("bridges failed: %s", exc)
    try:
        articulation = set(nx.articulation_points(g_sub))
    except Exception as exc:  # noqa: BLE001
        logger.debug("articulation_points failed: %s", exc)

    report["brokerage_nodes"] = len(constraint)
    report["bridges"] = len(bridges)
    report["articulation_points"] = len(articulation)

    # Merge onto rows.
    for n in tables.nodes:
        nid = n["Id"]
        if nid in constraint:
            n["sna_constraint"] = constraint[nid]
        if nid in effective:
            n["sna_effective_size"] = effective[nid]
        n["sna_is_articulation"] = nid in articulation
    for e in tables.edges:
        e["is_bridge"] = frozenset((e.get("Source"), e.get("Target"))) in bridges

    return report
