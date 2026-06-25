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


def _build_graph(node_ids, edges, keep_classes: set[str] | None,
                 drop_unsupported: bool = False):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for e in edges:
        if keep_classes is not None and e.get("tie_class") not in keep_classes:
            continue
        # Edges the relation verifier could not ground in their own evidence are
        # tagged (and stay in the export) but must not drive structural-hole or
        # bridge analysis - IF the verifier is trustworthy (quality.trust_verification).
        # A weak local self-verifier over-rejects, so this defaults off and the
        # flags only tag. No-op unless verify_relations ran (else field is "").
        if drop_unsupported and e.get("verification") == "unsupported":
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


def _signed_balance(edges, *, max_triangles: int = 200_000) -> dict[str, Any]:
    """Cartwright-Harary structural balance on the signed tie graph.

    A triangle is balanced when the product of its three edge signs is positive
    ("the friend of my friend is my friend; the enemy of my enemy is my friend").
    We only use edges that carry a polarity (positive/negative); neutral ties are
    not signed. Reports the balanced fraction - high = a polarized, balanced
    network; low = lots of frustrated triads. Bounded triangle enumeration.
    """
    import networkx as nx
    sign = {"positive": 1, "negative": -1}
    G = nx.Graph()
    for e in edges:
        s, t = e.get("Source"), e.get("Target")
        sg = sign.get(str(e.get("polarity") or ""))
        if not s or not t or s == t or sg is None:
            continue
        # Collapse parallel signed edges by sign sum; net 0 -> drop (ambiguous).
        if G.has_edge(s, t):
            G[s][t]["s"] += sg
        else:
            G.add_edge(s, t, s=sg)
    balanced = unbalanced = 0
    seen: set[frozenset] = set()
    for u, v in G.edges():
        if G[u][v]["s"] == 0:
            continue
        for w in set(G[u]) & set(G[v]):
            tri = frozenset((u, v, w))
            if w == u or w == v or tri in seen:
                continue
            s1, s2, s3 = G[u][v]["s"], G[u][w].get("s", 0), G[v][w].get("s", 0)
            if s1 == 0 or s2 == 0 or s3 == 0:
                continue
            seen.add(tri)
            if (1 if s1 > 0 else -1) * (1 if s2 > 0 else -1) * (1 if s3 > 0 else -1) > 0:
                balanced += 1
            else:
                unbalanced += 1
            if len(seen) >= max_triangles:
                break
        if len(seen) >= max_triangles:
            break
    total = balanced + unbalanced
    return {"signed_edges": G.number_of_edges(), "triangles": total,
            "balanced": balanced, "unbalanced": unbalanced,
            "balanced_pct": round(100.0 * balanced / total, 1) if total else 0.0}


def _polarity_conflicts(edges, id_to_name=None, *, sample: int = 20) -> dict[str, Any]:
    """Dyads carrying BOTH a positive and a negative tie - contradictory signed
    edges on the same pair (e.g. allied_with + fought_against). Either an
    extraction error or a genuinely ambivalent / over-time relationship; either
    way worth a look. Signed balance collapses these to net-zero and drops them,
    so they are otherwise invisible. Reported, not filtered - the analyst decides.
    """
    id_to_name = id_to_name or {}
    pos: dict[frozenset, set] = {}
    neg: dict[frozenset, set] = {}
    for e in edges:
        s, t = e.get("Source"), e.get("Target")
        if not s or not t or s == t:
            continue
        pol = str(e.get("polarity") or "")
        bucket = pos if pol == "positive" else neg if pol == "negative" else None
        if bucket is None:
            continue
        bucket.setdefault(frozenset((s, t)), set()).add(
            e.get("rel_type") or e.get("tie_class") or "?")
    conflicts = []
    for pair in set(pos) & set(neg):
        a, b = tuple(pair)
        conflicts.append({"source": id_to_name.get(a, a), "target": id_to_name.get(b, b),
                          "positive": sorted(pos[pair]), "negative": sorted(neg[pair])})
    conflicts.sort(key=lambda c: (c["source"], c["target"]))
    return {"conflicting_dyads": len(conflicts), "sample": conflicts[:sample]}


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


def quality_pillars(report: dict[str, Any], tables) -> dict[str, Any]:
    """KGC-2026-style quality summary over five pillars, derived from data the
    run already has - not new computation. provenance + consistency come
    straight from edge tiers and conflict counts; accuracy/completeness/
    timeliness are honest coverage proxies (there's no gold at run time),
    labelled as such so nobody reads them as scored. Fail-soft."""
    try:
        from postprocess import evidence_tiers as et
        edges = list(getattr(tables, "edges", []) or [])
        n = len(edges)
        pct = lambda k: round(100.0 * k / n, 1) if n else None

        # provenance: every edge should carry an edge_source.
        with_src = sum(1 for e in edges if (e.get("edge_source") or "").strip())
        # accuracy proxy: share of edges in the conservative (asserted) tier.
        asserted = sum(1 for e in edges
                       if et.tier_allows(e.get("edge_source", ""), "conservative"))
        # consistency: contradictory dyads + type-signature violations.
        conf = report.get("conflicts", {})
        n_conf = conf.get("conflicting_dyads", 0) if isinstance(conf, dict) else 0
        n_typeviol = sum(1 for e in edges if e.get("type_violation"))
        # Per-relation breakdown: which relation types violate most. Fast way to
        # tell a too-tight signature (one relation dominates) from a real
        # extraction problem (spread across many).
        tv_by_rel: dict[str, int] = {}
        for e in edges:
            if e.get("type_violation"):
                rt = e.get("rel_type", "?")
                tv_by_rel[rt] = tv_by_rel.get(rt, 0) + 1
        tv_by_rel = dict(sorted(tv_by_rel.items(), key=lambda kv: -kv[1]))
        # completeness proxy: connectivity (isolates flood => undercovered).
        qa = report.get("qa_substantive", {})
        # timeliness proxy: temporal coverage of edges.
        dated = sum(1 for e in edges
                    if e.get("period") or e.get("date") or e.get("year"))

        return {
            "accuracy_proxy": {
                "asserted_tier_pct": pct(asserted),
                "note": "share of edges stated in text or a verified record; "
                        "not gold-scored accuracy",
            },
            "completeness_proxy": {
                "largest_cc_pct": qa.get("largest_cc_pct"),
                "isolates": qa.get("isolates"),
                "note": "graph connectivity stands in for completeness; "
                        "no gold recall at run time",
            },
            "consistency": {
                "polarity_conflicts": n_conf,
                "type_violations": n_typeviol,
                "type_violations_by_relation": tv_by_rel,
                "clean_pct": pct(n - n_typeviol),
            },
            "provenance": {
                "edges_with_source_pct": pct(with_src),
            },
            "timeliness_proxy": {
                "edges_with_time_pct": pct(dated),
                "note": "share of edges carrying a period/date; corpus-dependent",
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("quality_pillars failed: %s", exc)
        return {}


def enrich(tables, *, trust_verification: bool = False,
           max_constraint_nodes: int = 6000) -> dict[str, Any]:
    """Attach brokerage/bridge columns to ``tables`` and return a QA report.

    Mutates ``tables.nodes`` (adds ``sna_constraint``, ``sna_effective_size``,
    ``sna_is_articulation``) and ``tables.edges`` (adds ``is_bridge``). Returns a
    diagnostics dict for logging / a report file. Fail-soft throughout.

    ``trust_verification`` (quality.trust_verification) gates whether
    verification=unsupported edges are pruned from the metric graphs; off for
    weak local verifiers that over-reject (see QualityConfig).
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
        g_sub = _build_graph(node_ids, tables.edges, _SUBSTANTIVE,
                             drop_unsupported=trust_verification)
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
        # Weighted: edge weight = corroboration (distinct docs). Burt's constraint
        # and effective size are defined on the weighted ego network, so pass it.
        try:
            for v, c in nx.constraint(active, weight="weight").items():
                if c == c:  # filter NaN
                    constraint[v] = round(float(c), 4)
        except Exception as exc:  # noqa: BLE001
            logger.debug("constraint failed: %s", exc)
        try:
            for v, s in nx.effective_size(active, weight="weight").items():
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

    # Verifier-flagged edges drop out of the signed analysis too - same trust
    # gate: an unsupported stance edge makes false balance triads, but only a
    # reliable verifier earns the drop. Weak local flags stay in (tagged, not
    # cut). No-op when verify_relations did not run (field empty).
    signed_edges = [e for e in tables.edges
                    if not (trust_verification and e.get("verification") == "unsupported")]

    # Signed structural balance over the full edge set (stance edges carry the
    # sign; substantive-only would drop them since stance is non-social).
    try:
        report["balance"] = _signed_balance(signed_edges)
    except Exception as exc:  # noqa: BLE001
        logger.debug("structural balance failed: %s", exc)

    # Contradictory signed dyads (same pair, both ally and enemy). A QA signal,
    # not a filter - balance drops them as net-zero, so surface the count + a
    # readable sample here for review.
    try:
        id_to_name = {n["Id"]: n.get("Label", n["Id"]) for n in tables.nodes}
        report["conflicts"] = _polarity_conflicts(signed_edges, id_to_name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("polarity conflicts failed: %s", exc)

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
