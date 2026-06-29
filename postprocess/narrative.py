# Narrative-sequence networks (Bearman & Stovel, "Becoming a Nazi: A model for
# narrative networks", Poetics 2000). Their move: stop treating a life story as a
# bag of entities and treat it as a *sequence* - elements (events/states) are nodes,
# and one element following another in the telling is a directed edge. Aggregated
# across many autobiographies, the recurring element-to-element transitions are the
# shape of the typical narrative (e.g. war -> hardship -> politics).
#
# v1, deliberately coarse: an "element" is a keyword-bucketed event category, and
# the sequence is the per-document timeline ordered by year (then extraction order;
# many events are undated, so order is approximate). This is the scaffold - the
# element scheme is the part a domain refines. Off by default (export.narrative_network).

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Coarse life-course / interwar categories. First match wins, so order specific
# before generic. EN + DE keywords (the Abel corpus is German). A domain can pass
# its own rules; this is the generic default.
_ELEMENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("war_combat", ("war", "front", "battle", "combat", "soldier", "regiment",
                    "trench", "wounded", "krieg", "soldat", "schlacht", "schützengraben")),
    ("violence", ("street fight", "beaten", "riot", "clash", "freikorps", "putsch",
                  "murder", "assault", "gewalt", "schlägerei")),
    ("politics_party", ("party", "nsdap", "rally", "election", "movement", "agitation",
                        "propaganda", "partei", "versammlung", "bewegung", "wahl")),
    ("education", ("school", "university", "studied", "gymnasium", "examination",
                   "studium", "schule", "universität", "lehrer")),
    ("work_economic", ("factory", "unemploy", "wage", "business", "farm", "trade",
                       "apprentice", "arbeit", "stelle", "fabrik", "arbeitslos")),
    ("family", ("father", "mother", "married", "wife", "child", "born", "family",
                "vater", "mutter", "heirat", "geboren", "familie")),
    ("migration", ("emigrat", "returned", "travel", "moved to", "zog", "auswander")),
    ("hardship_crisis", ("hunger", "poverty", "inflation", "crisis", "illness",
                         "death", "elend", "armut", "krise", "krankheit", "tod")),
    ("religion", ("church", "faith", "catholic", "protestant", "kirche", "glaube",
                  "katholisch", "evangelisch")),
]


# Dramatic / plot-beat scheme for fiction and scripts (novels, TV/film). Same
# Bearman-Stovel sequence machinery, but the elements are story beats (conflict ->
# revelation -> resolution) instead of life-course stages. Select it with
# export.narrative_scheme: fiction, or ship a domain narrative_rules.py. First match
# wins, so order specific before generic. English keywords (most fiction corpora).
FICTION_ELEMENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("death", ("death", "died", "killed", "murder", "slain", "funeral", "grave", "corpse")),
    ("violence_conflict", ("fight", "battle", "attack", "duel", "ambush", "struggle",
                           "blood", "wound", "war")),
    ("romance", ("love", "kiss", "embrace", "wedding", "married", "affair", "courtship",
                 "lover", "betrothed")),
    ("betrayal", ("betray", "treachery", "deceive", "trick", "double-cross", "traitor",
                  "backstab")),
    ("revelation", ("discover", "reveal", "realize", "uncover", "secret", "confess",
                    "truth", "prophecy")),
    ("journey", ("journey", "travel", "set out", "depart", "arrive", "voyage", "quest",
                 "road", "sail")),
    ("threat_danger", ("danger", "threat", "trap", "pursue", "chase", "escape", "flee",
                       "hunt", "captured")),
    ("crime", ("steal", "theft", "robbery", "kidnap", "ransom", "heist", "burglary",
               "smuggle")),
    ("power_politics", ("throne", "crown", "king", "queen", "rule", "power", "election",
                        "conspiracy", "coup", "council")),
    ("celebration", ("feast", "party", "celebration", "festival", "banquet", "triumph",
                     "victory", "toast")),
    ("loss_grief", ("loss", "grief", "mourning", "exile", "ruin", "despair", "abandon",
                    "orphan")),
    ("reconciliation", ("forgive", "reunite", "reconcile", "peace", "redemption",
                        "homecoming", "restored")),
]

# Named schemes selectable from config (export.narrative_scheme). A domain's
# narrative_rules() still wins over either.
ELEMENT_SCHEMES: dict[str, list] = {
    "life_course": _ELEMENT_RULES,
    "fiction": FICTION_ELEMENT_RULES,
}


def categorize(text: str, rules=_ELEMENT_RULES) -> str:
    """Bucket an event description into a narrative element category."""
    t = (text or "").lower()
    for label, kws in rules:
        if any(k in t for k in kws):
            return label
    return "other"


def _field(ev: Any, name: str, default=None):
    return ev.get(name, default) if isinstance(ev, dict) else getattr(ev, name, default)


def build_transitions(timeline: list, rules=_ELEMENT_RULES):
    """Per-document event sequences -> aggregated element->element transitions.

    Returns (transitions, element_docs):
      transitions[(a, b)] = {"weight": consecutive count, "docs": set(doc_id)}
      element_docs[cat]    = set(doc_id) the element appears in.
    """
    by_doc: dict[str, list] = defaultdict(list)
    for idx, ev in enumerate(timeline):
        by_doc[_field(ev, "doc_id", "") or ""].append((idx, ev))

    transitions: dict[tuple[str, str], dict] = {}
    element_docs: dict[str, set] = defaultdict(set)
    for doc, evs in by_doc.items():
        # Year when present, else push undated events to the end in extraction order.
        evs.sort(key=lambda e: (_field(e[1], "year") or 10**9, e[0]))
        cats = [categorize(_field(e[1], "description", "") or "", rules) for e in evs]
        for c in cats:
            element_docs[c].add(doc)
        for a, b in zip(cats, cats[1:]):
            if a == b:                       # collapse repeats; a transition is a change
                continue
            st = transitions.setdefault((a, b), {"weight": 0, "docs": set()})
            st["weight"] += 1
            st["docs"].add(doc)
    return transitions, element_docs


def write_narrative(out_dir: str | Path, timeline: list, rules=_ELEMENT_RULES) -> dict[str, str]:
    """Write narrative.gexf + narrative_transitions.csv from the timeline. Returns
    {artifact: path}; empty if there is nothing to sequence. Fail-soft."""
    out_dir = Path(out_dir)
    transitions, element_docs = build_transitions(timeline, rules)
    if not transitions:
        return {}
    written: dict[str, str] = {}

    from .exporter import _write_csv
    rows = [{"from": a, "to": b, "weight": st["weight"], "n_docs": len(st["docs"])}
            for (a, b), st in sorted(transitions.items(), key=lambda kv: -kv[1]["weight"])]
    p_csv = out_dir / "narrative_transitions.csv"
    _write_csv(rows, p_csv)
    written["narrative_transitions"] = str(p_csv)

    try:
        import networkx as nx
        G = nx.DiGraph()
        for cat, docs in element_docs.items():
            G.add_node(cat, n_docs=len(docs))
        for (a, b), st in transitions.items():
            G.add_edge(a, b, weight=st["weight"], n_docs=len(st["docs"]))
        p_gexf = out_dir / "narrative.gexf"
        nx.write_gexf(G, p_gexf)
        written["narrative_gexf"] = str(p_gexf)
    except Exception as exc:  # noqa: BLE001 - networkx missing / write error
        logger.debug("narrative gexf skipped: %s", exc)

    logger.info("Narrative network: %d elements, %d transitions across %d documents.",
                len(element_docs), len(transitions),
                len({d for st in transitions.values() for d in st["docs"]}))
    return written
