# Two-mode (affiliation) -> one-mode (actor) projection.
#
# In affiliation-dense corpora - modern political "dark money" (people share PAC
# boards / shell companies), multi-agency disaster response (agencies share a
# response event) - direct person-person ties are rare; actors connect THROUGH a
# shared group. This is the classic two-mode network (Breiger 1974): project the
# actor x group bipartite graph onto an actor x actor graph, where two actors tied
# to the same group get an edge.
#
# Weighting follows the same Newman 1/(k-1) scheme as the cross-document
# co-occurrence projection: sharing a 2-person board is a strong tie, sharing a
# 500-member party is not. A co_affiliated edge is a CO-PRESENCE, not a direct
# asserted tie (the two may never have met) - full tier, like co-occurrence.

from __future__ import annotations

import itertools
import logging
from collections import defaultdict

from core.schema import Entity, Relationship

from . import tie_classes

logger = logging.getLogger(__name__)

# Defaults: people are actors, orgs/institutions/events are the groups they share.
# Disaster-response corpora override actors to orgs/agencies sharing an event (set
# actor_labels={ORG,INSTITUTION}, group_labels={EVENT}) - then two agencies that
# responded to the same fire get the co_affiliated edge.
_GROUP_LABELS = frozenset({"ORG", "INSTITUTION", "EVENT"})
_ACTOR_LABELS = frozenset({"PERSON"})


def project_affiliations(
    entities: list[Entity], edges: list[Relationship], min_shared: int = 1,
    actor_labels: frozenset[str] | None = None,
    group_labels: frozenset[str] | None = None,
) -> list[Relationship]:
    """Return co_affiliated actor-actor edges from shared group memberships.

    An edge contributes if it's an affiliation/participation tie between an actor
    and a group. Two actors in the same group get a co_affiliated edge, Newman-
    weighted by group size and summed over shared groups. ``min_shared`` gates on
    the number of shared groups. ``actor_labels`` / ``group_labels`` default to
    PERSON actors over ORG/INSTITUTION/EVENT groups; override for org-as-actor
    domains. An edge with both endpoints in both sets resolves target=group.
    """
    actors = actor_labels or _ACTOR_LABELS
    groups = group_labels or _GROUP_LABELS
    label = {e.entity_id: e.label for e in entities}
    name = {e.entity_id: e.canonical_name for e in entities}

    # group_id -> set of actor ids tied to it.
    group_members: dict[str, set[str]] = defaultdict(set)
    for r in edges:
        sl, tl = label.get(r.source), label.get(r.target)
        if sl is None or tl is None:
            continue
        cls = tie_classes.classify(r.rel_type, sl, tl)
        if cls not in ("affiliation", "participation"):
            continue
        # The group is the org/institution/event endpoint, the actor the other.
        if tl in groups and sl in actors:
            group, actor = r.target, r.source
        elif sl in groups and tl in actors:
            group, actor = r.source, r.target
        else:
            continue
        if group == actor:
            continue
        group_members[group].add(actor)

    # Newman one-mode projection.
    pair_groups: dict[frozenset[str], set[str]] = defaultdict(set)
    pair_strength: dict[frozenset[str], float] = defaultdict(float)
    for group, members in group_members.items():
        k = len(members)
        if k < 2:
            continue
        w = 1.0 / (k - 1)
        for a, b in itertools.combinations(sorted(members), 2):
            fp = frozenset((a, b))
            pair_groups[fp].add(group)
            pair_strength[fp] += w

    out: list[Relationship] = []
    for pair, groups in pair_groups.items():
        if len(groups) < min_shared:
            continue
        a, b = tuple(pair)
        shared_names = ", ".join(sorted(name.get(g, g) for g in groups))
        out.append(
            Relationship(
                source=a, target=b, rel_type="co_affiliated",
                doc_id="",
                evidence=f"Share {len(groups)} affiliation(s): {shared_names}"[:200],
                confidence=min(1.0, 0.4 + 0.1 * len(groups)),
                directed=False, origin="inferred",
                attributes={"edge_source": "affiliation_projected",
                            "shared_groups": len(groups),
                            "affiliation_strength": round(pair_strength[pair], 4)},
            )
        )
    logger.info("Projected %d co_affiliated edges from %d shared groups",
                len(out), sum(1 for m in group_members.values() if len(m) >= 2))
    return out
