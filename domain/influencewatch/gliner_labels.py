# GLiNER zero-shot labels for modern political-influence text (InfluenceWatch
# profiles, FEC-style filings, investigative news). Money amounts are NOT entities
# here - they ride as the qual_monetary_value edge qualifier, not a node.

from __future__ import annotations

LABELS: list[str] = [
    "person",
    "politician",
    "lobbyist",
    "donor",
    "political action committee",
    "super pac",
    "nonprofit organization",
    "foundation",
    "think tank",
    "trade association",
    "labor union",
    "corporation",
    "shell company",
    "law or lobbying firm",
    "political party",
    "campaign",
    "government agency",
    "city",
    "state",
    "country",
    "date",
]

# GLiNER label (lowercase) -> canonical pipeline entity type. Everything that acts
# as a fundable/affiliating body folds to ORG; only a regulator/department is
# INSTITUTION. Keeping PACs and shells as ORG lets the bipartite projection treat
# them as the shared group two actors connect through.
LABEL_TO_TYPE_MAP: dict[str, str] = {
    "person": "PERSON",
    "politician": "PERSON",
    "lobbyist": "PERSON",
    "donor": "PERSON",
    "political action committee": "ORG",
    "super pac": "ORG",
    "nonprofit organization": "ORG",
    "foundation": "ORG",
    "think tank": "ORG",
    "trade association": "ORG",
    "labor union": "ORG",
    "corporation": "ORG",
    "shell company": "ORG",
    "law or lobbying firm": "ORG",
    "political party": "ORG",
    "campaign": "ORG",
    "government agency": "INSTITUTION",
    "city": "LOCATION",
    "state": "LOCATION",
    "country": "LOCATION",
    "date": "DATE",
}
