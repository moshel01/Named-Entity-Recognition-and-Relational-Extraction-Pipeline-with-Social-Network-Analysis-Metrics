# GLiNER zero-shot labels for Oregon disaster-response text. Jurisdiction/location
# scope rides as the qual_jurisdiction / qual_location edge qualifier; a grant or
# program is a node (actors connect to it), the dollar amount is qual_monetary_value.

from __future__ import annotations

LABELS: list[str] = [
    "person",
    "emergency management official",
    "government agency",
    "tribal government",
    "nonprofit or relief organization",
    "community organization",
    "faith-based organization",
    "county",
    "city",
    "region",
    "tribal land",
    "shelter or facility",
    "wildfire",
    "flood",
    "disaster or emergency",
    "funding program or grant",
    "date",
]

# Agencies/regulators -> INSTITUTION; NGOs/tribes/community/faith orgs -> ORG (the
# affiliating bodies the projection links actors through). Grants/programs map to
# EVENT so they read as the central node actors connect to (the hyperedge stand-in
# until event reification lands).
LABEL_TO_TYPE_MAP: dict[str, str] = {
    "person": "PERSON",
    "emergency management official": "PERSON",
    "government agency": "INSTITUTION",
    "tribal government": "INSTITUTION",
    "nonprofit or relief organization": "ORG",
    "community organization": "ORG",
    "faith-based organization": "ORG",
    "county": "LOCATION",
    "city": "LOCATION",
    "region": "LOCATION",
    "tribal land": "LOCATION",
    "shelter or facility": "LOCATION",
    "wildfire": "EVENT",
    "flood": "EVENT",
    "disaster or emergency": "EVENT",
    "funding program or grant": "EVENT",
    "date": "DATE",
}
