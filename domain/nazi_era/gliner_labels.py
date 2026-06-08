# Pre-configured GLiNER zero-shot labels for Nazi-era German text (1919-1945).

from __future__ import annotations

# The labels handed to GLiNER at inference time (order is irrelevant).
LABELS: list[str] = [
    "person",
    "nazi organization",
    "political party",
    "paramilitary unit",
    "military unit",
    "sa unit",
    "ss unit",
    "nsdap subdivision",
    "city",
    "region",
    "country",
    "venue",
    "historical event",
    "political event",
    "military event",
    "date",
    "military rank",
    "nazi rank",
    "government institution",
    "educational institution",
    "church or religious organization",
    "company or employer",
    "youth organization",
    "veterans organization",
]

# GLiNER label (lowercase) -> canonical pipeline entity type.
LABEL_TO_TYPE_MAP: dict[str, str] = {
    "person": "PERSON",
    "nazi organization": "ORG",
    "political party": "ORG",
    "paramilitary unit": "ORG",
    "military unit": "ORG",
    "sa unit": "ORG",
    "ss unit": "ORG",
    "nsdap subdivision": "ORG",
    "city": "LOCATION",
    "region": "LOCATION",
    "country": "LOCATION",
    "venue": "LOCATION",
    "historical event": "EVENT",
    "political event": "EVENT",
    "military event": "EVENT",
    "date": "DATE",
    "military rank": "RANK",
    "nazi rank": "RANK",
    "government institution": "INSTITUTION",
    "educational institution": "INSTITUTION",
    "church or religious organization": "ORG",
    "company or employer": "ORG",
    "youth organization": "ORG",
    "veterans organization": "ORG",
}
