# Entity config for OREM/OPAL. STOPWORDS stays small - "agency", "county",
# "shelter", "grant", "program" are load-bearing here and must survive.

from __future__ import annotations

# Oregon/federal response acronyms GLiNER mistypes or splits. Lowercased keys.
LABEL_OVERRIDES: dict[str, str] = {
    "odhs": "INSTITUTION",
    "oem": "INSTITUTION",
    "odem": "INSTITUTION",
    "fema": "INSTITUTION",
    "oha": "INSTITUTION",
    "orem": "INSTITUTION",
    "opal": "ORG",
    "arc": "ORG",
    "red cross": "ORG",
}

STOPWORDS: set[str] = {
    "he", "she", "they", "it", "we", "i",
    "man", "woman", "people", "person", "someone", "resident", "residents",
    "official", "officials", "responder", "responders", "volunteer", "volunteers",
    "survivor", "survivors", "victim", "victims",
    "mr", "mrs", "ms", "dr", "director", "secretary", "governor",
    "the agency", "the county", "the state", "the program", "the team",
}

ENTITY_SUBTYPES: dict[str, list[str]] = {
    "PERSON": ["emergency_manager", "official", "coordinator", "responder",
               "elected_official", "other"],
    "ORG": ["relief_org", "nonprofit", "community_org", "faith_based",
            "tribal_org", "other"],
    "INSTITUTION": ["federal_agency", "state_agency", "county_agency",
                    "tribal_government", "emergency_management", "health_authority",
                    "other"],
    "LOCATION": ["county", "city", "region", "tribal_land", "facility", "other"],
    "EVENT": ["wildfire", "flood", "storm", "public_health", "grant_program",
              "exercise", "other"],
}
