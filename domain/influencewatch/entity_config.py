# Entity-type config for InfluenceWatch. STOPWORDS is deliberately small: most
# generic-noise words (board, fund, committee, agency) are meaningful here and
# must NOT be dropped. Only bare honorifics and pure pronoun/common-noun noise go.

from __future__ import annotations

# Forced label by lowercased name. Acronyms GLiNER often types as PERSON or misses.
LABEL_OVERRIDES: dict[str, str] = {
    "pac": "ORG",
    "super pac": "ORG",
    "irs": "INSTITUTION",
    "fec": "INSTITUTION",
    "sec": "INSTITUTION",
    "doj": "INSTITUTION",
}

STOPWORDS: set[str] = {
    "he", "she", "they", "it", "we", "i", "you",
    "man", "woman", "people", "person", "someone", "anyone", "everyone",
    "official", "officials", "spokesman", "spokeswoman", "spokesperson",
    "mr", "mrs", "ms", "miss", "dr", "sen", "rep", "gov", "president",
    "the group", "the company", "the organization", "the agency", "the foundation",
}

# Controlled subtypes handed to the LLM enricher.
ENTITY_SUBTYPES: dict[str, list[str]] = {
    "PERSON": ["politician", "lobbyist", "donor", "executive", "operative",
               "activist", "official", "other"],
    "ORG": ["pac", "super_pac", "nonprofit", "foundation", "think_tank",
            "trade_association", "labor_union", "corporation", "shell_company",
            "law_firm", "political_party", "campaign", "other"],
    "INSTITUTION": ["federal_agency", "state_agency", "regulator", "legislature",
                    "court", "other"],
    "LOCATION": ["city", "state", "country", "district", "other"],
}
