# Entity types, subtypes, and label overrides for the Nazi-era domain.

from __future__ import annotations

# Canonical entity types this domain emits (superset of the generic four).
ENTITY_TYPES: list[str] = ["PERSON", "ORG", "LOCATION", "EVENT", "RANK", "DATE", "INSTITUTION"]

# Bare common nouns the NER/LLM mislabels as entities. Matched on the exact
# normalized name, so specific names ("Volksschule Berlin") are unaffected.
STOPWORDS: set[str] = {
    "stadt", "dorf", "sohn", "tochter", "vater", "mutter", "eltern", "bruder",
    "schwester", "familie", "mann", "frau", "herr", "kind", "kinder", "leute",
    "mensch", "menschen", "freund", "freunde", "kamerad", "kameraden",
    "jahr", "jahre", "tag", "tage", "woche", "monat", "zeit", "leben",
    "arbeit", "schule", "haus", "stelle", "beruf", "welt", "heimat",
    "soldat", "soldaten", "regierung", "wahl", "wahlen", "betrieb", "betrieben",
    "staat", "volk", "volksgenossen", "volksgenosse", "parteigenossen",
    "parteigenosse", "parteien", "ortsgruppe", "bevölkerung", "kameradschaft",
    "pg", "arbeiter", "verwaltung", "kreise", "kreis", "behörde", "gegner",
    "feind", "bürger", "genosse", "anhänger", "masse", "massen",
    # Bare ranks the EntityRuler tags generically (rank-with-name nodes survive).
    "general", "sturmführer", "unteroffizier", "leutnant", "hauptmann",
    "gefreiter", "feldwebel", "major", "oberst", "offizier", "rottenführer",
    "scharführer", "obersturmführer", "hauptsturmführer",
    # Salutations and generic school types (named schools survive exact match).
    "heil hitler", "volksschule", "gymnasium", "realschule", "oberrealschule",
    "hochschule", "universität", "bürgerschule", "mittelschule",
}

# Fine-grained subtypes for analysis/interpretation (informational).
ENTITY_SUBTYPES: dict[str, list[str]] = {
    "PERSON": [
        "nazi_leader", "political_figure", "military_figure", "monarch",
        "intellectual", "martyr", "ordinary_member", "opponent",
    ],
    "ORG": [
        "political_party", "paramilitary", "nazi_organization",
        "nsdap_subdivision", "sa_unit", "ss_unit", "military_unit",
        "youth_organization", "veterans_organization", "labor_organization",
        "religious_organization", "company_employer",
    ],
    "LOCATION": ["city", "region", "state", "country", "venue"],
    "EVENT": ["political_event", "military_event", "putsch", "election", "war"],
    "INSTITUTION": ["government", "educational", "judicial", "press"],
}

# Forced canonical-label overrides keyed by lowercased surface form. These are
# common generic references in NSDAP autobiographies that NER tools misread.
LABEL_OVERRIDES: dict[str, str] = {
    "the party": "ORG",
    "die partei": "ORG",
    "the movement": "ORG",
    "die bewegung": "ORG",
    "the führer": "PERSON",
    "der führer": "PERSON",
    "the fuhrer": "PERSON",
    "der fuehrer": "PERSON",
    "the reich": "LOCATION",
    "das reich": "LOCATION",
    "the front": "EVENT",
    "die front": "EVENT",
    "the war": "EVENT",
    "der krieg": "EVENT",
    "the system": "INSTITUTION",
    "das system": "INSTITUTION",
}
