# Domain-specific relationship types and verb->relation mappings.

from __future__ import annotations

# Canonical relationship types this domain recognizes and their directedness.
RELATIONSHIP_TYPES: dict[str, bool] = {
    # membership / affiliation
    "member_of": True,
    "joined": True,
    "founded": True,
    "led": True,
    "commanded": True,
    "served_in": True,
    "promoted_to": True,
    "appointed_by": True,
    "subordinate_to": True,
    "expelled_from": True,
    # political / ideological
    "supported": True,
    "opposed": True,
    "allied_with": False,
    "recruited": True,
    "influenced_by": True,
    "propagandized_for": True,
    # combat / events
    "fought_in": True,
    "fought_against": True,
    "wounded_at": True,
    "participated_in": True,
    "imprisoned_by": True,
    # social
    "met_with": False,
    "family_of": False,
    "employed_by": True,
    "studied_at": True,
    "located_in": True,
}

# German/English verb lemma -> (relation_type, directed). Merged with the
# generic backend lexicon at runtime via inference_rules.
DOMAIN_VERB_LEXICON: dict[str, tuple[str, bool]] = {
    "join": ("joined", True),
    "joined": ("joined", True),
    "beitreten": ("joined", True),
    "eintreten": ("joined", True),
    "found": ("founded", True),
    "gründen": ("founded", True),
    "gruenden": ("founded", True),
    "lead": ("led", True),
    "führen": ("led", True),
    "fuehren": ("led", True),
    "command": ("commanded", True),
    "befehligen": ("commanded", True),
    "promote": ("promoted_to", True),
    "befördern": ("promoted_to", True),
    "befoerdern": ("promoted_to", True),
    "appoint": ("appointed_by", True),
    "ernennen": ("appointed_by", True),
    "serve": ("served_in", True),
    "dienen": ("served_in", True),
    "fight": ("fought_in", True),
    "kämpfen": ("fought_in", True),
    "kaempfen": ("fought_in", True),
    "wound": ("wounded_at", True),
    "verwunden": ("wounded_at", True),
    "recruit": ("recruited", True),
    "werben": ("recruited", True),
    "support": ("supported", True),
    "unterstützen": ("supported", True),
    "unterstuetzen": ("supported", True),
    "oppose": ("opposed", True),
    "bekämpfen": ("opposed", True),
    "bekaempfen": ("opposed", True),
    "imprison": ("imprisoned_by", True),
    "verhaften": ("imprisoned_by", True),
    "employ": ("employed_by", True),
    "study": ("studied_at", True),
    "studieren": ("studied_at", True),
}

# Relation types whose presence signals an ideological/affinity edge (consumed
# by the generic Tagger's connection_quality logic if extended).
IDEOLOGICAL_RELATIONS: set[str] = {
    "supported", "opposed", "allied_with", "propagandized_for",
    "influenced_by", "recruited",
}


# Canonical relation ontology {canonical: [synonyms / phrasings]}
# Used by the OntologyAligner to normalize raw extracted relation types, and to
# constrain LLM relation extraction. Synonyms include English + German phrasings
# the LLM / dependency parser commonly emit. Built to cover the structural edges
# the pipeline also produces (co_occurs_with, subordinate_to, member_of).
def _ontology() -> dict[str, list[str]]:
    onto: dict[str, list[str]] = {rel: [] for rel in RELATIONSHIP_TYPES}
    # Verb-lexicon surface forms become synonyms of their canonical relation.
    for verb, (canon, _directed) in DOMAIN_VERB_LEXICON.items():
        onto.setdefault(canon, [])
        onto[canon].append(verb)
    # Hand-written phrasings the LLM tends to produce.
    extra = {
        "member_of": ["joined", "belonged to", "was a member of", "mitglied von",
                      "trat bei", "was in"],
        "fought_in": ["fought for", "was a fighter for", "fought as", "kämpfte für",
                      "served at the front", "saw combat in"],
        "fought_against": ["fought against", "battled", "clashed with", "kämpfte gegen"],
        "led": ["was leader of", "headed", "commanded", "led the", "führte"],
        "founded": ["co-founded", "established", "set up", "gründete"],
        "promoted_to": ["was promoted to", "rose to", "rose to the rank of",
                        "wurde befördert zu"],
        "supported": ["backed", "endorsed", "sympathized with", "unterstützte"],
        "opposed": ["was against", "fought politically against", "lehnte ab"],
        "located_in": ["lived in", "based in", "from", "wohnte in", "stationed in"],
        "employed_by": ["worked for", "worked at", "was employed by", "arbeitete für"],
        "family_of": ["brother of", "father of", "son of", "married to", "related to",
                      "bruder von", "vater von", "verheiratet mit"],
        "met_with": ["met", "spoke with", "encountered", "traf"],
        "co_occurs_with": ["co occurs with", "appears with", "mentioned with"],
        "subordinate_to": ["reported to", "under the command of", "part of", "unterstellt"],
    }
    for canon, syns in extra.items():
        onto.setdefault(canon, [])
        onto[canon].extend(syns)
    # De-duplicate.
    return {k: sorted(set(v)) for k, v in onto.items()}


RELATION_ONTOLOGY: dict[str, list[str]] = _ontology()
