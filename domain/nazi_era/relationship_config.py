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


# One-line definitions shown to the LLM next to the allowed labels. Contrastive
# on the pairs qwen confuses: joined/member_of/served_in, led/commanded,
# opposed/fought_against, participated_in/fought_in, met_with/co_occurs_with.
# Keep them short - 27 labels share the prompt budget with the passage.
RELATION_GUIDE: dict[str, str] = {
    "joined": "Became a member of an org/party at a point in time (the act of joining). Prefer this over member_of when the text gives a date or describes entering.",
    "member_of": "Ongoing membership in an org/party/group, no joining moment stated. Use joined for the act itself.",
    "served_in": "Duty or service in a military unit or official body (not paid civilian work - that is employed_by).",
    "employed_by": "Paid civilian work for a person or company.",
    "led": "Headed or directed an organization or group (non-military). Use commanded for troops.",
    "commanded": "Held military command over a unit or formation.",
    "founded": "Created or established the organization.",
    "joined_into": "",
    "opposed": "Political or ideological opposition, no physical combat. Use fought_against for armed conflict.",
    "fought_against": "Armed or physical conflict against a person, group, or force.",
    "fought_in": "Took part in combat in a named war or battle (the event).",
    "participated_in": "Took part in a non-combat event: rally, putsch, election, march, meeting.",
    "supported": "Backed, endorsed, or aided - without a formal alliance (allied_with) or propaganda (propagandized_for).",
    "allied_with": "Formal alliance or cooperation between roughly equal parties.",
    "propagandized_for": "Produced or spread propaganda on behalf of a cause or org.",
    "influenced_by": "This person's views were shaped by another (directional: subject is influenced).",
    "met_with": "A specific meeting or encounter between people.",
    "co_occurs_with": "Mentioned together with no stated relationship - the weakest tie. Use only when nothing more specific fits.",
    "family_of": "Any kin relationship (parent, sibling, spouse, relative).",
    "located_in": "A person or org is situated in a place.",
    "studied_at": "Attended or was educated at a school, university, or institution.",
    "appointed_by": "Was appointed or named to a post by someone.",
    "promoted_to": "Advanced to a rank or office.",
    "recruited": "Brought someone into an org or cause.",
    "imprisoned_by": "Was jailed or detained by an authority.",
    "expelled_from": "Was thrown out of an org, place, or country.",
    "wounded_at": "Was injured at a place or in a battle.",
    "subordinate_to": "Reported to or ranked under another person.",
    "propaganda_for": "",
}
# Keep only labels that exist in the ontology (typo guard); drop empty stubs.
RELATION_GUIDE = {k: v for k, v in RELATION_GUIDE.items()
                  if v and k in RELATION_ONTOLOGY}
