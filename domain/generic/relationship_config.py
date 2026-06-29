# Default relation vocabulary for the no-knowledge generic domain. Without this the
# generic domain extracts FREE-FORM relations that never align to a stable label, so
# the typed-RE machinery (tie classes, type_violation gate, projection) sits inert and
# the graph is a bag of one-off edge strings. This is the general-purpose backstop:
# the common interpersonal / organizational / biographical / spatial / stance ties any
# narrative or article carries. Copy it into a real domain and specialize. The
# canonical names match the tie-class buckets in postprocess/tie_classes.py.

from __future__ import annotations

# {canonical: [surface phrasings the LLM/parser emit]}. Constrains the extraction
# prompt and folds the verbose tail back to one label. co_occurs_with is kept so the
# aligner doesn't drop the co-occurrence floor.
RELATION_ONTOLOGY: dict[str, list[str]] = {
    # interpersonal / social. Canonicals match the existing tie-class + polarity +
    # symmetric maps (knew/led/mentored/allied_with) so they inherit correct sign and
    # directedness instead of leaning on the free-form fallback.
    "knew": ["knew", "knows", "is acquainted with", "is associated with"],
    "friend_of": ["friend of", "befriended", "is friends with", "companion of"],
    "family_of": ["related to", "father of", "mother of", "son of", "daughter of",
                  "brother of", "sister of", "parent of", "child of", "relative of"],
    "married_to": ["married to", "wife of", "husband of", "spouse of", "wed"],
    "allied_with": ["allied with", "ally of", "partnered with", "sided with",
                    "joined forces with"],
    "rival_of": ["rival of", "enemy of", "feuded with", "at odds with"],
    "mentored": ["mentored", "tutored", "taught", "is the mentor of", "trained"],
    # interaction (a recorded contact, not a standing relationship)
    "met_with": ["met", "met with", "visited", "encountered"],
    "spoke_to": ["spoke to", "spoke with", "talked to", "told", "addressed"],
    "wrote_to": ["wrote to", "corresponded with", "sent a letter to", "messaged"],
    # organizational / affiliation (feed the bipartite projection)
    "member_of": ["member of", "belongs to", "is part of", "joined"],
    "led": ["leads", "led", "heads", "is the leader of", "president of", "directs", "commands"],
    "founded": ["founded", "co-founded", "established", "created", "set up", "started"],
    "employed_by": ["works for", "employed by", "is on staff at", "serves"],
    "works_with": ["works with", "collaborated with", "is a colleague of"],
    # stance (signed; polarity is derived downstream)
    "supported": ["supported", "backed", "endorsed", "praised", "promoted", "defended"],
    "opposed": ["opposed", "criticized", "denounced", "attacked", "condemned"],
    # conflict (substantive but not cooperative)
    "fought_against": ["fought against", "fought", "battled", "warred with", "clashed with"],
    "killed": ["killed", "murdered", "assassinated", "slew"],
    # biographical / spatial
    "born_in": ["born in", "was born in", "a native of"],
    "died_in": ["died in", "passed away in", "perished in"],
    "lived_in": ["lived in", "resided in", "settled in", "dwelt in"],
    "located_in": ["located in", "based in", "situated in", "found in"],
    "traveled_to": ["traveled to", "went to", "journeyed to", "moved to", "arrived in"],
    # creation / possession / participation
    "created": ["created", "wrote", "authored", "built", "made", "composed"],
    "owns": ["owns", "possesses", "holds", "controls"],
    "participated_in": ["participated in", "took part in", "attended", "fought in", "joined in"],
    # generic causal (driver -> impact; surfaced, kept out of the interpersonal set)
    "caused": ["caused", "led to", "resulted in", "triggered", "brought about"],
    # co-occurrence floor (never a tie; kept so the aligner preserves it)
    "co_occurs_with": ["appears with", "mentioned with", "co occurs with"],
}

# One-line definitions for the pairs a model conflates. Short - they share the prompt
# budget with the passage.
RELATION_GUIDE: dict[str, str] = {
    "knew": "A bare acquaintance with no closer tie stated. Prefer friend_of/allied_with/family_of when the text is more specific.",
    "friend_of": "A stated friendship/companionship. Use allied_with for a tactical or political alliance, knew for a bare acquaintance.",
    "family_of": "Any kinship tie (parent, child, sibling, cousin). Use married_to for spouses specifically.",
    "allied_with": "A cooperative/tactical alliance (partners, allies, joined forces). Use friend_of for personal friendship.",
    "rival_of": "A standing rivalry/enmity between two parties. Use opposed for a one-directional stance, fought_against for physical conflict.",
    "met_with": "A discrete meeting/encounter (an event), not a standing relationship.",
    "spoke_to": "A recorded act of speaking/telling/addressing. Use wrote_to for written contact.",
    "member_of": "Stated membership in a group/org/movement, no leadership role. Use led for heading it.",
    "led": "Heads or directs the org/group (leader, president, commander). Use founded for creating it, member_of for rank-and-file.",
    "founded": "Created/established the org, work, or movement. Use led for running an existing one.",
    "employed_by": "Paid or formal service to a person or org. Use member_of for unpaid membership.",
    "supported": "Backed/endorsed/praised the target (positive stance). The polarity gate reads this as positive.",
    "opposed": "Criticized/denounced/attacked the target in word or politics (negative stance, not physical combat). Use fought_against for combat.",
    "fought_against": "Physical/armed conflict against the target. Use opposed for a verbal or political stance.",
    "born_in": "Birthplace of a person. Use lived_in for residence, located_in for a thing's location.",
    "located_in": "Spatial containment of a place/org/thing in a larger place. Use born_in/lived_in for a person.",
    "created": "Authored/built/made a work or object. Use founded for an organization or movement.",
    "participated_in": "Took part in an event/action (a person-to-event tie). Feeds the participation projection.",
    "caused": "A driver-to-impact causal link (not interpersonal). Surfaced and filterable, excluded from the interpersonal centrality set.",
}
