# Relation vocabulary for OREM/OPAL disaster response. The network of interest is
# inter-organizational: who coordinates with whom, who responded to which event,
# and how funding/resources flow. coordinated_with is symmetric (agency<->agency).
# Jurisdiction/location scope is carried on the edge as qual_jurisdiction, not as a
# relation. Money flows carry qual_monetary_value.

from __future__ import annotations

RELATION_ONTOLOGY: dict[str, list[str]] = {
    # inter-agency coordination (symmetric)
    "coordinated_with": ["coordinated with", "worked with", "collaborated with",
                         "partnered with", "in a joint operation with", "co-located with"],
    "supported": ["supported", "assisted", "backed up", "augmented", "reinforced"],
    "provided_resources_to": ["provided resources to", "supplied", "delivered aid to",
                              "sheltered", "fed", "staffed", "provided personnel to"],
    "contracted": ["contracted with", "hired", "subcontracted to", "tasked"],
    # event response (actor -> disaster/program event)
    "responded_to": ["responded to", "deployed to", "mobilized for", "activated for",
                     "was activated for", "stood up for"],
    "participated_in": ["participated in", "took part in", "was involved in",
                        "joined the response to"],
    "managed": ["managed", "administered", "oversaw", "directed the response to",
                "was incident command for"],
    # funding flows (carry qual_monetary_value)
    "funded": ["funded", "financed", "provided funding for", "appropriated for"],
    "granted": ["awarded a grant to", "allocated funds to", "disbursed to",
                "passed through funds to", "sub-granted to"],
    # structure / authority
    "part_of": ["part of", "division of", "unit of", "under", "a program of"],
    "member_of": ["member of", "a participating agency in", "belongs to",
                  "a partner in"],
    "led": ["leads", "heads", "directs", "is the lead agency for"],
    "reports_to": ["reports to", "under the authority of", "answerable to",
                   "accountable to"],
    "operates_in": ["operates in", "serves", "covers", "has jurisdiction over",
                    "is responsible for", "responds in"],
    "located_in": ["located in", "based in", "headquartered in", "sited in"],
    "co_occurs_with": ["appears with", "mentioned with", "co occurs with"],
}

RELATION_GUIDE: dict[str, str] = {
    "coordinated_with": "Two agencies/orgs working jointly on a response (mutual, no lead/subordinate). Use supported when one aids the other, managed when one runs the response.",
    "supported": "One actor assists/augments another's effort (not an equal joint operation). Use provided_resources_to for concrete supplies/personnel.",
    "provided_resources_to": "Delivered concrete resources - supplies, staff, sheltering, food - to an actor or population. Use funded/granted for money.",
    "contracted": "Engaged another party under contract/task order for services.",
    "responded_to": "An agency/org mobilized or deployed for a specific disaster or emergency event. Use participated_in for a looser involvement.",
    "participated_in": "Took part in a response/operation/program without leading it.",
    "managed": "Ran or had incident command/administrative control of a response or program.",
    "funded": "Provided general funding for a program or response (carries the amount in qual_monetary_value).",
    "granted": "Awarded/allocated/passed through a defined grant. Use funded for general financing.",
    "part_of": "Is a formal sub-unit/program of a larger body. Use member_of for a partnership/coalition seat.",
    "member_of": "Is a participating agency/partner in a coalition, task force, or program.",
    "led": "Is the lead/coordinating agency. Use managed for incident-level command.",
    "reports_to": "Sits under another body's authority in the response hierarchy.",
    "operates_in": "Has jurisdiction over or serves a place/population (the geographic scope). Use located_in for where the org physically sits.",
    "located_in": "Physical seat/headquarters of the org. Use operates_in for its service jurisdiction.",
    "co_occurs_with": "Named together with no stated relationship - the weakest tie.",
}
