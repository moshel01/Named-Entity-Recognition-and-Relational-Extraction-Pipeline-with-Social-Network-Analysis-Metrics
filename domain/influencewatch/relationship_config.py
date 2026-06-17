# Relation vocabulary for the InfluenceWatch domain. The analytically load-bearing
# layers are the money flow (who funds whom, with qual_monetary_value) and the
# affiliation structure (boards/memberships) the bipartite projection turns into a
# person<->person network. donated_to is kept distinct from the generic causal
# "contributed_to" on purpose - here a contribution is money, not causation.

from __future__ import annotations

# {canonical: [surface phrasings the LLM/parser emit]}. Constrains extraction and
# normalizes the verbose tail. co_occurs_with is listed so the aligner keeps it.
RELATION_ONTOLOGY: dict[str, list[str]] = {
    # money flows (carry qual_monetary_value when the amount is stated)
    "funded": ["financed", "bankrolled", "underwrote", "provided funding to",
               "gave money to", "seeded", "is a funder of"],
    "donated_to": ["donated to", "gave a donation to", "made a contribution to",
                   "contributed money to", "gave to the campaign of"],
    "granted": ["awarded a grant to", "gave a grant to", "made a grant to",
                "disbursed to"],
    # corporate / control structure
    "owns": ["owns", "is the parent of", "holds a stake in", "acquired"],
    "owned_by": ["owned by", "controlled by", "is a unit of", "managed by", "managed_by"],
    "subsidiary_of": ["subsidiary of", "division of", "affiliate of", "arm of",
                      "sister organization of"],
    "controls": ["controls", "operates", "manages", "runs", "oversees",
                 "is the parent of"],
    # fiscal-sponsorship / pop-up structure - the Arabella-style dark-money model
    # where one nonprofit hosts many "projects" that look independent.
    "fiscal_sponsor_of": ["fiscal sponsor of", "acts as a fiscal sponsor for",
                          "fiscally sponsors", "hosts the project"],
    "project_of": ["is a project of", "structured as a project of", "a project of",
                   "pop-up group of", "operated under"],
    # affiliation / governance (feed the two-mode -> one-mode projection)
    "board_member_of": ["sits on the board of", "board member of", "serves on the board of",
                        "trustee of", "is a board director of"],
    "director_of": ["director of", "executive director of", "officer of",
                    "general counsel of"],
    "chairs": ["chairs", "chairman of", "chairwoman of", "chair of"],
    "led": ["leads", "heads", "president of", "ceo of", "founder and ceo of"],
    "founded": ["founded", "co-founded", "established", "launched", "set up"],
    "employed_by": ["works for", "employed by", "is on staff at", "staffer at"],
    "member_of": ["member of", "belongs to", "is part of"],
    "affiliated_with": ["affiliated with", "associated with", "tied to", "linked to",
                        "aligned with"],
    # influence behavior
    "lobbied": ["lobbied", "lobbied on", "advocated to", "pressured", "petitioned"],
    "lobbied_for": ["lobbied on behalf of", "represents", "registered to lobby for"],
    "advised": ["advised", "is a consultant to", "counsel to", "adviser to"],
    "supported": ["backed", "endorsed", "supports", "promotes"],
    "opposed": ["opposed", "campaigned against", "targeted", "attacked"],
    "allied_with": ["allied with", "partnered with", "in coalition with",
                    "joined forces with"],
    # biographical / place
    "located_in": ["based in", "headquartered in", "located in", "operates out of"],
    "co_occurs_with": ["appears with", "mentioned with", "co occurs with"],
}

# One-line definitions for the pairs a model conflates. Short - they share the
# prompt budget with the passage.
RELATION_GUIDE: dict[str, str] = {
    "funded": "Sustained financial support from one org/person to another (a funder/grantee flow). Use donated_to for a political contribution to a campaign/candidate.",
    "donated_to": "A political/charitable contribution, typically to a candidate, campaign, or PAC. Use funded for sustained organizational financing.",
    "granted": "A foundation or agency awarding a defined grant. Use funded for general bankrolling.",
    "owns": "Holds ownership/parent stake in a company. Use controls when influence is operational, not equity.",
    "owned_by": "Is owned/controlled by the target (inverse of owns).",
    "subsidiary_of": "Is a formal sub-unit/affiliate/sister org of a larger org. Use owned_by for an equity parent.",
    "controls": "Operationally runs, manages, or oversees another org without necessarily owning it (shared staff, common control).",
    "fiscal_sponsor_of": "Source nonprofit provides legal/financial host status to the target group (the target operates under the source's tax status). The dark-money 'pop-up' pattern.",
    "project_of": "Target is the host; source is a project/pop-up group operating under it (inverse of fiscal_sponsor_of).",
    "board_member_of": "Holds a seat on the board/trustees of an org. Use director_of for a named officer role, employed_by for paid staff.",
    "director_of": "Holds a named officer/executive post (executive director, officer). Use board_member_of for a board seat.",
    "led": "Heads the organization (president/CEO/chief). Use founded for creating it.",
    "employed_by": "Paid staff of a person or org. Use board_member_of/director_of for governance roles.",
    "member_of": "Stated membership in an org/coalition, no governance role.",
    "affiliated_with": "A looser tie (associated/linked/aligned) when no membership or role is stated. Prefer a specific relation when one fits.",
    "lobbied": "Directly lobbied/pressured a target (agency, official, body) on an issue.",
    "lobbied_for": "Acts as a paid lobbyist/representative for a client. Use lobbied for the act of pressuring a target.",
    "advised": "Serves as adviser/consultant/counsel, not staff or board.",
    "supported": "Backed or endorsed without money or formal alliance. Use funded/donated_to for money, allied_with for a formal alliance.",
    "opposed": "Worked against a person/org/measure (no armed sense). The negative-polarity tie.",
    "allied_with": "A formal alliance/coalition/partnership between roughly equal parties.",
    "co_occurs_with": "Named together with no stated relationship - the weakest tie. Use only when nothing specific fits.",
}
