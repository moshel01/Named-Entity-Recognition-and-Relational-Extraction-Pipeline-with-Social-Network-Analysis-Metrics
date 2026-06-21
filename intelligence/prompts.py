# Prompt templates shared by the API (Mode 1) and Ollama (Mode 3) backends.

from __future__ import annotations

import json
from typing import Any

from core.schema import EntityMention

EXTRACTION_SYSTEM = """You are a precise information-extraction engine for \
social network analysis. You read a passage of text and return STRICT JSON \
describing the entities, the relationships between them, and any dated events.

Rules:
- Use ONLY information stated or directly implied by the passage. Never invent.
- Prefer the entity candidate names provided, but you may correct obvious \
errors, merge fragments, or add entities the candidate list missed.
- Relationship `type` must be a short lowercase verb phrase \
(e.g. "works_for", "member_of", "met_with", "related_to", "located_in").
- Set `directed` true only when the relation is inherently asymmetric \
(e.g. "works_for", "leads"); use false for symmetric relations (e.g. "met_with").
- `evidence` must be a verbatim sentence or short span from the passage.
- Keep entity names and evidence in the language of the passage; never \
translate them.
- `confidence` is a float in [0,1] reflecting how strongly the text supports it.
- Output a SINGLE JSON object and NOTHING else. No markdown, no commentary.

NEVER do these (negative examples):
- entity "he" / "the company" / "my father" -> WRONG: pronouns and bare common \
nouns are not entities; use the named referent or drop it.
- type "provided_grant_to_support_the_education_program" -> WRONG: the type is a \
short verb phrase ("funded"), never a sentence.
- a relationship the passage does not state -> WRONG: if it is not in the text, \
omit it; do not infer from world knowledge.
- evidence translated to English when the passage is German -> WRONG: keep names \
and evidence verbatim in the source language.
- ```json ... ``` or "Here is the JSON:" -> WRONG: emit the raw object only."""

_OUTPUT_SCHEMA = {
    "entities": [
        {"name": "string", "type": "PERSON|ORG|LOCATION|EVENT",
         "aliases": ["string"], "confidence": 0.0}
    ],
    "relationships": [
        {"source": "string", "target": "string", "type": "string",
         "directed": False, "evidence": "string", "confidence": 0.0}
    ],
    "timeline": [
        {"date": "string", "description": "string",
         "entities": ["string"], "confidence": 0.0}
    ],
}


def relationship_schema_str(edge_qualifiers: list[str] | None = None) -> str:
    """Output schema as a JSON string. Qualifier fields go IN the relationship
    example, not just the prose - the model copies the example literally and drops
    fields it doesn't see there (monetary_value left in the evidence, never lifted)."""
    schema = _OUTPUT_SCHEMA
    if edge_qualifiers:
        rel_ex = dict(_OUTPUT_SCHEMA["relationships"][0])
        for q in edge_qualifiers:
            rel_ex[q] = f"<{q}, only if stated>"
        schema = {**_OUTPUT_SCHEMA, "relationships": [rel_ex]}
    return json.dumps(schema, indent=2)


def relation_constraint_block(relation_types: list[str] | None,
                              relation_guide: dict[str, str] | None = None,
                              type_signatures: dict[str, str] | None = None) -> str:
    """The ALLOWED RELATION TYPES block (with guide defs + type-signature hints)."""
    if not relation_types:
        return ""
    guide = relation_guide or {}
    sigs = type_signatures or {}
    if guide or sigs:
        # Definitions pin a coding scheme the model would otherwise overrule with
        # its intuition (the classic "associate" labeled "friend"); the (type->type)
        # hint pins argument types so it forms fewer type-violating edges.
        lines = []
        for rt in relation_types:
            d = guide.get(rt)
            sig = sigs.get(rt)
            head = f"{rt} ({sig})" if sig else rt
            lines.append(f"  - {head}: {d}" if d else f"  - {head}")
        return (
            "\nALLOWED RELATION TYPES - set each relationship `type` to the "
            "single closest value below. The definitions are deliberate; "
            "follow them over your own intuition. A (type->type) hint is the "
            "allowed endpoint kinds. Use \"other\" only if none fit:\n"
            + "\n".join(lines) + "\n"
        )
    return (
        "\nALLOWED RELATION TYPES - set each relationship `type` to the "
        "single closest value from this list (use \"other\" only if none fit):\n  "
        + ", ".join(relation_types) + "\n"
    )


def qualifier_constraint_block(edge_qualifiers: list[str] | None) -> str:
    """The EDGE QUALIFIERS instruction (pairs with the schema-injected slots)."""
    if not edge_qualifiers:
        return ""
    return (
        "\nEDGE QUALIFIERS - add these as extra keys ON the relationship object "
        "(shown in the schema) whenever its evidence sentence states them; omit a "
        "field the text doesn't give, never guess: " + ", ".join(edge_qualifiers)
        + ". If the evidence names a dollar amount, date, or place, it belongs in "
        "the matching qualifier, not just left in the evidence string.\n"
    )


def build_extraction_prompt(
    text: str,
    candidates: list[EntityMention],
    label_types: list[str],
    relation_types: list[str] | None = None,
    author_name: str = "",
    relation_guide: dict[str, str] | None = None,
    edge_qualifiers: list[str] | None = None,
    type_signatures: dict[str, str] | None = None,
) -> str:
    """Construct the user prompt for relationship/entity extraction."""
    # Compact, de-duplicated candidate list to keep the prompt small.
    seen: set[tuple[str, str]] = set()
    cand_lines: list[str] = []
    for m in candidates:
        key = (m.text.strip(), m.label)
        if key in seen or not m.text.strip():
            continue
        seen.add(key)
        cand_lines.append(f'  - "{m.text.strip()}" [{m.label}]')
    cand_block = "\n".join(cand_lines) if cand_lines else "  (none detected)"

    schema_str = relationship_schema_str(edge_qualifiers)

    narrator = ""
    if author_name:
        narrator = (
            f"\nNARRATOR: this is a first-person account by \"{author_name}\". Use "
            f"\"{author_name}\" as the entity for every first-person reference "
            "(ich, mir, mein, wir); never output a pronoun as an entity.\n"
        )

    rel_constraint = relation_constraint_block(relation_types, relation_guide, type_signatures)
    qual_constraint = qualifier_constraint_block(edge_qualifiers)

    return f"""ENTITY CANDIDATES (from a base NER model - confirm, refine, extend):
{cand_block}

ENTITY TYPES IN USE: {", ".join(label_types)}
{narrator}{rel_constraint}{qual_constraint}
PASSAGE:
\"\"\"
{text}
\"\"\"

Return JSON exactly matching this schema (values illustrative):
{schema_str}
"""


def build_quality_review_prompt(entities_summary: str, edges_summary: str) -> tuple[str, str]:
    """Build (system, user) prompts for LLM-based quality review.

    The model returns JSON listing entity names and edge keys to drop, with
    reasons, plus optional alias merges it is confident about.
    """
    system = """You are a data-quality reviewer for an extracted social \
network. You are given aggregated entities and edges. Identify items that are \
clearly spurious: NER noise, generic terms mislabeled as entities, duplicate \
surface forms, or edges with no plausible basis. Be conservative - only flag \
items you are confident are wrong. Return STRICT JSON, nothing else."""

    user = f"""ENTITIES (name | type | mentions):
{entities_summary}

EDGES (source -- type --> target | weight):
{edges_summary}

Return JSON of this shape:
{{
  "drop_entities": ["exact entity name", ...],
  "drop_edges": ["source||type||target", ...],
  "merge_aliases": [{{"canonical": "name", "aliases": ["other", ...]}}]
}}
"""
    return system, user


def coerce_extraction(obj: Any) -> dict[str, list]:
    """Normalize a parsed LLM object into the expected dict-of-lists shape."""
    if not isinstance(obj, dict):
        return {"entities": [], "relationships": [], "timeline": []}
    return {
        "entities": obj.get("entities") or [],
        "relationships": obj.get("relationships") or obj.get("relations") or [],
        "timeline": obj.get("timeline") or obj.get("events") or [],
    }


# Enrichment: subtype / rank / attributes for resolved entities.
ENRICHMENT_SYSTEM = (
    "You enrich already-extracted entities. For each, pick the single best subtype "
    "from the allowed list shown next to it - a specific sub-category, NEVER the "
    "generic type itself. Also return any attributes (rank, office, role, "
    "affiliation) that are stated or directly implied. Never invent facts. Return "
    "STRICT JSON, nothing else."
)


def build_enrichment_prompt(rows: list[dict]) -> str:
    """rows: [{name, type, allowed?}]. Ask for subtype + attributes per name."""
    lines = []
    for r in rows:
        allowed = r.get("allowed") or []
        hint = f"  choose from: {', '.join(allowed)}" if allowed else ""
        lines.append(f'  - "{r["name"]}" [{r["type"]}]{hint}')
    return (
        "ENTITIES (subtype must come from each entity's allowed list):\n"
        + "\n".join(lines) + "\n\n"
        "Return JSON:\n"
        '{ "entities": [ {"name": "<exact name above>", "subtype": "<one allowed value>", '
        '"attributes": {"rank": "", "office": "", "role": "", "affiliation": ""}} ] }\n'
        "Use the entity's exact name. Omit attribute keys you cannot fill. Omit the "
        "subtype if none of the allowed values fit. Include only entities you can enrich."
    )


# LLM-assisted dedup: propose same-entity merge groups the rules missed.
MERGE_SYSTEM = (
    "You resolve duplicate entities in a social network. Given a list of entities "
    "of one type, group those that refer to the SAME real entity (spelling "
    "variants, abbreviations, name forms). Do NOT group distinct entities "
    "(different people, different organizations, different places). Be "
    "conservative. Return STRICT JSON, nothing else."
)


def build_merge_prompt(entity_type: str, names: list[str]) -> str:
    block = "\n".join(f"  - {n}" for n in names)
    return (
        f"ENTITY TYPE: {entity_type}\nENTITIES:\n{block}\n\n"
        "Return JSON:\n"
        '{ "groups": [ {"canonical": "<best full name>", "aliases": ["<other>", ...]} ] }\n'
        "Only include groups with at least one alias. Use names exactly as listed."
    )
