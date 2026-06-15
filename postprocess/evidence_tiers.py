# Canonical edge_source -> evidence-tier membership. SINGLE SOURCE OF TRUTH.
# The evaluator (tier filtering in evaluation/evaluate.py) and the codebook
# (postprocess/codebook.py) both import this so the documented tiers can't
# drift from what the pipeline actually stamps on edges.
#
# Trust ladder. Each wider tier admits one weaker class of evidence:
#   asserted  - stated in the source text, or copied from a verified record
#               (the metadata spreadsheet). The high-precision core.
#   inferred  - membership inferred from a detected textual signal (evidence
#               tier > 0). Not stated outright but evidence-backed.
#   assumption- a blanket prior with no per-edge evidence (mandatory NSDAP
#               membership of every autobiography author). Weaker than inferred.
#   proximity - mere co-presence in a shared context. Weakest layer, "not a
#               tie" (see tie_classes.NON_SOCIAL). It is the bulk of the edges
#               and floods precision.
# Both assumption and proximity live only in the widest tier.
#
# Emitters, for the record - keep this in sync when adding an edge source:
#   llm_extracted          intelligence/{api,ollama}_backend
#   langextract_extracted  intelligence/langextract_backend
#   rule_extracted         intelligence/relationship_patterns
#   metadata               domain/*/metadata + main.py metadata spec
#   canonical_inferred     domain/nazi_era canonical inference (signal detected)
#   pipeline_inferred      domain/nazi_era mandatory-membership assumption
#   rule_cooccurrence      postprocess/canonical_inference + python_only backend
#   sna_inferred           legacy co-occurrence tag (pre-2026-06 runs) -> proximity

from __future__ import annotations

ASSERTED = frozenset({
    "llm_extracted", "langextract_extracted", "rule_extracted", "metadata",
})
INFERRED = frozenset({"canonical_inferred"})
ASSUMPTION = frozenset({"pipeline_inferred"})
# sna_inferred is the old co-occurrence tag; recognised for filtering so runs
# made before the branding was unified still re-tier correctly, but kept out of
# the human-readable docs (current runs never emit it).
_LEGACY = frozenset({"sna_inferred"})
PROXIMITY = frozenset({"rule_cooccurrence"}) | _LEGACY

# Filterable tiers (cumulative). 'full'/'all' are deliberately absent: they
# admit *everything*, including unknown or future sources, and are handled as a
# short-circuit in tier_allows rather than a closed set.
TIER_SOURCES: dict[str, frozenset[str]] = {
    "conservative": ASSERTED,
    "moderate": ASSERTED | INFERRED,
}

# Human-readable tier table for the codebook (tier, sources it adds, why).
TIER_DOCS: list[tuple[str, str, str]] = [
    ("conservative", ", ".join(sorted(ASSERTED)),
     "Stated in the text or copied from a verified record. Precision-critical claims."),
    ("moderate", "+ " + ", ".join(sorted(INFERRED)),
     "Adds membership inferred from a detected textual signal."),
    ("full", "+ " + ", ".join(sorted((ASSUMPTION | PROXIMITY) - _LEGACY)),
     "Adds the mandatory-membership assumption and raw co-occurrence. Weakest layers; not asserted ties."),
]


def tier_allows(edge_source_field: str, tier: str) -> bool:
    """True if an edge with this ';'-joined edge_source field belongs in `tier`.

    'full'/'all' (and any unrecognised tier name) admit every edge, including
    unknown and legacy sources. An edge is kept if *any* of its sources is in
    the tier's allowed set (gephi_builder joins parallel edges with ';').
    """
    if tier in ("full", "all"):
        return True
    allowed = TIER_SOURCES.get(tier)
    if allowed is None:
        return True
    sources = {s for s in (edge_source_field or "").split(";") if s}
    return not sources.isdisjoint(allowed)
