# Acronym/short-form -> canonical name. Lowercased keys. Extend as a target
# corpus surfaces recurring orgs; kept short so it doesn't over-merge.

from __future__ import annotations

ALIASES: dict[str, str] = {
    "the koch network": "Koch Industries",
    "afp": "Americans for Prosperity",
    "alec": "American Legislative Exchange Council",
    "uschamber": "U.S. Chamber of Commerce",
    "us chamber": "U.S. Chamber of Commerce",
    "chamber of commerce": "U.S. Chamber of Commerce",
}

# Public figures so ubiquitous they form a symbolic-reference layer, separable
# from the lived network. Empty for now - populate per corpus.
REFERENCE_FIGURES: set[str] = set()
