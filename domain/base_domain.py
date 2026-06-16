# Abstract domain class and a dynamic loader.

from __future__ import annotations

import importlib
import logging
from abc import ABC
from typing import Any, Optional

from core.schema import Entity, Relationship

logger = logging.getLogger(__name__)


class BaseDomain(ABC):
    """Base interface for a domain knowledge plugin."""

    name: str = "base"

    # Aliases
    def aliases(self) -> dict[str, str]:
        """Return ``{alias_lower: canonical_name}`` mappings (may be empty)."""
        return {}

    def entity_label_overrides(self) -> dict[str, str]:
        """Return ``{name_lower: forced_label}`` overrides (may be empty)."""
        return {}

    def entity_stopwords(self) -> set[str]:
        """Lowercased entity names to drop as generic noise (may be empty)."""
        return set()

    def reference_figures(self) -> set[str]:
        """Lowercased canonical names of public/historical figures (may be empty).

        Tagged in the graph so analysts can separate the symbolic-reference
        network (everyone -> Hitler) from the lived interpersonal network.
        """
        return set()

    def entity_subtypes(self) -> dict[str, list[str]]:
        """Controlled subtype vocabulary per entity type for LLM enrichment."""
        return {}

    def temporal_period(self, year: int) -> str:
        """Map a year to a named period for temporal slicing, or "" if none."""
        return ""

    def narrative_rules(self):
        """Element-category keyword rules for the narrative-sequence network
        (list[(label, (keywords,))]), or None to use the generic default."""
        return None

    # Foundation tuning
    def gliner_labels(self) -> Optional[list[str]]:
        """Return domain GLiNER labels, or None to keep the config defaults."""
        return None

    def gliner_label_map(self) -> dict[str, str]:
        """Return ``{gliner_label_lower: CANONICAL_TYPE}`` overrides."""
        return {}

    def spacy_patterns(self) -> list[dict[str, Any]]:
        """Return spaCy EntityRuler patterns to merge into the pipeline."""
        return []

    def temporal_vocab(self) -> dict[str, dict[str, int]]:
        """Return non-English date vocabulary for the date extractor.

        Shape: ``{"months": {name_lower: 1..12}, "seasons": {name_lower: month}}``.
        Used to recognize and normalize dates the English parser would miss.
        """
        return {"months": {}, "seasons": {}}

    def relation_ontology(self) -> dict[str, list[str]]:
        """Return ``{canonical_relation: [synonyms...]}`` for alignment, or {}."""
        return {}

    def relation_guide(self) -> dict[str, str]:
        """Return ``{relation: one-line definition}`` shown to the LLM, or {}."""
        return {}

    def narrator_name(self, filename: str, doc_id: str) -> Optional[str]:
        """Author/narrator node name from a filename, or None for the default."""
        return None

    def load_metadata(self, path: str) -> dict[str, dict[str, Any]]:
        """Per-document metadata keyed by letter_id, or {} if unsupported."""
        return {}

    def metadata_edges(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        """Verified edges from a metadata row: [{target, type, rel, attrs}]."""
        return []

    # LLM prompt overrides
    def extraction_system_prompt(self) -> Optional[str]:
        """Return a domain system prompt for extraction, or None for default."""
        return None

    def quality_review_system_prompt(self) -> Optional[str]:
        """Return a domain system prompt for quality review, or None."""
        return None

    # Canonical inference
    def infer_canonical_edges(
        self, entities: list[Entity], edges: list[Relationship],
        options: Optional[dict[str, Any]] = None,
    ) -> list[Relationship]:
        """Return additional ``origin="canonical"`` edges. Default: none.

        ``options`` carries pipeline settings such as ``mandatory_membership``.
        """
        return []


class GenericDomain(BaseDomain):
    """Default no-knowledge domain assembled by reflection over a package.

    Works for any domain package that follows the standard module layout; the
    nazi_era domain reuses it directly. The loader sets ``name`` after init.
    """

    name = "generic"

    def __init__(self, package: str = "domain.generic") -> None:
        self.package = package
        self._aliases = _safe_attr(package, "aliases", "ALIASES", {})
        self._labels = _safe_attr(package, "entity_config", "LABEL_OVERRIDES", {})
        self._infer_fn = _safe_attr(package, "inference_rules", "infer_edges", None)
        # Foundation tuning (optional).
        self._gliner_labels = _safe_attr(package, "gliner_labels", "LABELS", None)
        self._gliner_map = _safe_attr(package, "gliner_labels", "LABEL_TO_TYPE_MAP", {})
        self._spacy_patterns = _safe_attr(package, "spacy_patterns", "PATTERNS", [])
        # Prompt overrides: probe known prompt module names.
        self._sys_extraction = None
        self._sys_quality = None
        for modname in ("domain_prompts", "prompts", "prompts_nazi_era"):
            if self._sys_extraction is None:
                self._sys_extraction = _safe_attr(package, modname, "SYSTEM_EXTRACTION", None)
            if self._sys_quality is None:
                self._sys_quality = _safe_attr(package, modname, "SYSTEM_QUALITY_REVIEW", None)

    def aliases(self) -> dict[str, str]:
        return {k.lower(): v for k, v in self._aliases.items()}

    def entity_label_overrides(self) -> dict[str, str]:
        return {k.lower(): v for k, v in self._labels.items()}

    def entity_stopwords(self) -> set[str]:
        sw = _safe_attr(self.package, "entity_config", "STOPWORDS", set())
        return {s.lower() for s in (sw or set())}

    def reference_figures(self) -> set[str]:
        figs = _safe_attr(self.package, "aliases", "REFERENCE_FIGURES", set())
        return {f.lower() for f in (figs or set())}

    def temporal_period(self, year: int) -> str:
        fn = _safe_attr(self.package, "historical_context", "temporal_period", None)
        return fn(year) if callable(fn) and year else ""

    def entity_subtypes(self) -> dict[str, list[str]]:
        sub = _safe_attr(self.package, "entity_config", "ENTITY_SUBTYPES", {})
        return {k: list(v or []) for k, v in (sub or {}).items()}

    def gliner_labels(self) -> Optional[list[str]]:
        return list(self._gliner_labels) if self._gliner_labels else None

    def gliner_label_map(self) -> dict[str, str]:
        return {k.lower(): v for k, v in (self._gliner_map or {}).items()}

    def spacy_patterns(self) -> list[dict[str, Any]]:
        return list(self._spacy_patterns or [])

    def extraction_system_prompt(self) -> Optional[str]:
        return self._sys_extraction

    def quality_review_system_prompt(self) -> Optional[str]:
        return self._sys_quality

    def temporal_vocab(self) -> dict[str, Any]:
        months = _safe_attr(self.package, "historical_context", "GERMAN_MONTHS", {})
        seasons = _safe_attr(self.package, "historical_context", "GERMAN_SEASONS", {})
        pivot = _safe_attr(self.package, "historical_context", "PERIOD_END", None)
        return {"months": dict(months or {}), "seasons": dict(seasons or {}), "pivot_max": pivot}

    def relation_ontology(self) -> dict[str, list[str]]:
        onto = _safe_attr(self.package, "relationship_config", "RELATION_ONTOLOGY", None)
        if onto is None:
            onto = _safe_attr(self.package, "relation_config", "RELATION_ONTOLOGY", {})
        return {k: list(v or []) for k, v in (onto or {}).items()}

    def relation_guide(self) -> dict[str, str]:
        guide = _safe_attr(self.package, "relationship_config", "RELATION_GUIDE", None)
        if guide is None:
            guide = _safe_attr(self.package, "relation_config", "RELATION_GUIDE", {})
        return {str(k): str(v) for k, v in (guide or {}).items() if v}

    def narrator_name(self, filename: str, doc_id: str) -> Optional[str]:
        fn = _safe_attr(self.package, "german_nlp", "author_from_filename", None)
        return fn(filename) if callable(fn) else None

    def load_metadata(self, path: str) -> dict[str, dict[str, Any]]:
        fn = _safe_attr(self.package, "metadata", "load_metadata", None)
        return fn(path) if callable(fn) else {}

    def metadata_edges(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        fn = _safe_attr(self.package, "metadata", "metadata_edges", None)
        return fn(row) if callable(fn) else []

    def infer_canonical_edges(
        self, entities: list[Entity], edges: list[Relationship],
        options: Optional[dict[str, Any]] = None,
    ) -> list[Relationship]:
        if not callable(self._infer_fn):
            return []
        # Support both new (entities, edges, options) and legacy (entities, edges).
        try:
            return self._infer_fn(entities, edges, options)
        except TypeError:
            return self._infer_fn(entities, edges)


def _safe_attr(package: str, module: str, attr: str, default: Any) -> Any:
    """Import ``package.module`` and return ``attr`` or a default."""
    try:
        mod = importlib.import_module(f"{package}.{module}")
        return getattr(mod, attr, default)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Domain attr %s.%s.%s unavailable: %s", package, module, attr, exc)
        return default


def load_domain(name: str) -> BaseDomain:
    """Load a domain plugin by folder name under ``domain/``.

    Falls back to :class:`GenericDomain` if the named package is missing.
    """
    package = f"domain.{name}"
    try:
        importlib.import_module(package)
    except Exception:  # noqa: BLE001
        logger.warning("Domain '%s' not found; using generic domain.", name)
        package = "domain.generic"
    domain = GenericDomain(package)
    domain.name = name
    return domain
