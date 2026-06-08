# Historical validation rules for extracted entities, edges, and timeline.

from __future__ import annotations

from dataclasses import dataclass, field

from core.schema import Entity, Relationship, TimelineEvent

from .historical_context import (
    ORGANIZATION_EXISTENCE,
    validate_date_range,
)
from .rank_systems import identify_rank_org


@dataclass
class ValidationFinding:
    """A single validation warning."""

    item_id: str
    kind: str               # "anachronism" | "rank_mismatch" | "implausible_date"
    message: str
    severity: str = "warning"   # "warning" | "error"


@dataclass
class ValidationReport:
    """Collected validation findings."""

    findings: list[ValidationFinding] = field(default_factory=list)

    def add(self, item_id: str, kind: str, message: str, severity: str = "warning") -> None:
        self.findings.append(ValidationFinding(item_id, kind, message, severity))

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error")

    def to_rows(self) -> list[dict]:
        return [
            {"item_id": f.item_id, "kind": f.kind,
             "message": f.message, "severity": f.severity}
            for f in self.findings
        ]


def _canonical_org_in(text: str) -> str | None:
    """Return the first known organization whose name appears in ``text``."""
    low = text.lower()
    for org in ORGANIZATION_EXISTENCE:
        if org.lower() in low:
            return org
    return None


def validate_timeline(events: list[TimelineEvent]) -> ValidationReport:
    """Flag timeline events with implausible or anachronistic dates."""
    report = ValidationReport()
    for ev in events:
        if ev.year is None:
            continue
        org = _canonical_org_in(ev.description)
        ok, reason = validate_date_range(ev.year, org)
        if not ok:
            kind = "anachronism" if org else "implausible_date"
            report.add(f"{ev.doc_id}:{ev.date_text}", kind, reason)
    return report


def validate_rank_consistency(
    entities: list[Entity], relationships: list[Relationship]
) -> ValidationReport:
    """Flag persons whose stated rank's organization is not reflected in edges.

    A person tagged with, say, an SS rank in their attributes but never linked
    to the SS is flagged for review (possible missed membership edge).
    """
    report = ValidationReport()
    id_to_entity = {e.entity_id: e for e in entities}

    # Map person -> set of org names they are linked to.
    person_orgs: dict[str, set[str]] = {e.entity_id: set() for e in entities
                                        if e.label == "PERSON"}
    for r in relationships:
        if r.source in person_orgs and r.target in id_to_entity:
            person_orgs[r.source].add(id_to_entity[r.target].canonical_name)
        if r.target in person_orgs and r.source in id_to_entity:
            person_orgs[r.target].add(id_to_entity[r.source].canonical_name)

    for e in entities:
        if e.label != "PERSON":
            continue
        rank = e.attributes.get("rank")
        if not rank:
            continue
        res = identify_rank_org(rank)
        if not res:
            continue
        org, _canon, _level = res
        if org in ("SA", "SS"):
            org_name = "SA (Sturmabteilung)" if org == "SA" else "SS (Schutzstaffel)"
            linked = person_orgs.get(e.entity_id, set())
            if not any(org_name.split(" ")[0] in ln for ln in linked):
                report.add(
                    e.entity_id, "rank_mismatch",
                    f"{e.canonical_name} holds rank '{rank}' ({org}) but is not "
                    f"linked to {org_name}.",
                )
    return report


def validate_all(
    entities: list[Entity],
    relationships: list[Relationship],
    timeline: list[TimelineEvent],
) -> ValidationReport:
    """Run every validation check and merge the findings."""
    report = ValidationReport()
    report.findings.extend(validate_timeline(timeline).findings)
    report.findings.extend(validate_rank_consistency(entities, relationships).findings)
    return report
