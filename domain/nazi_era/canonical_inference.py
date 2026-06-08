# Evidence-based membership inference for the Nazi-era domain.

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from core.schema import Entity, Relationship, stable_id

from .rank_systems import identify_rank_org

# Tier -> confidence assigned to a canonical_inferred edge.
_TIER_CONFIDENCE = {1: 0.85, 2: 0.78, 3: 0.68, 4: 0.50}
_MANDATORY_CONFIDENCE = 0.95


@dataclass
class OrgEvidence:
    """Evidence configuration for one organization."""

    canonical: str
    entity_keywords: list[str]
    text_patterns: list[str]
    rank_keywords: list[str]
    context_keywords: list[str]
    mandatory: bool = False

    def compiled(self) -> list[re.Pattern]:
        return [re.compile(p, re.IGNORECASE) for p in self.text_patterns]


# Organization evidence definitions
ORG_EVIDENCE: list[OrgEvidence] = [
    OrgEvidence(
        canonical="NSDAP",
        entity_keywords=["nsdap", "national socialist", "nationalsozialist",
                         "nazi party", "die partei", "the party", "movement",
                         "bewegung", "hitler party", "hitlerpartei"],
        text_patterns=[
            r"\bNSDAP[-\s]?(?:Nr\.?|Nummer|No\.?)\s*\d{2,7}",
            r"\b(?:joined|trat .* bei|eintritt in die) .*\b(?:NSDAP|Partei|Bewegung)\b",
            r"\bParteigenosse\b", r"\bPg\.\s",
            r"\bMitglied der (?:NSDAP|Partei|Bewegung)\b",
        ],
        rank_keywords=["ortsgruppenleiter", "kreisleiter", "gauleiter",
                       "reichsleiter", "blockleiter", "zellenleiter"],
        context_keywords=["party member", "parteimitglied", "old fighter",
                          "alter kämpfer", "alter kaempfer"],
        mandatory=True,
    ),
    OrgEvidence(
        canonical="SA (Sturmabteilung)",
        entity_keywords=["sa", "sturmabteilung", "storm trooper", "stormtrooper",
                         "brownshirt", "brown shirt", "braunhemd", "sturmtrupp"],
        text_patterns=[
            r"\bSA[-\s]?(?:Sturm|Sturmbann|Standarte|Brigade|Gruppe)\b",
            r"\bin die SA\b", r"\bjoined the SA\b", r"\bSA-Mann\b",
            r"\bDienst in der SA\b",
        ],
        rank_keywords=["sa-sturmführer", "sa-sturmfuehrer", "sa-truppführer",
                       "sa-scharführer", "sa-standartenführer", "sturmführer",
                       "rottenführer", "sa-mann"],
        context_keywords=["marched with the sa", "sa duty", "sa-dienst",
                          "saal protection", "saalschutz", "ordnertruppe"],
    ),
    OrgEvidence(
        canonical="SS (Schutzstaffel)",
        entity_keywords=["ss", "schutzstaffel", "blackshirt", "black shirt",
                         "allgemeine ss", "waffen-ss"],
        text_patterns=[
            r"\bSS[-\s]?(?:Nr\.?|Nummer)\s*\d{1,7}",
            r"\bSS[-\s]?(?:Sturm|Sturmbann|Standarte|Abschnitt)\b",
            r"\bin die SS\b", r"\bjoined the SS\b", r"\bSS-Mann\b",
        ],
        rank_keywords=["ss-sturmführer", "ss-sturmfuehrer", "ss-untersturmführer",
                       "ss-obersturmführer", "ss-hauptsturmführer",
                       "ss-standartenführer", "reichsführer-ss", "ss-mann"],
        context_keywords=["ss duty", "ss-dienst", "elite guard", "leibstandarte"],
    ),
    OrgEvidence(
        canonical="Freikorps",
        entity_keywords=["freikorps", "free corps", "freecorps"],
        text_patterns=[
            r"\bFreikorps\b", r"\bin ein Freikorps\b", r"\bFreikorps[-\s]?\w+\b",
        ],
        rank_keywords=[],
        context_keywords=["baltic campaign", "baltikum", "border fighting",
                          "grenzschutz", "ruhr fighting", "kapp"],
    ),
    OrgEvidence(
        canonical="Stahlhelm",
        entity_keywords=["stahlhelm", "steel helmet", "frontsoldaten"],
        text_patterns=[r"\bStahlhelm\b", r"\bBund der Frontsoldaten\b"],
        rank_keywords=[],
        context_keywords=["veterans league", "frontsoldat", "front soldier"],
    ),
    OrgEvidence(
        canonical="Hitler Youth (Hitlerjugend)",
        entity_keywords=["hitler youth", "hitlerjugend", "hitler-jugend", "hj"],
        text_patterns=[r"\bHitlerjugend\b", r"\bHitler[-\s]?Jugend\b",
                       r"\bin die HJ\b"],
        rank_keywords=["hj-führer", "hj-fuehrer", "bannführer", "gefolgschaftsführer"],
        context_keywords=["jungvolk", "youth group", "jugendgruppe"],
    ),
    OrgEvidence(
        canonical="NSKK",
        entity_keywords=["nskk", "kraftfahrkorps", "motor corps"],
        text_patterns=[r"\bNSKK\b", r"\bKraftfahrkorps\b"],
        rank_keywords=["nskk-sturmführer", "nskk-staffelführer"],
        context_keywords=["motor unit", "motorstaffel", "driver corps"],
    ),
    OrgEvidence(
        canonical="Reichswehr",
        entity_keywords=["reichswehr", "100,000 man army", "hunderttausend-mann-heer"],
        text_patterns=[r"\bReichswehr\b", r"\bin die Reichswehr\b"],
        rank_keywords=[],
        context_keywords=["regular army", "berufssoldat", "professional soldier",
                          "schwarze reichswehr", "black reichswehr"],
    ),
]


class MembershipInferenceEngine:
    """Infer organizational-membership edges from textual evidence."""

    def __init__(self) -> None:
        self._compiled = {oe.canonical: oe.compiled() for oe in ORG_EVIDENCE}

    # Evidence assembly
    @staticmethod
    def _person_evidence(
        entities: list[Entity], edges: list[Relationship]
    ) -> dict[str, str]:
        """Build a per-person evidence string from names, aliases, and edges."""
        id_to_entity = {e.entity_id: e for e in entities}
        buf: dict[str, list[str]] = defaultdict(list)
        for e in entities:
            if e.label == "PERSON":
                buf[e.entity_id].append(e.canonical_name)
                buf[e.entity_id].extend(e.aliases)
                rank = e.attributes.get("rank")
                if rank:
                    buf[e.entity_id].append(str(rank))
        for r in edges:
            if not r.evidence:
                continue
            for endpoint in (r.source, r.target):
                ent = id_to_entity.get(endpoint)
                if ent is not None and ent.label == "PERSON":
                    buf[endpoint].append(r.evidence)
                    # The other endpoint's name is also evidence of association.
                    other = r.target if endpoint == r.source else r.source
                    oent = id_to_entity.get(other)
                    if oent is not None:
                        buf[endpoint].append(oent.canonical_name)
        return {pid: " \n ".join(parts).lower() for pid, parts in buf.items()}

    # Org node resolution
    @staticmethod
    def _org_node_index(entities: list[Entity]) -> dict[str, str]:
        """Map a canonical org name (lower) -> entity_id, if it exists as a node."""
        idx: dict[str, str] = {}
        for e in entities:
            if e.label in ("ORG", "INSTITUTION"):
                idx[e.canonical_name.lower()] = e.entity_id
                for a in e.aliases:
                    idx.setdefault(a.lower(), e.entity_id)
        return idx

    # Tiered detection
    def _detect(self, evidence: str, oe: OrgEvidence) -> tuple[int, str] | None:
        """Return (tier, matched_signal) for the strongest evidence, or None."""
        # Tier 1: org-specific rank.
        for rk in oe.rank_keywords:
            if rk in evidence:
                return 1, rk
        # Tier 2: text patterns.
        for pat in self._compiled[oe.canonical]:
            m = pat.search(evidence)
            if m:
                return 2, m.group(0).strip()
        # Tier 3: explicit org mention (word-boundary to avoid 'ss' in 'class').
        for kw in oe.entity_keywords:
            if re.search(rf"\b{re.escape(kw)}\b", evidence):
                return 3, kw
        # Tier 4: contextual cues.
        for kw in oe.context_keywords:
            if kw in evidence:
                return 4, kw
        return None

    # Main entry point
    def infer(
        self,
        entities: list[Entity],
        edges: list[Relationship],
        mandatory_scope: str = "authors_only",
    ) -> list[Relationship]:
        """Return new membership edges (origin='canonical')."""
        person_evidence = self._person_evidence(entities, edges)
        org_index = self._org_node_index(entities)

        # Avoid duplicating an already-extracted membership edge.
        existing: set[tuple[str, str]] = {
            (r.source, r.target) for r in edges if r.rel_type in
            ("member_of", "joined", "served_in")
        }

        new_edges: list[Relationship] = []
        persons = [e for e in entities if e.label == "PERSON"]

        for oe in ORG_EVIDENCE:
            org_id = org_index.get(oe.canonical.lower())
            if org_id is None:
                continue  # org not represented as a node; cannot link
            for person in persons:
                if person.entity_id == org_id:
                    continue
                if (person.entity_id, org_id) in existing:
                    continue
                evidence = person_evidence.get(person.entity_id, "")

                mandatory_here = oe.mandatory and self._mandatory_applies(
                    person, mandatory_scope
                )
                if mandatory_here:
                    scope_note = ("document author" if mandatory_scope == "authors_only"
                                  else "corpus member")
                    tier, signal, source, conf = (
                        0, f"corpus assumption: {scope_note} of NSDAP autobiography",
                        "pipeline_inferred", _MANDATORY_CONFIDENCE,
                    )
                else:
                    detected = self._detect(evidence, oe)
                    if detected is None:
                        continue
                    tier, signal = detected
                    source, conf = "canonical_inferred", _TIER_CONFIDENCE[tier]

                existing.add((person.entity_id, org_id))
                new_edges.append(
                    Relationship(
                        source=person.entity_id,
                        target=org_id,
                        rel_type="member_of",
                        doc_id=";".join(person.doc_ids),
                        evidence=signal,
                        confidence=conf,
                        directed=True,
                        origin="canonical",
                        attributes={
                            "edge_source": source,
                            "evidence_tier": tier,
                            "inferred_org": oe.canonical,
                        },
                    )
                )
        return new_edges

    @staticmethod
    def _mandatory_applies(person: Entity, scope: str) -> bool:
        """Whether the mandatory-membership assumption applies to ``person``."""
        if scope == "off":
            return False
        if scope == "all":
            return True
        # "authors_only": entity must be flagged as a document author/narrator.
        return bool(person.attributes.get("is_author") or person.tags.get("is_author"))
