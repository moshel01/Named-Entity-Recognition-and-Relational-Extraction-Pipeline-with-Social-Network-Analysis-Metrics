# Re-DocRED adapter (tonytan48/Re-DocRED).

from __future__ import annotations

import re
from collections import Counter

from .common import BenchDoc, BenchEntity, BenchRelation

HF_ID = "tonytan48/Re-DocRED"

# The substantive SNA types to score by default (NUM/MISC/DATE are noisy in
# DocRED and GLiNER is weak on them - they drag the headline down). Override
# with --types on the runner.
DEFAULT_TARGET_TYPES = ["PERSON", "ORG", "LOCATION"]

# Re-DocRED entity type -> canonical pipeline type.
TYPE_MAP = {
    "PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION",
    "TIME": "DATE", "NUM": "NUM", "MISC": "MISC",
}

# GLiNER labels + label_map the pipeline should use for this dataset.
GLINER_LABELS = ["person", "organization", "location", "date", "number", "miscellaneous"]
LABEL_MAP = {
    "person": "PERSON", "organization": "ORG", "location": "LOCATION",
    "date": "DATE", "number": "NUM", "miscellaneous": "MISC",
}

# Wikidata property id -> readable label (DocRED rel_info.json, snake_cased).
# Gold relations carry these names instead of opaque Pxxx codes, so
# --constrain-relations gives the LLM an inventory it can actually emit and
# typed relation F1 becomes meaningful for this dataset.
REL_INFO = {
    "p6": "head_of_government",
    "p17": "country",
    "p19": "place_of_birth",
    "p20": "place_of_death",
    "p22": "father",
    "p25": "mother",
    "p26": "spouse",
    "p27": "country_of_citizenship",
    "p30": "continent",
    "p31": "instance_of",
    "p35": "head_of_state",
    "p36": "capital",
    "p37": "official_language",
    "p39": "position_held",
    "p40": "child",
    "p50": "author",
    "p54": "member_of_sports_team",
    "p57": "director",
    "p58": "screenwriter",
    "p69": "educated_at",
    "p86": "composer",
    "p102": "member_of_political_party",
    "p108": "employer",
    "p112": "founded_by",
    "p118": "league",
    "p123": "publisher",
    "p127": "owned_by",
    "p131": "located_in_the_administrative_territorial_entity",
    "p136": "genre",
    "p137": "operator",
    "p140": "religion",
    "p150": "contains_administrative_territorial_entity",
    "p155": "follows",
    "p156": "followed_by",
    "p159": "headquarters_location",
    "p161": "cast_member",
    "p162": "producer",
    "p166": "award_received",
    "p170": "creator",
    "p171": "parent_taxon",
    "p172": "ethnic_group",
    "p175": "performer",
    "p176": "manufacturer",
    "p178": "developer",
    "p179": "series",
    "p190": "sister_city",
    "p194": "legislative_body",
    "p205": "basin_country",
    "p206": "located_in_or_next_to_body_of_water",
    "p241": "military_branch",
    "p264": "record_label",
    "p272": "production_company",
    "p276": "location",
    "p279": "subclass_of",
    "p355": "subsidiary",
    "p361": "part_of",
    "p364": "original_language_of_work",
    "p400": "platform",
    "p403": "mouth_of_the_watercourse",
    "p449": "original_network",
    "p463": "member_of",
    "p488": "chairperson",
    "p495": "country_of_origin",
    "p527": "has_part",
    "p551": "residence",
    "p569": "date_of_birth",
    "p570": "date_of_death",
    "p571": "inception",
    "p576": "dissolved_abolished_or_demolished",
    "p577": "publication_date",
    "p580": "start_time",
    "p582": "end_time",
    "p585": "point_in_time",
    "p607": "conflict",
    "p674": "characters",
    "p676": "lyrics_by",
    "p706": "located_on_terrain_feature",
    "p710": "participant",
    "p737": "influenced_by",
    "p740": "location_of_formation",
    "p749": "parent_organization",
    "p800": "notable_work",
    "p807": "separated_from",
    "p840": "narrative_location",
    "p937": "work_location",
    "p1001": "applies_to_jurisdiction",
    "p1056": "product_or_material_produced",
    "p1198": "unemployment_rate",
    "p1336": "territory_claimed_by",
    "p1344": "participant_of",
    "p1365": "replaces",
    "p1366": "replaced_by",
    "p1376": "capital_of",
    "p1412": "languages_spoken_written_or_signed",
    "p1441": "present_in_work",
    "p3373": "sibling",
}

_NO_SPACE_BEFORE = re.compile(r"^[.,;:!?)\]}'\"%]")
_NO_SPACE_AFTER = {"(", "[", "{", "$", "``"}


def _detokenize(tokens: list[str]) -> str:
    out = ""
    prev = ""
    for i, tok in enumerate(tokens):
        if i > 0 and not _NO_SPACE_BEFORE.match(tok) and prev not in _NO_SPACE_AFTER:
            out += " "
        out += tok
        prev = tok
    return out


def _entity_repr(cluster: list[dict]) -> tuple[str, str, list[str]]:
    """Return (representative_name, canonical_type, aliases) for a cluster."""
    names = [m["name"].strip() for m in cluster if m.get("name", "").strip()]
    types = [m.get("type", "MISC") for m in cluster]
    canon = TYPE_MAP.get(Counter(types).most_common(1)[0][0], "MISC")
    if names:
        # Representative = most frequent surface; tie-break to longest.
        freq = Counter(names)
        rep = max(names, key=lambda n: (freq[n], len(n)))
    else:
        rep = "UNKNOWN"
    aliases = sorted({n for n in names if n != rep})
    return rep, canon, aliases


def load(split: str = "test", limit: int = 0, **_) -> list[BenchDoc]:
    """Load Re-DocRED into BenchDocs. ``split`` in {train,validation,test}."""
    from datasets import load_dataset
    ds = load_dataset(HF_ID, split=split)
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    docs: list[BenchDoc] = []
    for i, ex in enumerate(ds):
        text = "\n".join(_detokenize(s) for s in ex["sents"])
        vset = ex["vertexSet"]
        entities: list[BenchEntity] = []
        reps: list[str] = []
        for cluster in vset:
            rep, canon, aliases = _entity_repr(cluster)
            reps.append(rep)
            entities.append(BenchEntity(name=rep, type=canon, aliases=aliases))

        relations: list[BenchRelation] = []
        for lab in ex.get("labels", []):
            h, t = lab.get("h"), lab.get("t")
            if h is None or t is None or h >= len(reps) or t >= len(reps):
                continue
            code = str(lab.get("r", "")).lower()
            relations.append(BenchRelation(source=reps[h], target=reps[t],
                                           type=REL_INFO.get(code, code)))

        title = ex.get("title") or f"redocred_{split}_{i}"
        docs.append(BenchDoc(doc_id=f"{i:05d}_{title}", text=text,
                             entities=entities, relations=relations))
    return docs
