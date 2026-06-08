# Load the Abel metadata spreadsheet, keyed by LetterID (matches the trailing
# digits in each filename). Columns are matched fuzzily so header typos are fine.

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def _key(h: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(h or "").lower())


# wanted snake_case field -> substrings to find in the header (typo-tolerant)
_FIELDS = {
    "last_name": ["lastname"],
    "first_name": ["firstname"],
    "gender": ["gender"],
    "birth_date": ["birthdate"],
    "place_of_birth": ["placeofbirth"],
    "place_of_residence": ["placeofresidence"],
    "education": ["education"],
    "current_position": ["currentposition"],
    "profession_father": ["professionfather"],
    "marital_status": ["maritalstatus"],
    "number_of_children": ["numberofchildren"],
    "religion": ["religion"],
    "membership_number": ["membershipnumber", "membernumber"],
    "join_date": ["joindate"],
    "prior_party": ["partymembershipbefore", "memberhsipbefore"],
    "war_veteran": ["warveteran"],
    "war_injury": ["warinjury"],
    "letter_id": ["letterid"],
}

_YEAR = re.compile(r"(1[89]\d{2})")


def load_metadata(path: str) -> dict[str, dict[str, str]]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    headers = [_key(h) for h in next(rows)]

    # resolve each wanted field to a column index
    col = {}
    for field, needles in _FIELDS.items():
        for i, h in enumerate(headers):
            if any(n in h for n in needles):
                col[field] = i
                break

    if "letter_id" not in col:
        logger.warning("metadata: no LetterID column found in %s", path)
        return {}

    out: dict[str, dict[str, str]] = {}
    for row in rows:
        lid = row[col["letter_id"]]
        if lid is None:
            continue
        lid = re.sub(r"\D", "", str(lid))
        if not lid:
            continue
        rec = {}
        for field, idx in col.items():
            if field == "letter_id":
                continue
            v = row[idx] if idx < len(row) else None
            if v is not None and str(v).strip():
                rec[field] = str(v).strip()
        bd = rec.get("birth_date", "")
        m = _YEAR.search(bd)
        if m:
            rec["birth_year"] = m.group(1)
        if rec.get("first_name") or rec.get("last_name"):
            rec["meta_name"] = f"{rec.get('first_name','')} {rec.get('last_name','')}".strip()
        out[lid] = rec
    logger.info("metadata: loaded %d rows from %s", len(out), path)
    return out


def _ok(v) -> bool:
    return bool(v) and str(v).strip().lower() not in ("na", "n/a", "none", "unknown", "-", "")


def metadata_edges(row: dict) -> list[dict]:
    # Verified edges straight from the spreadsheet (edge_source=metadata).
    edges = []
    if _ok(row.get("place_of_birth")):
        edges.append({"target": row["place_of_birth"], "type": "LOCATION", "rel": "born_in"})
    if _ok(row.get("place_of_residence")):
        edges.append({"target": row["place_of_residence"], "type": "LOCATION", "rel": "resided_in"})
    if _ok(row.get("prior_party")):
        edges.append({"target": row["prior_party"], "type": "ORG", "rel": "member_of",
                      "attrs": {"prior_party": True}})
    nsdap = {}
    if _ok(row.get("membership_number")):
        nsdap["membership_number"] = row["membership_number"]
    if _ok(row.get("join_date")):
        nsdap["join_date"] = row["join_date"]
    edges.append({"target": "NSDAP", "type": "ORG", "rel": "member_of", "attrs": nsdap})
    return edges
