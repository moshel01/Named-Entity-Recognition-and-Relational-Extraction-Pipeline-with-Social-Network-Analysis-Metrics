# LittleSis connector. LittleSis (Public Accountability Initiative) is a curated,
# sourced GRAPH of powerful people and organizations - the InfluenceWatch problem,
# but already as explicit typed relationships instead of prose. So we import the edges
# DIRECTLY as asserted ties (edge_source=littlesis), not crawl profiles and re-extract
# with the LLM. Each LittleSis relationship resolves BOTH endpoints (name + person/org)
# from one call, carries the category, and - for donations - the amount/currency/dates.
#
# Public API v2, no token. robots allows it (Crawl-delay 10 -> be polite). LICENSE:
# CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/) - attribution AND
# share-alike are REQUIRED. The license + attribution ride on every source Document's
# meta; surface them in any published network (the codebook note).
#
# spec = "Target". forms:
#   "search:Koch Industries"     top entities matching the term + their relationships
#   "id:28220" / "entity:28220"  that entity by LittleSis id + its relationships
#   "Koch Industries"            bare term == search
# depth 1 (default) = the seed entities' direct relationships (an ego network); the
# other endpoint of each edge becomes a node but is not itself expanded.

from __future__ import annotations

import json
import logging
import time
from urllib.parse import quote, unquote, urlsplit

from core.schema import Document, EntityMention, Relationship, stable_id

logger = logging.getLogger(__name__)

_UA = "SNA-Extraction-Pipeline/1.0 (academic research)"
_BASE = "https://littlesis.org/api"
_LICENSE = "CC BY-SA 4.0"
_ATTRIBUTION = "LittleSis / Public Accountability Initiative (littlesis.org)"
_SYMMETRIC = {"family_of", "knew"}


def _http_get(url: str, user_agent: str = _UA, timeout: int = 30) -> str:
    import requests
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_spec(spec: str) -> tuple[str, str]:
    """'search:Koch' -> ('search','Koch'); 'id:28220' -> ('id','28220'); bare -> search."""
    s = (spec or "").strip()
    kind, sep, val = s.partition(":")
    kind = kind.strip().lower()
    if sep and kind in ("search", "id", "entity", "name"):
        return ("id" if kind == "entity" else kind), val.strip()
    return "search", s


def _endpoint(url: str) -> tuple[str, str, str]:
    """LittleSis entity URL -> (LABEL, id, name).
    'https://littlesis.org/org/28220-Koch_Industries,_Inc.' -> ('ORG','28220','Koch Industries, Inc.')."""
    parts = urlsplit(url or "").path.strip("/").split("/")
    ext = parts[0].lower() if parts else ""
    last = parts[-1] if parts else ""
    lsid, _, name = last.partition("-")          # id is numeric, name may hold more '-'
    name = unquote(name).replace("_", " ").strip()
    label = "PERSON" if ext == "person" else "ORG"
    return label, lsid, name


def _ext_label(primary_ext: str) -> str:
    return "PERSON" if (primary_ext or "").lower() == "person" else "ORG"


def _map_relation(attrs: dict) -> str:
    """LittleSis category_id (+ position flags) -> our canonical relation."""
    cid = attrs.get("category_id")
    ca = attrs.get("category_attributes") or {}
    if cid == 1:                                  # Position
        if ca.get("is_board"):
            return "board_member_of"
        if ca.get("is_executive"):
            return "director_of"
        return "employed_by"
    return {
        2: "studied_at", 3: "member_of", 4: "family_of", 5: "donated_to",
        6: "contracted", 7: "lobbied", 8: "knew", 9: "advised", 10: "owns",
        11: "affiliated_with", 12: "affiliated_with",
    }.get(cid, "affiliated_with")


def _search(term: str, limit: int, get) -> list[dict]:
    url = f"{_BASE}/entities/search?q={quote(term)}"
    try:
        data = json.loads(get(url))
    except Exception as exc:  # noqa: BLE001
        logger.warning("littlesis: search failed (%s): %s", term, exc)
        return []
    out = []
    for e in (data.get("data") or [])[:limit]:
        a = e.get("attributes") or {}
        out.append({"id": str(a.get("id") or e.get("id") or ""), "name": a.get("name", ""),
                    "ext": a.get("primary_ext", ""), "blurb": a.get("blurb", "") or "",
                    "summary": a.get("summary", "") or ""})
    return [e for e in out if e["id"]]


def _entity(eid: str, get) -> dict:
    try:
        a = (json.loads(get(f"{_BASE}/entities/{eid}")).get("data") or {}).get("attributes") or {}
    except Exception:  # noqa: BLE001
        return {"id": str(eid), "name": "", "ext": "", "blurb": "", "summary": ""}
    return {"id": str(a.get("id") or eid), "name": a.get("name", ""),
            "ext": a.get("primary_ext", ""), "blurb": a.get("blurb", "") or "",
            "summary": a.get("summary", "") or ""}


def _relationships(eid: str, get, max_pages: int, delay: float) -> list[dict]:
    """All of an entity's relationships -> normalized edge dicts (paginated, capped)."""
    edges: list[dict] = []
    page = 1
    while page <= max_pages:
        url = f"{_BASE}/entities/{eid}/relationships?page={page}"
        try:
            data = json.loads(get(url))
        except Exception as exc:  # noqa: BLE001
            logger.debug("littlesis: relationships failed (%s p%d): %s", eid, page, exc)
            break
        rows = data.get("data") or []
        for r in rows:
            a = r.get("attributes") or {}
            s_label, s_id, s_name = _endpoint(r.get("entity", ""))
            t_label, t_id, t_name = _endpoint(r.get("related", ""))
            if not s_name or not t_name:
                continue
            edges.append({
                "src": s_name, "src_label": s_label, "src_id": s_id,
                "tgt": t_name, "tgt_label": t_label, "tgt_id": t_id,
                "rel": _map_relation(a),
                "amount": a.get("amount"), "currency": a.get("currency"),
                "start_date": a.get("start_date"), "end_date": a.get("end_date"),
                "evidence": (a.get("description") or "").strip(),
            })
        total = (data.get("meta") or {}).get("pageCount", page)
        if page >= total:
            break
        page += 1
        if delay:
            time.sleep(delay)
    return edges


def _entity_document(ent: dict, edges: list[dict]) -> Document:
    blurb, summary = ent.get("blurb", "") or "", ent.get("summary", "") or ""
    text = f"{ent['name']}. {blurb} {summary}".strip()
    ext = ent.get("ext", "") or ""
    url = f"https://littlesis.org/{(ext or 'entity').lower()}/{ent['id']}"
    # Node attributes surfaced as Gephi attr_* columns (filterable). ls_blurb/ls_types/
    # ls_website come from entities.json enrichment; absent on the API/edge-only path.
    ls_attrs = {}
    if blurb:
        ls_attrs["ls_blurb"] = blurb
    if ent.get("types"):
        ls_attrs["ls_types"] = ent["types"]
    if ent.get("website"):
        ls_attrs["ls_website"] = ent["website"]
    if ext:
        ls_attrs["ls_ext"] = ext
    meta = {"filename": url, "source_type": "littlesis", "platform": "littlesis",
            "littlesis_id": ent["id"], "name": ent["name"],
            "primary_ext": ext, "label": _ext_label(ext),
            "license": _LICENSE, "attribution": _ATTRIBUTION,
            "ls_edges": edges, "ls_attrs": ls_attrs, "n_chars": len(text)}
    if ent.get("aliases"):
        meta["ls_aliases"] = ent["aliases"]
    return Document(
        doc_id=stable_id(f"littlesis:{ent['id']}", prefix="ls_", length=10),
        source_path=url, text=text, meta=meta)


def fetch_littlesis(spec: str, *, limit: int = 10, depth: int = 1, fetch=None,
                    delay: float = 1.0, max_pages: int = 3, **_) -> list[Document]:
    """Resolve a LittleSis spec to seed-entity Documents carrying their relationships."""
    get = fetch or _http_get
    kind, val = parse_spec(spec)
    if not val:
        raise ValueError("littlesis spec needs a target, e.g. littlesis:search:Koch Industries "
                         "or littlesis:id:28220")
    if kind == "id":
        seeds = [_entity(val, get)]
    else:
        seeds = _search(val, limit, get)
    docs: list[Document] = []
    for ent in seeds:
        if not ent.get("id"):
            continue
        edges = _relationships(ent["id"], get, max_pages, delay) if depth >= 1 else []
        docs.append(_entity_document(ent, edges))
        if delay and fetch is None:
            time.sleep(delay)
    logger.info("littlesis: %d entit(ies) for %s (%d edges).",
                len(docs), spec, sum(len(d.meta.get("ls_edges") or []) for d in docs))
    return docs


# ---- bulk dump import -------------------------------------------------------
# The full CC BY-SA database is two gzipped JSON ARRAYS (not JSONL):
#   https://littlesis.org/database/public_data/{entities,relationships}.json.gz
# relationships.json.gz is self-contained (each record's entity/related URLs carry
# both endpoint names + person/org), so the whole edge graph builds from it alone.
# The files are large, so we STREAM the array element-by-element (no json.load of GB).

def _iter_json_array(fh, chunk_size: int = 1 << 20):
    """Yield each top-level object of a big JSON array, streaming a text file handle."""
    dec = json.JSONDecoder()
    buf = ""
    started = False
    while True:
        data = fh.read(chunk_size)
        if not data:
            break
        buf += data
        if not started:
            i = buf.find("[")
            if i < 0:
                buf = ""
                continue
            buf = buf[i + 1:]
            started = True
        while True:
            t = buf.lstrip()
            if t[:1] == ",":
                t = t[1:].lstrip()
            if not t or t[0] == "]":
                buf = t
                break
            try:
                obj, end = dec.raw_decode(t)
            except ValueError:        # object spans the chunk boundary - read more
                buf = t
                break
            yield obj
            buf = t[end:]


def _norm_name(name: str) -> str:
    n = (name or "").strip().lower()
    if n.startswith("the "):
        n = n[4:]
    for suf in (", inc.", " inc.", ", inc", " inc", ", llc", " llc", " co.", " corp."):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n.strip()


def _bulk_open(path):
    import gzip
    return (gzip.open if str(path).endswith(".gz") else open)(path, "rt", encoding="utf-8")


def load_bulk(relationships_path, *, entities_path=None, ids=None, names=None,
              categories=None, min_amount=None, max_edges: int = 0,
              both_endpoints: bool = False, include_isolated: bool = False) -> list[Document]:
    """Import the LittleSis bulk dump as entity Documents carrying their edges (same shape
    as the API connector, so the structure hook + dedup/merge are unchanged). Two passes:
    relationships.json builds the edges, entities.json (optional) enriches each node with
    blurb/types/website/aliases. Filters carve the slice out of the ~1.7M-edge full graph:
      ids/names      - keep an edge if an endpoint matches (pass your scraped entity names);
                       both_endpoints=True keeps only edges where BOTH match (induced subgraph)
      categories     - LittleSis category_ids (e.g. {1,5,10} = positions/donations/ownership)
      min_amount     - keep only money edges at/above this amount
      max_edges      - hard cap (0 = unlimited)
      include_isolated - with entities_path, also emit entities that have no kept edge
                         (only meaningful for the whole-dump variant)."""
    id_set = {str(i) for i in ids} if ids else None
    name_set = {_norm_name(n) for n in names} if names else None
    cat_set = set(categories) if categories else None
    has_filter = bool(id_set or name_set)
    nodes: dict[str, dict] = {}

    def _ensure(eid, name, label):
        n = nodes.get(eid)
        if n is None:
            n = nodes[eid] = {"id": eid, "name": name,
                              "ext": "person" if label == "PERSON" else "org",
                              "blurb": "", "summary": "", "types": "", "website": "",
                              "aliases": [], "edges": []}
        return n

    def _match(eid, name):
        return (id_set is not None and eid in id_set) or \
               (name_set is not None and _norm_name(name) in name_set)

    # Pass 1: relationships -> edges (grouped on the source entity) + the involved node set.
    kept = 0
    with _bulk_open(relationships_path) as fh:
        for rec in _iter_json_array(fh):
            a = rec.get("attributes") or {}
            if cat_set and a.get("category_id") not in cat_set:
                continue
            s_label, s_id, s_name = _endpoint(rec.get("entity", ""))
            t_label, t_id, t_name = _endpoint(rec.get("related", ""))
            if not s_name or not t_name:
                continue
            if has_filter:
                s_ok, t_ok = _match(s_id, s_name), _match(t_id, t_name)
                keep = (s_ok and t_ok) if both_endpoints else (s_ok or t_ok)
                if not keep:
                    continue
            amt = a.get("amount")
            if min_amount is not None and not (amt is not None and amt >= min_amount):
                continue
            src = _ensure(s_id, s_name, s_label)
            _ensure(t_id, t_name, t_label)        # target is a node too (enriched in pass 2)
            src["edges"].append({
                "src": s_name, "src_label": s_label, "src_id": s_id,
                "tgt": t_name, "tgt_label": t_label, "tgt_id": t_id,
                "rel": _map_relation(a),
                "amount": amt, "currency": a.get("currency"),
                "start_date": a.get("start_date"), "end_date": a.get("end_date"),
                "evidence": (a.get("description") or "").strip(),
            })
            kept += 1
            if max_edges and kept >= max_edges:
                break

    # Pass 2: entities.json -> node attributes (and isolated nodes if asked).
    if entities_path:
        with _bulk_open(entities_path) as fh:
            for rec in _iter_json_array(fh):
                a = rec.get("attributes") or {}
                eid = str(a.get("id") or rec.get("id") or "")
                if not eid:
                    continue
                n = nodes.get(eid)
                if n is None:
                    if not include_isolated:
                        continue
                    n = _ensure(eid, a.get("name", ""),
                                "PERSON" if (a.get("primary_ext") or "").lower() == "person" else "ORG")
                n["name"] = n["name"] or a.get("name", "")
                if a.get("primary_ext"):
                    n["ext"] = a["primary_ext"].lower()
                n["blurb"] = (a.get("blurb") or "")[:200]
                n["summary"] = (a.get("summary") or "")[:300]
                n["website"] = a.get("website") or ""
                t = a.get("types") or []
                n["types"] = ";".join(t) if isinstance(t, list) else str(t)
                n["aliases"] = a.get("aliases") or []

    # Emit a Document per node when enriching (so target/isolated nodes carry attributes);
    # edge-only mode keeps it lean (source entities only - targets ride in via edge mentions).
    emit_all = bool(entities_path) or include_isolated
    docs = [_entity_document(n, n["edges"]) for n in nodes.values() if n["edges"] or emit_all]
    logger.info("littlesis bulk: %d edges, %d nodes -> %d documents%s.",
                kept, len(nodes), len(docs), " (enriched)" if entities_path else "")
    return docs


def _mention(name: str, label: str, ls_id: str, doc_id: str, extra: dict | None = None) -> EntityMention:
    attrs = {"littlesis": True, "littlesis_id": ls_id}
    if extra:
        attrs.update(extra)
    return EntityMention(
        text=name, label=label, start_char=0, end_char=0, chunk_id=doc_id,
        doc_id=doc_id, confidence=1.0, sources=["littlesis"], attributes=attrs)


def littlesis_structure(doc: Document) -> tuple[list[EntityMention], list[Relationship]]:
    """A littlesis Document's meta -> (PERSON/ORG mentions, asserted typed edges).
    Each edge is edge_source=littlesis (ASSERTED tier); donations carry
    qual_monetary_value. Appended to the doc's extraction in run_extract."""
    meta = doc.meta or {}
    if meta.get("source_type") != "littlesis":
        return [], []
    did = doc.doc_id
    mentions: list[EntityMention] = []
    edges: list[Relationship] = []
    seen: set[tuple[str, str]] = set()

    # The anchor entity always exists as a node, even with no relationships. It carries
    # the entities.json enrichment (ls_blurb/ls_types/ls_website) so the node gets attr_*
    # columns; edge-endpoint mentions stay minimal and fold into their own anchor by name.
    anchor, alabel = (meta.get("name") or "").strip(), meta.get("label") or "ORG"
    if anchor:
        mentions.append(_mention(anchor, alabel, str(meta.get("littlesis_id") or ""), did,
                                 extra=dict(meta.get("ls_attrs") or {})))
        seen.add((anchor.lower(), alabel))

    for e in (meta.get("ls_edges") or []):
        for nm, lb, lid in ((e["src"], e["src_label"], e.get("src_id", "")),
                            (e["tgt"], e["tgt_label"], e.get("tgt_id", ""))):
            key = (nm.lower(), lb)
            if nm and key not in seen:
                mentions.append(_mention(nm, lb, str(lid), did))
                seen.add(key)
        attrs = {"edge_source": "littlesis"}
        if e.get("amount") is not None:
            attrs["qual_monetary_value"] = e["amount"]
            if e.get("currency"):
                attrs["qual_currency"] = e["currency"]
        if e.get("start_date"):
            attrs["qual_date"] = e["start_date"]
        edges.append(Relationship(
            source=e["src"], target=e["tgt"], rel_type=e["rel"], doc_id=did,
            evidence=e.get("evidence", "") or f"{e['src']} - {e['rel']} - {e['tgt']} (LittleSis)",
            confidence=1.0, directed=e["rel"] not in _SYMMETRIC,
            origin="extracted", attributes=attrs))
    return mentions, edges
