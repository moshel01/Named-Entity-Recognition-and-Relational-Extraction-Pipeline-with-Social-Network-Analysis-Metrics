# CLEF HIPE-2022 adapter (hipe-eval/HIPE-2022-data): historical-newspaper NER.
# German by default - the closest public proxy for the Abel corpus (historical
# German, OCR noise). Entity gold only; HIPE has no relation annotations, so
# score the `entities` sections and ignore relations.

from __future__ import annotations

import logging
from pathlib import Path

from .common import BenchDoc, BenchEntity

logger = logging.getLogger(__name__)

_RAW_BASE = ("https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/"
             "data/v2.1/{dataset}/{lang}/HIPE-2022-v2.1-{dataset}-{split}-{lang}.tsv")
_CACHE_DIR = Path("data/bench/hipe/raw")

# HIPE coarse tag -> canonical pipeline type.
TYPE_MAP = {"pers": "PERSON", "org": "ORG", "loc": "LOCATION",
            "time": "DATE", "prod": "MISC"}

DEFAULT_TARGET_TYPES = ["PERSON", "ORG", "LOCATION"]

# German historical text: multilingual GLiNER2 + the German spaCy pipeline.
DEFAULT_SPACY_MODEL = "de_core_news_lg"
DEFAULT_GLINER_MODEL = "fastino/gliner2-multi-v1"


def _fetch(dataset: str, split: str, lang: str) -> Path:
    """Download (and cache) one HIPE TSV; returns the local path."""
    name = f"HIPE-2022-v2.1-{dataset}-{split}-{lang}.tsv"
    local = _CACHE_DIR / name
    if local.exists() and local.stat().st_size > 1000:
        return local
    import requests
    url = _RAW_BASE.format(dataset=dataset, split=split, lang=lang)
    logger.info("Downloading %s ...", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    if resp.text.startswith("404"):
        raise FileNotFoundError(f"HIPE file not found: {url}")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local.write_text(resp.text, encoding="utf-8")
    return local


def _parse(tsv_path: Path) -> list[BenchDoc]:
    """Parse a HIPE TSV into BenchDocs (text rebuilt token by token)."""
    docs: list[BenchDoc] = []
    doc_id = ""
    tokens: list[tuple[str, str, bool]] = []   # (token, coarse_tag, space_after)

    def flush() -> None:
        nonlocal tokens
        if not doc_id or not tokens:
            tokens = []
            return
        text_parts: list[str] = []
        entities: list[BenchEntity] = []
        cur_surface: list[str] = []
        cur_type = ""
        pos = 0

        def close_entity() -> None:
            nonlocal cur_surface, cur_type
            if cur_surface and cur_type:
                entities.append(BenchEntity(name=" ".join(cur_surface), type=cur_type))
            cur_surface, cur_type = [], ""

        for tok, tag, space_after in tokens:
            text_parts.append(tok)
            if space_after:
                text_parts.append(" ")
            pos += len(tok) + (1 if space_after else 0)
            if tag.startswith("B-"):
                close_entity()
                cur_type = TYPE_MAP.get(tag[2:].lower(), "")
                cur_surface = [tok] if cur_type else []
            elif tag.startswith("I-") and cur_type:
                cur_surface.append(tok)
            else:
                close_entity()
        close_entity()

        # Collapse duplicate surface+type pairs (scorer clusters them anyway).
        seen: set[tuple[str, str]] = set()
        uniq: list[BenchEntity] = []
        for e in entities:
            key = (e.name.lower(), e.type)
            if e.name and e.type and key not in seen:
                seen.add(key)
                uniq.append(e)
        docs.append(BenchDoc(doc_id=doc_id, text="".join(text_parts).strip(),
                             entities=uniq, relations=[]))
        tokens = []

    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# hipe2022:document_id"):
            flush()
            doc_id = line.split("=", 1)[1].strip()
            continue
        if not line or line.startswith("#") or line.startswith("TOKEN\t"):
            continue
        cols = line.split("\t")
        if len(cols) < 10:
            continue
        tok, coarse, misc = cols[0], cols[1], cols[9]
        space_after = "NoSpaceAfter" not in misc
        tokens.append((tok, coarse, space_after))
    flush()
    return docs


def load(split: str = "dev", limit: int = 0, dataset: str = "hipe2020",
         lang: str = "de", path: str = "") -> list[BenchDoc]:
    """Load HIPE-2022 into BenchDocs. ``split`` in {sample,train,dev,test}."""
    tsv = Path(path) if path else _fetch(dataset, split, lang)
    docs = [d for d in _parse(tsv) if len(d.text) > 200]
    if limit and limit > 0:
        docs = docs[:limit]
    return docs
