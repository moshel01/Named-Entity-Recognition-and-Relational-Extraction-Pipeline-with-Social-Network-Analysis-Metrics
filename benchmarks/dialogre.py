# DialogRE adapter (nlpdata/dialogre, v2 English): relation extraction from
# dialogue transcripts - interpersonal ties (friends, siblings, boss) stated in
# conversation, the closest public benchmark to narrative interaction ties.
#
# Caveats baked into the conversion:
# - "Speaker N" slots get per-dialogue synthetic names ("Alan Abbott") in BOTH
#   the text and the gold. 96% of gold relations involve a speaker slot and
#   half the (Speaker i, Speaker j) pairs collide across dialogues, so leaving
#   the literal slots makes corpus-level scoring merge different people into
#   one node. Names are deterministic: given name by slot, surname by dialogue.
# - STRING/VALUE arguments (titles, dates) and `unanswerable` pairs are
#   dropped - they are not social ties.

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .common import BenchDoc, BenchEntity, BenchRelation

logger = logging.getLogger(__name__)

_RAW_BASE = "https://raw.githubusercontent.com/nlpdata/dialogre/master/data_v2/en/data/{split}.json"
_CACHE_DIR = Path("data/bench/dialogre/raw")

TYPE_MAP = {"PER": "PERSON", "ORG": "ORG", "GPE": "LOCATION", "LOC": "LOCATION"}

DEFAULT_TARGET_TYPES = ["PERSON", "ORG", "LOCATION"]

_SPEAKER_RE = re.compile(r"\bSpeaker (\d+)\b")
_GIVEN = ["Alan", "Beth", "Carl", "Dana", "Eric", "Fern", "Glen", "Hope",
          "Ivan", "June", "Kurt", "Lena"]
_SURNAMES = ["Abbott", "Barker", "Cooper", "Dawson", "Ellis", "Foster",
             "Garner", "Hayes", "Ingram", "Jarvis", "Keller", "Lawson",
             "Mercer", "Nolan", "Osborn", "Parker", "Quinn", "Reeves",
             "Sutton", "Tanner", "Upton", "Vance", "Walton", "Yates"]


def _speaker_name(dialogue_idx: int, slot: int) -> str:
    given = _GIVEN[(slot - 1) % len(_GIVEN)]
    surname = _SURNAMES[dialogue_idx % len(_SURNAMES)]
    gen = dialogue_idx // len(_SURNAMES)
    return f"{given} {surname}{gen if gen else ''}"


def _rel_label(r: str) -> str:
    """'per:positive_impression' -> 'positive_impression'."""
    return r.split(":", 1)[-1].strip().lower()


def _fetch(split: str) -> Path:
    name = f"{split}.json"
    local = _CACHE_DIR / name
    if local.exists() and local.stat().st_size > 1000:
        return local
    import requests
    url = _RAW_BASE.format(split=split)
    logger.info("Downloading %s ...", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local.write_text(resp.text, encoding="utf-8")
    return local


def load(split: str = "dev", limit: int = 0, path: str = "") -> list[BenchDoc]:
    """Load DialogRE into BenchDocs. ``split`` in {train,dev,test}."""
    src = Path(path) if path else _fetch(split)
    data = json.loads(src.read_text(encoding="utf-8"))
    if limit and limit > 0:
        data = data[:limit]

    docs: list[BenchDoc] = []
    for i, (turns, pairs) in enumerate(data):
        def named(s: str) -> str:
            return _SPEAKER_RE.sub(lambda m: _speaker_name(i, int(m.group(1))), s)
        slots = {int(m.group(1)) for t in turns for m in _SPEAKER_RE.finditer(t)}
        turns = [named(t) for t in turns]
        entities: dict[tuple[str, str], BenchEntity] = {}
        relations: list[BenchRelation] = []
        # Gold pairs only cover relation arguments, but every speaker is a real
        # person named in the text now - count them all or entity precision is
        # charged for correctly extracting the unpaired ones.
        for s in sorted(slots):
            nm = _speaker_name(i, s)
            entities.setdefault((nm.lower(), "PERSON"), BenchEntity(name=nm, type="PERSON"))
        for p in pairs:
            xt = TYPE_MAP.get(str(p.get("x_type", "")))
            yt = TYPE_MAP.get(str(p.get("y_type", "")))
            x, y = named(str(p.get("x", "")).strip()), named(str(p.get("y", "")).strip())
            if not x or not y or xt is None or yt is None:
                continue
            entities.setdefault((x.lower(), xt), BenchEntity(name=x, type=xt))
            entities.setdefault((y.lower(), yt), BenchEntity(name=y, type=yt))
            for r in p.get("r", []):
                lab = _rel_label(str(r))
                if lab and lab != "unanswerable":
                    relations.append(BenchRelation(source=x, target=y, type=lab))
        docs.append(BenchDoc(
            doc_id=f"{i:05d}_dialogre_{split}",
            text="\n".join(turns),
            entities=list(entities.values()),
            relations=relations,
        ))
    return docs
