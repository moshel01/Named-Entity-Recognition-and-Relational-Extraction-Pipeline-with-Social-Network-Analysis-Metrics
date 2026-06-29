# Screenplay / TV-script structure parser. A script is not prose: scene-heading slug
# lines (INT./EXT. ...) bound the units and an all-caps line above dialogue is a
# speaker cue. The people who speak in a scene are CO-PRESENT whether or not the text
# states a relationship between them - and scene co-presence is the standard character-
# network signal (Moretti/Stanford Lit Lab; the screenplay analogue of who-shares-a-
# scene). The proximity window only approximates it; here we read it straight from the
# format and emit Newman-weighted co_present_in_scene edges (a scene is a group, the
# characters in it get a tie). Co-presence, not an asserted interaction - the edges ride
# the weakest evidence tier (script_copresence, see postprocess/evidence_tiers), tagged
# and filterable, so an analyst can lift the character graph in Gephi by tie class.

from __future__ import annotations

import logging
import re
from collections import defaultdict
from itertools import combinations

from .schema import Relationship

logger = logging.getLogger(__name__)

# Scene boundary slug lines (INT./EXT. variants) and the common transitions.
_SLUG_RE = re.compile(r"^[ \t]*(?:INT|EXT|INT\.?/EXT|I/E)[\.\s/]", re.M)
_TRANS_RE = re.compile(r"^[ \t]*(?:FADE IN|FADE OUT|FADE TO|CUT TO|DISSOLVE TO|"
                       r"SMASH CUT|MATCH CUT|BACK TO)\b", re.M | re.I)
# A speaker cue: a SHORT all-caps line (no lowercase), optionally with a (V.O.)/
# (CONT'D) parenthetical. The lack of lowercase in the char class is what separates a
# cue from an action/prose line.
_CUE_RE = re.compile(r"^[ \t]{0,20}([A-Z][A-Z0-9 .'\-]{0,30})(?:[ \t]*\([^)]*\))?[ \t]*$",
                     re.M)

# All-caps lines that are directions/transitions, never characters.
_NOT_CHARACTER = {
    "INT", "EXT", "FADE IN", "FADE OUT", "FADE TO", "CUT TO", "DISSOLVE TO",
    "SMASH CUT", "MATCH CUT", "BACK TO", "THE END", "CONTINUED", "CONT'D",
    "INTERCUT", "MONTAGE", "SUPER", "TITLE", "ANGLE ON", "CLOSE ON", "WIDE",
    "CLOSE", "INSERT", "POV", "V.O", "O.S", "OMITTED", "LATER", "MOMENTS LATER",
    "PRELAP", "BEAT", "END", "ACT ONE", "ACT TWO", "ACT THREE", "TEASER",
}


def _clean_cue(raw: str) -> str:
    """Normalize a candidate cue to a character name, or "" if it isn't one."""
    name = re.sub(r"\s*\([^)]*\)", "", raw).strip()      # drop (V.O.), (CONT'D)
    name = name.strip(" .-")
    if not name:
        return ""
    up = name.upper()
    if up in _NOT_CHARACTER:
        return ""
    if re.match(r"^(?:INT|EXT|I/E)\b", up):              # scene slug remnant
        return ""
    if " - " in name or "--" in name:                    # "KITCHEN - DAY" etc.
        return ""
    if len(name) > 30 or len(name.split()) > 4:          # cues are short
        return ""
    if not re.search(r"[A-Z]", up):
        return ""
    # Display form: BILBO -> Bilbo, GANDALF THE GREY -> Gandalf The Grey.
    return name.title()


def looks_like_script(text: str, min_scenes: int = 2, min_cues: int = 4) -> bool:
    """Heuristic: enough scene slugs/transitions AND enough distinct speaker cues."""
    if not text:
        return False
    scenes = len(_SLUG_RE.findall(text)) + len(_TRANS_RE.findall(text))
    cues = {c for m in _CUE_RE.finditer(text) if (c := _clean_cue(m.group(1)))}
    return scenes >= min_scenes and len(cues) >= min_cues


def parse_scenes(text: str) -> list[list[str]]:
    """Segment into scenes; return each scene's distinct speaking characters in order.

    Boundaries are slug + transition lines. Within a scene a character is anyone with
    a speaker cue (an all-caps line above dialogue). Two characters who both speak in a
    scene are co-present."""
    bounds = sorted({m.start() for m in _SLUG_RE.finditer(text)}
                    | {m.start() for m in _TRANS_RE.finditer(text)}
                    | {0, len(text)})
    scenes: list[list[str]] = []
    for a, b in zip(bounds, bounds[1:]):
        block = text[a:b]
        chars: list[str] = []
        seen: set[str] = set()
        for m in _CUE_RE.finditer(block):
            name = _clean_cue(m.group(1))
            if name and name not in seen:
                seen.add(name)
                chars.append(name)
        if chars:
            scenes.append(chars)
    return scenes


def copresence_edges(scenes: list[list[str]], doc_id: str,
                     max_scene_size: int = 25) -> list[Relationship]:
    """Newman-weighted co_present_in_scene edges from per-scene character sets.

    Weight 1/(k-1) per scene (sharing a 2-hander is a strong tie; sharing a crowd
    scene is weak), summed across the document's scenes. Undirected. Co-presence, not
    an asserted tie - stamped script_copresence (weakest evidence tier)."""
    agg: dict[tuple[str, str], dict] = defaultdict(lambda: {"w": 0.0, "n": 0})
    for chars in scenes:
        uniq = sorted(set(chars))
        k = len(uniq)
        if k < 2 or k > max_scene_size:        # singletons and parse-error crowds
            continue
        w = 1.0 / (k - 1)
        for a, b in combinations(uniq, 2):
            cell = agg[(a, b)]
            cell["w"] += w
            cell["n"] += 1
    edges: list[Relationship] = []
    for (a, b), cell in agg.items():
        edges.append(Relationship(
            source=a, target=b, rel_type="co_present_in_scene", doc_id=doc_id,
            evidence=f"co-present in {cell['n']} scene(s)",
            confidence=1.0, directed=False, origin="inferred",
            attributes={"edge_source": "script_copresence",
                        "scene_weight": round(cell["w"], 4),
                        "n_scenes": cell["n"]},
        ))
    return edges


def script_copresence(text: str, doc_id: str) -> list[Relationship]:
    """One-shot: parse a script's text and return its co_present_in_scene edges, or []
    if it doesn't look like a script. The opt-in hook the extractor calls."""
    if not looks_like_script(text):
        return []
    scenes = parse_scenes(text)
    edges = copresence_edges(scenes, doc_id)
    if edges:
        logger.info("Script co-presence: %d scenes, %d character pairs in %s.",
                    len(scenes), len(edges), doc_id)
    return edges
