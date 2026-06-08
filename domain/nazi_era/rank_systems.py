# Complete rank ladders for SA, SS, and the Wehrmacht (Heer), plus a resolver.

from __future__ import annotations

import re
from typing import Optional


def _norm(text: str) -> str:
    """Lowercase + umlaut/ß normalization for matching."""
    t = text.lower().strip()
    t = (t.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
           .replace("ß", "ss"))
    t = re.sub(r"[\s\-_.]+", "", t)
    return t


# SA - Sturmabteilung (20 ranks)
SA_RANKS: list[dict] = [
    {"canonical": "SA-Anwärter", "level": 1, "english": "candidate", "variants": ["SA-Anwaerter", "Anwärter"]},
    {"canonical": "SA-Sturmmann", "level": 2, "english": "stormtrooper / private", "variants": ["Sturmmann"]},
    {"canonical": "SA-Obersturmmann", "level": 3, "english": "senior stormtrooper", "variants": ["Obersturmmann"]},
    {"canonical": "SA-Rottenführer", "level": 4, "english": "section leader", "variants": ["SA-Rottenfuehrer", "Rottenführer"]},
    {"canonical": "SA-Scharführer", "level": 5, "english": "squad leader", "variants": ["SA-Scharfuehrer", "Scharführer"]},
    {"canonical": "SA-Oberscharführer", "level": 6, "english": "senior squad leader", "variants": ["SA-Oberscharfuehrer", "Oberscharführer"]},
    {"canonical": "SA-Truppführer", "level": 7, "english": "troop leader", "variants": ["SA-Truppfuehrer", "Truppführer"]},
    {"canonical": "SA-Obertruppführer", "level": 8, "english": "senior troop leader", "variants": ["SA-Obertruppfuehrer", "Obertruppführer"]},
    {"canonical": "SA-Haupttruppführer", "level": 9, "english": "head troop leader", "variants": ["SA-Haupttruppfuehrer", "Haupttruppführer"]},
    {"canonical": "SA-Sturmführer", "level": 10, "english": "storm leader (lieutenant)", "variants": ["SA-Sturmfuehrer", "Sturmführer"]},
    {"canonical": "SA-Obersturmführer", "level": 11, "english": "senior storm leader", "variants": ["SA-Obersturmfuehrer", "Obersturmführer"]},
    {"canonical": "SA-Hauptsturmführer", "level": 12, "english": "head storm leader (captain)", "variants": ["SA-Hauptsturmfuehrer", "Hauptsturmführer"]},
    {"canonical": "SA-Sturmbannführer", "level": 13, "english": "assault unit leader (major)", "variants": ["SA-Sturmbannfuehrer", "Sturmbannführer"]},
    {"canonical": "SA-Obersturmbannführer", "level": 14, "english": "senior assault unit leader", "variants": ["SA-Obersturmbannfuehrer", "Obersturmbannführer"]},
    {"canonical": "SA-Standartenführer", "level": 15, "english": "regiment leader (colonel)", "variants": ["SA-Standartenfuehrer", "Standartenführer"]},
    {"canonical": "SA-Oberführer", "level": 16, "english": "senior leader", "variants": ["SA-Oberfuehrer", "Oberführer"]},
    {"canonical": "SA-Brigadeführer", "level": 17, "english": "brigade leader (brigadier)", "variants": ["SA-Brigadefuehrer", "Brigadeführer"]},
    {"canonical": "SA-Gruppenführer", "level": 18, "english": "group leader (lieutenant general)", "variants": ["SA-Gruppenfuehrer", "Gruppenführer"]},
    {"canonical": "SA-Obergruppenführer", "level": 19, "english": "senior group leader (general)", "variants": ["SA-Obergruppenfuehrer", "Obergruppenführer"]},
    {"canonical": "Oberster SA-Führer", "level": 20, "english": "Supreme SA Leader (Hitler) / Stabschef", "variants": ["Oberster SA-Fuehrer", "Stabschef", "Stabschef der SA"]},
]

# SS - Schutzstaffel (22 ranks)
SS_RANKS: list[dict] = [
    {"canonical": "SS-Anwärter", "level": 1, "english": "candidate", "variants": ["SS-Anwaerter", "Anwärter"]},
    {"canonical": "SS-Bewerber", "level": 2, "english": "applicant", "variants": ["Bewerber"]},
    {"canonical": "SS-Mann", "level": 3, "english": "private", "variants": ["SS-Mann", "SS-Schütze", "SS-Schuetze"]},
    {"canonical": "SS-Sturmmann", "level": 4, "english": "stormtrooper", "variants": ["Sturmmann"]},
    {"canonical": "SS-Rottenführer", "level": 5, "english": "section leader", "variants": ["SS-Rottenfuehrer", "Rottenführer"]},
    {"canonical": "SS-Unterscharführer", "level": 6, "english": "junior squad leader (corporal)", "variants": ["SS-Unterscharfuehrer", "Unterscharführer"]},
    {"canonical": "SS-Scharführer", "level": 7, "english": "squad leader (sergeant)", "variants": ["SS-Scharfuehrer", "Scharführer"]},
    {"canonical": "SS-Oberscharführer", "level": 8, "english": "senior squad leader", "variants": ["SS-Oberscharfuehrer", "Oberscharführer"]},
    {"canonical": "SS-Hauptscharführer", "level": 9, "english": "head squad leader (sergeant major)", "variants": ["SS-Hauptscharfuehrer", "Hauptscharführer"]},
    {"canonical": "SS-Sturmscharführer", "level": 10, "english": "staff sergeant major", "variants": ["SS-Sturmscharfuehrer", "Sturmscharführer"]},
    {"canonical": "SS-Untersturmführer", "level": 11, "english": "junior storm leader (2nd lieutenant)", "variants": ["SS-Untersturmfuehrer", "Untersturmführer"]},
    {"canonical": "SS-Obersturmführer", "level": 12, "english": "senior storm leader (1st lieutenant)", "variants": ["SS-Obersturmfuehrer", "Obersturmführer"]},
    {"canonical": "SS-Hauptsturmführer", "level": 13, "english": "head storm leader (captain)", "variants": ["SS-Hauptsturmfuehrer", "Hauptsturmführer"]},
    {"canonical": "SS-Sturmbannführer", "level": 14, "english": "assault unit leader (major)", "variants": ["SS-Sturmbannfuehrer", "Sturmbannführer"]},
    {"canonical": "SS-Obersturmbannführer", "level": 15, "english": "senior assault unit leader (lt. colonel)", "variants": ["SS-Obersturmbannfuehrer", "Obersturmbannführer"]},
    {"canonical": "SS-Standartenführer", "level": 16, "english": "regiment leader (colonel)", "variants": ["SS-Standartenfuehrer", "Standartenführer"]},
    {"canonical": "SS-Oberführer", "level": 17, "english": "senior leader", "variants": ["SS-Oberfuehrer", "Oberführer"]},
    {"canonical": "SS-Brigadeführer", "level": 18, "english": "brigade leader (major general)", "variants": ["SS-Brigadefuehrer", "Brigadeführer"]},
    {"canonical": "SS-Gruppenführer", "level": 19, "english": "group leader (lieutenant general)", "variants": ["SS-Gruppenfuehrer", "Gruppenführer"]},
    {"canonical": "SS-Obergruppenführer", "level": 20, "english": "senior group leader (general)", "variants": ["SS-Obergruppenfuehrer", "Obergruppenführer"]},
    {"canonical": "SS-Oberstgruppenführer", "level": 21, "english": "supreme group leader (colonel general)", "variants": ["SS-Oberstgruppenfuehrer", "Oberstgruppenführer"]},
    {"canonical": "Reichsführer-SS", "level": 22, "english": "Reich Leader SS (Himmler)", "variants": ["Reichsfuehrer-SS", "Reichsführer SS", "RFSS"]},
]

# Wehrmacht / Heer (25 ranks, Schütze -> Reichsmarschall)
WEHRMACHT_RANKS: list[dict] = [
    {"canonical": "Schütze", "level": 1, "english": "rifleman / private", "variants": ["Schuetze", "Soldat", "Grenadier", "Musketier"]},
    {"canonical": "Oberschütze", "level": 2, "english": "senior private", "variants": ["Oberschuetze"]},
    {"canonical": "Gefreiter", "level": 3, "english": "lance corporal", "variants": ["Gefr."]},
    {"canonical": "Obergefreiter", "level": 4, "english": "corporal", "variants": ["Obergefr."]},
    {"canonical": "Stabsgefreiter", "level": 5, "english": "staff corporal", "variants": ["Stabsgefr."]},
    {"canonical": "Unteroffizier", "level": 6, "english": "junior NCO / sergeant", "variants": ["Uffz.", "Uffz"]},
    {"canonical": "Unterfeldwebel", "level": 7, "english": "staff sergeant", "variants": ["Unterfw."]},
    {"canonical": "Feldwebel", "level": 8, "english": "sergeant first class", "variants": ["Fw.", "Feldw."]},
    {"canonical": "Oberfeldwebel", "level": 9, "english": "master sergeant", "variants": ["Oberfw."]},
    {"canonical": "Stabsfeldwebel", "level": 10, "english": "senior master sergeant", "variants": ["Stabsfw."]},
    {"canonical": "Hauptfeldwebel", "level": 11, "english": "first sergeant (appointment)", "variants": ["Hauptfw.", "Spieß", "Spiess"]},
    {"canonical": "Fähnrich", "level": 12, "english": "officer cadet (ensign)", "variants": ["Faehnrich", "Fähnr."]},
    {"canonical": "Oberfähnrich", "level": 13, "english": "senior officer cadet", "variants": ["Oberfaehnrich"]},
    {"canonical": "Leutnant", "level": 14, "english": "second lieutenant", "variants": ["Lt.", "Leutn."]},
    {"canonical": "Oberleutnant", "level": 15, "english": "first lieutenant", "variants": ["Oblt.", "Oberlt."]},
    {"canonical": "Hauptmann", "level": 16, "english": "captain", "variants": ["Hptm.", "Rittmeister"]},
    {"canonical": "Major", "level": 17, "english": "major", "variants": ["Maj."]},
    {"canonical": "Oberstleutnant", "level": 18, "english": "lieutenant colonel", "variants": ["Oberstlt.", "Obstlt."]},
    {"canonical": "Oberst", "level": 19, "english": "colonel", "variants": ["Obst."]},
    {"canonical": "Generalmajor", "level": 20, "english": "brigadier general", "variants": ["Gen.Maj.", "Generalmaj."]},
    {"canonical": "Generalleutnant", "level": 21, "english": "major general", "variants": ["Gen.Lt.", "Generallt."]},
    {"canonical": "General der Infanterie", "level": 22, "english": "general (branch general)", "variants": ["General der Artillerie", "General der Kavallerie", "General der Panzertruppe", "General"]},
    {"canonical": "Generaloberst", "level": 23, "english": "colonel general", "variants": ["Gen.Oberst", "Generaloberst"]},
    {"canonical": "Generalfeldmarschall", "level": 24, "english": "field marshal", "variants": ["GFM", "Feldmarschall", "Generalfeldm."]},
    {"canonical": "Reichsmarschall", "level": 25, "english": "Reich Marshal (Göring, unique)", "variants": ["Reichsmarschall des Grossdeutschen Reiches"]},
]

ALL_LADDERS: dict[str, list[dict]] = {
    "SA": SA_RANKS,
    "SS": SS_RANKS,
    "Wehrmacht": WEHRMACHT_RANKS,
}

# Lookup indexes (built once)
def _build_index() -> dict[str, list[tuple[str, str, int]]]:
    """norm_form -> list of (org, canonical, level). Lists handle ambiguity."""
    idx: dict[str, list[tuple[str, str, int]]] = {}
    for org, ladder in ALL_LADDERS.items():
        for rank in ladder:
            forms = [rank["canonical"], *rank.get("variants", [])]
            for f in forms:
                key = _norm(f)
                idx.setdefault(key, []).append((org, rank["canonical"], rank["level"]))
    return idx


_RANK_INDEX = _build_index()


def identify_rank_org(rank_text: str) -> Optional[tuple[str, str, int]]:
    """Resolve a rank mention to ``(organization, canonical_rank, level)``.

    The SA-/SS- prefix disambiguates the many shared ``...führer`` compounds.
    For an unprefixed shared compound the result is reported as organization
    ``"SA/SS"`` (ambiguous) so the caller can treat it as weak evidence.

    Returns ``None`` if the text is not a recognized rank.
    """
    if not rank_text:
        return None
    key = _norm(rank_text)
    matches = _RANK_INDEX.get(key)
    if not matches:
        # Try stripping a leading org prefix that was glued differently.
        for prefix in ("sa", "ss"):
            if key.startswith(prefix):
                sub = _RANK_INDEX.get(key[len(prefix):])
                if sub:
                    org = prefix.upper()
                    for m in sub:
                        if m[0] == org:
                            return m
                    return (org, sub[0][1], sub[0][2])
        return None

    if len(matches) == 1:
        return matches[0]

    # Ambiguous: same surface form in multiple ladders.
    orgs = {m[0] for m in matches}
    if orgs == {"SA", "SS"}:
        # Report ambiguous; use the SS canonical name and shared level.
        ss = next(m for m in matches if m[0] == "SS")
        return ("SA/SS", ss[1], ss[2])
    # Prefer the explicitly prefixed reading if the text carried one.
    return matches[0]


def rank_level(rank_text: str) -> Optional[int]:
    """Convenience: return just the seniority level of a rank, or None."""
    res = identify_rank_org(rank_text)
    return res[2] if res else None
