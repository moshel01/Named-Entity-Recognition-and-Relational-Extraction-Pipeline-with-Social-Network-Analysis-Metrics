# Map doc_id -> source info (letter_id, author). letter_id = trailing digits of
# the filename (Hoover/Abel id); author via the domain (else filename stem).

from __future__ import annotations

import re
from pathlib import Path

_DIGITS_END = re.compile(r"(\d+)$")


def source_info(source_path: str) -> tuple[str, str, str]:
    name = Path(source_path).name
    stem = name.rsplit(".", 1)[0] if "." in name else name
    m = _DIGITS_END.search(stem)
    return name, stem, (m.group(1) if m else "")


def build_manifest(extractions, domain=None) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for ex in extractions:
        name, stem, letter_id = source_info(ex.source_path)
        author = domain.narrator_name(name, ex.doc_id) if domain is not None else None
        out[ex.doc_id] = {"letter_id": letter_id, "author": author or stem,
                          "filename": name, "source_path": ex.source_path}
    return out
