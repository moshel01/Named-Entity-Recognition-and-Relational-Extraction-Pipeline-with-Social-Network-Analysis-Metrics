# Recover malformed LLM JSON: fences, commas, control chars, truncation.

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# Missing comma between a value-ending token and the start of the next value,
# e.g. `}{`, `] [`, `" "`, `1 "b"`, `true {`. The first group only matches real
# value terminators (close brackets, a string close, a number, or the literal
# words true/false/null) so individual letters inside strings are never matched.
_MISSING_COMMA_RE = re.compile(
    r'(["}\]\d]|\btrue\b|\bfalse\b|\bnull\b)\s*\n?\s*(["{\[])'
)


def _extract_json_blob(text: str) -> str:
    """Return the substring most likely to be the JSON payload."""
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    # Find the outermost { } or [ ] span.
    starts = [i for i, c in enumerate(text) if c in "{["]
    ends = [i for i, c in enumerate(text) if c in "}]"]
    if starts and ends:
        return text[starts[0]: ends[-1] + 1]
    return text.strip()


def _balance_and_close(text: str) -> str:
    """Truncate at last sane point and append closing brackets in order."""
    stack: list[str] = []
    in_str = False
    escape = False
    last_safe = 0
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                last_safe = i + 1
        elif ch == "," and not stack:
            continue
    # Keep up to the last balanced top-level close if we have one.
    if last_safe:
        candidate = text[:last_safe]
    else:
        candidate = text
    # Re-scan candidate for any still-open brackets and close them.
    stack = []
    in_str = False
    escape = False
    for ch in candidate:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        candidate += '"'
    closers = {"{": "}", "[": "]"}
    candidate += "".join(closers[c] for c in reversed(stack))
    return candidate


def repair_json(text: str) -> Optional[Any]:
    """Attempt to parse possibly-malformed JSON, returning the object or None."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None

    # Level 0: direct.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Level 1: extract blob.
    blob = _extract_json_blob(text)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass

    # Level 2: drop trailing commas.
    fixed = _TRAILING_COMMA_RE.sub(r"\1", blob)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Level 3: insert missing commas.
    fixed3 = _MISSING_COMMA_RE.sub(r"\1,\2", fixed)
    try:
        return json.loads(fixed3)
    except json.JSONDecodeError:
        pass

    # Level 4: strip control characters.
    fixed4 = _CONTROL_RE.sub("", fixed3)
    try:
        return json.loads(fixed4)
    except json.JSONDecodeError:
        pass

    # Level 5: truncate + close.
    fixed5 = _balance_and_close(fixed4)
    try:
        return json.loads(fixed5)
    except json.JSONDecodeError as exc:
        logger.warning("JSON repair exhausted all levels: %s", exc)
        return None
