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
# e.g. `}{`, `] [`, `" "`, `true {`. The first group only matches real value
# terminators (close brackets, a string close, or the literal words
# true/false/null) so individual letters inside strings are never matched.
_MISSING_COMMA_RE = re.compile(
    # (?<!\\) so an escaped quote (`\""`) is never treated as a value
    # terminator - inserting a comma there manufactures `\",",` garbage.
    r'((?<!\\)["}\]]|\btrue\b|\bfalse\b|\bnull\b)\s*\n?\s*(["{\[])'
)
# Digits are terminators only across a newline. Same-line `1"` is usually a
# digit inside a string right before its close quote ("born 1903") - treating
# it as a missing comma writes a comma INTO the string.
_MISSING_COMMA_NUM_RE = re.compile(r'(\d)\s*\n(\s*["{\[])')


def _outermost_span(text: str) -> str:
    """Trim a blob to its outermost { } / [ ] span, dropping prose either side."""
    starts = [i for i, c in enumerate(text) if c in "{["]
    ends = [i for i, c in enumerate(text) if c in "}]"]
    if starts and ends:
        return text[starts[0]: ends[-1] + 1]
    return text.strip()


_DANGLING_KEY_RE = re.compile(r',?\s*"[^"\n]*"?\s*:?\s*$')
# Content beginning/ending with a straight quote doubles up against the JSON
# delimiter (`: ""Now go on!" ..."` / `...he said.""`). Escape the inner one.
# Lookarounds keep legitimate empty strings (`: "",`) untouched.
_DOUBLED_OPEN_RE = re.compile(r'(:\s*)""(?=[^\s,}\]])')
_DOUBLED_CLOSE_RE = re.compile(r'(?<=[^\s:,{\[\\])""(\s*[,}\]\n])')
# Python-style literals in value position ("directed": False).
_PY_LITERALS = [(re.compile(r"(:\s*)True\b"), r"\1true"),
                (re.compile(r"(:\s*)False\b"), r"\1false"),
                (re.compile(r"(:\s*)None\b"), r"\1null")]
# Unquoted bare-word value ("type": enemy,). Quotes it; skips JSON keywords.
_BARE_VALUE_RE = re.compile(
    r'(:\s*)(?!true\b|false\b|null\b)([A-Za-z_][A-Za-z0-9_\- ]*?)(\s*[,}\]])')
# Model commentary between a string close and the delimiter:
# `"...house arrest" (implied residence),`. Drop the parenthetical.
_PAREN_ANNOTATION_RE = re.compile(r'"\s*\([^()"\n]*\)(\s*[,}\]])')
# Reasoning leaked after a string close via an arrow: `"NSDSP" -> Note: NSDAP is
# the party. NSV was... assuming standard entities:` running to the array/object
# close. A weak model thinking out loud inside the JSON. Drop arrow-to-delimiter;
# the char class excludes quotes/brackets so it can't swallow a following value.
_ARROW_ANNOTATION_RE = re.compile(r'"\s*->[^"\[\]{}]*?(\s*[,}\]])')
# Bare prose leaked after a comma inside an array, same line, before the next
# element: `"Frohlichsein", Frohlichsein is an activity.\n "Liberalismus",`. Only
# fires when a non-space, non-quote char follows the comma (legit `, "x"` has a
# quote there, multiline `,\n "x"` has a newline) so it can't eat real values.
# Leaves a trailing comma the trailing-comma level cleans up.
_ARRAY_PROSE_RE = re.compile(r',[ \t]*[^\s"\]}][^"\n]*(\r?\n)')
# A stray sentence-punctuation char the model leaks right after a value's close
# quote, before the next member: `"...All-powerful;".` then a newline + the next
# key. `. ; : ( )` are never legal between a value and the next member - the parens
# are a parenthetical whose close leaked outside the quote (`"(1948 Indian film")`
# - the model shut the string early and left the `)` behind). Two shapes: before
# the next KEY it is the missing comma; before a CLOSE bracket just drop it. The
# lookahead (a key-quote, i.e. `"..."` then `:`, or `}`/`]`) keeps punctuation
# INSIDE strings - always followed by more content - untouched.
_STRAY_PUNCT_BEFORE_KEY_RE = re.compile(
    r'("\s*)[.;:()]+(\s*\n\s*"(?:[^"\\]|\\.)*"\s*:)')
_STRAY_PUNCT_BEFORE_CLOSE_RE = re.compile(r'("\s*)[.;:()]+(\s*[}\]])')
# Several comma-separated strings as one value, no array brackets:
# `"evidence": "s1", "s2", "s3",`. Merge them. The anchor on `:` keeps real
# array elements out; the lookahead keeps the next key (always followed by a
# colon) out. Applied repeatedly for 3+ strings.
_MULTI_STRING_VALUE_RE = re.compile(
    r'(:\s*"(?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*"(?!\s*:))')
# Value opens with an escaped quote: `"evidence": \"text...`. qwen pre-escapes
# the JSON delimiter when the evidence is itself a book quote. Outside a string
# a backslash is invalid, so this only fires on real malformations. Strip the
# opening backslash and let _escape_inner_quotes find the true close. (`\\"` -
# an escaped backslash then quote - does not match: the pattern needs `\` then
# `"` adjacent.)
_VALUE_OPEN_ESCAPED_RE = re.compile(r'(:\s*)\\"')


def _fix_literal_values(text: str) -> str:
    for pat, rep in _PY_LITERALS:
        text = pat.sub(rep, text)
    return _BARE_VALUE_RE.sub(r'\1"\2"\3', text)


def _fix_escaped_delimiters(text: str) -> str:
    """Fix strings delimited by escaped quotes: `"evidence": \\"text...\\",`.

    qwen emits the value's surrounding quotes pre-escaped (outside any string,
    where `\\` is invalid JSON). Outside a string, `\\"` becomes an opening
    quote; inside such a string, a `\\"` standing directly before a value
    terminator (`,` `}` `]` or end of line) becomes the closing quote.
    """
    out: list[str] = []
    in_str = False
    opened_by_escape = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if not in_str:
            if ch == "\\" and nxt == '"':
                out.append('"')
                in_str = True
                opened_by_escape = True
                i += 2
                continue
            if ch == '"':
                in_str = True
                opened_by_escape = False
        else:
            if ch == "\\" and nxt == '"':
                j = i + 2
                while j < n and text[j] in " \t":
                    j += 1
                follow = text[j] if j < n else ""
                # `\"` before a bare quote is content - the bare quote closes
                # the string naturally (`...caves.\""` = quoted dialogue).
                if follow == '"':
                    out.append('\\"')
                    i += 2
                    continue
                # `\"` before a colon is a mis-escaped KEY closer
                # (`"evidence\":`) - keys never contain escaped quotes.
                # Before , } ] or end of line it is a mis-escaped VALUE closer,
                # but only when the string was opened by `\"` too; a normally
                # opened string can legitimately contain `\",` in content.
                if follow == ":" or (opened_by_escape and
                                     (follow in ",}]\n" or follow == "")):
                    out.append('"')
                    in_str = False
                    opened_by_escape = False
                    i += 2
                    continue
                out.append('\\"')
                i += 2
                continue
            if ch == "\\":
                out.append(text[i:i + 2])
                i += 2
                continue
            if ch == '"':
                in_str = False
                opened_by_escape = False
        out.append(ch)
        i += 1
    return "".join(out)


def _escape_inner_quotes(text: str) -> str:
    """Escape unescaped quotes inside string values (`"He said "no" here"`).

    State machine: inside a string, a quote whose next non-space char is not a
    JSON delimiter cannot be the closing quote - escape it.
    """
    out: list[str] = []
    in_str = False
    escape = False
    n = len(text)
    for i, ch in enumerate(text):
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            if not in_str:
                in_str = True
                out.append(ch)
                continue
            j = i + 1
            while j < n and text[j] in " \t":
                j += 1
            nxt = text[j] if j < n else ""
            if nxt in ",:}]\n" or nxt == "":
                in_str = False
                out.append(ch)
            else:
                out.append('\\"')        # content quote, not a terminator
            continue
        out.append(ch)
    return "".join(out)


# Curly/smart double quotes a model uses as a string delimiter (`: "When ...,"
# noted ...,"`) - JSON needs straight quotes. Singles/guillemets included; smart
# singles stay valid string content so they are left alone.
_SMART_DQUOTE = {ord(c): '"' for c in "“”„‟«»″"}


def _normalize_smart_quotes(text: str) -> str:
    return text.translate(_SMART_DQUOTE)


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


def _repair_blob(blob: str) -> Optional[Any]:
    """Run the repair ladder on one candidate blob; None if unrecoverable.

    No failure dump here - repair_json tries several candidates and dumps once
    only if all of them fail.
    """
    blob = _outermost_span(blob)

    # Level 1: direct parse of the trimmed blob.
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

    # Level 2.5: doubled quotes where content starts/ends with a quote char.
    # Must run before the missing-comma level, which would corrupt `""` pairs.
    fixed = _DOUBLED_CLOSE_RE.sub(r'\\""\1', _DOUBLED_OPEN_RE.sub(r'\1"\\"', fixed))
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Level 3: insert missing commas.
    fixed3 = _MISSING_COMMA_NUM_RE.sub(r"\1,\2", _MISSING_COMMA_RE.sub(r"\1,\2", fixed))
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

    # Level 4.5: Python literals (False/None) and unquoted bare-word values.
    # Not string-aware, so keep fixed4 pristine for the later fallback levels.
    fixed45 = _fix_literal_values(fixed4)
    try:
        return json.loads(fixed45)
    except json.JSONDecodeError:
        pass

    # Level 4.6: escaped-quote string delimiters (`: \"text\",`).
    fixed46 = _fix_escaped_delimiters(fixed4)
    try:
        return json.loads(fixed46)
    except json.JSONDecodeError:
        pass

    # Level 4.7: parenthetical annotation after a string close. Must run
    # before level 5, whose state machine would swallow the close quote.
    fixed47 = _PAREN_ANNOTATION_RE.sub(r'"\1', fixed4)
    try:
        return json.loads(fixed47)
    except json.JSONDecodeError:
        pass

    # Level 4.72: arrow-annotation reasoning leak (`"x" -> Note: ...`). Chain off
    # 4.7 so a blob carrying both annotation shapes is cleaned in one pass.
    fixed472 = _ARROW_ANNOTATION_RE.sub(r'"\1', fixed47)
    try:
        return json.loads(fixed472)
    except json.JSONDecodeError:
        pass

    # Level 4.73: bare prose between array elements (weak model commenting on each
    # alias). Chain off 4.72; trailing comma it leaves is cleaned at the next try.
    fixed473 = _TRAILING_COMMA_RE.sub(r"\1", _ARRAY_PROSE_RE.sub(r",\1", fixed472))
    try:
        return json.loads(fixed473)
    except json.JSONDecodeError:
        pass

    # Level 4.75: stray sentence punctuation the model left between a value's
    # close quote and the next member (`"...powerful;".` + newline + key). Make
    # it the missing comma before a key, drop it before a close bracket.
    fixed475 = _STRAY_PUNCT_BEFORE_CLOSE_RE.sub(r'\1\2',
               _STRAY_PUNCT_BEFORE_KEY_RE.sub(r'\1,\2', fixed4))
    try:
        return json.loads(fixed475)
    except json.JSONDecodeError:
        pass

    # Level 4.8: multi-string value -> single string, " ... " separated
    # (matches the model's own ellipsis style, keeps the verbatim check happy).
    fixed48 = fixed4
    for _ in range(30):
        merged = _MULTI_STRING_VALUE_RE.sub(r'\1 ... \2', fixed48)
        if merged == fixed48:
            break
        fixed48 = merged
    try:
        return json.loads(fixed48)
    except json.JSONDecodeError:
        pass

    # Level 5: escape unescaped quotes inside string values.
    fixed5 = _escape_inner_quotes(fixed4)
    try:
        return json.loads(fixed5)
    except json.JSONDecodeError:
        pass

    # Level 5.5: escaped delimiters THEN inner-quote escaping. A response can
    # mix a mis-escaped value closer (`\",`) on one edge with an unescaped
    # dialogue quote on another; neither 4.6 nor 5 alone fixes both, composed
    # they do.
    fixed55 = _escape_inner_quotes(_fix_escaped_delimiters(fixed4))
    try:
        return json.loads(fixed55)
    except json.JSONDecodeError:
        pass

    # Level 5.6: value-opening escaped quote, then inner-quote escaping. Handles
    # book quotes the model wraps in `\"...\"` with embedded dialogue quotes and
    # commas (`: \"...quietly,\" said Gandalf.\""`) that 4.6 mis-segments.
    fixed56 = _escape_inner_quotes(_VALUE_OPEN_ESCAPED_RE.sub(r'\1"', fixed4))
    try:
        return json.loads(fixed56)
    except json.JSONDecodeError:
        pass

    # Level 5.7: curly/smart quotes used as a value delimiter (`: "When ...,"
    # noted ...,"`). Straighten them, then let the inner-quote escaper tell the
    # real closing delimiter from the quote marks now inside the value.
    fixed57 = _escape_inner_quotes(_normalize_smart_quotes(fixed4))
    try:
        return json.loads(fixed57)
    except json.JSONDecodeError:
        pass

    # Level 6: truncation - close an open string and any open brackets.
    fixed6 = _balance_and_close(fixed5)
    try:
        return json.loads(fixed6)
    except json.JSONDecodeError:
        pass

    # Level 7: drop a dangling trailing key first, then close.
    fixed7 = _balance_and_close(_DANGLING_KEY_RE.sub("", fixed5))
    try:
        return json.loads(fixed7)
    except json.JSONDecodeError:
        pass

    # Level 8: as 7 but without inner-quote escaping (in case it misfired).
    fixed8 = _balance_and_close(_DANGLING_KEY_RE.sub("", fixed4))
    try:
        return json.loads(fixed8)
    except json.JSONDecodeError:
        return None


def repair_json(text: str) -> Optional[Any]:
    """Attempt to parse possibly-malformed JSON, returning the object or None.

    A reasoning model (qwen3.5, with think:false ignored) emits visible working
    with several ```json blocks: a discarded first attempt, then the corrected
    answer last. Try each fenced block last-first, then the whole response; dump
    the raw only if every candidate fails.
    """
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

    # Candidate blobs: each fenced block (last first), then the whole response.
    candidates = list(reversed(_FENCE_RE.findall(text)))
    candidates.append(text)
    for cand in candidates:
        obj = _repair_blob(cand)
        if obj is not None:
            return obj

    # All candidates exhausted; save the raw for offline analysis.
    try:
        json.loads(_balance_and_close(_outermost_span(text)))
    except json.JSONDecodeError as exc:
        _dump_failure(text, exc)
    return None


def _dump_failure(raw: str, exc: Exception, cap: int = 50) -> None:
    """Save an unrepairable LLM response for offline analysis (bounded)."""
    try:
        from pathlib import Path
        import hashlib
        d = Path("scratch/json_failures")
        d.mkdir(parents=True, exist_ok=True)
        existing = list(d.glob("*.txt"))
        name = hashlib.md5(raw.encode("utf-8", "replace")).hexdigest()[:12] + ".txt"
        if len(existing) < cap and not (d / name).exists():
            (d / name).write_text(raw, encoding="utf-8")
        logger.warning("JSON repair exhausted all levels (%s); raw saved to %s",
                       exc, d / name)
    except Exception:  # noqa: BLE001 - diagnostics must never break extraction
        logger.warning("JSON repair exhausted all levels: %s", exc)
