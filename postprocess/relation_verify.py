# LLM relation self-verification. Verbatim grounding (evidence_unverified) only
# checks the evidence is a real span of the source text; it cannot tell whether that
# sentence actually ASSERTS the relation. This pass asks the model, per edge, "does
# this evidence state that <source> <rel> <target>?" and tags the unsupported ones
# (or drops them). The post-hoc half of accuracy, after the pre-emptive ontology +
# type-signature constraints. Opt-in, LLM modes only (duck-types backend._complete),
# guarded + batched. Only LLM-asserted text edges are checked; rule/co-occurrence/
# metadata edges are deterministic and left alone.

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from intelligence.json_repair import repair_json

logger = logging.getLogger(__name__)


def _verdict_key(s: str, t: str, rel: str, evidence: str) -> str:
    raw = "\x1f".join((s, t, rel, (evidence or "").strip()[:300]))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_verdicts(path: Path) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    verdicts[d["k"]] = d["v"]
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 - a corrupt cache just means re-verifying
        logger.warning("verify cache unreadable (%s): %s", path, exc)
    return verdicts

_SYSTEM = (
    "You verify relation extractions against their evidence. Each numbered item gives "
    "a SOURCE entity, a RELATION, a TARGET entity, and the EVIDENCE sentence it was "
    "drawn from. Judge from the text, not world knowledge. Answer \"yes\" when the "
    "evidence states the relation OR clearly paraphrases/implies it (a dated 'joined "
    "the party', 'was attacked by communists' for fought_against KPD - these are yes). "
    "Answer \"no\" only when the evidence is about something else, names the wrong "
    "entity, or states a different relation than the one given. When the evidence "
    "plausibly supports the relation, prefer yes. "
    "Examples: EVIDENCE=\"trat am 15. Marz 1930 in die NSDAP ein\" RELATION=joined "
    "TARGET=NSDAP -> yes. EVIDENCE=\"Tausende Fabriken schlossen ihre Tore\" "
    "RELATION=opposed TARGET=Fabriken -> no (describes factory closures, not "
    "opposition). Output one JSON object mapping each item number to \"yes\" or "
    "\"no\". Output nothing but the JSON object."
)

# Edge sources this pass is allowed to judge: the LLM-asserted text tier. Rule,
# co-occurrence, metadata, inferred edges are deterministic - never LLM-verified.
# Must match the exact edge_source the backends stamp (langextract_backend emits
# "langextract_extracted", not "langextract" - mismatched here, langextract edges
# slip through unverified).
_VERIFIABLE = {"llm_extracted", "langextract_extracted", "", None}


def verify_relations(
    relationships: list, backend: Any, id_to_name: dict[str, str], *,
    batch_size: int = 20, max_relations: int = 0, drop: bool = False,
    cache_path: str | Path | None = None,
) -> tuple[list, int]:
    """Tag/drop LLM edges whose evidence does not assert the relation.

    Returns (relationships, flagged). Tags `verification=unsupported|supported` on
    each judged edge (filterable in Gephi); drops the unsupported when ``drop``.
    Fail-safe: a batch the model botches is skipped, not dropped.

    ``cache_path``: verdict checkpoint (jsonl, keyed by name/rel/evidence hash).
    Verification is the most expensive analyze-stage step (~one call per 20
    edges); without the cache every `--stage analyze` re-run either redoes hours
    of LLM calls or - backend unreachable - silently loses all the tags."""
    complete = getattr(backend, "_complete", None)
    if not callable(complete):
        return relationships, 0
    cands = [r for r in relationships
             if (getattr(r, "evidence", "") or "").strip()
             and (r.attributes or {}).get("edge_source") in _VERIFIABLE]
    if max_relations and len(cands) > max_relations:
        cands = cands[:max_relations]
    if not cands:
        return relationships, 0

    cache = _load_verdicts(Path(cache_path)) if cache_path else {}
    cache_fh = None
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        cache_fh = Path(cache_path).open("a", encoding="utf-8")

    def key_of(r) -> str:
        return _verdict_key(id_to_name.get(r.source, r.source),
                            id_to_name.get(r.target, r.target),
                            r.rel_type, getattr(r, "evidence", "") or "")

    flagged = 0
    drop_ids: set[int] = set()

    def apply(r, ans: str) -> None:
        nonlocal flagged
        if ans.startswith("n"):
            flagged += 1
            if drop:
                drop_ids.add(id(r))
            else:
                r.attributes["verification"] = "unsupported"
        elif ans.startswith("y"):
            r.attributes.setdefault("verification", "supported")

    pending = []
    hits = 0
    for r in cands:
        ans = cache.get(key_of(r))
        if ans is not None:
            apply(r, ans)
            hits += 1
        else:
            pending.append(r)
    if hits:
        logger.info("Relation verify: %d/%d verdicts from cache.", hits, len(cands))

    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        lines = []
        for n, r in enumerate(batch, 1):
            s = id_to_name.get(r.source, r.source)
            t = id_to_name.get(r.target, r.target)
            ev = (r.evidence or "").strip().replace("\n", " ")[:300]
            lines.append(f'{n}. SOURCE="{s}" RELATION="{r.rel_type}" TARGET="{t}" EVIDENCE="{ev}"')
        try:
            raw = complete(_SYSTEM, "Verify each item:\n" + "\n".join(lines))
        except Exception as exc:  # noqa: BLE001 - one batch failing must not drop edges
            logger.warning("relation verify batch failed: %s", exc)
            continue
        obj = repair_json(raw)
        if not isinstance(obj, dict):
            continue
        for n, r in enumerate(batch, 1):
            ans = str(obj.get(str(n), obj.get(n, ""))).strip().lower()
            apply(r, ans)
            if cache_fh and (ans.startswith("n") or ans.startswith("y")):
                cache_fh.write(json.dumps(
                    {"k": key_of(r), "v": "no" if ans.startswith("n") else "yes"}) + "\n")
        if cache_fh:
            cache_fh.flush()
    if cache_fh:
        cache_fh.close()
    if drop and drop_ids:
        relationships = [r for r in relationships if id(r) not in drop_ids]
    logger.info("Relation verify: %d/%d LLM edges unsupported by evidence (%s).",
                flagged, len(cands), "dropped" if drop else "tagged")
    return relationships, flagged
