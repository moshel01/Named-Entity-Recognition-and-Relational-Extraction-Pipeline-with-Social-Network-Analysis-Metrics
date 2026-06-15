# fastcoref coreference microservice.
#
# Runs in an ISOLATED venv: fastcoref needs transformers <5, which conflicts with
# the main pipeline's GLiNER2 (transformers 5.x). Keeping coref out-of-process
# lets each side pin its own deps, and the service env is light (fastcoref +
# FastAPI, no spaCy/GLiNER/langextract). The pipeline POSTs chunk text here and
# re-attaches the returned clusters; an unreachable service falls back to
# in-process fastcoref, then the heuristic resolver, so this is optional.
#
#   python -m venv .venv-coref
#   . .venv-coref/Scripts/activate              # Windows; or .venv-coref/bin/activate
#   pip install -r services/requirements-coref.txt
#   uvicorn services.coref_service:app --host 127.0.0.1 --port 8000
#
# Then point the pipeline at it (config):
#   coreference:
#     enabled: true
#     pronoun_resolution: true
#     service_url: "http://127.0.0.1:8000"
#
# Env knobs: COREF_MODEL (default biu-nlp/f-coref), COREF_DEVICE (default cpu).

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="fastcoref service")

_MODEL = os.environ.get("COREF_MODEL", "biu-nlp/f-coref")
_DEVICE = os.environ.get("COREF_DEVICE", "cpu")
_model = None


def _get_model():
    global _model
    if _model is None:
        from fastcoref import FCoref
        _model = FCoref(model_name_or_path=_MODEL, device=_DEVICE)
    return _model


class ResolveRequest(BaseModel):
    texts: list[str]


class ResolveResponse(BaseModel):
    # texts -> clusters -> spans -> [start_char, end_char]
    clusters: list[list[list[list[int]]]]


@app.get("/health")
def health():
    return {"status": "ok", "model": _MODEL, "device": _DEVICE,
            "loaded": _model is not None}


@app.post("/resolve", response_model=ResolveResponse)
def resolve(req: ResolveRequest):
    """Char-offset coref clusters per input text. Mirrors fastcoref's
    ``predict(...).get_clusters(as_strings=False)`` so the pipeline's cluster
    re-attachment logic is identical whether coref runs here or in-process."""
    if not req.texts:
        return {"clusters": []}
    preds = _get_model().predict(texts=req.texts)
    out = [[[list(span) for span in cluster]
            for cluster in pred.get_clusters(as_strings=False)]
           for pred in preds]
    return {"clusters": out}
