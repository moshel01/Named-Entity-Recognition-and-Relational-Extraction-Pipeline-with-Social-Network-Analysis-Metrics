# CoNLL-2003 English NER (Reuters newswire) - the classic NER baseline. Tags
# PER/ORG/LOC/MISC; we score the three social types (MISC = misc/nationalities,
# not a network node). datasets 4.x dropped script loading, so this pulls the
# auto-converted parquet (see common.load_token_dataset), which keeps the
# ClassLabel ner_tags. Override the HF id with --path if the mirror moves.

from __future__ import annotations

import logging

from .common import build_ner_docs, classlabel_names, load_token_dataset

logger = logging.getLogger(__name__)

_HF_ID = "eriktks/conll2003"

TYPE_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION"}

DEFAULT_TARGET_TYPES = ["PERSON", "ORG", "LOCATION"]
DEFAULT_SPACY_MODEL = "en_core_web_trf"
DEFAULT_GLINER_MODEL = "fastino/gliner2-large-v1"


def load(split: str = "test", limit: int = 0, path: str = "") -> list:
    ds = load_token_dataset(path or _HF_ID, split)
    names = classlabel_names(ds.features["ner_tags"])
    if not names:
        raise RuntimeError("conll2003: ner_tags has no ClassLabel names; pass "
                           "--path to a mirror that exposes them.")
    sents = [(r["tokens"], [names[t] for t in r["ner_tags"]]) for r in ds]
    docs = build_ner_docs(sents, TYPE_MAP, dataset="conll2003", split=split,
                          limit=limit)
    logger.info("CoNLL-2003: %d docs, %d gold entities",
                len(docs), sum(len(d.entities) for d in docs))
    return docs
