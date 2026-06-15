# GermEval 2014 NER (gwlms/germeval2014 parquet mirror - the original script
# dataset no longer loads under datasets 4.x). Modern German from news/wiki:
# together with HIPE (historical OCR newspapers) it brackets the Abel corpus
# register from both sides. Entities only - no relation gold.
#
# Sentences are grouped into pseudo-docs so corpus-level scoring has document
# units. Only core PER/LOC/ORG spans enter the gold: OTH and the deriv/part
# variants ("deutschen" = LOCderiv) are adjectives/derivations, not nodes a
# social network would carry.

from __future__ import annotations

import logging

from .common import BenchDoc, BenchEntity

logger = logging.getLogger(__name__)

_HF_ID = "gwlms/germeval2014"

TYPE_MAP = {"PER": "PERSON", "LOC": "LOCATION", "ORG": "ORG"}

DEFAULT_TARGET_TYPES = ["PERSON", "ORG", "LOCATION"]
DEFAULT_SPACY_MODEL = "de_core_news_lg"
DEFAULT_GLINER_MODEL = "fastino/gliner2-multi-v1"

_SENTS_PER_DOC = 25


def _decode_bio(tokens: list[str], tags: list[str]) -> list[tuple[str, str]]:
    """BIO -> (surface, base_type) spans, core types only."""
    spans: list[tuple[str, str]] = []
    cur_toks: list[str] = []
    cur_type = ""

    def flush() -> None:
        nonlocal cur_toks, cur_type
        if cur_toks and cur_type in TYPE_MAP:
            spans.append((" ".join(cur_toks), cur_type))
        cur_toks, cur_type = [], ""

    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            flush()
            cur_type = tag[2:]
            cur_toks = [tok]
        elif tag.startswith("I-") and cur_toks and tag[2:] == cur_type:
            cur_toks.append(tok)
        else:
            flush()
    flush()
    # deriv/part variants carry suffixes ("LOCderiv") -> filtered by TYPE_MAP.
    return spans


def load(split: str = "validation", limit: int = 0, path: str = "") -> list[BenchDoc]:
    """``limit`` counts pseudo-docs of ~25 sentences, matching other adapters."""
    import datasets

    ds = datasets.load_dataset(_HF_ID, split=split)
    names = ds.features["ner_tags"].feature.names

    docs: list[BenchDoc] = []
    n_docs = (len(ds) + _SENTS_PER_DOC - 1) // _SENTS_PER_DOC
    if limit and limit > 0:
        n_docs = min(n_docs, limit)
    for d in range(n_docs):
        rows = ds.select(range(d * _SENTS_PER_DOC,
                               min((d + 1) * _SENTS_PER_DOC, len(ds))))
        sents: list[str] = []
        entities: dict[tuple[str, str], BenchEntity] = {}
        for row in rows:
            toks = row["tokens"]
            tags = [names[t] for t in row["ner_tags"]]
            sents.append(" ".join(toks))
            for surface, base in _decode_bio(toks, tags):
                etype = TYPE_MAP[base]
                entities.setdefault((surface.lower(), etype),
                                    BenchEntity(name=surface, type=etype))
        docs.append(BenchDoc(
            doc_id=f"{d:05d}_germeval_{split}",
            text="\n".join(sents),
            entities=list(entities.values()),
            relations=[],
        ))
    logger.info("GermEval: %d docs, %d gold entities",
                len(docs), sum(len(x.entities) for x in docs))
    return docs
