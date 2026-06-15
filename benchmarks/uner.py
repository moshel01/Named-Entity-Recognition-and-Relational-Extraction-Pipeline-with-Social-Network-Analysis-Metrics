# Universal NER (UNER v1) - PER/ORG/LOC spans on Universal Dependencies
# treebanks, one scheme across many languages. The HF repo (universalner/
# universal_ner) is script-only with no data files and no parquet branch, so it
# does not load under datasets 4.x. Feed it a local treebank instead: download an
# .iob2 from github.com/UniversalNER (e.g. en_ewt/en_ewt-ud-test.iob2) and pass
# --path. Two-column token<TAB>tag, which parse_iob2 reads. Set --spacy-model to
# the language (en_core_web_trf for English, de_core_news_lg for German, ...).

from __future__ import annotations

import logging

from .common import build_ner_docs, parse_iob2

logger = logging.getLogger(__name__)

TYPE_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION"}

DEFAULT_TARGET_TYPES = ["PERSON", "ORG", "LOCATION"]
DEFAULT_SPACY_MODEL = "en_core_web_trf"
DEFAULT_GLINER_MODEL = "fastino/gliner2-large-v1"


def load(split: str = "test", limit: int = 0, path: str = "") -> list:
    if not path:
        raise RuntimeError(
            "uner: pass --path to a UNER .iob2 file. The HF repo is script-only "
            "and won't load under datasets 4.x; download a treebank from "
            "github.com/UniversalNER, e.g. en_ewt/en_ewt-ud-test.iob2.")
    sents = parse_iob2(path)
    docs = build_ner_docs(sents, TYPE_MAP, dataset="uner", split=split, limit=limit)
    logger.info("UNER: %d docs, %d gold entities (from %s)",
                len(docs), sum(len(d.entities) for d in docs), path)
    return docs
