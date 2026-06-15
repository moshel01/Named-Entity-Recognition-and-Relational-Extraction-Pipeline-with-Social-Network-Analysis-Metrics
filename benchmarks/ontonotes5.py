# OntoNotes 5.0 English NER (multi-genre: newswire, web, broadcast, telephone).
# 18 entity types; we score PERSON/ORG and GPE+LOC -> LOCATION (the types a
# social network carries). Source is the tner parquet mirror, which stores tags
# as raw ints; the id->tag map is fetched from the dataset README at load time
# (common.hf_iob_label_map), with a confirmed core-type fallback if that scrape
# fails. If the prepared gold shows 0 entities, the map is wrong - check the card.

from __future__ import annotations

import logging

from .common import (build_ner_docs, classlabel_names, hf_iob_label_map,
                     load_token_dataset)

logger = logging.getLogger(__name__)

_HF_ID = "tner/ontonotes5"

# Confirmed core-type ids (tner/ontonotes5 README); rest -> O. Fallback only;
# the README scrape is preferred and supplies the full map.
_FALLBACK_ID2TAG = {4: "B-PERSON", 5: "I-PERSON", 7: "B-GPE", 8: "I-GPE",
                    11: "B-ORG", 12: "I-ORG", 23: "B-LOC", 27: "I-LOC"}

TYPE_MAP = {"PERSON": "PERSON", "ORG": "ORG", "GPE": "LOCATION", "LOC": "LOCATION"}

DEFAULT_TARGET_TYPES = ["PERSON", "ORG", "LOCATION"]
DEFAULT_SPACY_MODEL = "en_core_web_trf"
DEFAULT_GLINER_MODEL = "fastino/gliner2-large-v1"


def load(split: str = "test", limit: int = 0, path: str = "") -> list:
    repo = path or _HF_ID
    ds = load_token_dataset(repo, split)
    tag_field = "tags" if "tags" in ds.features else "ner_tags"
    names = classlabel_names(ds.features[tag_field])
    if names:
        def to_tags(ids):
            return [names[i] for i in ids]
    else:
        id2tag = hf_iob_label_map(repo) or _FALLBACK_ID2TAG

        def to_tags(ids):
            return [id2tag.get(i, "O") for i in ids]

    sents = [(r["tokens"], to_tags(r[tag_field])) for r in ds]
    docs = build_ner_docs(sents, TYPE_MAP, dataset="ontonotes5", split=split,
                          limit=limit)
    n_ent = sum(len(d.entities) for d in docs)
    if docs and n_ent == 0:
        logger.warning("OntoNotes: 0 gold entities - the tag id map is wrong for "
                       "this mirror; check the dataset card.")
    logger.info("OntoNotes5: %d docs, %d gold entities", len(docs), n_ent)
    return docs
