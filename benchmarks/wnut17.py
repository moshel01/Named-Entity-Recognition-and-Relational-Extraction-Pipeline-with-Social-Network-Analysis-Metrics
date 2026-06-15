# WNUT-17 emerging-entity NER (noisy user text: Twitter, Reddit, YouTube,
# StackExchange). The opposite register from CoNLL newswire - informal, OOV-heavy
# surface forms - so it stresses recall on unseen names. Types person/location/
# corporation/group/creative-work/product; we score person->PERSON,
# location->LOCATION, corporation+group->ORG (companies, teams, bands), dropping
# creative-work/product (not network nodes). tner parquet mirror stores raw ints;
# the id->tag map is fetched from the README (common.hf_iob_label_map). NOTE tner
# orders O last (O=12), not first - the fallback reflects that.

from __future__ import annotations

import logging

from .common import (build_ner_docs, classlabel_names, hf_iob_label_map,
                     load_token_dataset)

logger = logging.getLogger(__name__)

_HF_ID = "tner/wnut2017"

# Confirmed full map (tner/wnut2017 README): O is 12, not 0. Fallback only.
_FALLBACK_ID2TAG = {0: "B-corporation", 1: "B-creative-work", 2: "B-group",
                    3: "B-location", 4: "B-person", 5: "B-product",
                    6: "I-corporation", 7: "I-creative-work", 8: "I-group",
                    9: "I-location", 10: "I-person", 11: "I-product", 12: "O"}

TYPE_MAP = {"person": "PERSON", "location": "LOCATION",
            "corporation": "ORG", "group": "ORG"}

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
    # Posts are short; group fewer per pseudo-doc so a "document" is a handful of
    # posts rather than 25 unrelated ones.
    docs = build_ner_docs(sents, TYPE_MAP, dataset="wnut17", split=split,
                          sents_per_doc=10, limit=limit)
    n_ent = sum(len(d.entities) for d in docs)
    if docs and n_ent == 0:
        logger.warning("WNUT-17: 0 gold entities - check the tag id map.")
    logger.info("WNUT-17: %d docs, %d gold entities", len(docs), n_ent)
    return docs
