# Entity config for the social domain. Users -> PERSON, communities -> ORG (set by the
# connectors). Stopwords reuse the generic literary/common-noun list plus social chrome.

from __future__ import annotations

from domain.generic.entity_config import ENTITY_SUBTYPES as _GEN_SUBTYPES
from domain.generic.entity_config import STOPWORDS as _GEN_STOPWORDS

LABEL_OVERRIDES: dict[str, str] = {}

DEFAULT_TYPES: list[str] = ["PERSON", "ORG", "LOCATION", "EVENT"]

# Social chrome the NER/LLM mislabels as entities.
STOPWORDS: set[str] = set(_GEN_STOPWORDS) | {
    "op", "edit", "tldr", "tl;dr", "imo", "imho", "fyi", "afaik", "lol", "thread",
    "post", "comment", "reply", "upvote", "downvote", "karma", "sub", "subreddit",
    "dm", "rt", "via", "follow", "like", "share", "repost", "deleted", "removed",
}

ENTITY_SUBTYPES: dict[str, list[str]] = {
    **_GEN_SUBTYPES,
    "PERSON": ["user", "public_figure", "organization_account", "bot", "other"],
    "ORG": ["community", "subreddit", "instance", "company", "media", "political", "other"],
}
