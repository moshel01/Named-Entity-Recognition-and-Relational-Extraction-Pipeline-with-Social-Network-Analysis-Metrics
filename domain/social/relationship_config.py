# Social domain relations = the platform-structural ties (set by core/social, not the
# LLM) PLUS the generic interpersonal/org vocabulary for relations stated INSIDE post
# text (a tweet that says "X met Y"). The structural ones are listed here too so the
# ontology aligner keeps them and the type-signature gate knows them.

from __future__ import annotations

from domain.generic.relationship_config import RELATION_GUIDE as _GEN_GUIDE
from domain.generic.relationship_config import RELATION_ONTOLOGY as _GEN_ONTO

# Platform-structural relations (emitted by the connectors as social_graph edges).
_STRUCTURAL: dict[str, list[str]] = {
    "replied_to": ["replied to", "in reply to", "responded to"],
    "mentions": ["mentioned", "tagged", "at-mentioned"],
    "quoted": ["quoted", "quote tweeted", "quote-posted"],
    "retweeted": ["retweeted", "reposted", "boosted", "shared"],
    "follows": ["follows", "is following", "subscribed to"],
    "posted_in": ["posted in", "submitted to", "commented in"],
}

RELATION_ONTOLOGY: dict[str, list[str]] = {**_GEN_ONTO, **_STRUCTURAL}

RELATION_GUIDE: dict[str, str] = {
    **_GEN_GUIDE,
    "replied_to": "Author replied directly to another user's post (a directed conversational tie).",
    "mentions": "Author @-mentioned/tagged another user in the post text.",
    "quoted": "Author quoted/quote-posted another user's post.",
    "retweeted": "Author reposted/boosted another user's post verbatim.",
    "follows": "Author follows the target account (a subscription tie, if available).",
    "posted_in": "Author posted/commented in this community (subreddit/instance/thread). Feeds the co-affiliation projection so co-posters link.",
}
