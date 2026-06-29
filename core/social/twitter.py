# Twitter / X connector via the OFFICIAL API v2 only. No UI scraping (login-walled,
# anti-botted, account-banning, and breaks weekly). Needs a bearer token in
# $TWITTER_BEARER_TOKEN (or opts["bearer_token"]). HONEST LIMIT: X's free tier is
# essentially write-only - reads need at least Basic ($100/mo at time of writing). This
# adapter is correct and ToS-compliant; whether it returns anything depends on your tier.
#
# target forms:
#   "from:nasa"                  -> a recent-search query (any valid v2 query works)
#   "@nasa"                      -> rewritten to from:nasa
#   "climate policy"             -> a keyword search
# Edges: author -> in_reply_to user (replied_to), author -> @mentions (mentions),
# author -> #hashtag is NOT emitted as posted_in (a hashtag isn't a closed community);
# use the query itself as the community node so a pull's tweets co-affiliate.

from __future__ import annotations

import json
import logging
import os

from .base import SocialPost, extract_mentions

logger = logging.getLogger(__name__)

_SEARCH = "https://api.twitter.com/2/tweets/search/recent"
_FIELDS = ("tweet.fields=created_at,author_id,public_metrics,entities,referenced_tweets"
           "&expansions=author_id,referenced_tweets.id.author_id"
           "&user.fields=username&max_results=")


def _http_get_factory(bearer: str):
    def _get(url: str, timeout: int = 30) -> str:
        import requests
        resp = requests.get(url, headers={"Authorization": f"Bearer {bearer}",
                                          "User-Agent": "SNA-Extraction-Pipeline/1.0"},
                            timeout=timeout)
        resp.raise_for_status()
        return resp.text
    return _get


def fetch(target: str, *, limit: int = 100, depth: int = 1, fetch=None,
          bearer_token: str = "", **_) -> list[SocialPost]:
    """Recent-search the v2 API for `target`. Needs a bearer token (env or opts)."""
    q = (target or "").strip()
    if q.startswith("@"):
        q = "from:" + q[1:]
    if not q:
        raise ValueError("twitter target must be a query, e.g. twitter:from:nasa")
    get = fetch
    if get is None:
        bearer = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN", "")
        if not bearer:
            raise ValueError(
                "twitter needs a bearer token: set $TWITTER_BEARER_TOKEN (official API "
                "v2). The free tier is read-limited - a Basic tier or higher is required "
                "for search to return results.")
        get = _http_get_factory(bearer)

    n = max(10, min(limit, 100))  # v2 recent search: 10..100 per page
    import urllib.parse
    url = f"{_SEARCH}?query={urllib.parse.quote(q)}&{_FIELDS}{n}"
    try:
        payload = json.loads(get(url))
    except Exception as exc:  # noqa: BLE001
        logger.warning("twitter: search failed for %r: %s", q, exc)
        return []

    users = {u["id"]: u.get("username", "")
             for u in ((payload.get("includes") or {}).get("users") or [])}
    community = f"query:{q}"
    posts: list[SocialPost] = []
    for tw in (payload.get("data") or []):
        author = users.get(tw.get("author_id", ""), tw.get("author_id", ""))
        text = tw.get("text", "") or ""
        ents = tw.get("entities") or {}
        mentions = [m.get("username", "") for m in (ents.get("mentions") or [])
                    if m.get("username")] or extract_mentions(text)
        parent_author = ""
        for ref in (tw.get("referenced_tweets") or []):
            if ref.get("type") == "replied_to":
                parent_author = users.get(ref.get("id", ""), "")  # best-effort
        metrics = tw.get("public_metrics") or {}
        posts.append(SocialPost(
            platform="twitter", post_id=str(tw.get("id", "")), author=author,
            text=text, community=community, parent_author=parent_author,
            url=f"https://twitter.com/{author}/status/{tw.get('id','')}",
            score=int(metrics.get("like_count", 0) or 0),
            mentions=mentions,
        ))
    logger.info("twitter: %d tweets for %r.", len(posts), q)
    return posts
