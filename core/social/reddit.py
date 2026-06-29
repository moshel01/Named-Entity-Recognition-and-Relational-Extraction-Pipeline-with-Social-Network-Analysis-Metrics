# Reddit connector via the public read-only JSON endpoints (append .json to any
# listing/permalink). No login: just a descriptive User-Agent, which Reddit requires.
# A subreddit listing gives submissions; each submission's permalink + .json gives the
# comment tree, which we flatten. parent_id on a comment ("t1_x"/"t3_x") is the reply
# edge - resolved to the parent's author in base.posts_to_documents.
#
# Politeness: a small inter-request delay, fail-soft per URL. For heavy/authenticated
# use, set up an OAuth app and pass a token via opts (left as an extension); the public
# endpoints are fine for bounded research pulls.

from __future__ import annotations

import json
import logging
import time

from .base import SocialPost, extract_mentions

logger = logging.getLogger(__name__)

_UA = "SNA-Extraction-Pipeline/1.0 (academic research)"
_BASE = "https://www.reddit.com"


def _http_get(url: str, user_agent: str = _UA, timeout: int = 30) -> str:
    import requests
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _strip_prefix(fullname: str) -> str:
    """t3_abc / t1_def -> abc / def (Reddit thing-id prefix)."""
    return fullname.split("_", 1)[1] if fullname and "_" in fullname else (fullname or "")


def _submission(data: dict) -> SocialPost:
    sub = data.get("subreddit", "")
    return SocialPost(
        platform="reddit",
        post_id=data.get("id", ""),
        author=data.get("author", "") or "",
        title=data.get("title", "") or "",
        text=data.get("selftext", "") or "",
        community=f"r/{sub}" if sub else "",
        url=_BASE + data.get("permalink", "") if data.get("permalink") else "",
        score=int(data.get("score", 0) or 0),
        created_utc=float(data.get("created_utc", 0) or 0),
        mentions=extract_mentions(data.get("selftext", "")),
        extra={"num_comments": data.get("num_comments", 0)},
    )


def _walk_comments(children: list, sub: str, out: list[SocialPost]) -> None:
    for ch in children or []:
        if ch.get("kind") != "t1":
            continue
        d = ch.get("data") or {}
        body = d.get("body", "") or ""
        out.append(SocialPost(
            platform="reddit",
            post_id=d.get("id", ""),
            author=d.get("author", "") or "",
            text=body,
            community=f"r/{sub}" if sub else "",
            parent_id=_strip_prefix(d.get("parent_id", "")),
            url=_BASE + d.get("permalink", "") if d.get("permalink") else "",
            score=int(d.get("score", 0) or 0),
            created_utc=float(d.get("created_utc", 0) or 0),
            mentions=extract_mentions(body),
        ))
        replies = d.get("replies")
        if isinstance(replies, dict):
            _walk_comments((replies.get("data") or {}).get("children", []), sub, out)


def fetch(target: str, *, limit: int = 100, depth: int = 1, fetch=None,
          sort: str = "hot", delay: float = 1.0, **_) -> list[SocialPost]:
    """Pull a subreddit. target='datascience' (or 'r/datascience'). depth>=1 also pulls
    each submission's comment tree (the reply network). limit caps submissions."""
    get = fetch or _http_get
    sub = target.strip().lstrip("/").removeprefix("r/")
    if not sub:
        raise ValueError("reddit target must be a subreddit, e.g. reddit:datascience")
    posts: list[SocialPost] = []
    try:
        raw = get(f"{_BASE}/r/{sub}/{sort}.json?limit={min(limit, 100)}")
        listing = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - fail soft
        logger.warning("reddit: subreddit fetch failed for r/%s: %s", sub, exc)
        return []
    subs = [c.get("data") or {} for c in (listing.get("data") or {}).get("children", [])
            if c.get("kind") == "t3"][:limit]
    for sd in subs:
        posts.append(_submission(sd))
    if depth >= 1:
        for sd in subs:
            pid = sd.get("id", "")
            if not pid:
                continue
            if delay:
                time.sleep(delay)
            try:
                raw = get(f"{_BASE}/r/{sub}/comments/{pid}.json?limit=200")
                blocks = json.loads(raw)
                if isinstance(blocks, list) and len(blocks) > 1:
                    children = (blocks[1].get("data") or {}).get("children", [])
                    _walk_comments(children, sub, posts)
            except Exception as exc:  # noqa: BLE001 - one thread failing isn't fatal
                logger.debug("reddit: comments fetch failed for %s: %s", pid, exc)
    logger.info("reddit: %d posts from r/%s (depth=%d).", len(posts), sub, depth)
    return posts
