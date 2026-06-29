# Hacker News connector via the official Firebase API (fully open, no auth). A feed
# (top/new/best) gives story ids; each item gives the story and its comment ids (kids),
# walked breadth-first to a bounded count. The reply network is parent->child (a comment
# replies to its parent's author); co-participation in a thread is captured by posting
# everyone into the story as a community node (HN:<title>), so commenters on one story
# get projected co-affiliated edges.

from __future__ import annotations

import json
import logging
from collections import deque

from .base import SocialPost, extract_mentions

logger = logging.getLogger(__name__)

_BASE = "https://hacker-news.firebaseio.com/v0"
_UA = "SNA-Extraction-Pipeline/1.0 (academic research)"
_FEEDS = {"top": "topstories", "new": "newstories", "best": "beststories",
          "ask": "askstories", "show": "showstories", "job": "jobstories"}


def _http_get(url: str, user_agent: str = _UA, timeout: int = 30) -> str:
    import requests
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _item(get, item_id) -> dict:
    return json.loads(get(f"{_BASE}/item/{item_id}.json")) or {}


def fetch(target: str, *, limit: int = 30, depth: int = 1, fetch=None,
          max_comments: int = 200, **_) -> list[SocialPost]:
    """Pull an HN feed. target='top'|'new'|'best'|... or a numeric story id. limit caps
    stories; depth>=1 walks comments (bounded by max_comments per story)."""
    get = fetch or _http_get
    posts: list[SocialPost] = []
    tgt = (target or "top").strip().lower()
    try:
        if tgt.isdigit():
            story_ids = [int(tgt)]
        else:
            feed = _FEEDS.get(tgt, "topstories")
            story_ids = (json.loads(get(f"{_BASE}/{feed}.json")) or [])[:limit]
    except Exception as exc:  # noqa: BLE001
        logger.warning("hackernews: feed fetch failed (%s): %s", tgt, exc)
        return []

    for sid in story_ids:
        try:
            story = _item(get, sid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("hackernews: story %s failed: %s", sid, exc)
            continue
        if not story or story.get("type") not in ("story", "job", "poll", None):
            continue
        title = story.get("title", "") or ""
        community = f"HN:{title[:60]}" if title else ""
        posts.append(SocialPost(
            platform="hackernews", post_id=str(story.get("id", sid)),
            author=story.get("by", "") or "", title=title,
            text=story.get("text", "") or "", community=community,
            url=story.get("url") or f"https://news.ycombinator.com/item?id={sid}",
            score=int(story.get("score", 0) or 0),
            created_utc=float(story.get("time", 0) or 0),
            mentions=extract_mentions(story.get("text", "")),
        ))
        if depth < 1:
            continue
        # BFS the comment tree, capped.
        q = deque((story.get("kids") or [])[:max_comments])
        fetched = 0
        while q and fetched < max_comments:
            cid = q.popleft()
            try:
                c = _item(get, cid)
            except Exception:  # noqa: BLE001
                continue
            fetched += 1
            if not c or c.get("type") != "comment" or c.get("deleted"):
                continue
            body = c.get("text", "") or ""
            posts.append(SocialPost(
                platform="hackernews", post_id=str(c.get("id", cid)),
                author=c.get("by", "") or "", text=body, community=community,
                parent_id=str(c.get("parent", "")),
                url=f"https://news.ycombinator.com/item?id={cid}",
                created_utc=float(c.get("time", 0) or 0),
                mentions=extract_mentions(body),
            ))
            for k in (c.get("kids") or []):
                if fetched < max_comments:
                    q.append(k)
    logger.info("hackernews: %d posts from '%s' (depth=%d).", len(posts), tgt, depth)
    return posts
