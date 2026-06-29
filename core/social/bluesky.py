# Bluesky connector via the AT Protocol PUBLIC AppView (public.api.bsky.app). Bluesky is
# built to be openly readable: public posts, author feeds, and thread trees come back
# with no auth and no games. target forms:
#   "from:alice.bsky.social"     -> that account's feed (also bare "@alice..." / "alice...")
#   "climate"                    -> a public post search
#   "#science"                   -> a hashtag search
# The reply graph is record.reply.parent (resolved to the parent author within the batch);
# the thread root is used as the community so co-repliers project together. Mentions come
# from the post text (handles carry dots, so a dot-aware regex, not base.extract_mentions).

from __future__ import annotations

import json
import logging
import re
import urllib.parse

from .base import SocialPost

logger = logging.getLogger(__name__)

_BASE = "https://public.api.bsky.app/xrpc"
_UA = "SNA-Extraction-Pipeline/1.0 (academic research)"
# Bluesky handles are domains: @alice.bsky.social, @nytimes.com.
_HANDLE_RE = re.compile(r"@([a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?(?:\.[a-zA-Z0-9-]+)+)")


def _http_get(url: str, user_agent: str = _UA, timeout: int = 30) -> str:
    import requests
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _handle(author: dict) -> str:
    return author.get("handle", "") or author.get("did", "")


def _post_view(pv: dict) -> SocialPost:
    rec = pv.get("record") or {}
    reply = rec.get("reply") or {}
    parent_uri = ((reply.get("parent") or {}).get("uri")) or ""
    root_uri = ((reply.get("root") or {}).get("uri")) or ""
    text = rec.get("text", "") or ""
    return SocialPost(
        platform="bluesky",
        post_id=pv.get("uri", ""),
        author=_handle(pv.get("author") or {}),
        text=text,
        community=root_uri,                 # the thread; co-repliers co-affiliate
        parent_id=parent_uri,
        url=pv.get("uri", ""),
        score=int(pv.get("likeCount", 0) or 0),
        mentions=[m for m in _HANDLE_RE.findall(text)],
        extra={"replyCount": pv.get("replyCount", 0)},
    )


def fetch(target: str, *, limit: int = 100, depth: int = 1, fetch=None, **_) -> list[SocialPost]:
    """Search posts or pull an author feed from the public AppView (no auth)."""
    get = fetch or _http_get
    t = (target or "").strip()
    if not t:
        raise ValueError("bluesky target needed, e.g. bluesky:from:alice.bsky.social or bluesky:climate")
    n = max(1, min(limit, 100))
    if t.startswith("from:") or t.startswith("@"):
        actor = t.split(":", 1)[1] if t.startswith("from:") else t[1:]
        url = f"{_BASE}/app.bsky.feed.getAuthorFeed?actor={urllib.parse.quote(actor)}&limit={n}"
        key = "feed"
    else:
        q = t[1:] if t.startswith("#") else t
        url = f"{_BASE}/app.bsky.feed.searchPosts?q={urllib.parse.quote(q)}&limit={n}"
        key = "posts"
    try:
        data = json.loads(get(url))
    except Exception as exc:  # noqa: BLE001
        logger.warning("bluesky: fetch failed for %r: %s", t, exc)
        return []
    items = data.get(key) or []
    posts: list[SocialPost] = []
    for it in items:
        pv = it.get("post") if key == "feed" else it      # author feed wraps in {post:...}
        if isinstance(pv, dict) and pv.get("uri"):
            posts.append(_post_view(pv))
    logger.info("bluesky: %d posts for %r.", len(posts), t)
    return posts
