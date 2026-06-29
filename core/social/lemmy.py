# Lemmy connector via the open /api/v3 read API (no auth for public communities). Lemmy
# is the fediverse's Reddit: communities (!tech@instance) with threaded comments. target:
#   "lemmy.world/technology"     -> that community's posts (+ comments at depth>=1)
# The comment `path` ("0.<root>.<...>.<id>") gives the reply parent; the community node
# (!name) makes co-commenters co-affiliate, same as the Reddit/HN model.

from __future__ import annotations

import json
import logging
import time

from .base import SocialPost, extract_mentions

logger = logging.getLogger(__name__)

_UA = "SNA-Extraction-Pipeline/1.0 (academic research)"


def _http_get(url: str, user_agent: str = _UA, timeout: int = 30) -> str:
    import requests
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _submission(pv: dict, community: str) -> SocialPost:
    post = pv.get("post") or {}
    return SocialPost(
        platform="lemmy", post_id=f"post_{post.get('id', '')}",
        author=(pv.get("creator") or {}).get("name", "") or "",
        title=post.get("name", "") or "", text=post.get("body", "") or "",
        community=community, url=post.get("ap_id", ""),
        score=int((pv.get("counts") or {}).get("score", 0) or 0),
        mentions=extract_mentions(post.get("body", "")),
        extra={"post_id_raw": post.get("id")},
    )


def _comment(cv: dict, community: str) -> SocialPost:
    c = cv.get("comment") or {}
    path = (c.get("path") or "").split(".")
    # path = 0.<root>...<this>; parent is the element before this one.
    parent = ""
    if len(path) >= 3:
        parent = f"comment_{path[-2]}"
    elif c.get("post_id"):
        parent = f"post_{c.get('post_id')}"
    body = c.get("content", "") or ""
    return SocialPost(
        platform="lemmy", post_id=f"comment_{c.get('id', '')}",
        author=(cv.get("creator") or {}).get("name", "") or "",
        text=body, community=community, parent_id=parent,
        url=c.get("ap_id", ""), mentions=extract_mentions(body),
    )


def fetch(target: str, *, limit: int = 50, depth: int = 1, fetch=None,
          sort: str = "Hot", delay: float = 0.5, **_) -> list[SocialPost]:
    """Pull a Lemmy community. target='instance/community'."""
    get = fetch or _http_get
    t = (target or "").strip().strip("/")
    if "/" not in t:
        raise ValueError("lemmy target must be instance/community, e.g. lemmy:lemmy.world/technology")
    instance, community_name = t.split("/", 1)
    community = f"!{community_name}"
    base = f"https://{instance}/api/v3"
    try:
        listing = json.loads(get(
            f"{base}/post/list?community_name={community_name}&limit={min(limit, 50)}&sort={sort}"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("lemmy: community fetch failed (%s): %s", t, exc)
        return []
    subs = (listing.get("posts") or [])[:limit]
    posts: list[SocialPost] = [_submission(pv, community) for pv in subs]
    if depth >= 1:
        for pv in subs:
            pid = (pv.get("post") or {}).get("id")
            if not pid:
                continue
            if delay:
                time.sleep(delay)
            try:
                cl = json.loads(get(f"{base}/comment/list?post_id={pid}&limit=300&max_depth=8"))
                for cv in (cl.get("comments") or []):
                    posts.append(_comment(cv, community))
            except Exception as exc:  # noqa: BLE001
                logger.debug("lemmy: comments fetch failed for %s: %s", pid, exc)
    logger.info("lemmy: %d posts from %s (depth=%d).", len(posts), t, depth)
    return posts
