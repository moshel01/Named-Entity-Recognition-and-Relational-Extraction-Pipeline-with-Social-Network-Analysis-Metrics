# Mastodon / Fediverse connector via the open public REST API (no auth for public
# timelines). target forms:
#   mastodon.social              -> the instance public timeline
#   mastodon.social/tag/ai       -> a hashtag timeline
# Status text is HTML; we strip tags. mentions come from the status `mentions` array
# (acct handles). in_reply_to_id is the reply edge; the parent's author resolves within
# the batch (cross-instance parents may not be present - partial, as expected).

from __future__ import annotations

import json
import logging
import re

from .base import SocialPost

logger = logging.getLogger(__name__)

_UA = "SNA-Extraction-Pipeline/1.0 (academic research)"
_TAG_RE = re.compile(r"<[^>]+>")


def _http_get(url: str, user_agent: str = _UA, timeout: int = 30) -> str:
    import requests
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _strip_html(html: str) -> str:
    text = _TAG_RE.sub(" ", html or "")
    return re.sub(r"\s+", " ", text).strip()


def _status_to_post(s: dict, community: str) -> SocialPost:
    acct = (s.get("account") or {})
    return SocialPost(
        platform="mastodon",
        post_id=str(s.get("id", "")),
        author=acct.get("acct", "") or acct.get("username", "") or "",
        text=_strip_html(s.get("content", "")),
        community=community,
        parent_id=str(s.get("in_reply_to_id") or ""),
        url=s.get("url", "") or s.get("uri", ""),
        score=int(s.get("favourites_count", 0) or 0)
        + int(s.get("reblogs_count", 0) or 0),
        mentions=[m.get("acct", "") for m in (s.get("mentions") or []) if m.get("acct")],
        extra={"in_reply_to_account_id": s.get("in_reply_to_account_id")},
    )


def fetch(target: str, *, limit: int = 100, depth: int = 1, fetch=None, **_) -> list[SocialPost]:
    """Pull a Mastodon public/hashtag timeline. target='instance' or 'instance/tag/NAME'."""
    get = fetch or _http_get
    t = (target or "").strip().strip("/")
    if not t:
        raise ValueError("mastodon target must be an instance, e.g. mastodon:mastodon.social")
    if "/tag/" in t:
        instance, tag = t.split("/tag/", 1)
        url = f"https://{instance}/api/v1/timelines/tag/{tag}?limit={min(limit, 40)}"
        community = f"#{tag}"
    else:
        instance = t
        url = f"https://{instance}/api/v1/timelines/public?limit={min(limit, 40)}"
        community = instance
    try:
        data = json.loads(get(url))
    except Exception as exc:  # noqa: BLE001
        logger.warning("mastodon: timeline fetch failed (%s): %s", t, exc)
        return []
    posts = [_status_to_post(s, community) for s in (data or []) if isinstance(s, dict)][:limit]
    logger.info("mastodon: %d posts from %s.", len(posts), t)
    return posts
