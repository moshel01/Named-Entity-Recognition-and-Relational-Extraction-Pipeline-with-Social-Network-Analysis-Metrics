# Telegram connector via the PUBLIC channel web preview (t.me/s/<channel>). That preview
# is Telegram's own no-auth, public, embeddable view of a broadcast channel - the same
# kind of open public read as a Mastodon timeline. Channels only (broadcast); public
# groups have no /s/ preview. target forms:
#   "durov"                      -> https://t.me/s/durov (recent messages, paged back)
#   "@durov"                     -> same (leading @ stripped)
#
# The channel IS the author (broadcast), so the signal isn't replies - it's FORWARDS:
# channel A reposting a message that originated in channel B is the directed propagation
# edge (forwarded_from, A->B) that maps how content moves across the channel ecosystem.
# Mentions (@handle / t.me/<handle> links in the text) are the other edge. No community
# node (a channel only posts in itself - a self-loop carries nothing).
#
# HONEST LIMIT: the web preview only exposes PUBLIC channels and the recent window we page
# through; it has no search. For groups, full history, or non-public channels, use the
# official MTProto client API (Telethon, api_id/api_hash from my.telegram.org) or the Bot
# API (bot added as a channel admin) - sanctioned official paths, an extension. We do NOT
# bypass anything: /s/ is public by design and the connector fails soft if it's withdrawn.

from __future__ import annotations

import html
import logging
import re
import time

from .base import SocialPost

logger = logging.getLogger(__name__)

_UA = "SNA-Extraction-Pipeline/1.0 (academic research)"
_TAG_RE = re.compile(r"<[^>]+>")
# One message: from its data-post id up to the next message's data-post (or end).
_MSG_RE = re.compile(r'data-post="([^"]+)"(.*?)(?=data-post="|\Z)', re.S)
# Linked, public forward source: the forwarded-from anchor's t.me/<handle>.
_FWD_RE = re.compile(
    r'tgme_widget_message_forwarded_from_name"\s+href="https?://t\.me/([A-Za-z0-9_]{3,32})"')
# The message body text div (non-greedy to the first closing div).
_TEXT_RE = re.compile(r'tgme_widget_message_text[^>]*>(.*?)</div>', re.S)
_TIME_RE = re.compile(r'<time[^>]+datetime="([^"]+)"')
_VIEWS_RE = re.compile(r'tgme_widget_message_views"[^>]*>([^<]+)')
# A bare t.me/<handle> link (NOT a /<handle>/<digits> message permalink - the closing
# quote must follow the handle directly).
_TME_LINK_RE = re.compile(r'href="https?://t\.me/([A-Za-z0-9_]{3,32})"')
# @handle written in plain text.
_AT_RE = re.compile(r"(?<![\w@])@([A-Za-z0-9_]{3,32})\b")


def _http_get(url: str, user_agent: str = _UA, timeout: int = 30) -> str:
    import requests
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _strip_html(s: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", _TAG_RE.sub(" ", s or ""))).strip()


def _num(post_id: str) -> int:
    tail = (post_id or "").rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _parse_message(post_id: str, body: str, channel: str) -> SocialPost:
    fwd_m = _FWD_RE.search(body)
    forwarded = (fwd_m.group(1) if fwd_m else "").lower()
    text_m = _TEXT_RE.search(body)
    raw_text = text_m.group(1) if text_m else ""
    text = _strip_html(raw_text)

    # Mentions: t.me/<handle> links in the message text + @handles in plain text.
    # Exclude the channel itself and the forward source (already its own edge).
    drop = {channel.lower(), forwarded}
    seen: set[str] = set()
    mentions: list[str] = []
    for h in _TME_LINK_RE.findall(raw_text) + _AT_RE.findall(text):
        low = h.lower()
        if low in drop or low in seen:
            continue
        seen.add(low)
        mentions.append(h)

    tm = _TIME_RE.search(body)
    views = _VIEWS_RE.search(body)
    return SocialPost(
        platform="telegram",
        post_id=post_id,
        author=channel,
        text=text,
        forwarded_from=forwarded,
        url=f"https://t.me/{post_id}",
        mentions=mentions,
        extra={"datetime": tm.group(1) if tm else "", "views": views.group(1).strip() if views else ""},
    )


def _parse_page(html_text: str, channel: str) -> list[SocialPost]:
    out: list[SocialPost] = []
    for post_id, body in _MSG_RE.findall(html_text or ""):
        if "/" in post_id:                       # data-post is "channel/<id>"
            out.append(_parse_message(post_id, body, channel))
    return out


def fetch(target: str, *, limit: int = 50, depth: int = 1, fetch=None,
          delay: float = 0.5, **_) -> list[SocialPost]:
    """Pull a public Telegram channel's recent messages, paging back with ?before=."""
    get = fetch or _http_get
    channel = (target or "").strip().lstrip("@").strip("/").split("/")[0]
    if not channel:
        raise ValueError("telegram target must be a public channel, e.g. telegram:durov")

    posts: list[SocialPost] = []
    seen: set[str] = set()
    before = 0
    max_pages = max(1, (limit + 19) // 20) + 1
    for _page in range(max_pages):
        url = f"https://t.me/s/{channel}" + (f"?before={before}" if before else "")
        try:
            page = get(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram: fetch failed (%s): %s", channel, exc)
            break
        batch = [p for p in _parse_page(page, channel) if p.post_id not in seen]
        if not batch:
            break
        for p in batch:
            seen.add(p.post_id)
        posts.extend(batch)
        nums = [n for n in (_num(p.post_id) for p in batch) if n]
        if len(posts) >= limit or not nums:
            break
        before = min(nums)
        if delay:
            time.sleep(delay)
    posts = posts[:limit]
    logger.info("telegram: %d messages from @%s.", len(posts), channel)
    return posts
