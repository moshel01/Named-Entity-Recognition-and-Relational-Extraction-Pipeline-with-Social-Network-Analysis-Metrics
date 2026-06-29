# Truth Social connector. Truth Social is a Mastodon fork (Soapbox) and speaks the
# Mastodon API, so this just points the Mastodon connector at truthsocial.com - no
# subversion, only their own public API. target forms:
#   truthsocial:                 -> truthsocial.com public/federated timeline
#   truthsocial:tag/news         -> a hashtag timeline
#
# HONEST LIMIT: Truth Social gates unauthenticated API access more tightly than a stock
# Mastodon instance and sits behind a CDN, so a public pull may return little or be
# refused. We do NOT bypass that (no CDN/anti-bot evasion); the connector fails soft. For
# fuller access, register a Truth Social app and pass an OAuth token via opts (extension),
# the same sanctioned path as any Mastodon client.

from __future__ import annotations

from . import mastodon

_INSTANCE = "truthsocial.com"


def fetch(target: str, *, limit: int = 100, depth: int = 1, fetch=None, **opts):
    t = (target or "").strip().strip("/")
    # Rewrite the target onto the truthsocial.com instance for the Mastodon connector.
    if t.startswith("tag/"):
        masto_target = f"{_INSTANCE}/{t}"
    elif not t:
        masto_target = _INSTANCE
    else:
        masto_target = f"{_INSTANCE}/{t}" if "/" in t else _INSTANCE
    posts = mastodon.fetch(masto_target, limit=limit, depth=depth, fetch=fetch, **opts)
    for p in posts:
        p.platform = "truthsocial"
    return posts
