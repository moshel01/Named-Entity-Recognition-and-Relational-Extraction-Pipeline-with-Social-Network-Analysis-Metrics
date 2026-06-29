# MediaWiki connector. Crawling a wiki's rendered HTML works (the generic crawler +
# trafilatura), but it drags in infobox/template/nav cruft and citation chrome. The
# MediaWiki API hands back clean article PROSE directly (action=query&prop=extracts&
# explaintext), which is exactly what NER/relation-extraction wants. So for wikis we go
# through the API, not the page HTML. Works on any MediaWiki (Wikipedia, Fandom, a
# self-hosted wiki) - only the host changes.
#
# spec = "host:Target". Target forms:
#   "en.wikipedia.org:Ada Lovelace|Charles Babbage"   explicit pages (| separated)
#   "en.wikipedia.org:Category:German resistance"     a category's member pages
#   "harrypotter.fandom.com:Category:Death Eaters"    same, any MediaWiki host
# Only public read endpoints, descriptive UA, polite. No login, no scraping the UI.

from __future__ import annotations

import json
import logging
import time
import urllib.parse

from core.schema import Document, stable_id

logger = logging.getLogger(__name__)

_UA = "SNA-Extraction-Pipeline/1.0 (academic research)"
# Section headers that begin the reference/navigation tail in a plaintext extract.
_TAIL = ("references", "see also", "external links", "further reading", "notes",
         "bibliography", "citations", "sources", "footnotes")


def _http_get(url: str, user_agent: str = _UA, timeout: int = 30) -> str:
    import requests
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_spec(spec: str) -> tuple[str, str]:
    """'en.wikipedia.org:Category:Foo' -> ('en.wikipedia.org', 'Category:Foo')."""
    host, _, target = (spec or "").strip().partition(":")
    return host.strip(), target.strip()


def _api(host: str) -> str:
    host = host.strip().strip("/")
    if "/" in host:                       # caller gave host + script path
        return f"https://{host}/api.php" if not host.endswith("api.php") else f"https://{host}"
    return f"https://{host}/w/api.php"


def _strip_tail(text: str) -> str:
    """Cut the trailing == References == / == See also == sections off an extract."""
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        h = ln.strip().strip("=").strip().lower()
        if h in _TAIL and i > len(lines) // 3:
            return "\n".join(lines[:i]).strip()
    return text.strip()


def _category_titles(api: str, category: str, limit: int, get) -> list[str]:
    cat = category if category.lower().startswith("category:") else f"Category:{category}"
    titles: list[str] = []
    cont = ""
    while len(titles) < limit:
        url = (f"{api}?action=query&format=json&list=categorymembers"
               f"&cmtitle={urllib.parse.quote(cat)}&cmlimit={min(500, limit)}&cmtype=page")
        if cont:
            url += f"&cmcontinue={urllib.parse.quote(cont)}"
        try:
            data = json.loads(get(url))
        except Exception as exc:  # noqa: BLE001
            logger.warning("wiki: category fetch failed (%s): %s", cat, exc)
            break
        for m in (data.get("query", {}).get("categorymembers") or []):
            if m.get("title"):
                titles.append(m["title"])
        cont = (data.get("continue") or {}).get("cmcontinue", "")
        if not cont:
            break
    return titles[:limit]


def _extract_page(api: str, host: str, title: str, get) -> Document | None:
    url = (f"{api}?action=query&format=json&prop=extracts&explaintext=1&redirects=1"
           f"&titles={urllib.parse.quote(title)}")
    try:
        data = json.loads(get(url))
    except Exception as exc:  # noqa: BLE001
        logger.debug("wiki: extract failed (%s): %s", title, exc)
        return None
    pages = (data.get("query", {}) or {}).get("pages", {}) or {}
    for _pid, page in pages.items():
        text = _strip_tail(page.get("extract", "") or "")
        real = page.get("title", title)
        if not text.strip():
            return None
        page_url = f"https://{host}/wiki/{urllib.parse.quote(real.replace(' ', '_'))}"
        return Document(
            doc_id=stable_id(page_url, prefix="wiki_", length=10),
            source_path=page_url, text=text,
            meta={"filename": page_url, "source_type": "wiki", "platform": "mediawiki",
                  "host": host, "title": real, "n_chars": len(text)})
    return None


def fetch_wiki(spec: str, *, limit: int = 50, fetch=None, delay: float = 0.2,
               **_) -> list[Document]:
    """Resolve a 'host:Target' wiki spec to clean article Documents via the API."""
    host, target = parse_spec(spec)
    if not host or not target:
        raise ValueError("wiki spec must be 'host:Target', e.g. "
                         "en.wikipedia.org:Ada Lovelace or en.wikipedia.org:Category:Physicists")
    get = fetch or _http_get
    api = _api(host)
    if target.lower().startswith("category:"):
        titles = _category_titles(api, target, limit, get)
    else:
        titles = [t.strip() for t in target.split("|") if t.strip()]
    docs: list[Document] = []
    for t in titles[:limit]:
        doc = _extract_page(api, host, t, get)
        if doc is not None:
            docs.append(doc)
        if delay and fetch is None:
            time.sleep(delay)
    logger.info("wiki: %d page(s) from %s (%s).", len(docs), host, target)
    return docs
