# Bounded, polite web crawler. Discovers subpages from seed URLs (sitemap first,
# then scoped breadth-first link following) and returns ready-to-process
# Documents. Fetch-once: the page we read to find links is the page we keep, so
# the pipeline never re-fetches.
#
# Hard guarantees (a crawl must never run away or take the pipeline down):
#   - stays in scope (same host, optional path prefix, allow/deny regex)
#   - obeys robots.txt + crawl-delay (opt-out via config)
#   - rate-limited per host, page-size capped, request-count capped
#   - visited-set dedup with URL normalization; depth + page caps
#   - fail-soft per URL: one bad page is logged and skipped, never raised
#
# Network is injectable (`fetch=`) so the whole thing is testable offline.

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from urllib import robotparser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from .schema import Document, stable_id

logger = logging.getLogger(__name__)

_DEFAULT_UA = "SNA-Extraction-Pipeline/1.0 (academic research)"

# Tracking junk stripped during normalization so ?utm_*=... doesn't defeat dedup.
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "mc_cid", "mc_eid",
    "igshid", "ref", "ref_src", "ref_url", "_ga", "yclid", "_hsenc", "_hsmi",
}

# Binary / media we never enqueue as crawl targets (PDF is handled separately).
_SKIP_EXT = (
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg", ".ico", ".webp",
    ".css", ".js", ".json", ".xml", ".rss", ".atom", ".zip", ".gz", ".tar",
    ".rar", ".7z", ".exe", ".dmg", ".iso", ".mp3", ".mp4", ".avi", ".mov",
    ".wmv", ".flv", ".ogg", ".wav", ".woff", ".woff2", ".ttf", ".eot",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
)


@dataclass
class CrawlOptions:
    max_pages: int = 50            # documents to return (the deliverable cap)
    max_depth: int = 3            # link hops from a seed
    stay_on_host: bool = True     # only follow links on the seed's host
    stay_under_path: bool = False  # also require the seed's directory prefix
    allow: tuple[str, ...] = ()   # if set, URL must match one (regex, search)
    deny: tuple[str, ...] = ()    # URL dropped if it matches any (regex, search)
    delay: float = 1.0            # min seconds between requests to one host
    respect_robots: bool = True
    use_sitemap: bool = True
    user_agent: str = _DEFAULT_UA
    timeout: int = 30
    max_bytes: int = 5_000_000    # per-page download ceiling
    max_fetches: int = 0          # hard request cap; 0 -> derived from max_pages


@dataclass
class FetchResult:
    """What the (injectable) fetcher hands back. `url` is post-redirect."""
    url: str
    status: int = 200
    content_type: str = "text/html"
    text: str = ""
    content: bytes = b""
    ok: bool = True


def _reg(host: str) -> str:
    """Registrable-ish host key: lowercase, drop a leading www. (no PSL dep, so
    www vs apex is folded but example.co.uk-style siblings are not)."""
    host = (host or "").lower()
    return host[4:] if host.startswith("www.") else host


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _dir_prefix(path: str) -> str:
    """Directory prefix of a seed path, for stay_under_path scope.

    A page seed ('/docs/intro') scopes to its parent directory ('/docs/') so its
    siblings are in scope. The gotcha: normalize_url strips a trailing slash, so a
    single-segment seed ('/docs/' or '/docs') has parent '/' - which would widen
    scope to the whole host. For a single-segment seed use the segment itself as
    the directory instead. A bare-root seed ('/') legitimately covers the host."""
    path = path or "/"
    parent = path.rsplit("/", 1)[0]
    if parent in ("", "/"):               # single segment: don't widen to the host
        seg = path.strip("/")
        return f"/{seg}/" if seg else "/"
    return parent + "/"


def normalize_url(url: str) -> str:
    """Canonical form for dedup: lowercase scheme/host, drop default port,
    collapse // in path, strip a trailing slash (except root), drop the fragment
    and tracking params. Bad input is returned stripped, not raised."""
    url = (url or "").strip()
    try:
        p = urlsplit(url)
    except ValueError:
        return url
    scheme = (p.scheme or "http").lower()
    host = (p.hostname or "").lower()
    if not host:
        return url
    netloc = host
    if p.port and not ((scheme == "http" and p.port == 80) or
                       (scheme == "https" and p.port == 443)):
        netloc = f"{host}:{p.port}"
    path = re.sub(r"/{2,}", "/", p.path or "/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    query = ""
    if p.query:
        kept = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                if k.lower() not in _TRACKING_PARAMS]
        query = urlencode(kept)
    return urlunsplit((scheme, netloc, path, query, ""))


class Crawler:
    def __init__(self, opts: CrawlOptions, fetch=None) -> None:
        self.opts = opts
        self._fetch = fetch or self._http_fetch
        self._allow = [re.compile(p) for p in opts.allow]
        self._deny = [re.compile(p) for p in opts.deny]
        self._robots: dict[str, object] = {}      # host key -> RobotFileParser|None
        self._crawl_delay: dict[str, float] = {}  # host key -> seconds
        self._last: dict[str, float] = {}         # host key -> monotonic ts
        self._hosts: set[str] = set()             # seed host keys (scope)
        self._dirs: set[str] = set()              # seed dir prefixes (scope)
        self._seeds: set[str] = set()             # normalized seeds (always in scope)
        self.visited: set[str] = set()
        self.fetched = 0

    # -- public --------------------------------------------------------------
    def crawl(self, seeds: list[str]) -> list[Document]:
        seeds = [normalize_url(s) for s in (seeds or [])
                 if s and urlsplit(s).scheme in ("http", "https")]
        if not seeds:
            logger.warning("Crawler: no usable http(s) seeds.")
            return []

        self._seeds = set(seeds)
        for s in seeds:
            self._hosts.add(_reg(_host(s)))
            self._dirs.add(_dir_prefix(urlsplit(s).path))

        budget = self.opts.max_fetches or max(self.opts.max_pages * 5,
                                              self.opts.max_pages + 10)

        # Preload robots per seed host (also yields Sitemap: directives + delay).
        for s in seeds:
            self._ensure_robots(s)

        frontier: deque[tuple[str, int]] = deque((s, 0) for s in seeds)
        if self.opts.use_sitemap:
            for u in self._sitemap_seed(seeds):
                frontier.append((u, 0))

        docs: list[Document] = []
        while frontier and len(docs) < self.opts.max_pages and self.fetched < budget:
            url, depth = frontier.popleft()
            url = normalize_url(url)
            if url in self.visited:
                continue
            self.visited.add(url)
            try:
                self._process(url, depth, frontier, docs)
            except Exception as exc:  # noqa: BLE001 - one page must not kill the crawl
                logger.debug("Crawler: skipping %s: %s", url, exc)

        logger.info("Crawler: %d documents from %d seeds (%d pages fetched, %d urls seen).",
                    len(docs), len(seeds), self.fetched, len(self.visited))
        return docs[:self.opts.max_pages]

    # -- per-url work --------------------------------------------------------
    def _process(self, url, depth, frontier, docs) -> None:
        if not self._in_scope(url) or not self._robots_ok(url):
            return
        res = self._get(url)
        if not res or not res.ok:
            return
        final = normalize_url(res.url or url)
        if final != url:  # redirect: re-check scope + dedup on the destination
            if final in self.visited:
                return
            self.visited.add(final)
            if not self._in_scope(final):
                return

        ctype = (res.content_type or "").lower()
        ct = ctype.split(";", 1)[0].strip()
        is_pdf = "pdf" in ct or final.split("?")[0].lower().endswith(".pdf")
        if is_pdf:
            text = self._pdf_text(res.content)
            if text.strip():
                docs.append(self._doc(final, text))
            return  # never parse links out of a PDF

        # Trust an explicit content-type; only sniff the body when it's missing or
        # generic (a wrong image/* etc. must not be parsed as HTML just because the
        # bytes happen to start with '<').
        is_html = ct in ("text/html", "application/xhtml+xml")
        if ct in ("", "text/plain", "application/octet-stream"):
            is_html = (res.text or "").lstrip()[:1] == "<"
        if not is_html:
            return  # not html, not pdf -> not a document, no links

        from .preprocessor import _clean_html
        text = _clean_html(res.text)
        if text.strip():
            docs.append(self._doc(final, text))
        if depth < self.opts.max_depth:
            for link in self._links(res.text, final):
                nl = normalize_url(link)
                if nl not in self.visited and self._in_scope(nl):
                    frontier.append((nl, depth + 1))

    def _doc(self, url, text) -> Document:
        from .preprocessor import normalize_text
        text = normalize_text(text)
        # Same id scheme as documents_from_urls so a URL also listed in io.urls
        # dedups by doc_id instead of producing a twin node.
        return Document(
            doc_id=stable_id(url, prefix="url_", length=10),
            source_path=url,
            text=text,
            meta={"filename": url, "source_type": "url", "n_chars": len(text)},
        )

    # -- scope / robots ------------------------------------------------------
    def _in_scope(self, url) -> bool:
        if url in self._seeds:  # explicit entry points define scope; never filtered
            return True
        try:
            sp = urlsplit(url)
        except ValueError:
            return False
        if sp.scheme not in ("http", "https"):
            return False
        path = sp.path.lower()
        if any(path.endswith(ext) for ext in _SKIP_EXT):
            return False
        if self.opts.stay_on_host and _reg(sp.hostname or "") not in self._hosts:
            return False
        if self.opts.stay_under_path and not any(sp.path.startswith(d) for d in self._dirs):
            return False
        if self._allow and not any(rx.search(url) for rx in self._allow):
            return False
        if any(rx.search(url) for rx in self._deny):
            return False
        return True

    def _ensure_robots(self, url) -> None:
        if not self.opts.respect_robots:
            return
        key = _reg(_host(url))
        if key in self._robots:
            return
        sp = urlsplit(url)
        robots_url = urlunsplit((sp.scheme, sp.netloc, "/robots.txt", "", ""))
        res = self._get(robots_url)
        rp = None
        if res and res.ok and res.text:
            rp = robotparser.RobotFileParser()
            try:
                rp.parse(res.text.splitlines())
                cd = rp.crawl_delay(self.opts.user_agent)
                if cd:
                    self._crawl_delay[key] = float(cd)
            except Exception:  # noqa: BLE001 - malformed robots -> treat as absent
                rp = None
        self._robots[key] = rp  # None = unavailable -> allow (conventional)

    def _robots_ok(self, url) -> bool:
        if not self.opts.respect_robots:
            return True
        self._ensure_robots(url)
        rp = self._robots.get(_reg(_host(url)))
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.opts.user_agent, url)
        except Exception:  # noqa: BLE001
            return True

    # -- sitemaps ------------------------------------------------------------
    def _sitemap_seed(self, seeds) -> list[str]:
        cap = max(self.opts.max_pages * 3, self.opts.max_pages)
        found: list[str] = []
        seen_hosts: set[str] = set()
        seen_sm: set[str] = set()
        for s in seeds:
            key = _reg(_host(s))
            if key in seen_hosts:
                continue
            seen_hosts.add(key)
            sp = urlsplit(s)
            candidates: list[str] = []
            rp = self._robots.get(key)
            if rp is not None:
                try:
                    candidates.extend(rp.site_maps() or [])
                except Exception:  # noqa: BLE001
                    pass
            candidates.append(urlunsplit((sp.scheme, sp.netloc, "/sitemap.xml", "", "")))
            for sm in candidates:
                if sm in seen_sm:
                    continue
                seen_sm.add(sm)
                for u in self._parse_sitemap(sm, 0, seen_sm):
                    if self._in_scope(u):
                        found.append(u)
                        if len(found) >= cap:
                            return found
        return found

    def _parse_sitemap(self, sm_url, depth, seen_sm) -> list[str]:
        if depth > 2:
            return []
        res = self._get(sm_url)
        if not res or not res.ok or not res.text:
            return []
        try:
            root = ET.fromstring(res.text.encode("utf-8", "replace"))
        except Exception:  # noqa: BLE001 - malformed xml
            return []
        locs = [e.text.strip() for e in root.iter()
                if e.tag.lower().endswith("loc") and e.text and e.text.strip()]
        if root.tag.lower().endswith("sitemapindex"):
            out: list[str] = []
            for child in locs:
                if child in seen_sm:
                    continue
                seen_sm.add(child)
                out.extend(self._parse_sitemap(child, depth + 1, seen_sm))
            return out
        return locs

    # -- link extraction -----------------------------------------------------
    def _links(self, html, base) -> list[str]:
        out: list[str] = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = (a["href"] or "").strip()
                if not href or href[:1] in ("#",) or \
                        href.lower().startswith(("mailto:", "tel:", "javascript:", "data:")):
                    continue
                absu = urljoin(base, href)
                if urlsplit(absu).scheme in ("http", "https"):
                    out.append(absu)
        except Exception:  # noqa: BLE001
            pass
        return out

    # -- fetch + rate limit --------------------------------------------------
    def _get(self, url):
        key = _reg(_host(url))
        delay = max(self.opts.delay, self._crawl_delay.get(key, 0.0))
        if delay > 0:
            last = self._last.get(key)
            if last is not None:
                wait = delay - (time.monotonic() - last)
                if wait > 0:
                    time.sleep(wait)
        res = self._fetch(url)
        self._last[key] = time.monotonic()
        self.fetched += 1
        return res

    def _http_fetch(self, url):
        import requests
        try:
            resp = requests.get(url, headers={"User-Agent": self.opts.user_agent},
                                timeout=self.opts.timeout, stream=True, allow_redirects=True)
        except Exception as exc:  # noqa: BLE001 - network failure -> fail soft
            logger.debug("Crawler fetch failed %s: %s", url, exc)
            return None
        try:
            status = resp.status_code
            ctype = (resp.headers.get("Content-Type") or "").lower()
            final = resp.url or url
            buf = bytearray()
            for chunk in resp.iter_content(8192):
                if chunk:
                    buf.extend(chunk)
                    if len(buf) > self.opts.max_bytes:
                        break
            body = bytes(buf)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Crawler read failed %s: %s", url, exc)
            return None
        finally:
            resp.close()
        if status >= 400:
            return None
        if "pdf" in ctype or final.split("?")[0].lower().endswith(".pdf"):
            return FetchResult(url=final, status=status, content_type=ctype,
                               content=body, ok=True)
        text = self._decode(body, ctype)
        return FetchResult(url=final, status=status, content_type=ctype,
                           text=text, content=body, ok=True)

    @staticmethod
    def _decode(body: bytes, ctype: str) -> str:
        m = re.search(r"charset=([\w\-]+)", ctype or "")
        if m:
            try:
                return body.decode(m.group(1), errors="replace")
            except (LookupError, TypeError):
                pass
        from .preprocessor import _decode  # chardet sniff, header had no charset
        return _decode(body, "auto")

    @staticmethod
    def _pdf_text(content: bytes) -> str:
        try:
            import fitz
            parts = []
            with fitz.open(stream=content, filetype="pdf") as doc:
                for page in doc:
                    parts.append(page.get_text("text"))
            return "\n".join(parts)
        except Exception:  # noqa: BLE001 - corrupt/empty pdf
            return ""


def crawl_documents(seeds: list[str], opts: CrawlOptions | None = None,
                    fetch=None) -> list[Document]:
    """Convenience wrapper: crawl `seeds` and return Documents."""
    return Crawler(opts or CrawlOptions(), fetch=fetch).crawl(seeds)
