# Offline regression tests: no models, no network, no LLM. Run after touching
# json_repair, checkpoint scoring, or dedup folds:
#   python -m tests.run_tests

from __future__ import annotations

import sys
from pathlib import Path

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


def test_json_repair() -> None:
    from intelligence.json_repair import repair_json
    print("-- json_repair")
    cases = [
        ("valid", '{"a": 1}', {"a": 1}),
        ("trailing comma", '{"a": [1, 2,], }', {"a": [1, 2]}),
        ("missing comma str", '{"a": "x"\n"b": "y"}', {"a": "x", "b": "y"}),
        ("missing comma num", '{"a": 1\n"b": 2}', {"a": 1, "b": 2}),
        # digit before close quote must NOT get a comma written into the string
        ("digit in string", '{"e": "born 1903"\n"k": 2}', {"e": "born 1903", "k": 2}),
        ("py literals", '{"a": True, "b": None}', {"a": True, "b": None}),
        ("bare enum", '{"type": enemy}', {"type": "enemy"}),
        ("truncated", '{"a": [1, 2', {"a": [1, 2]}),
        ("dangling key", '{"a": 1, "b":', {"a": 1}),
        ("paren annotation", '{"e": "arrest" (implied), "k": 2}', {"e": "arrest", "k": 2}),
        ("paren in string", '{"e": "real (aside) inside", "k": 3}',
         {"e": "real (aside) inside", "k": 3}),
        ("multi-string 2", '{"e": "s1", "s2", "k": 1}', {"e": "s1 ... s2", "k": 1}),
        ("multi-string digit", '{"e": "in 1903", "more", "k": 1}',
         {"e": "in 1903 ... more", "k": 1}),
        ("multi-string 4", '{"e": "a", "b", "c", "d", "k": 1}',
         {"e": "a ... b ... c ... d", "k": 1}),
        ("array untouched", '{"aliases": ["x", "y"], "k": 1}',
         {"aliases": ["x", "y"], "k": 1}),
        ("obj untouched", '{"a": "v1", "b": "v2"}', {"a": "v1", "b": "v2"}),
        ("fenced", 'noise ```json\n{"a": 1}\n``` more', {"a": 1}),
        # Reasoning model: a discarded first ```json block (invalid - paren
        # annotation in an array) then the corrected answer last. Take the last.
        ("two fences last wins",
         '```json\n{"groups": [{"canonical": "34", "aliases": ["x" (bad)]}]}\n```\n'
         'On reflection:\n```json\n{"groups": [{"canonical": "1918", '
         '"aliases": ["I918"]}]}\n```',
         {"groups": [{"canonical": "1918", "aliases": ["I918"]}]}),
        # Value opens with an escaped quote (book quote as evidence).
        ("value open escaped", '{"e": \\"hello\\"", "k": 1}', {"e": 'hello"', "k": 1}),
        # Escaped-quote wrapper with an embedded dialogue comma (4.6 mis-segments
        # the close; the openfix+inner-quote fallback recovers it).
        ("dialogue comma quote", '{"e": \\"go away,\\" he said.\\"", "k": 2}',
         {"e": 'go away," he said."', "k": 2}),
    ]
    for name, raw, want in cases:
        got = repair_json(raw)
        check(name, got == want, f"got {got!r}")
    # Every captured real failure must recover. New dumps that fail here are
    # new shapes - fix the ladder, don't delete the dump.
    for f in sorted(Path("scratch/json_failures").glob("*.txt")):
        obj = repair_json(f.read_text(encoding="utf-8"))
        check(f"dump {f.name}", obj is not None)


def test_checkpoint_scoring() -> None:
    from checkpoint.manager import _failure_score
    from core.schema import DocumentExtraction, EntityMention
    print("-- checkpoint failure scoring")
    m = [EntityMention(text="X", label="PERSON", start_char=0, end_char=1,
                       chunk_id="c", doc_id="d")]

    def rec(meta, mentions=(), rels=()):
        return DocumentExtraction("d", "p", mentions=list(mentions),
                                  relationships=list(rels), meta=meta)

    check("clean", _failure_score(rec({"n_chunks": 3, "chunks_failed": 0})) == 0)
    check("partial", _failure_score(rec({"n_chunks": 3, "chunks_failed": 1})) == 1)
    check("full fail", _failure_score(rec({"n_chunks": 3, "chunks_failed": 3})) == 2)
    check("legacy empty", _failure_score(rec({"backend": "ollama"})) == 2)
    check("legacy 1-chunk mentions-only llm",
          _failure_score(rec({"backend": "ollama", "n_chunks": 1}, mentions=m)) == 2)
    check("legacy 1-chunk python_only not failure",
          _failure_score(rec({"backend": "python_only", "n_chunks": 1}, mentions=m)) == 1)


def test_dedup_folds() -> None:
    from config import DedupConfig
    from core.schema import Entity
    from postprocess.deduplicator import Deduplicator

    print("-- dedup folds")

    def ent(name, label="PERSON", mentions=1, **attrs):
        e = Entity(entity_id="", canonical_name=name, label=label)
        e.mention_count = mentions
        e.attributes.update(attrs)
        return e

    d = Deduplicator(DedupConfig())

    # Subset persons: middle-name variants fold into the longest unique form.
    ents = [ent("Theodore Fred Abel", mentions=3), ent("Theodore Abel", mentions=7)]
    out = d._fold_subset_persons(ents)
    names = {e.canonical_name for e in out}
    check("subset fold", names == {"Theodore Fred Abel"}, str(names))

    # Family block holds: different first name, shared surname - no fold.
    ents = [ent("Theodore Fred Abel"), ent("Fred Abel")]
    out = d._fold_subset_persons(ents)
    check("family block respected", len(out) == 2, str([e.canonical_name for e in out]))

    # Acronym orgs.
    ents = [ent("American Enterprise Institute", "ORG", 22), ent("AEI", "ORG", 69)]
    out = d._fold_org_acronyms(ents)
    check("acronym fold", len(out) == 1 and "AEI" in out[0].aliases,
          str([(e.canonical_name, e.aliases) for e in out]))

    # Ambiguous initials: no fold.
    ents = [ent("American Enterprise Institute", "ORG"),
            ent("Allied Engineering Initiative", "ORG"), ent("AEI", "ORG")]
    out = d._fold_org_acronyms(ents)
    check("ambiguous acronym kept", len(out) == 3)

    # Distinct acronyms never treated as one (DVP vs DNVP regression guard).
    ents = [ent("DVP", "ORG"), ent("DNVP", "ORG")]
    out = d._fold_org_acronyms(ents)
    check("distinct acronyms kept", len(out) == 2)

    # Article bucketing: "the X" and "X" must land in the same fuzzy bucket.
    ents = [ent("the American Enterprise Institute", "ORG", 3),
            ent("American Enterprise Institute", "ORG", 22)]
    merged, _, _ = d.resolve(ents, [])
    check("article variants merge", len(merged) == 1,
          str([e.canonical_name for e in merged]))


def test_relation_guide() -> None:
    import types

    from intelligence.prompts import build_extraction_prompt
    from postprocess.ontology import resolve_relation_guide

    print("-- relation guide")
    labels = ["associate", "friend", "enemy"]
    guide = {"associate": "companions, NOT friends", "friend": "stated affection only"}

    p = build_extraction_prompt("x", [], ["PERSON"], relation_types=labels,
                                relation_guide=guide)
    check("guided definition rendered", "associate: companions, NOT friends" in p)
    check("undefined label still listed", "- enemy" in p)

    # No guide -> bare comma list (unchanged behavior).
    bare = build_extraction_prompt("x", [], ["PERSON"], relation_types=labels)
    check("bare list when no guide", "associate, friend, enemy" in bare)

    # Resolver reads a config-style ontology object.
    cfg = types.SimpleNamespace(ontology=types.SimpleNamespace(
        relations=labels, relation_guide=guide))
    check("resolver from config", resolve_relation_guide(cfg) == guide)

    # Abel domain guide covers its whole ontology.
    from domain.base_domain import load_domain
    dom = load_domain("nazi_era")
    g = dom.relation_guide()
    onto = dom.relation_ontology()
    check("abel guide non-empty", len(g) > 0)
    check("abel guide labels all in ontology", set(g) <= set(onto),
          str(set(g) - set(onto)))


def test_evidence_tiers() -> None:
    from postprocess.evidence_tiers import tier_allows

    print("-- evidence tiers")
    # Asserted text + verified-record edges are conservative.
    check("llm conservative", tier_allows("llm_extracted", "conservative"))
    check("langextract conservative", tier_allows("langextract_extracted", "conservative"))
    check("metadata conservative", tier_allows("metadata", "conservative"))
    # Signal-detected domain inference is moderate, not conservative.
    check("canonical not conservative", not tier_allows("canonical_inferred", "conservative"))
    check("canonical moderate", tier_allows("canonical_inferred", "moderate"))
    # The mandatory-membership assumption (no per-edge evidence) is full-only.
    check("assumption not moderate", not tier_allows("pipeline_inferred", "moderate"))
    check("assumption full", tier_allows("pipeline_inferred", "full"))
    # Co-occurrence (and its legacy tag) is the weakest layer: full-only.
    check("cooccurrence not moderate", not tier_allows("rule_cooccurrence", "moderate"))
    check("legacy sna_inferred not moderate", not tier_allows("sna_inferred", "moderate"))
    check("cooccurrence full", tier_allows("rule_cooccurrence", "full"))
    # full admits unknown/future sources; conservative does not.
    check("unknown full", tier_allows("something_new", "full"))
    check("unknown not conservative", not tier_allows("something_new", "conservative"))
    # Joined sources (parallel-edge merge): any match admits the edge.
    check("joined any-match conservative",
          tier_allows("rule_cooccurrence;llm_extracted", "conservative"))


def test_exclude_edge_source() -> None:
    from evaluation.evaluate import _filter_edges

    print("-- exclude edge source")
    edges = [
        {"edge_source": "metadata"},               # pure injection
        {"edge_source": "metadata;llm_extracted"}, # text also asserts it
        {"edge_source": "llm_extracted"},
        {"edge_source": ""},                        # unknown
    ]
    kept = {e["edge_source"]
            for e in _filter_edges(edges, "conservative", {"metadata"})}
    # metadata-only drops; metadata;llm survives (llm remains); llm stays;
    # unknown was never conservative anyway.
    check("metadata-only excluded", "metadata" not in kept)
    check("metadata+llm survives exclude", "metadata;llm_extracted" in kept)
    check("llm kept", "llm_extracted" in kept)
    check("no exclude is a no-op",
          len(_filter_edges(edges, "full", set())) == len(edges))


def test_proximity_edges() -> None:
    from config import InferenceConfig
    from core.schema import EntityMention
    from postprocess.canonical_inference import InferenceEngine

    print("-- proximity co-occurrence")

    def men(text, pos, doc="d1"):
        return EntityMention(text=text, label="PERSON", start_char=pos,
                             end_char=pos + len(text), chunk_id="c0", doc_id=doc)

    name_to_id = {"alice": "idA", "bob": "idB", "carol": "idC"}
    # Alice@0, Bob@100 (in 600-window), Carol@1000 (out). Bob<->Carol gap 900 (out).
    mentions = [men("Alice", 0), men("Bob", 100), men("Carol", 1000)]
    eng = InferenceEngine(InferenceConfig(proximity_window_chars=600))
    edges = eng.proximity_edges(mentions, name_to_id)
    pairs = {frozenset((e.source, e.target)) for e in edges}
    check("near pair linked", frozenset(("idA", "idB")) in pairs)
    check("far pair not linked", frozenset(("idA", "idC")) not in pairs)
    check("only one edge", len(edges) == 1)
    check("tagged rule_cooccurrence",
          all(e.attributes.get("edge_source") == "rule_cooccurrence" for e in edges))
    # Cross-chunk: same window math holds across a chunk boundary (positions are
    # doc-absolute), and unresolved surfaces are skipped.
    m2 = [men("Alice", 5500), men("Bob", 5550), men("Ghost", 5560)]
    edges2 = eng.proximity_edges(m2, name_to_id)
    check("cross-boundary pair linked",
          {frozenset((e.source, e.target)) for e in edges2} == {frozenset(("idA", "idB"))})
    # Window 0 disables.
    eng0 = InferenceEngine(InferenceConfig(proximity_window_chars=0))
    check("window 0 disables", eng0.proximity_edges(mentions, name_to_id) == [])
    # proximity_min_count floors the weakest layer: Alice<->Bob co-occur twice
    # (two docs), Alice<->Carol once -> floor=2 keeps only the repeated pair.
    rep = [men("Alice", 0, "d1"), men("Bob", 50, "d1"), men("Carol", 80, "d1"),
           men("Alice", 0, "d2"), men("Bob", 50, "d2")]
    engf = InferenceEngine(InferenceConfig(proximity_window_chars=600, proximity_min_count=2))
    pf = {frozenset((e.source, e.target)) for e in engf.proximity_edges(rep, name_to_id)}
    check("floor keeps repeated pair", frozenset(("idA", "idB")) in pf)
    check("floor drops single-adjacency pairs", frozenset(("idA", "idC")) not in pf and len(pf) == 1)
    # The metadata mojibake fix leans on this repair covering umlauts.
    from postprocess.aggregator import clean_surface
    check("mojibake repaired", clean_surface("Stallup√∂nen") == "Stallupönen")


def test_org_marker_suspect_exemption() -> None:
    from config import QualityConfig
    from core.schema import Entity
    from postprocess.quality_review import QualityReviewer, _has_org_marker

    print("-- org-marker suspect exemption")

    # Distinctive org-form endings (German proper-org names + English) vs common
    # nouns. German names like Volkspartei tag as NOUN, so the propn gate would
    # otherwise flag the real party.
    check("party/movement/front marked",
          _has_org_marker("Deutschnationale Volkspartei")
          and _has_org_marker("Deutsch-Völkische Freiheitsbewegung")
          and _has_org_marker("Deutsche Arbeiterfront"))
    check("units marked",
          _has_org_marker("Freikorps") and _has_org_marker("Infanterie-Regiment"))
    check("english party marked", _has_org_marker("Social Democratic Party"))
    check("common nouns not marked",
          not _has_org_marker("Bankgewerbe") and not _has_org_marker("Armee")
          and not _has_org_marker("Bauernhaus"))

    def org(name):
        e = Entity(entity_id="", canonical_name=name, label="ORG")
        e.mention_count = 2
        e.confidence = 1.0
        e.attributes["propn_ratio"] = 0.0
        return e

    qr = QualityReviewer(QualityConfig(pos_gate=True, min_entity_mentions=1,
                                       min_entity_confidence=0.0))
    ents, _ = qr.rule_filter(
        [org("Deutschnationale Volkspartei"), org("Bankgewerbe")], [])
    flagged = {e.canonical_name: e.attributes.get("suspect_common_noun", False)
               for e in ents}
    check("real party not flagged",
          flagged.get("Deutschnationale Volkspartei") is not True, str(flagged))
    check("common noun still flagged",
          flagged.get("Bankgewerbe") is True, str(flagged))


def test_bench_bio() -> None:
    from benchmarks.common import decode_bio, build_ner_docs

    print("-- benchmark BIO decode")
    tm = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION"}
    toks = ["Adolf", "Hitler", "led", "the", "NSDAP", "in", "Munich"]
    tags = ["B-PER", "I-PER", "O", "O", "B-ORG", "O", "B-LOC"]
    spans = decode_bio(toks, tags, tm)
    check("bio spans",
          spans == [("Adolf Hitler", "PERSON"), ("NSDAP", "ORG"),
                    ("Munich", "LOCATION")], str(spans))
    # Base not in type_map dropped; orphan I- with no B- dropped.
    check("unmapped + orphan dropped",
          decode_bio(["a", "b"], ["B-MISC", "I-PER"], tm) == [])
    # Pseudo-doc grouping + per-doc dedup of a repeated surface.
    sents = [(["NSDAP"], ["B-ORG"]), (["NSDAP"], ["B-ORG"]), (["Munich"], ["B-LOC"])]
    docs = build_ner_docs(sents, tm, dataset="t", split="x", sents_per_doc=2)
    check("pseudo-doc split", len(docs) == 2, str(len(docs)))
    check("dedup within doc",
          sorted(e.name for e in docs[0].entities) == ["NSDAP"],
          str([e.name for e in docs[0].entities]))

    # IOB2 file parsing (UNER local path).
    import tempfile, os
    from benchmarks.common import parse_iob2
    fd, fp = tempfile.mkstemp(suffix=".iob2")
    os.write(fd, b"# sent_id = 1\nAdolf\tB-PER\nHitler\tI-PER\nin\tO\nMunich\tB-LOC\n\nNSDAP\tB-ORG\n")
    os.close(fd)
    try:
        sents = parse_iob2(fp)
        check("iob2 sentence split", len(sents) == 2, str(len(sents)))
        check("iob2 decode",
              decode_bio(*sents[0], tm) == [("Adolf Hitler", "PERSON"),
                                            ("Munich", "LOCATION")],
              str(decode_bio(*sents[0], tm)))
    finally:
        os.unlink(fp)


def test_coref_clusters() -> None:
    from types import SimpleNamespace
    from config import CoreferenceConfig
    from core.coreference import CoreferenceResolver
    from core.schema import EntityMention

    print("-- coref cluster re-attach")
    cfg = SimpleNamespace(coreference=CoreferenceConfig(
        enabled=True, pronoun_resolution=True, service_url="http://x:8000/"))
    r = CoreferenceResolver(cfg)
    check("service_url normalized", r.service_url == "http://x:8000", r.service_url)

    text = "Hitler spoke. He left."   # "He" at [14,16]
    m = EntityMention(text="Hitler", label="PERSON", start_char=0, end_char=6,
                      chunk_id="c0", doc_id="d1")
    clusters = [[[0, 6], [14, 16]]]   # service/fcoref shape: spans as [s,e]
    extra = r._mentions_from_clusters(clusters, text, [m], "d1", "c0", 0)
    check("one pronoun re-emitted", len(extra) == 1, str(len(extra)))
    if extra:
        e = extra[0]
        check("named identity carried", e.text == "Hitler", e.text)
        check("at pronoun span", (e.start_char, e.end_char) == (14, 16),
              f"{e.start_char},{e.end_char}")
        check("resolved_from he", e.attributes.get("resolved_from") == "he",
              str(e.attributes))
    # No named identity overlapping the cluster -> nothing emitted.
    check("no identity -> empty",
          r._mentions_from_clusters([[[14, 16]]], text, [], "d1", "c0", 0) == [])


def test_html_extraction() -> None:
    import sys
    from core.preprocessor import _clean_html

    print("-- html main-content extraction")
    # Article-sized body: trafilatura's content model needs real paragraph text to
    # tell body from chrome (it is built for web pages, not 1-line fragments).
    html = (
        "<html><head><title>News</title></head><body>"
        "<nav>Home About Contact Login</nav>"
        "<main><article><h1>The Meeting</h1>"
        "<p>Bilbo Baggins met Gandalf in the Shire on a grey morning. "
        "The wizard had come a long way and would not say why.</p>"
        "<p>They spoke for an hour about the road east, about dwarves, and "
        "about a door that opened on only one day of the year.</p>"
        "<p>By evening the bargain was struck and the company was thirteen "
        "strong, with Bilbo the fourteenth and least willing of them all.</p>"
        "</article></main>"
        "<footer>Copyright 2020 ACME. All rights reserved.</footer>"
        "</body></html>"
    )

    # Path 1: trafilatura main-content extraction (only when installed).
    try:
        import trafilatura  # noqa: F401
        have_traf = True
    except Exception:  # noqa: BLE001
        have_traf = False
    if have_traf:
        out = _clean_html(html)
        check("traf keeps body", "Bilbo Baggins met Gandalf" in out, out[:120])
        check("traf drops nav", "About Contact" not in out, out[:120])
        check("traf drops footer", "All rights reserved" not in out, out[-160:])

    # Path 2: BeautifulSoup fallback. Force trafilatura unavailable so the tag
    # blocklist path is exercised regardless of what's installed in the test env.
    saved = sys.modules.get("trafilatura", "MISSING")
    sys.modules["trafilatura"] = None  # `import trafilatura` now raises ImportError
    try:
        out = _clean_html(html)
    finally:
        if saved == "MISSING":
            sys.modules.pop("trafilatura", None)
        else:
            sys.modules["trafilatura"] = saved
    check("bs4 keeps body", "Bilbo Baggins met Gandalf" in out, out[:120])
    check("bs4 drops nav", "About Contact" not in out, out[:120])
    check("bs4 drops footer", "Copyright 2020" not in out, out[-160:])


def test_connection_type() -> None:
    from postprocess.tie_classes import connection_type

    print("-- connection-type axis")
    check("met -> physical", connection_type("met_with") == "physical")
    # The point of the axis: fought_against is a stance in tie_class but a
    # physical connection; influenced_by is a stance but ideological.
    check("fought -> physical", connection_type("fought_against") == "physical")
    check("influenced -> ideological", connection_type("influenced_by") == "ideological")
    check("opposed -> ideological", connection_type("opposed") == "ideological")
    check("member -> organizational", connection_type("member_of") == "organizational")
    check("born -> biographical", connection_type("born_in") == "biographical")
    check("cooccur -> cooccurrence", connection_type("co_occurs_with") == "cooccurrence")
    # Free-form fallbacks (belief checked before act) + target-type fallback.
    check("freeform funded -> physical", connection_type("secretly_funded") == "physical")
    check("freeform endorsed -> ideological",
          connection_type("publicly_endorsed") == "ideological")
    check("unknown + LOCATION -> biographical",
          connection_type("blorp", "LOCATION") == "biographical")
    check("unknown -> unspecified", connection_type("blorp") == "unspecified")


def _fake_site(pages, robots=None, sitemaps=None, redirects=None):
    """Build an injectable crawler fetcher over an in-memory site (no network).

    pages: normalized-url -> (content_type, html). robots: robots-url -> text.
    sitemaps: sitemap-url -> xml. redirects: requested-url -> final-url.
    """
    from core.crawler import FetchResult, normalize_url
    robots, sitemaps, redirects = robots or {}, sitemaps or {}, redirects or {}

    def fetch(url):
        if url.endswith("/robots.txt"):
            txt = robots.get(url)
            return FetchResult(url=url, content_type="text/plain", text=txt, ok=True) if txt is not None else None
        if url in sitemaps:
            return FetchResult(url=url, content_type="application/xml", text=sitemaps[url], ok=True)
        if url.endswith("/sitemap.xml"):
            return None
        final = redirects.get(url, url)
        rec = pages.get(normalize_url(final)) or pages.get(final)
        if rec is None:
            return None
        ctype, text = rec
        return FetchResult(url=final, content_type=ctype, text=text, ok=True)
    return fetch


def test_crawl_url_norm() -> None:
    from core.crawler import normalize_url
    print("-- crawler url normalization")
    check("lowercase + default port + fragment",
          normalize_url("HTTP://Ex.COM:80/p#frag") == "http://ex.com/p")
    check("trailing slash stripped", normalize_url("https://ex.com/a/") == "https://ex.com/a")
    check("root slash kept", normalize_url("https://ex.com/") == "https://ex.com/")
    check("double slash collapsed", normalize_url("https://ex.com/a//b") == "https://ex.com/a/b")
    check("tracking params dropped",
          normalize_url("https://ex.com/p?utm_source=x&q=1&fbclid=z") == "https://ex.com/p?q=1")
    check("bad input not raised", normalize_url("not a url") == "not a url")


def test_crawler() -> None:
    from core.crawler import Crawler, CrawlOptions
    print("-- crawler")

    def urls(docs):
        return sorted(d.source_path for d in docs)

    # A. BFS + depth, cycle safety, off-host exclusion.
    site = {
        "https://s.org/": ("text/html", "<a href='/a'>a</a><a href='/b'>b</a><a href='https://x.com/o'>off</a>"),
        "https://s.org/a": ("text/html", "<p>Alice met Bob.</p><a href='/c'>c</a>"),
        "https://s.org/b": ("text/html", "<p>Carol.</p><a href='/a'>loop</a>"),
        "https://s.org/c": ("text/html", "<p>Dave.</p>"),
    }
    f = _fake_site(site)
    docs = Crawler(CrawlOptions(delay=0, max_depth=3), fetch=f).crawl(["https://s.org/"])
    check("bfs reaches all in-scope",
          urls(docs) == ["https://s.org/", "https://s.org/a", "https://s.org/b", "https://s.org/c"], urls(docs))
    check("off-host excluded", all("x.com" not in u for u in urls(docs)))

    # B. max_pages hard cap.
    docs = Crawler(CrawlOptions(delay=0, max_pages=2), fetch=f).crawl(["https://s.org/"])
    check("max_pages caps docs", len(docs) == 2, str(len(docs)))

    # C. max_depth=0 -> only the seed.
    docs = Crawler(CrawlOptions(delay=0, max_depth=0), fetch=f).crawl(["https://s.org/"])
    check("depth 0 -> seed only", urls(docs) == ["https://s.org/"], urls(docs))

    # D. stay_under_path keeps the seed's directory prefix.
    site2 = {
        "https://s.org/docs/intro": ("text/html", "<a href='/docs/a'>a</a><a href='/about'>about</a>"),
        "https://s.org/docs/a": ("text/html", "<p>under docs</p>"),
        "https://s.org/about": ("text/html", "<p>elsewhere</p>"),
    }
    docs = Crawler(CrawlOptions(delay=0, stay_under_path=True),
                   fetch=_fake_site(site2)).crawl(["https://s.org/docs/intro"])
    check("stay_under_path includes sibling", "https://s.org/docs/a" in urls(docs))
    check("stay_under_path excludes /about", "https://s.org/about" not in urls(docs))

    # E. deny / allow regex.
    site3 = {
        "https://s.org/": ("text/html", "<a href='/keep/1'>k</a><a href='/private/x'>p</a><a href='/keep/2'>k2</a>"),
        "https://s.org/keep/1": ("text/html", "<p>1</p>"),
        "https://s.org/keep/2": ("text/html", "<p>2</p>"),
        "https://s.org/private/x": ("text/html", "<p>secret</p>"),
    }
    docs = Crawler(CrawlOptions(delay=0, deny=(r"/private",)),
                   fetch=_fake_site(site3)).crawl(["https://s.org/"])
    check("deny excludes", all("/private" not in u for u in urls(docs)))
    docs = Crawler(CrawlOptions(delay=0, allow=(r"/keep/",)),
                   fetch=_fake_site(site3)).crawl(["https://s.org/"])
    check("allow keeps seed entry point", "https://s.org/" in urls(docs))
    check("allow keeps matching link", "https://s.org/keep/1" in urls(docs))
    check("allow drops non-matching link", "https://s.org/private/x" not in urls(docs))

    # F. robots.txt Disallow respected.
    robots = {"https://s.org/robots.txt": "User-agent: *\nDisallow: /no\nCrawl-delay: 0\n"}
    site4 = {
        "https://s.org/": ("text/html", "<a href='/no/x'>no</a><a href='/yes'>yes</a>"),
        "https://s.org/no/x": ("text/html", "<p>blocked</p>"),
        "https://s.org/yes": ("text/html", "<p>ok</p>"),
    }
    docs = Crawler(CrawlOptions(delay=0, respect_robots=True),
                   fetch=_fake_site(site4, robots=robots)).crawl(["https://s.org/"])
    check("robots disallow respected", all("/no/" not in u for u in urls(docs)), urls(docs))
    check("robots allows the rest", "https://s.org/yes" in urls(docs))
    # opt-out fetches the disallowed path
    docs = Crawler(CrawlOptions(delay=0, respect_robots=False),
                   fetch=_fake_site(site4, robots=robots)).crawl(["https://s.org/"])
    check("robots opt-out fetches blocked", "https://s.org/no/x" in urls(docs))

    # G. sitemap-only discovery (page links nowhere; url lives in sitemap).
    sm = {"https://s.org/sitemap.xml":
          "<urlset><url><loc>https://s.org/hidden</loc></url></urlset>"}
    site5 = {
        "https://s.org/": ("text/html", "<p>no links here</p>"),
        "https://s.org/hidden": ("text/html", "<p>only in sitemap</p>"),
    }
    docs = Crawler(CrawlOptions(delay=0, use_sitemap=True),
                   fetch=_fake_site(site5, sitemaps=sm)).crawl(["https://s.org/"])
    check("sitemap seeds frontier", "https://s.org/hidden" in urls(docs), urls(docs))

    # H. sitemap index recursion.
    smx = {
        "https://s.org/sitemap.xml":
            "<sitemapindex><sitemap><loc>https://s.org/sm2.xml</loc></sitemap></sitemapindex>",
        "https://s.org/sm2.xml":
            "<urlset><url><loc>https://s.org/deep</loc></url></urlset>",
    }
    site6 = {"https://s.org/": ("text/html", "<p>x</p>"),
             "https://s.org/deep": ("text/html", "<p>deep</p>")}
    docs = Crawler(CrawlOptions(delay=0), fetch=_fake_site(site6, sitemaps=smx)).crawl(["https://s.org/"])
    check("sitemap index recursion", "https://s.org/deep" in urls(docs), urls(docs))

    # I. redirects: in-scope followed, out-of-scope dropped.
    site7 = {
        "https://s.org/": ("text/html", "<a href='/old'>old</a><a href='/leave'>leave</a>"),
        "https://s.org/new": ("text/html", "<p>moved here</p>"),
    }
    redir = {"https://s.org/old": "https://s.org/new", "https://s.org/leave": "https://evil.com/x"}
    docs = Crawler(CrawlOptions(delay=0),
                   fetch=_fake_site(site7, redirects=redir)).crawl(["https://s.org/"])
    check("in-scope redirect followed", "https://s.org/new" in urls(docs), urls(docs))
    check("out-of-scope redirect dropped", all("evil.com" not in u for u in urls(docs)))

    # J. non-html skipped; links not followed from non-html.
    site8 = {
        "https://s.org/": ("text/html", "<a href='/img'>img</a><a href='/ok'>ok</a>"),
        "https://s.org/img": ("image/png", "<a href='/buried'>should not follow</a>"),
        "https://s.org/ok": ("text/html", "<p>real</p>"),
    }
    docs = Crawler(CrawlOptions(delay=0), fetch=_fake_site(site8)).crawl(["https://s.org/"])
    check("non-html not a doc", "https://s.org/img" not in urls(docs))
    check("no links followed from non-html", "https://s.org/buried" not in urls(docs))

    # K. doc_id matches the URL ingestion path (so crawl + io.urls dedup).
    from core.schema import stable_id
    docs = Crawler(CrawlOptions(delay=0, max_depth=0), fetch=f).crawl(["https://s.org/"])
    check("doc_id scheme matches url ingestion",
          docs[0].doc_id == stable_id("https://s.org/", prefix="url_", length=10))


def test_coref_service_warmup() -> None:
    import json
    import types
    import urllib.error
    import urllib.request
    from config import CoreferenceConfig
    from core.coreference import CoreferenceResolver

    print("-- coref service warmup/fallback")

    class _Resp:
        def __init__(self, payload):
            self._p = json.dumps(payload).encode("utf-8")
        def read(self):
            return self._p
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def shim():
        return types.SimpleNamespace(
            coreference=CoreferenceConfig(service_url="http://x:8000",
                                          pronoun_resolution=True))

    real = urllib.request.urlopen

    # Healthy + loaded service: cluster comes back, warmed flag set, no fcoref load.
    def ok(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        if url.endswith("/health"):
            return _Resp({"status": "ok", "loaded": True})
        if url.endswith("/resolve"):
            return _Resp({"clusters": [[[[0, 3], [8, 11]]]]})
        raise ValueError(url)
    try:
        urllib.request.urlopen = ok
        r = CoreferenceResolver(shim())
        r._fcoref_failed = True  # ensure we never touch in-process in the test
        clusters = r._get_clusters("some chunk of text")
        check("service cluster returned", clusters == [[[0, 3], [8, 11]]], str(clusters))
        check("service marked warmed", r._service_warmed and not r._service_failed)

        # Unreachable service: warmup gives up cleanly, no crash, falls through.
        def down(req, timeout=None):
            raise urllib.error.URLError("connection refused")
        urllib.request.urlopen = down
        r2 = CoreferenceResolver(shim())
        r2._fcoref_failed = True
        check("unreachable -> None", r2._get_clusters("text") is None)
        check("unreachable -> service_failed", r2._service_failed)
    finally:
        urllib.request.urlopen = real


def test_citation_artifact_tagger() -> None:
    from core.schema import Entity
    from postprocess.tagger import Tagger

    print("-- citation/bibliography artifact tagging")

    def ent(name, label, **attrs):
        e = Entity(entity_id=name, canonical_name=name, label=label)
        e.attributes.update(attrs)
        return e

    tagged = [
        ent("Oxford University Press", "ORG"),
        ent("University of Chicago Press", "ORG"),
        ent("John Wiley & Sons", "ORG"),
        ent("Routledge", "ORG"),
        ent("the Wayback Machine", "ORG"),
        ent("Weeks, Marcus", "PERSON"),
        ent("Olsson PE", "PERSON"),
        ent("Todd M.", "PERSON"),
    ]
    # Must NOT be tagged: real actors that share surface features.
    clean = [
        ent("J. R. R. Tolkien", "PERSON"),
        ent("George W. Bush", "PERSON"),
        ent("Malcolm X", "PERSON"),
        ent("Martin Luther King Jr.", "PERSON"),
        ent("Max Weber", "PERSON"),
        ent("Facebook", "ORG"),
        ent("New York", "LOCATION"),
        ent("Marx, Karl", "PERSON", is_author=True),   # narrator exempt
    ]
    Tagger().tag(tagged + clean, [])

    miss = [e.canonical_name for e in tagged if not e.tags.get("citation_artifact")]
    check("all artifacts tagged", not miss, f"untagged: {miss}")
    wrong = [e.canonical_name for e in clean if e.tags.get("citation_artifact")]
    check("real actors untagged", not wrong, f"wrongly tagged: {wrong}")


def test_crawler_dir_prefix() -> None:
    from core.crawler import _dir_prefix
    print("-- crawler stay_under_path dir prefix")
    # A single-segment seed (trailing slash stripped by normalize) must not widen to '/'.
    check("single-segment seed -> self dir", _dir_prefix("/docs") == "/docs/", _dir_prefix("/docs"))
    # A page seed scopes to its parent so siblings are in scope.
    check("page seed -> parent dir", _dir_prefix("/docs/intro") == "/docs/",
          _dir_prefix("/docs/intro"))
    check("deep page -> parent dir", _dir_prefix("/a/b/page.html") == "/a/b/",
          _dir_prefix("/a/b/page.html"))
    check("root stays root", _dir_prefix("/") == "/", _dir_prefix("/"))


def test_scorer_directed_relations() -> None:
    from evaluation.gold_schema import GoldDocument, GoldEntity, GoldRelation, GoldSet
    from evaluation.scorer import score_relations
    print("-- scorer directed relation matching")
    gold = GoldSet([GoldDocument("d",
        entities=[GoldEntity("Alice", "PERSON"), GoldEntity("Bob", "PERSON")],
        relations=[GoldRelation("Alice", "Bob", "recruited"),     # asymmetric
                   GoldRelation("Alice", "Bob", "married_to")])])  # symmetric
    pred_entities = [{"canonical_name": "Alice", "label": "PERSON"},
                     {"canonical_name": "Bob", "label": "PERSON"}]
    # Both predicted edges are REVERSED (Bob->Alice).
    pred_edges = [{"source_name": "Bob", "target_name": "Alice", "rel_type": "recruited"},
                  {"source_name": "Bob", "target_name": "Alice", "rel_type": "married_to"}]
    d = score_relations(gold, pred_entities, pred_edges, type_sensitive=True, directed=True)
    check("directed: reversed asymmetric is wrong, symmetric ok (tp=1)",
          d["overall"]["tp"] == 1, str(d["overall"]))
    u = score_relations(gold, pred_entities, pred_edges, type_sensitive=True, directed=False)
    check("undirected: both match (tp=2)", u["overall"]["tp"] == 2, str(u["overall"]))


def test_scorer_entity_one_to_one() -> None:
    from evaluation.gold_schema import GoldDocument, GoldEntity, GoldSet
    from evaluation.scorer import score_entities
    print("-- scorer 1:1 entity matching (penalize over-split)")
    gold = GoldSet([GoldDocument("d", entities=[GoldEntity("Alice", "PERSON")])])
    # Two predicted nodes both claim the one gold entity: a split.
    pred = [{"canonical_name": "Alice", "label": "PERSON"},
            {"canonical_name": "Alice", "label": "PERSON"}]
    r = score_entities(gold, pred, type_sensitive=True)
    o = r["overall"]
    check("precision penalized for split (0.5)", o["precision"] == 0.5, str(o))
    check("recall still full (1.0)", o["recall"] == 1.0, str(o))


def test_newman_cooccurrence() -> None:
    from config import InferenceConfig
    from core.schema import Entity
    from postprocess.canonical_inference import InferenceEngine
    print("-- Newman bipartite-projection co-occurrence weight")

    def ent(eid, docs):
        e = Entity(entity_id=eid, canonical_name=eid, label="PERSON")
        e.doc_ids = docs
        return e
    # X,Y alone in a 2-entity doc (strength 1.0); P,Q in a 10-entity doc (1/9 each).
    big = [f"f{i}" for i in range(8)]
    ents = [ent("X", ["small"]), ent("Y", ["small"]),
            ent("P", ["big"]), ent("Q", ["big"])] + [ent(b, ["big"]) for b in big]
    cfg = InferenceConfig(cooccurrence_min_shared_docs=1, enable_proximity_edges=False)
    edges = InferenceEngine(cfg).cooccurrence_edges(ents)
    by_pair = {frozenset((e.source, e.target)): e.attributes.get("cooccur_strength") for e in edges}
    sxy = by_pair.get(frozenset(("X", "Y")))
    spq = by_pair.get(frozenset(("P", "Q")))
    check("small-doc pair strength 1.0", sxy == 1.0, str(sxy))
    check("big-doc pair down-weighted", spq is not None and spq < 0.2, str(spq))


def test_disparity_backbone() -> None:
    from core.schema import Relationship
    from postprocess.backbone import disparity_filter
    print("-- disparity-filter backbone")

    def coo(a, b, w):
        return Relationship(source=a, target=b, rel_type="co_occurs_with", doc_id="d",
                            origin="inferred",
                            attributes={"edge_source": "rule_cooccurrence", "cooccur_count": w})
    # K4 on A,B,C strong (10); D attached to each by a lone weight-1 tie.
    strong = [coo("A", "B", 10), coo("A", "C", 10), coo("B", "C", 10)]
    weak = [coo("A", "D", 1), coo("B", "D", 1), coo("C", "D", 1)]
    kept, dropped = disparity_filter(list(strong + weak), alpha=0.3)
    kept_pairs = {frozenset((r.source, r.target)) for r in kept}
    check("weak ties dropped", dropped == 3, f"dropped={dropped}")
    check("strong triangle kept",
          all(frozenset(p) in kept_pairs for p in (("A", "B"), ("A", "C"), ("B", "C"))),
          str(kept_pairs))
    # alpha=0 stamps but drops nothing.
    kept0, dropped0 = disparity_filter(list(strong + weak), alpha=0.0)
    check("alpha=0 drops nothing", dropped0 == 0 and len(kept0) == 6, str(dropped0))
    check("alpha stamped on edges",
          all("disparity_alpha" in r.attributes for r in kept0), "missing alpha")


def test_signed_balance() -> None:
    from postprocess.graph_metrics import _signed_balance
    print("-- signed structural balance")
    edges = [
        {"Source": "X", "Target": "Y", "polarity": "positive"},
        {"Source": "Y", "Target": "Z", "polarity": "positive"},
        {"Source": "X", "Target": "Z", "polarity": "positive"},   # all + -> balanced
        {"Source": "P", "Target": "Q", "polarity": "positive"},
        {"Source": "P", "Target": "R", "polarity": "positive"},
        {"Source": "Q", "Target": "R", "polarity": "negative"},   # +,+,- -> unbalanced
    ]
    b = _signed_balance(edges)
    check("one balanced triangle", b["balanced"] == 1, str(b))
    check("one unbalanced triangle", b["unbalanced"] == 1, str(b))
    check("balanced_pct 50", b["balanced_pct"] == 50.0, str(b))


def test_polarity_conflicts() -> None:
    from postprocess.graph_metrics import _polarity_conflicts
    print("-- polarity conflicts")
    edges = [
        {"Source": "A", "Target": "B", "polarity": "positive", "rel_type": "allied_with"},
        {"Source": "B", "Target": "A", "polarity": "negative", "rel_type": "fought_against"},
        {"Source": "C", "Target": "D", "polarity": "positive", "rel_type": "allied_with"},
        {"Source": "C", "Target": "E", "polarity": "neutral", "rel_type": "met_with"},
    ]
    c = _polarity_conflicts(edges, {"A": "Alpha", "B": "Beta"})
    check("one conflicting dyad", c["conflicting_dyads"] == 1, str(c))
    s = c["sample"][0]
    check("conflict pair names mapped", {s["source"], s["target"]} == {"Alpha", "Beta"}, str(s))
    check("conflict lists both signs",
          s["positive"] == ["allied_with"] and s["negative"] == ["fought_against"], str(s))


def test_generic_ontology() -> None:
    from types import SimpleNamespace
    from postprocess.ontology import (resolve_relation_ontology, OntologyAligner,
                                       GENERIC_RELATION_ONTOLOGY)
    print("-- generic relation ontology")
    on = SimpleNamespace(ontology=SimpleNamespace(relations=None, relation_guide=None, enabled=True))
    onto = resolve_relation_ontology(on, None)
    check("generic ontology is the fallback", "funded" in onto and "employed_by" in onto, str(len(onto)))
    off = SimpleNamespace(ontology=SimpleNamespace(relations=None, relation_guide=None, enabled=False))
    check("ontology disabled -> free-form", resolve_relation_ontology(off, None) == {}, "")
    al = OntologyAligner(GENERIC_RELATION_ONTOLOGY, fuzzy_threshold=0.82)
    # Verbose rel_types pulled from a real crawl; the tail folds to canonical.
    want = {"provided_funding_to": "funded", "president_of": "led",
            "works_for": "employed_by", "trustee_of": "member_of",
            "former_employee_of": "employed_by", "controlled_by_initially": "owned_by",
            "sent_letters_to_requesting_compliance_info_about_funding": "met_with"}
    for raw, exp in want.items():
        check(f"align {raw[:22]}", al.align(raw) == exp, f"got {al.align(raw)}")
    # Direction must survive: funded_by must NOT collapse into funded.
    check("funded_by keeps direction", al.align("funded_by") == "funded_by", str(al.align("funded_by")))


def test_org_name_cleanup() -> None:
    from config import DedupConfig
    from core.schema import Entity
    from postprocess.deduplicator import (Deduplicator, _strip_leading_the,
                                          _singularize_org)
    print("-- org name cleanup")
    for raw, want in {"the Lilly Endowment": "Lilly Endowment",
                      "The Manhattan Institute": "Manhattan Institute",
                      "Die Linke": "Die Linke",            # German article kept
                      "Ford Foundation": "Ford Foundation"}.items():
        check(f"strip-the {raw!r}", _strip_leading_the(raw) == want, f"got {_strip_leading_the(raw)!r}")
    for raw, want in {"Knight Foundations": "Knight Foundation",
                      "Carnegie Universities": "Carnegie University",
                      "Heritage Societies": "Heritage Society"}.items():
        check(f"singularize {raw!r}", _singularize_org(raw) == want, f"got {_singularize_org(raw)!r}")
    # Safety: singularize only when a singular sibling exists; real plural names stay.
    d = Deduplicator(DedupConfig())
    ents = [Entity(entity_id="e1", canonical_name="Knight Foundations", label="ORG", mention_count=2),
            Entity(entity_id="e2", canonical_name="Knight Foundation", label="ORG", mention_count=5),
            Entity(entity_id="e3", canonical_name="Open Society Foundations", label="ORG", mention_count=4),
            Entity(entity_id="e4", canonical_name="the Lilly Endowment", label="ORG", mention_count=3)]
    names = sorted(e.canonical_name for e in d._clean_org_surfaces(ents))
    check("plural folds onto existing singular",
          "Knight Foundation" in names and "Knight Foundations" not in names, str(names))
    check("real plural name kept (no sibling)", "Open Society Foundations" in names, str(names))
    check("leading 'the' stripped", "Lilly Endowment" in names, str(names))
    check("knight pair merged (4 -> 3)", len(names) == 3, str(names))


def test_sparse_chunk_gate() -> None:
    from intelligence.base import _dense_enough
    from core.schema import EntityMention
    print("-- sparse chunk cost gate")
    def m(t, sc):
        return EntityMention(text=t, label="PERSON", start_char=sc, end_char=sc + len(t),
                             chunk_id="c", doc_id="d")
    txt = "Alice " + "x " * 40 + "Bob " + "y " * 300 + "Carol"
    ms = [m("Alice", 0), m("Bob", txt.find("Bob")), m("Carol", txt.find("Carol"))]
    check("two entities in one window -> send", _dense_enough(ms, txt, 0, 200, 2) is True, "")
    check("single entity -> skip", _dense_enough([ms[0]], txt, 0, 200, 2) is False, "")
    far = [m("Alice", 0), m("Carol", txt.find("Carol"))]
    check("entities >window apart -> skip", _dense_enough(far, txt, 0, 200, 2) is False, "")


def test_edge_consolidation() -> None:
    from core.schema import Relationship
    from postprocess.aggregator import _consolidate_relationships
    print("-- cross-chunk edge consolidation")
    def R(ev, doc="d"):
        return Relationship(source="A", target="B", rel_type="met_with", doc_id=doc, evidence=ev)
    out = _consolidate_relationships([R("They met in Berlin."), R("They met in Berlin."),
                                      R("Later they met again.")])
    check("overlap dup dropped, corroboration kept", len(out) == 2, str(len(out)))
    out2 = _consolidate_relationships([R("They met in Berlin."), R("They met in Berlin.", doc="d2")])
    check("cross-doc same evidence kept", len(out2) == 2, str(len(out2)))


def test_relation_type_signatures() -> None:
    from core.schema import Relationship
    from postprocess.ontology import check_relation_types
    print("-- ASP-style relation type signatures")
    type_of = {"p1": "PERSON", "p2": "PERSON", "o1": "ORG",
               "l1": "LOCATION", "x1": "ROLE"}  # ROLE not a core type -> wildcard

    def R(src, tgt, rt):
        return Relationship(source=src, target=tgt, rel_type=rt, doc_id="d", evidence="e")

    good = R("p1", "o1", "led")          # person -> org: ok
    bad = R("p1", "l1", "led")           # person -> place: violation
    wild = R("p1", "x1", "born_in")      # exotic target type: wildcard, ok
    loose = R("p1", "p2", "supported")   # no signature: never flagged
    out, n = check_relation_types([good, bad, wild, loose], type_of, drop=False)
    check("one violation flagged", n == 1, str(n))
    check("violating edge tagged", bad.attributes.get("type_violation") is True, "")
    check("valid edge untouched", "type_violation" not in good.attributes, "")
    check("wildcard target untouched", "type_violation" not in wild.attributes, "")
    check("loose relation untouched", "type_violation" not in loose.attributes, "")
    check("nothing dropped when drop=False", len(out) == 4, str(len(out)))
    out2, n2 = check_relation_types([R("p1", "l1", "led")], type_of, drop=True)
    check("violation dropped when drop=True", len(out2) == 0 and n2 == 1, str(len(out2)))


def test_quality_pillars() -> None:
    from types import SimpleNamespace
    from postprocess.graph_metrics import quality_pillars
    print("-- KGC quality pillars")
    edges = [
        {"edge_source": "llm_extracted"},                 # asserted
        {"edge_source": "metadata"},                       # asserted
        {"edge_source": "rule_cooccurrence", "type_violation": True},  # weak + bad
        {"edge_source": ""},                               # no provenance
    ]
    tables = SimpleNamespace(edges=edges)
    report = {"qa_substantive": {"largest_cc_pct": 80.0, "isolates": 2},
              "conflicts": {"conflicting_dyads": 1}}
    qp = quality_pillars(report, tables)
    check("asserted proxy 2/4 = 50%", qp["accuracy_proxy"]["asserted_tier_pct"] == 50.0,
          str(qp["accuracy_proxy"]))
    check("provenance 3/4 = 75%", qp["provenance"]["edges_with_source_pct"] == 75.0,
          str(qp["provenance"]))
    check("type violations counted", qp["consistency"]["type_violations"] == 1,
          str(qp["consistency"]))
    check("polarity conflicts surfaced", qp["consistency"]["polarity_conflicts"] == 1, "")
    check("completeness carries cc%", qp["completeness_proxy"]["largest_cc_pct"] == 80.0, "")


def test_evidence_grounding() -> None:
    from core.schema import Relationship
    from intelligence.base import _tag_ungrounded_evidence, _name_in_evidence
    print("-- AEVS anchor / evidence grounding")
    check("token grounds full name", _name_in_evidence("Joseph Goebbels", "goebbels spoke"), "")
    # A shared stopword token must not ground when the significant tokens are absent.
    check("article token ignored",
          not _name_in_evidence("the Ford Foundation", "the committee met today"), "")
    check("org acronym grounds", _name_in_evidence("the NSDAP", "joined the nsdap"), "")

    def R(s, t, ev, origin="extracted"):
        return Relationship(source=s, target=t, rel_type="met_with", doc_id="d",
                            evidence=ev, origin=origin)
    grounded = R("Goebbels", "Hitler", "Goebbels met Hitler in Berlin.")
    by_token = R("Joseph Goebbels", "Propaganda Ministry", "Goebbels ran the ministry.")
    bad = R("Alice", "Bob", "The committee approved the annual budget.")
    narr = R("Johann Alff", "NSDAP", "I joined the party in 1931.")  # author endpoint
    inferred = R("X", "Y", "", origin="inferred")
    n = _tag_ungrounded_evidence([grounded, by_token, bad, narr, inferred], author="Johann Alff")
    check("one ungrounded flagged", n == 1, str(n))
    check("bad edge tagged", bad.attributes.get("evidence_ungrounded") == "true", "")
    check("grounded untagged", "evidence_ungrounded" not in grounded.attributes, "")
    check("token-grounded untagged", "evidence_ungrounded" not in by_token.attributes, "")
    check("narrator endpoint exempt", "evidence_ungrounded" not in narr.attributes, "")
    check("inferred/no-evidence skipped", "evidence_ungrounded" not in inferred.attributes, "")


def test_qid_consolidation() -> None:
    from core.schema import Entity, Relationship
    from postprocess.wikidata import consolidate_by_qid
    print("-- Wikidata QID consolidation")

    def ent(eid, name, qid, mentions):
        e = Entity(entity_id=eid, canonical_name=name, label="PERSON", mention_count=mentions)
        e.attributes["wikidata_qid"] = qid
        return e
    ents = [ent("ent_a", "Goebbels", "Q1", 5),
            ent("ent_b", "Joseph Goebbels", "Q1", 10),
            ent("ent_c", "Hitler", "Q2", 20)]
    rels = [Relationship(source="ent_a", target="ent_c", rel_type="met_with", doc_id="d")]
    new_e, new_r, _ = consolidate_by_qid(ents, rels, {})
    check("same-QID entities merged (3 -> 2)", len(new_e) == 2, str(len(new_e)))
    surviving = {e.entity_id for e in new_e}
    check("edge remapped onto surviving id",
          new_r and new_r[0].source in surviving and new_r[0].source != "ent_a", str(new_r))
    goeb = [e for e in new_e if "Goebbels" in e.canonical_name][0]
    check("absorbed name kept as alias",
          "Goebbels" in {goeb.canonical_name, *goeb.aliases}, str(goeb.aliases))


def test_reference_stripping() -> None:
    from core.preprocessor import _strip_trailing_sections
    print("-- reference-section stripping")
    body = "\n".join([f"Body sentence number {i} with real content here." for i in range(20)])
    refs = "References\n" + "\n".join([f"Smith, J. ({1990+i}). A paper. Oxford University Press."
                                       for i in range(15)])
    cut = _strip_trailing_sections(body + "\n" + refs)
    check("reference tail removed", "Oxford University Press" not in cut, "refs survived")
    check("body kept", "Body sentence number 19" in cut, "body lost")
    # A 'Notes' heading in the first half must NOT truncate the document.
    early = "Notes\n" + body
    check("early heading not cut", _strip_trailing_sections(early) == early, "early cut")
    check("short doc untouched", _strip_trailing_sections("a\nb\nReferences\nc") ==
          "a\nb\nReferences\nc", "short doc cut")


def test_narrative_transitions() -> None:
    from postprocess.narrative import build_transitions, categorize
    print("-- narrative-sequence transitions")
    check("categorize war", categorize("served at the front in the war") == "war_combat")
    check("categorize politics", categorize("joined the party rally") == "politics_party")
    seq = [{"year": 1916, "description": "war front soldier"},
           {"year": 1919, "description": "unemployed in a factory"},
           {"year": 1921, "description": "joined the party"}]
    timeline = [{**s, "doc_id": "d1"} for s in seq] + [{**s, "doc_id": "d2"} for s in seq]
    trans, elems = build_transitions(timeline)
    t = trans.get(("war_combat", "work_economic"))
    check("transition aggregated across docs", t and t["weight"] == 2 and len(t["docs"]) == 2,
          str(t))
    check("self-transitions collapsed", ("war_combat", "war_combat") not in trans)


def test_gexf_parallel_edges() -> None:
    import tempfile
    from pathlib import Path
    try:
        import networkx as nx
    except Exception:  # noqa: BLE001 - networkx optional
        print("-- gexf parallel edges (skipped: no networkx)")
        return
    from postprocess.exporter import Exporter
    from postprocess.gephi_builder import GraphTables

    print("-- gexf parallel-edge weight preservation")
    nodes = [{"Id": "a", "Label": "A"}, {"Id": "b", "Label": "B"}]
    # Same unordered pair, two relation types: a plain Graph would drop one.
    edges = [
        {"Source": "a", "Target": "b", "Type": "Undirected", "Label": "met_with",
         "rel_type": "met_with", "tie_class": "interaction", "Weight": 3},
        {"Source": "a", "Target": "b", "Type": "Undirected", "Label": "supported",
         "rel_type": "supported", "tie_class": "stance", "Weight": 1},
    ]
    tables = GraphTables(nodes=nodes, edges=edges)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "g.gexf"
        Exporter._write_gexf(tables, p)
        G = nx.read_gexf(p)
        check("both endpoints kept", G.number_of_nodes() == 2, str(G.nodes))
        check("parallel edge not dropped", G.number_of_edges() == 1, str(G.edges))
        w = G["a"]["b"].get("weight")
        check("weights summed (3+1=4)", w == 4, f"weight={w}")
        rt = G["a"]["b"].get("rel_type", "")
        check("both rel_types retained",
              "met_with" in rt and "supported" in rt, f"rel_type={rt}")


def main() -> int:
    test_json_repair()
    test_checkpoint_scoring()
    test_dedup_folds()
    test_relation_guide()
    test_evidence_tiers()
    test_exclude_edge_source()
    test_proximity_edges()
    test_org_marker_suspect_exemption()
    test_citation_artifact_tagger()
    test_gexf_parallel_edges()
    test_crawler_dir_prefix()
    test_scorer_directed_relations()
    test_scorer_entity_one_to_one()
    test_newman_cooccurrence()
    test_disparity_backbone()
    test_signed_balance()
    test_polarity_conflicts()
    test_generic_ontology()
    test_org_name_cleanup()
    test_sparse_chunk_gate()
    test_edge_consolidation()
    test_relation_type_signatures()
    test_quality_pillars()
    test_evidence_grounding()
    test_qid_consolidation()
    test_reference_stripping()
    test_narrative_transitions()
    test_bench_bio()
    test_coref_clusters()
    test_connection_type()
    test_html_extraction()
    test_crawl_url_norm()
    test_crawler()
    test_coref_service_warmup()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("all tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
