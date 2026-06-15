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


def main() -> int:
    test_json_repair()
    test_checkpoint_scoring()
    test_dedup_folds()
    test_relation_guide()
    test_evidence_tiers()
    test_exclude_edge_source()
    test_proximity_edges()
    test_org_marker_suspect_exemption()
    test_bench_bio()
    test_coref_clusters()
    test_connection_type()
    test_html_extraction()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("all tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
