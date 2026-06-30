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
        # Weak model leaks reasoning into the JSON: an arrow note after a string,
        # and bare prose between array elements. Both lost the whole doc before.
        ("arrow annotation", '{"aliases": ["NSV", "NSDSP" -> Note: a typo for NSDAP]}',
         {"aliases": ["NSV", "NSDSP"]}),
        ("array element prose",
         '{"aliases": [\n"Froh", Froh is an activity.\n"Lib", refers to the party.\n]}',
         {"aliases": ["Froh", "Lib"]}),
        # Stray sentence punctuation the model leaks after a value's close quote,
        # before the next key (`"...powerful;".` then a newline + "confidence").
        ("stray dot after value", '{"e": "all-powerful;".\n  "k": 2}',
         {"e": "all-powerful;", "k": 2}),
        # Parenthetical close leaked outside the quote: the model opened `(` inside
        # the value, shut the string early, and left the `)` before the close
        # (`"evidence": "(1948 Indian film")`). Lost a whole redocred doc.
        ("stray paren before close", '{"e": "(1948 film")\n}', {"e": "(1948 film"}),
        # Gemma ignores responseMimeType and prepends a prose paraphrase of the
        # prompt - which carries stray brackets ("confidence float [0, 1].") - then
        # the real JSON. Anchoring on the first bracket lands in the prose; the object
        # span must win. Lost every gemma-4-31b doc before this.
        ("prose preamble with stray bracket",
         'Task: extract. confidence float [0, 1].\n{"doc_a": {"entities": []}}',
         {"doc_a": {"entities": []}}),
        ("stray paren before key", '{"e": "(1948 film")\n  "k": 2}',
         {"e": "(1948 film", "k": 2}),
        # A period that legitimately ENDS the string content must be left alone.
        ("period inside string", '{"e": "He left.", "k": 2}', {"e": "He left.", "k": 2}),
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

    # Cross-type folds (real InfluenceWatch / OREM artifacts). "the Berger Action
    # Fund" (ORG) and "Berger Action Fund" (PERSON) are one entity; the "Fund"
    # marker makes ORG the winner over the default person-preference.
    ents = [ent("the Berger Action Fund", "ORG", 1), ent("Berger Action Fund", "PERSON", 1)]
    out = d._resolve_cross_type(ents)
    check("the-prefixed org folds with bare person", len(out) == 1,
          str([(e.canonical_name, e.label) for e in out]))
    check("org marker wins the cross-type tie", out and out[0].label == "ORG",
          str([(e.canonical_name, e.label) for e in out]))

    # Trailing acronym gloss: "X (OEM)" folds with "X"; higher mentions wins.
    ents = [ent("Oregon Department of Emergency Management", "ORG", 3),
            ent("Oregon Department of Emergency Management (OEM)", "INSTITUTION", 1)]
    out = d._resolve_cross_type(ents)
    check("parenthetical-acronym variant folds", len(out) == 1,
          str([(e.canonical_name, e.label) for e in out]))

    # A real person mistyped ORG (no org marker) still resolves to PERSON.
    ents = [ent("Eric Kessler", "PERSON", 2), ent("Eric Kessler", "ORG", 1)]
    out = d._resolve_cross_type(ents)
    check("person without org marker stays person", out and out[0].label == "PERSON",
          str([(e.canonical_name, e.label) for e in out]))


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


def test_coref_prompt_hint() -> None:
    # Coref clusters never reached the LLM (the resolved name deduped into the
    # candidate list). The REFERENCE KEY surfaces pronoun->name so the model can
    # attribute a cross-sentence tie. Gated: no resolved mentions -> no block, so a
    # non-coref prompt is byte-identical.
    from intelligence.prompts import coref_reference_block, build_extraction_prompt
    from core.schema import EntityMention
    print("-- coref reference-key prompt hint")
    plain = EntityMention(text="John Smith", label="PERSON", start_char=0, end_char=10,
                          chunk_id="c", doc_id="d")
    resolved = EntityMention(text="John Smith", label="PERSON", start_char=40, end_char=42,
                             chunk_id="c", doc_id="d", attributes={"resolved_from": "he"})
    resolved2 = EntityMention(text="John Smith", label="PERSON", start_char=60, end_char=63,
                              chunk_id="c", doc_id="d", attributes={"resolved_from": "his"})
    blk = coref_reference_block([plain, resolved, resolved2])
    check("reference key lists the name", "John Smith" in blk and "REFERENCE KEY" in blk, blk)
    check("both pronouns grouped", '"he"' in blk and '"his"' in blk, blk)
    # No resolved mentions -> empty, and the prompt is unchanged vs candidates-only.
    check("no coref -> no block", coref_reference_block([plain]) == "", "")
    p_plain = build_extraction_prompt("Some text.", [plain], ["PERSON"])
    p_coref = build_extraction_prompt("Some text.", [plain, resolved], ["PERSON"])
    check("non-coref prompt has no reference key", "REFERENCE KEY" not in p_plain, "")
    check("coref prompt injects the key", "REFERENCE KEY" in p_coref, "")


def test_coref_observability() -> None:
    # A silent heuristic fallback (no fastcoref) looked like coref "not activating".
    # The resolver now records the live backend + a mention count so a run can show
    # it. With no service and no fastcoref, pronoun_mentions resolves to heuristic.
    from types import SimpleNamespace
    from config import CoreferenceConfig
    from core.coreference import CoreferenceResolver
    from core.schema import EntityMention
    print("-- coref observability (backend + counts)")
    cfg = SimpleNamespace(coreference=CoreferenceConfig(
        enabled=True, pronoun_resolution=True))   # no service_url, no fastcoref
    r = CoreferenceResolver(cfg)
    r._fcoref_failed = True        # force the fastcoref-unavailable path
    check("backend unset before use", r.pronoun_backend == "", r.pronoun_backend)
    text = "Hitler spoke at length. He then left the hall."   # one unambiguous PERSON
    m = EntityMention(text="Hitler", label="PERSON", start_char=0, end_char=6,
                      chunk_id="c0", doc_id="d1")
    out = r.pronoun_mentions(text, [m], "d1", "c0", 0)
    check("heuristic backend recorded", r.pronoun_backend == "heuristic", r.pronoun_backend)
    check("pronoun count tracked", r.n_pronoun_added == len(out) and len(out) >= 1,
          f"added={r.n_pronoun_added} out={len(out)}")


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

    # L. enqueue-time dedup: a fully cross-linked site (every page links to every other)
    # must not balloon the frontier - each url is queued once, fetched once.
    from collections import Counter
    nodes = ["https://s.org/", "https://s.org/a", "https://s.org/b", "https://s.org/c"]
    links = "".join(f"<a href='{u}'>l</a>" for u in nodes)
    mesh = {u: ("text/html", f"<p>{u}</p>{links}") for u in nodes}
    base = _fake_site(mesh)
    hits: Counter = Counter()

    def counting(url):
        r = base(url)
        if r is not None and r.url in mesh:
            hits[r.url] += 1
        return r
    c = Crawler(CrawlOptions(delay=0, max_depth=5), fetch=counting)
    docs = c.crawl(["https://s.org/"])
    check("cross-linked: all pages reached", len(docs) == 4, urls(docs))
    check("cross-linked: each page fetched once", all(v == 1 for v in hits.values()), str(dict(hits)))
    check("cross-linked: frontier deduped (queued == uniques)", len(c._queued) == 4, str(len(c._queued)))


def test_crawl_checkpoint() -> None:
    import shutil
    import tempfile
    from pathlib import Path
    from core.crawler import Crawler, CrawlOptions
    print("-- crawler checkpoint + progress")

    site = {
        "https://s.org/": ("text/html", "<a href='/a'>a</a><a href='/b'>b</a>"),
        "https://s.org/a": ("text/html", "<p>Alice.</p><a href='/c'>c</a>"),
        "https://s.org/b": ("text/html", "<p>Bob.</p><a href='/d'>d</a>"),
        "https://s.org/c": ("text/html", "<p>Carol.</p>"),
        "https://s.org/d": ("text/html", "<p>Dave.</p>"),
    }

    def counting_site():
        from collections import Counter
        base = _fake_site(site)
        hits: Counter = Counter()

        def fetch(url):
            res = base(url)
            if res is not None and res.url in site:
                hits[res.url] += 1
            return res
        return fetch, hits

    cpdir = Path(tempfile.mkdtemp(prefix="crawlckpt_"))
    try:
        # Run 1: cap at 2 pages, checkpoint every page. Frontier left non-empty.
        events1: list[dict] = []
        f1, hits1 = counting_site()
        c1 = Crawler(CrawlOptions(delay=0, max_pages=2, checkpoint_every=1),
                     fetch=f1, on_progress=events1.append,
                     checkpoint_dir=str(cpdir), resume=False)
        docs1 = c1.crawl(["https://s.org/"])
        check("capped run stops at max_pages", len(docs1) == 2, str(len(docs1)))
        check("capped run is not complete", c1.complete is False, "")
        check("checkpoint files written",
              (cpdir / "state.json").exists() and (cpdir / "pages.jsonl").exists(), "")
        check("progress page events fired",
              any(e.get("event") == "page" for e in events1), str(events1[:2]))

        # Run 2: resume with a higher cap; must NOT refetch run-1 pages and must finish.
        events2: list[dict] = []
        f2, hits2 = counting_site()
        c2 = Crawler(CrawlOptions(delay=0, max_pages=10, checkpoint_every=1),
                     fetch=f2, on_progress=events2.append,
                     checkpoint_dir=str(cpdir), resume=True)
        docs2 = c2.crawl(["https://s.org/"])
        got = sorted(d.source_path for d in docs2)
        check("resume reaches all pages", len(docs2) == 5, got)
        check("resume marks complete", c2.complete is True, "")
        check("resume did not refetch run-1 pages",
              hits2["https://s.org/"] == 0 and hits2["https://s.org/a"] == 0, str(dict(hits2)))
        check("resume only fetched the remainder",
              hits2["https://s.org/b"] == 1 and hits2["https://s.org/c"] == 1
              and hits2["https://s.org/d"] == 1, str(dict(hits2)))
        check("resume start event flagged",
              any(e.get("event") == "start" and e.get("resumed") for e in events2), "")
        check("done event reports complete",
              any(e.get("event") == "done" and e.get("complete") for e in events2), "")

        # Fingerprint guard: different seeds/scope must ignore the checkpoint.
        c3 = Crawler(CrawlOptions(delay=0, max_pages=10, deny=("/x",)),
                     fetch=counting_site()[0], checkpoint_dir=str(cpdir), resume=True)
        c3._fingerprint = c3._make_fingerprint(["https://s.org/"])
        check("changed scope -> different fingerprint",
              c3._fingerprint != c2._fingerprint, "")
    finally:
        shutil.rmtree(cpdir, ignore_errors=True)


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


def test_relation_family_scoring() -> None:
    from evaluation.gold_schema import GoldDocument, GoldEntity, GoldRelation, GoldSet
    from evaluation.scorer import score_relations
    print("-- scorer relation-family (tie-class) matching")
    gold = GoldSet([GoldDocument("d",
        entities=[GoldEntity("Hans", "PERSON"), GoldEntity("Berlin", "LOCATION"),
                  GoldEntity("NSDAP", "ORG")],
        relations=[GoldRelation("Hans", "Berlin", "born_in"),      # biographical
                   GoldRelation("Hans", "NSDAP", "member_of")])])  # affiliation
    pred_entities = [{"canonical_name": "Hans", "label": "PERSON"},
                     {"canonical_name": "Berlin", "label": "LOCATION"},
                     {"canonical_name": "NSDAP", "label": "ORG"}]
    # Text labels differ but the tie-class is the same: located_in~born_in
    # (biographical), joined~member_of (affiliation).
    pred_edges = [{"source_name": "Hans", "target_name": "Berlin", "rel_type": "located_in"},
                  {"source_name": "Hans", "target_name": "NSDAP", "rel_type": "joined"}]
    typed = score_relations(gold, pred_entities, pred_edges, type_sensitive=True)
    check("typed: differing labels miss (tp=0)", typed["overall"]["tp"] == 0,
          str(typed["overall"]))
    fam = score_relations(gold, pred_entities, pred_edges, type_sensitive=True, family=True)
    check("family: same tie-class credited (tp=2)", fam["overall"]["tp"] == 2,
          str(fam["overall"]))
    # Wrong family must not be credited: located_in (biographical) vs gold
    # member_of (affiliation) on the same pair.
    wrong = [{"source_name": "Hans", "target_name": "NSDAP", "rel_type": "located_in"}]
    fam2 = score_relations(gold, pred_entities, wrong, type_sensitive=True, family=True)
    check("family: wrong tie-class not credited (tp=0)", fam2["overall"]["tp"] == 0,
          str(fam2["overall"]))

    # Unmodeled vocab (e.g. a benchmark's Wikidata types) all fall into the "other"
    # catch-all. Family must NOT let two different unknown types collide - it falls
    # back to the exact label there, degrading to typed, never above it.
    g2 = GoldSet([GoldDocument("d",
        entities=[GoldEntity("A", "PERSON"), GoldEntity("B", "ORG")],
        relations=[GoldRelation("A", "B", "country")])])  # unmodeled type
    pe2 = [{"canonical_name": "A", "label": "PERSON"}, {"canonical_name": "B", "label": "ORG"}]
    diff = [{"source_name": "A", "target_name": "B", "rel_type": "has_part"}]  # also unmodeled
    f3 = score_relations(g2, pe2, diff, type_sensitive=True, family=True)
    check("family: unmodeled types don't collide (tp=0)", f3["overall"]["tp"] == 0,
          str(f3["overall"]))
    same = [{"source_name": "A", "target_name": "B", "rel_type": "country"}]
    f4 = score_relations(g2, pe2, same, type_sensitive=True, family=True)
    check("family: unmodeled exact label still matches (tp=1)", f4["overall"]["tp"] == 1,
          str(f4["overall"]))


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


def test_affiliation_projection() -> None:
    from core.schema import Entity, Relationship
    from postprocess.bipartite import project_affiliations
    print("-- two-mode affiliation projection")
    ents = [Entity("a", "Alice", "PERSON"), Entity("b", "Bob", "PERSON"),
            Entity("c", "Carol", "PERSON"), Entity("pac", "PAC X", "ORG"),
            Entity("ev", "Almeda Fire", "EVENT")]
    # Alice & Bob share PAC X (board); Bob & Carol share the response event.
    edges = [Relationship("a", "pac", "member_of", "d"),
             Relationship("b", "pac", "member_of", "d"),
             Relationship("b", "ev", "participated_in", "d"),
             Relationship("c", "ev", "participated_in", "d")]
    proj = project_affiliations(ents, edges, min_shared=1)
    pairs = {frozenset((r.source, r.target)): r for r in proj}
    check("Alice-Bob linked via shared org", frozenset(("a", "b")) in pairs, str(list(pairs)))
    check("Bob-Carol linked via shared event", frozenset(("b", "c")) in pairs, "")
    check("Alice-Carol not linked (no shared group)", frozenset(("a", "c")) not in pairs, "")
    check("tagged affiliation_projected (full tier)",
          all(r.attributes["edge_source"] == "affiliation_projected" for r in proj), "")
    check("rel_type co_affiliated", all(r.rel_type == "co_affiliated" for r in proj), "")
    ab = pairs[frozenset(("a", "b"))]
    check("Newman strength 1.0 for a 2-person group",
          ab.attributes["affiliation_strength"] == 1.0, str(ab.attributes))
    # Cross-tier check: co_affiliated must NOT pass the conservative filter.
    from postprocess.evidence_tiers import tier_allows
    check("co_affiliated excluded from conservative tier",
          not tier_allows("affiliation_projected", "conservative"), "")
    proj2 = project_affiliations(ents, edges, min_shared=2)
    check("min_shared=2 drops single-group pairs", proj2 == [], str(proj2))

    # Co-funding must NOT project: two donors to the same PAC are not co-members.
    fund_edges = [Relationship("a", "pac", "funded", "d"),
                  Relationship("b", "pac", "funded", "d"),
                  Relationship("c", "pac", "member_of", "d")]
    pf = project_affiliations(ents, fund_edges, min_shared=1)
    pf_pairs = {frozenset((r.source, r.target)) for r in pf}
    check("co-funders not projected as co_affiliated",
          frozenset(("a", "b")) not in pf_pairs, str(pf_pairs))
    check("a funder + a member still not co-membership",
          frozenset(("a", "c")) not in pf_pairs, str(pf_pairs))


def test_org_actor_projection() -> None:
    # OREM/OPAL: agencies (orgs) are the actors, sharing a disaster EVENT.
    from core.schema import Entity, Relationship
    from postprocess.bipartite import project_affiliations
    print("-- org-as-actor projection (disaster response)")
    ents = [Entity("od", "ODHS", "INSTITUTION"), Entity("rc", "Red Cross", "ORG"),
            Entity("ngo", "Local NGO", "ORG"), Entity("fire", "Almeda Fire", "EVENT"),
            Entity("mgr", "A. Manager", "PERSON")]
    edges = [Relationship("od", "fire", "responded_to", "d"),
             Relationship("rc", "fire", "responded_to", "d"),
             Relationship("ngo", "fire", "responded_to", "d"),
             Relationship("mgr", "od", "employed_by", "d")]
    # Default (PERSON actors) finds nothing - the responders are orgs.
    default = project_affiliations(ents, edges, min_shared=1)
    check("default person-actor projection finds no agency tie", default == [], str(default))
    # Org-as-actor: the three responders link pairwise through the shared event.
    proj = project_affiliations(ents, edges, min_shared=1,
                                actor_labels=frozenset({"ORG", "INSTITUTION"}),
                                group_labels=frozenset({"EVENT"}))
    pairs = {frozenset((r.source, r.target)) for r in proj}
    check("ODHS-RedCross linked via shared event", frozenset(("od", "rc")) in pairs, str(pairs))
    check("three responders -> three pairs", len(proj) == 3, str(len(proj)))
    check("co_affiliated edge_source", all(
        r.attributes["edge_source"] == "affiliation_projected" for r in proj), "")


def test_new_domain_packages() -> None:
    from domain.base_domain import load_domain
    print("-- influencewatch / orem_opal domain packages")
    for name, must_have in (
        ("influencewatch", ("funded", "donated_to", "board_member_of", "lobbied")),
        ("orem_opal", ("coordinated_with", "responded_to", "operates_in", "granted")),
    ):
        dom = load_domain(name)
        onto = dom.relation_ontology()
        guide = dom.relation_guide()
        labels = dom.gliner_labels() or []
        lmap = dom.gliner_label_map()
        check(f"{name} loaded (not generic fallback)", dom.name == name, dom.name)
        check(f"{name} has gliner labels", len(labels) > 5, str(len(labels)))
        check(f"{name} every label maps to a type", all(l in lmap for l in labels),
              str([l for l in labels if l not in lmap]))
        for rt in must_have:
            check(f"{name} ontology has {rt}", rt in onto, str(list(onto)[:6]))
        check(f"{name} guide covers its ontology",
              all(k in onto for k in guide), str([k for k in guide if k not in onto]))

    # The domain relations classify into the intended tie classes (global map).
    from postprocess import tie_classes as tc
    check("lobbied is stance (not affiliation)", tc.classify("lobbied", "PERSON", "ORG") == "stance", "")
    check("responded_to org->event is participation",
          tc.classify("responded_to", "ORG", "EVENT") == "participation", "")
    check("board_member_of person->org is affiliation",
          tc.classify("board_member_of", "PERSON", "ORG") == "affiliation", "")
    check("coordinated_with is symmetric", tc.is_symmetric("coordinated_with"), "")
    check("donated_to is a physical (material) connection",
          tc.connection_type("donated_to") == "physical", "")


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


def test_unsupported_excluded_from_substantive() -> None:
    # Edges the relation verifier flagged "unsupported" stay in the export (tagged)
    # but must not drive brokerage / bridges / signed balance. They were inflating
    # the substantive graph - 9% of substantive edges on the Abel pilot, 40 phantom
    # bridges. Gated: no-op when verification field is empty (verify did not run).
    from postprocess.graph_metrics import _build_graph, _signed_balance, _polarity_conflicts
    print("-- unsupported edges excluded from substantive analytics")
    nodes = ["A", "B", "C"]
    edges = [
        {"Source": "A", "Target": "B", "tie_class": "affiliation", "verification": ""},
        # The only edge tying C in is unsupported -> dropping it isolates C.
        {"Source": "B", "Target": "C", "tie_class": "affiliation", "verification": "unsupported"},
    ]
    g_keep = _build_graph(nodes, edges, {"affiliation"}, drop_unsupported=False)
    g_drop = _build_graph(nodes, edges, {"affiliation"}, drop_unsupported=True)
    check("kept: unsupported edge present", g_keep.number_of_edges() == 2, str(g_keep.number_of_edges()))
    check("dropped: unsupported edge gone", g_drop.number_of_edges() == 1, str(g_drop.number_of_edges()))
    check("dropped: C isolated", g_drop.degree("C") == 0, str(g_drop.degree("C")))

    # An unsupported stance edge must not create a phantom polarity conflict.
    sedges = [
        {"Source": "X", "Target": "Y", "polarity": "positive", "rel_type": "supported", "verification": "unsupported"},
        {"Source": "X", "Target": "Y", "polarity": "negative", "rel_type": "fought_against", "verification": "supported"},
    ]
    filt = [e for e in sedges if e.get("verification") != "unsupported"]
    c = _polarity_conflicts(filt)
    check("phantom conflict removed", c["conflicting_dyads"] == 0, str(c))


def test_trust_verification_gates_metric_drop() -> None:
    # A weak local verifier over-rejects (qwen ~50% unsupported on the Abel pilot),
    # so its flags must only TAG - not prune the metric graph - unless the run
    # opts in via quality.trust_verification (set when a strong api/gemini verifier
    # ran). enrich() carries the gate; default off keeps the ties in the structure.
    from postprocess.graph_metrics import enrich
    from postprocess.gephi_builder import GraphTables
    print("-- trust_verification gates the metric-graph prune")
    nodes = [{"Id": "A", "Label": "A"}, {"Id": "B", "Label": "B"}, {"Id": "C", "Label": "C"}]
    # B-C is the only tie holding C in, and the verifier flagged it unsupported.
    edges = [
        {"Source": "A", "Target": "B", "tie_class": "affiliation", "verification": "supported", "Weight": 1},
        {"Source": "B", "Target": "C", "tie_class": "affiliation", "verification": "unsupported", "Weight": 1},
    ]
    soft = enrich(GraphTables(nodes=list(nodes), edges=[dict(e) for e in edges]),
                  trust_verification=False)
    hard = enrich(GraphTables(nodes=list(nodes), edges=[dict(e) for e in edges]),
                  trust_verification=True)
    check("soft (default): unsupported tie kept, C connected",
          soft["qa_substantive"]["edges"] == 2 and soft["qa_substantive"]["isolates"] == 0,
          str(soft["qa_substantive"]))
    check("trusted: unsupported tie pruned, C isolated",
          hard["qa_substantive"]["edges"] == 1 and hard["qa_substantive"]["isolates"] == 1,
          str(hard["qa_substantive"]))


def test_causal_tie_class() -> None:
    from postprocess import tie_classes
    from postprocess.ontology import OntologyAligner, GENERIC_RELATION_ONTOLOGY
    print("-- causal tie class + vocabulary")
    check("caused -> causal", tie_classes.classify("caused") == "causal", "")
    check("caused_by -> causal", tie_classes.classify("caused_by") == "causal", "")
    check("contributed_to -> causal", tie_classes.classify("contributed_to") == "causal", "")
    check("causal is directed", not tie_classes.is_symmetric("caused"), "")
    check("causal not in interpersonal substantive set",
          "causal" not in (tie_classes.SOCIAL | tie_classes.STRUCTURAL), "")
    al = OntologyAligner(GENERIC_RELATION_ONTOLOGY, 0.82)
    check("'led to' aligns to caused", al.align("led to") == "caused", str(al.align("led to")))
    check("'resulted from' aligns to caused_by",
          al.align("resulted from") == "caused_by", str(al.align("resulted from")))
    # The builder once KeyError'd on a causal edge because _TIE_CLASSES omitted it.
    # A causal edge must build clean and get a deg_causal node column like any class.
    from core.schema import Entity, Relationship
    from postprocess.gephi_builder import GephiBuilder
    cents = [Entity(entity_id="d1", canonical_name="War", label="EVENT"),
             Entity(entity_id="d2", canonical_name="Hardship", label="EVENT")]
    crel = Relationship(source="d1", target="d2", rel_type="caused", doc_id="d",
                        evidence="the war caused hardship")
    ctab = GephiBuilder().build(cents, [crel], [])
    check("causal edge builds without crash", len(ctab.edges) == 1, str(len(ctab.edges)))
    check("deg_causal column on every node",
          all("deg_causal" in n for n in ctab.nodes), "")
    src = next(n for n in ctab.nodes if n["Id"] == "d1")
    check("causal degree counted", src["deg_causal"] == 1, str(src.get("deg_causal")))


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

    # located_in is permissive on the source (place-in-place containment and
    # person/org-in-place are all valid - the domain treats person->place as
    # biographical). The target must be a place: located_in pointing at a
    # person/org is the misextraction.
    lt = {"l1": "LOCATION", "l2": "LOCATION", "p1": "PERSON", "o1": "ORG"}
    pip = R("l1", "l2", "located_in")    # place in place: valid containment
    ppl = R("p1", "l1", "located_in")    # person in place: valid (biographical)
    bad = R("p1", "o1", "located_in")    # located_in into an org: violation
    _, nlo = check_relation_types([pip, ppl, bad], lt, drop=False)
    check("place-in-place not flagged", "type_violation" not in pip.attributes, "")
    check("person-in-place not flagged", "type_violation" not in ppl.attributes, "")
    check("located_in into org flagged", bad.attributes.get("type_violation") is True, "")
    check("located_in: one of three flagged", nlo == 1, str(nlo))

    # promoted_to targets a RANK, not an org/place.
    rt = {"p1": "PERSON", "r1": "RANK", "o1": "ORG"}

    def Rp(s, t):
        return Relationship(source=s, target=t, rel_type="promoted_to",
                            doc_id="d", evidence="e")
    okp, badp = Rp("p1", "r1"), Rp("p1", "o1")
    _, np = check_relation_types([okp, badp], rt, drop=False)
    check("promoted_to->rank ok", "type_violation" not in okp.attributes, "")
    check("promoted_to->org flagged", badp.attributes.get("type_violation") is True, "")
    check("promoted_to: one flagged", np == 1, str(np))


def test_type_hint_prompt() -> None:
    from intelligence.prompts import build_extraction_prompt
    from postprocess.ontology import relation_signature_hints
    print("-- structure-aware type hints in prompt")

    hints = relation_signature_hints(["born_in", "employed_by", "promoted_to",
                                      "located_in", "supported"])
    check("born_in signature rendered", hints.get("born_in") == "person->place", str(hints))
    check("employed_by signature rendered", hints.get("employed_by") == "person->org", "")
    check("promoted_to signature rendered", hints.get("promoted_to") == "person->rank", "")
    check("located_in permissive source", hints.get("located_in") == "person/org/place->place", "")
    check("loose relation has no signature", "supported" not in hints, str(hints))

    labels = ["born_in", "supported"]
    # On: the hint rides next to the type even with no guide text.
    p = build_extraction_prompt("x", [], ["PERSON"], relation_types=labels,
                                type_signatures=hints)
    check("hint shown in prompt", "born_in (person->place)" in p, "")
    check("unconstrained type listed plain", "- supported" in p and "supported (" not in p, "")
    # Off: bare comma list, no parenthetical signatures (default behavior).
    off = build_extraction_prompt("x", [], ["PERSON"], relation_types=labels)
    check("no hints when disabled", "(person->place)" not in off, "")


def test_biographical_inference() -> None:
    from core.schema import Entity, EntityMention, Relationship
    from domain.nazi_era.canonical_inference import infer_biographical_edges
    print("-- nazi_era birth/residence inference")
    author = Entity(entity_id="a1", canonical_name="Hans Müller", label="PERSON",
                    doc_ids=["d1"], attributes={"is_author": True, "author_doc": "d1"})
    berlin = Entity(entity_id="p1", canonical_name="Berlin", label="LOCATION", doc_ids=["d1"])
    hamburg = Entity(entity_id="p2", canonical_name="Hamburg", label="LOCATION", doc_ids=["d1"])
    ents = [author, berlin, hamburg]
    name_to_id = {"berlin": "p1", "hamburg": "p2"}

    def M(text, sent):
        return EntityMention(text=text, label="LOCATION", start_char=0, end_char=0,
                             chunk_id="c", doc_id="d1", sentence=sent)

    edges = infer_biographical_edges(ents, [], [
        M("Berlin", "Geboren bin ich am 5.5.1898 in Berlin."),  # narrator birth
        M("Hamburg", "Mein Vater wurde in Hamburg geboren."),   # relative -> skip
    ], name_to_id)
    by = {(r.source, r.target): r.rel_type for r in edges}
    check("birth cue -> narrator born_in place", by.get(("a1", "p1")) == "born_in", str(by))
    check("relative's birthplace skipped", ("a1", "p2") not in by, str(by))
    check("edge is conservative-tier rule_extracted",
          all(r.attributes.get("edge_source") == "rule_extracted" for r in edges), "")
    res = infer_biographical_edges(ents, [], [
        M("Berlin", "Ich wohnte bis 1930 in Berlin.")], name_to_id)
    check("residence cue -> resided_in", bool(res) and res[0].rel_type == "resided_in", str(res))
    dup = infer_biographical_edges(
        ents, [Relationship(source="a1", target="p1", rel_type="born_in", doc_id="d1")],
        [M("Berlin", "Geboren in Berlin.")], name_to_id)
    check("existing biographical edge not duplicated", dup == [], str(dup))
    # Upgrade: the model already emitted a generic located_in for the birthplace; the
    # cue rewrites it in place to born_in instead of being blocked (the gemini_batch
    # gap that cost typed recall). Edge mutated, not duplicated.
    loc = Relationship(source="a1", target="p1", rel_type="located_in", doc_id="d1",
                       attributes={"edge_source": "llm_extracted"})
    up = infer_biographical_edges(ents, [loc],
                                  [M("Berlin", "Geboren bin ich in Berlin.")], name_to_id)
    check("located_in upgraded in place to born_in", loc.rel_type == "born_in", loc.rel_type)
    check("upgrade adds no duplicate edge", up == [], str(up))
    check("upgrade tagged with origin type",
          loc.attributes.get("type_upgraded_from") == "located_in", str(loc.attributes))
    # Batch path: NO mentions (gemini_batch gives none with sentences), but the
    # located_in edge's evidence carries the cue. The first-person "als Sohn des ...
    # geboren" must still upgrade (relative-filter skipped on an author-sourced edge).
    loc2 = Relationship(source="a1", target="p1", rel_type="located_in", doc_id="d1",
                        evidence="Bin am 18.2.1898 als Sohn des Adolf in Berlin geboren.",
                        attributes={"edge_source": "llm_extracted"})
    ev = infer_biographical_edges(ents, [loc2], [], name_to_id)
    check("evidence cue upgrades located_in (batch, no mentions)",
          loc2.rel_type == "born_in", loc2.rel_type)
    check("evidence upgrade adds no new edge", ev == [], str(ev))
    # A bare nee 'geb.' (maiden name) is NOT a birth - must not upgrade.
    loc3 = Relationship(source="a1", target="p2", rel_type="located_in", doc_id="d1",
                        evidence="Meine Mutter Marie, geb. Krotzki, stammte aus Hamburg.",
                        attributes={"edge_source": "llm_extracted"})
    infer_biographical_edges(ents, [loc3], [], name_to_id)
    check("nee 'geb.' does not upgrade", loc3.rel_type == "located_in", loc3.rel_type)


def test_author_anchoring() -> None:
    from core.schema import Entity, Relationship
    from postprocess.identity_resolution import link_known_authors
    print("-- cross-doc author anchoring")

    def E(eid, name, author=False, docs=("d1",)):
        return Entity(entity_id=eid, canonical_name=name, label="PERSON",
                      doc_ids=list(docs), attributes={"is_author": True} if author else {})

    # Author "August Spanku" (letter B); a lone "Spanku" in letter A is the same person.
    auth = E("a_sp", "August Spanku", author=True, docs=("dB",))
    mention = E("m_sp", "Spanku", docs=("dA",))
    other = E("p_hans", "Hans Vogel", docs=("dA",))
    rels = [Relationship(source="p_hans", target="m_sp", rel_type="fought_with", doc_id="dA")]
    ents, out = link_known_authors([auth, mention, other], rels)
    ids = {e.entity_id for e in ents}
    check("surname mention folded into author", "m_sp" not in ids, str(ids))
    check("edge remapped to author", out and out[0].target == "a_sp", str(out))
    check("alias recorded on author", "Spanku" in auth.aliases, str(auth.aliases))
    # Ambiguity: a second person shares the surname -> do NOT auto-link.
    a2 = E("a_sp", "August Spanku", author=True, docs=("dB",))
    m2 = E("m_sp", "Spanku", docs=("dA",))
    sib = E("p_klara", "Klara Spanku", docs=("dC",))
    ents2, _ = link_known_authors([a2, m2, sib], [])
    check("ambiguous surname not linked", "m_sp" in {e.entity_id for e in ents2}, "")
    # Short surname guarded.
    li = E("a_li", "Li", author=True)
    ml = E("m_li", "Li")
    ents3, _ = link_known_authors([li, ml], [], min_len=4)
    check("short surname guarded", "m_li" in {e.entity_id for e in ents3}, "")


def test_relation_verify() -> None:
    from core.schema import Relationship
    from postprocess.relation_verify import verify_relations
    print("-- relation self-verification")

    class FakeBackend:
        def _complete(self, system, user):
            return '{"1": "no", "2": "yes"}'   # item 1 unsupported, item 2 supported

    def R(s, t, rt, ev, src="llm_extracted"):
        return Relationship(source=s, target=t, rel_type=rt, doc_id="d", evidence=ev,
                            attributes={"edge_source": src})

    id_to_name = {"a": "Alice", "b": "Bob", "c": "Cap", "d": "Dee"}
    r1 = R("a", "b", "met_with", "They met.")
    r2 = R("a", "c", "funded", "A funded C.")
    rrule = R("a", "d", "co_occurs_with", "", src="rule_cooccurrence")  # no evidence -> skipped
    out, flagged = verify_relations([r1, r2, rrule], FakeBackend(), id_to_name)
    check("one edge flagged unsupported", flagged == 1, str(flagged))
    check("unsupported tagged", r1.attributes.get("verification") == "unsupported", str(r1.attributes))
    check("supported tagged", r2.attributes.get("verification") == "supported", str(r2.attributes))
    check("non-llm edge not verified", "verification" not in rrule.attributes, str(rrule.attributes))
    # drop mode removes the unsupported edge; supported stays.
    r1b, r2b = R("a", "b", "met_with", "They met."), R("a", "c", "funded", "A funded C.")
    out2, _ = verify_relations([r1b, r2b], FakeBackend(), id_to_name, drop=True)
    check("drop removes unsupported", r1b not in out2 and r2b in out2, str(len(out2)))
    # A backend without _complete (python_only) is a no-op, never raises.
    _, f3 = verify_relations([R("a", "b", "x", "ev")], object(), id_to_name)
    check("no _complete -> no-op", f3 == 0, str(f3))


def test_functional_consistency() -> None:
    from core.schema import Relationship
    from postprocess.ontology import check_functional_consistency
    print("-- functional-property consistency")

    def R(s, t, rt, conf=0.8):
        return Relationship(source=s, target=t, rel_type=rt, doc_id="d", confidence=conf)

    # One person, two different birthplaces -> contradiction. Two residences -> fine.
    rels = [R("p1", "Berlin", "born_in", 0.9),
            R("p1", "Hamburg", "born_in", 0.6),   # conflict with Berlin
            R("p1", "Munich", "resided_in"),
            R("p1", "Cologne", "resided_in"),      # resided_in not functional -> ok
            R("p2", "Bonn", "born_in")]            # single -> ok
    out, flagged = check_functional_consistency(rels)
    check("both conflicting born_in tagged", flagged == 2, str(flagged))
    byt = {(r.source, r.target): r for r in rels}
    check("conflict edges carry tag",
          byt[("p1", "Berlin")].attributes.get("functional_conflict") is True
          and byt[("p1", "Hamburg")].attributes.get("functional_conflict") is True, "")
    check("non-functional relation untouched",
          "functional_conflict" not in byt[("p1", "Munich")].attributes, "")
    check("single-valued subject untouched",
          "functional_conflict" not in byt[("p2", "Bonn")].attributes, "")
    # drop=True keeps the best-supported target (Berlin, higher confidence), drops Hamburg.
    rels2 = [R("p1", "Berlin", "born_in", 0.9), R("p1", "Hamburg", "born_in", 0.6)]
    out2, _ = check_functional_consistency(rels2, drop=True)
    kept = {(r.source, r.target) for r in out2}
    check("drop keeps best target only", kept == {("p1", "Berlin")}, str(kept))


def test_recall_pass() -> None:
    from core.schema import EntityMention, Relationship
    from intelligence.relation_recall import recall_relations
    print("-- recall pass (missed cross-chunk relations)")

    def M(name, label="PERSON"):
        return EntityMention(text=name, label=label, start_char=0, end_char=0,
                             chunk_id="c", doc_id="d")

    mentions = [M("Alice"), M("Bob"), M("Acme", "ORG")]
    existing = [Relationship(source="Alice", target="Acme", rel_type="works_at", doc_id="d")]
    reply = ('{"relationships": ['
             '{"source":"Alice","target":"Bob","type":"met_with","evidence":"Alice met Bob."},'
             '{"source":"Alice","target":"Acme","type":"works_at","evidence":"x"},'
             '{"source":"Alice","target":"Zorg","type":"met_with","evidence":"x"}]}')
    out = recall_relations(
        "Alice met Bob. Alice works at Acme.", mentions, existing, lambda s, u: reply,
        label_types=["PERSON", "ORG"], relation_types=["met_with", "works_at"],
        relation_guide={}, edge_qualifiers=[], type_signatures={}, doc_id="d")
    rts = [(r.source, r.target, r.rel_type) for r in out]
    check("missed cross-entity relation recovered", ("Alice", "Bob", "met_with") in rts, str(rts))
    check("duplicate not re-added", ("Alice", "Acme", "works_at") not in rts, str(rts))
    check("unknown-entity relation dropped", all(r.target != "Zorg" for r in out), str(rts))
    check("recall edges tagged", out and all(r.attributes.get("recall_pass") for r in out), "")
    # Oversized doc is now WINDOWED (sectioned recall), not skipped: it still recovers
    # relations among entities whose mentions fall inside a window.
    big = recall_relations("x" * 30000, mentions, existing, lambda s, u: reply,
                           label_types=["PERSON", "ORG"], relation_types=["met_with", "works_at"],
                           relation_guide={}, edge_qualifiers=[], type_signatures={},
                           doc_id="d", max_chars=24000)
    big_rts = [(r.source, r.target, r.rel_type) for r in big]
    check("oversized doc windowed, not skipped", ("Alice", "Bob", "met_with") in big_rts, str(big_rts))

    # Window-local entity gating: a doc split into windows only prompts windows that
    # hold >=2 entities, and entities are filtered by doc-absolute offset.
    def Mo(name, label, start):
        return EntityMention(text=name, label=label, start_char=start, end_char=start + len(name),
                             chunk_id="c", doc_id="d")
    win_doc = "z" * 8000
    win_ments = [Mo("Alice", "PERSON", 10), Mo("Acme", "ORG", 60),
                 Mo("Bob", "PERSON", 5000), Mo("Globex", "ORG", 5050)]
    seen_windows = []
    recall_relations(win_doc, win_ments, [],
                     lambda s, u: seen_windows.append("A" if "Alice" in u else "B") or "{}",
                     label_types=["PERSON", "ORG"], relation_types=["knew"], relation_guide={},
                     edge_qualifiers=[], type_signatures={}, doc_id="d",
                     max_chars=2000, overlap=500)
    check("long doc prompts both entity regions", "A" in seen_windows and "B" in seen_windows,
          str(seen_windows))


def test_expansion_schema_load() -> None:
    import json as _json
    import csv as _csv
    import tempfile
    from pathlib import Path
    from postprocess.expansion import load_network_schema
    print("-- network expansion schema load")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "entities.json").write_text(_json.dumps([
            {"entity_id": "e1", "canonical_name": "Valjean", "label": "PERSON"},
            {"entity_id": "e2", "canonical_name": "Paris", "label": "LOCATION"},
        ]), encoding="utf-8")
        with (d / "gephi_edges.csv").open("w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["Source", "Target", "rel_type"])
            w.writeheader()
            w.writerow({"Source": "e1", "Target": "e2", "rel_type": "lived_in"})
            # co_occurs_with is structural, must not become a locked relation type
            w.writerow({"Source": "e1", "Target": "e2", "rel_type": "co_occurs_with"})
        sch = load_network_schema(str(d))
        check("relation types loaded", sch.relation_types == {"lived_in"}, str(sch.relation_types))
        check("co_occurs_with excluded", "co_occurs_with" not in sch.relation_types, "")
        check("entity kinds loaded", sch.entity_types == {"PERSON", "LOCATION"},
              str(sch.entity_types))
        check("names mapped to type", sch.entity_names.get("Valjean") == "PERSON", "")
    empty = load_network_schema("does/not/exist")
    check("missing source -> empty (locks no-op)", empty.empty, "")


def test_quality_pillars() -> None:
    from types import SimpleNamespace
    from postprocess.graph_metrics import quality_pillars
    print("-- KGC quality pillars")
    edges = [
        {"edge_source": "llm_extracted"},                 # asserted
        {"edge_source": "metadata"},                       # asserted
        {"edge_source": "rule_cooccurrence", "rel_type": "led",
         "type_violation": True},                          # weak + bad
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
    check("type violations broken out by relation",
          qp["consistency"]["type_violations_by_relation"] == {"led": 1},
          str(qp["consistency"].get("type_violations_by_relation")))
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


def test_extraction_schema() -> None:
    # Schema-constrained generation (structured_output): the grammar must close the
    # slots a weak model leaks prose into - chiefly the entity aliases array, where
    # qwen3.5 wrote '"NSDSP" -> Note: ...' and lost the whole document. Enforce
    # shape + the canonical entity-type enum, but keep relation type a free string
    # (the ontology aligner maps surface forms).
    from intelligence.prompts import extraction_json_schema
    print("-- extraction json schema")
    s = extraction_json_schema(["PERSON", "ORG", "LOCATION", "EVENT"], ["monetary_value"])
    ent = s["properties"]["entities"]["items"]["properties"]
    rel = s["properties"]["relationships"]["items"]
    check("aliases is a string array (prose-leak slot closed)",
          ent["aliases"]["items"]["type"] == "string", str(ent.get("aliases")))
    check("entity type enum-locked to canonical set",
          ent["type"].get("enum") == ["EVENT", "LOCATION", "ORG", "PERSON"], str(ent["type"]))
    check("relation type stays a free string (aligner maps it)",
          rel["properties"]["type"] == {"type": "string"}, str(rel["properties"]["type"]))
    check("declared qualifier reaches the schema",
          "monetary_value" in rel["properties"], str(list(rel["properties"])))
    check("source+target+type required on a relation",
          set(rel["required"]) == {"source", "target", "type"}, str(rel["required"]))
    # No label types -> type is an unconstrained string (generic domain).
    s2 = extraction_json_schema(None, None)
    check("no labels -> entity type unconstrained",
          "enum" not in s2["properties"]["entities"]["items"]["properties"]["type"], "")


def test_llm_review_protects_high_degree() -> None:
    # A weak LLM reviewer flags a low-mention entity for drop, but that entity
    # anchors many edges (NER tagged it once, the LLM cited it as the endpoint of
    # 20 ties). Dropping it orphaned all those edges - the 31% edge loss traced on
    # the ollama pilot. Degree must protect it like mention_count/doc_count do.
    from core.schema import Entity, Relationship
    from postprocess.quality_review import QualityReviewer
    from config import QualityConfig
    print("-- LLM review protects high-degree entities")

    class StubBackend:
        def review(self, ents, edges):
            return {"drop_entities": ["Schreiber"], "drop_edges": []}

    hub = Entity(entity_id="hub", canonical_name="Schreiber", label="PERSON",
                 mention_count=1, doc_ids=["d1"])
    others = [Entity(entity_id=f"p{i}", canonical_name=f"Person {i}", label="PERSON",
                     mention_count=3, doc_ids=["d1"]) for i in range(5)]
    # Schreiber anchors 5 edges; degree 5 >= protect threshold.
    edges = [Relationship(source="hub", target=f"p{i}", rel_type="met_with",
                          doc_id="d1") for i in range(5)]
    qr = QualityReviewer(QualityConfig(), stopwords=set())
    kept, kept_edges = qr.llm_filter([hub, *others], edges, StubBackend())
    names = {e.canonical_name for e in kept}
    check("high-degree entity survives LLM drop", "Schreiber" in names, str(names))
    check("its edges survive too", len(kept_edges) == 5, str(len(kept_edges)))

    # Control: a low-mention, low-degree entity the LLM flags IS dropped.
    lonely = Entity(entity_id="x", canonical_name="Schreiber", label="PERSON",
                    mention_count=1, doc_ids=["d1"])
    kept2, _ = qr.llm_filter([lonely, *others], [], StubBackend())
    check("low-degree flagged entity is dropped",
          "Schreiber" not in {e.canonical_name for e in kept2},
          str({e.canonical_name for e in kept2}))


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


def test_faithfulness_tags_exported() -> None:
    # Regression: the type gate (ontology) and the anchor check (intelligence)
    # set r.attributes["type_violation"] / ["evidence_ungrounded"], but the edge
    # table only copies an allowlist of attributes. Both tags were set upstream
    # yet never reached the table, so quality_pillars always read 0 and Gephi
    # had no column to filter on. Lock the propagation here.
    from core.schema import Entity, Relationship
    from postprocess.gephi_builder import GephiBuilder

    print("-- faithfulness tags reach the edge table")
    ents = [Entity(entity_id="p1", canonical_name="Hans", label="PERSON"),
            Entity(entity_id="l1", canonical_name="Berlin", label="LOCATION")]

    def R(rt, **attrs):
        return Relationship(source="p1", target="l1", rel_type=rt, doc_id="d",
                            evidence="e", attributes=dict(attrs))

    rels = [R("led", type_violation=True),
            R("born_in", evidence_ungrounded="true"),
            R("employed_by", verification="unsupported"),
            R("died_in", functional_conflict=True),
            R("met_with")]  # clean
    tables = GephiBuilder().build(ents, rels, [])
    by_rt = {e["rel_type"]: e for e in tables.edges}
    check("type_violation exported", by_rt["led"].get("type_violation") is True,
          str(by_rt["led"].get("type_violation")))
    check("verification tag exported",
          by_rt["employed_by"].get("verification") == "unsupported",
          str(by_rt["employed_by"].get("verification")))
    check("functional_conflict tag exported",
          by_rt["died_in"].get("functional_conflict") is True,
          str(by_rt["died_in"].get("functional_conflict")))
    check("evidence_ungrounded exported",
          by_rt["born_in"].get("evidence_ungrounded") is True,
          str(by_rt["born_in"].get("evidence_ungrounded")))
    check("clean edge flags default False",
          by_rt["met_with"].get("type_violation") is False
          and by_rt["met_with"].get("evidence_ungrounded") is False, "")
    check("both columns present on every edge",
          all("type_violation" in e and "evidence_ungrounded" in e
              for e in tables.edges), "")

    # Same allowlist class: the two-mode projection weight (affiliation_strength /
    # shared_groups) on a co_affiliated edge must reach the table, not be dropped.
    ents2 = [Entity(entity_id="a", canonical_name="Alice", label="PERSON"),
             Entity(entity_id="b", canonical_name="Bob", label="PERSON")]
    co = Relationship(source="a", target="b", rel_type="co_affiliated", doc_id="",
                      directed=False, evidence="share 2",
                      attributes={"edge_source": "affiliation_projected",
                                  "affiliation_strength": 1.5, "shared_groups": 2})
    t2 = GephiBuilder().build(ents2, [co], [])
    ce = t2.edges[0]
    check("affiliation_strength exported", ce.get("affiliation_strength") == 1.5,
          str(ce.get("affiliation_strength")))
    check("shared_groups exported", ce.get("shared_groups") == 2, str(ce.get("shared_groups")))

    # GEXF round-trip: Gephi reads flags from the GEXF, not just the CSV. The
    # three edges share one pair, so the writer merges them - the merged edge must
    # OR the flags (tainted by led's type_violation and born_in's ungrounded).
    try:
        import networkx as nx
        import tempfile
        from pathlib import Path
        from postprocess.exporter import Exporter
    except Exception:  # noqa: BLE001
        print("   (gexf flag round-trip skipped: no networkx)")
        return
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "g.gexf"
        Exporter._write_gexf(tables, p)
        G = nx.read_gexf(p)
        ed = G["p1"]["l1"]
        check("gexf edge carries type_violation",
              str(ed.get("type_violation")).lower() == "true", str(ed.get("type_violation")))
        check("gexf edge carries evidence_ungrounded",
              str(ed.get("evidence_ungrounded")).lower() == "true",
              str(ed.get("evidence_ungrounded")))


def test_edge_qualifiers() -> None:
    from intelligence.api_backend import _map_extraction
    from core.schema import Entity, Relationship
    from postprocess.gephi_builder import GephiBuilder
    print("-- per-edge qualifiers (qual_ namespace)")
    data = {"entities": [], "relationships": [
        {"source": "PAC X", "target": "Shell Co", "type": "funded",
         "monetary_value": "$50,000", "jurisdiction": "Oregon", "noise": "drop me"}]}
    _, rels, _ = _map_extraction(data, [], "c", "d", ["PERSON", "ORG"],
                                 qualifiers=["monetary_value", "jurisdiction"])
    attrs = rels[0].attributes
    check("declared qualifier captured", attrs.get("qual_monetary_value") == "$50,000", str(attrs))
    check("second qualifier captured", attrs.get("qual_jurisdiction") == "Oregon", "")
    check("undeclared key not captured", "qual_noise" not in attrs, str(attrs))
    # No qualifiers configured -> nothing captured (no behavior change).
    _, rels2, _ = _map_extraction(data, [], "c", "d", ["PERSON", "ORG"], qualifiers=None)
    check("no qualifiers -> none captured",
          not any(k.startswith("qual_") for k in rels2[0].attributes), "")
    # Passthrough to the edge table.
    ents = [Entity("p1", "PAC X", "ORG"), Entity("s1", "Shell Co", "ORG")]
    rel = Relationship(source="p1", target="s1", rel_type="funded", doc_id="d",
                       attributes={"edge_source": "llm_extracted",
                                   "qual_monetary_value": "$50,000"})
    e = GephiBuilder().build(ents, [rel], []).edges[0]
    check("qualifier reaches edge table column", e.get("qual_monetary_value") == "$50,000", str(e))

    # Schema stability: a declared qualifier no edge fills still gets an (empty)
    # column, so the CSV header is fixed run-to-run. orem_smoke emitted zero
    # qual_jurisdiction edges and the column silently vanished before this.
    plain = Relationship(source="p1", target="s1", rel_type="funded", doc_id="d",
                         attributes={"edge_source": "llm_extracted"})
    e2 = GephiBuilder().build(ents, [plain], [],
                              edge_qualifiers=["monetary_value", "jurisdiction"]).edges[0]
    check("declared qualifier seeds empty column",
          e2.get("qual_jurisdiction") == "" and "qual_monetary_value" in e2, str(e2))
    e3 = GephiBuilder().build(ents, [rel], [],
                              edge_qualifiers=["monetary_value", "jurisdiction"]).edges[0]
    check("filled value overrides the seed",
          e3.get("qual_monetary_value") == "$50,000" and e3.get("qual_jurisdiction") == "", str(e3))

    # Regression: the model only fills a qualifier if it sees the slot in the JSON
    # schema example (it copies the example literally). A real ollama run left
    # $1,415,274 in the evidence with no monetary_value key because the schema
    # example lacked the slot. The qualifier keys must be IN the schema block.
    from intelligence.prompts import build_extraction_prompt
    p = build_extraction_prompt("x", [], ["PERSON", "ORG"], relation_types=["funded"],
                                edge_qualifiers=["monetary_value", "jurisdiction"])
    i = p.find('"relationships"')
    sch = p[i:i + 320]
    check("qualifier slot in schema example", '"monetary_value"' in sch and '"jurisdiction"' in sch, sch)
    bare = build_extraction_prompt("x", [], ["PERSON", "ORG"], relation_types=["funded"])
    check("no qualifier slot when none declared", '"monetary_value"' not in bare, "")


def test_run_meta_provenance() -> None:
    # run_meta records the model that produced the checkpoint AT EXTRACT TIME. An
    # analyze-only re-run carries default settings (the user rarely re-passes
    # --batch-model), so it must not clobber the extraction model. Also: gemini_batch
    # keeps its model in batch_model, not a `gemini_batch` sub-config.
    import json
    import tempfile
    from pathlib import Path
    from main import Pipeline
    from config import load_config
    print("-- run_meta preserves the extraction model across analyze re-runs")
    cfg = load_config("domain/nazi_era/config_nazi_era.yaml")
    cfg.mode = "gemini_batch"
    cfg.intelligence.batch_model = "gemini-3.5-flash"
    p = Pipeline(cfg)
    p.run_dir = Path(tempfile.mkdtemp())
    p._write_run_meta("extract", resume=False, limit=40)
    m1 = json.loads((p.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    check("extract records batch_model (not blank)", m1["model"] == "gemini-3.5-flash", m1["model"])
    # Analyze re-run carrying the DEFAULT model must preserve the extraction model.
    cfg.intelligence.batch_model = "gemini-2.5-flash"
    p._write_run_meta("analyze", resume=False, limit=40)
    m2 = json.loads((p.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    check("analyze preserves the extraction model", m2["model"] == "gemini-3.5-flash", m2["model"])
    check("analyze still updates the stage", m2["stage"] == "analyze", m2["stage"])
    check("preserved config keeps the extract model",
          m2["config"]["intelligence"]["batch_model"] == "gemini-3.5-flash",
          m2["config"]["intelligence"]["batch_model"])


def test_span_reconcile() -> None:
    # gemini_batch entities are span-less (0,0); reconciliation folds local GLiNER
    # spans onto them (span transfer, relabeled to the LLM type so they aggregate to
    # the same node) and adds GLiNER-only mentions as a recall net (tagged ner_only).
    from core.schema import DocumentExtraction, EntityMention
    from postprocess.span_reconcile import reconcile_spans
    print("-- span reconciliation (gemini_batch GLiNER fold-in)")

    def llm_m(name, label):
        return EntityMention(text=name, label=label, start_char=0, end_char=0,
                             chunk_id="c", doc_id="d", sources=["llm"])
    ex = DocumentExtraction("d", "p", mentions=[llm_m("Adolf Hitler", "PERSON"),
                                                llm_m("NSDAP", "ORG")])
    # GLiNER (span-grounded): "Hitler" matches the LLM person but GLiNER mistyped it
    # ORG; "Munich" is GLiNER-only (recall net). normalize_name folds "Adolf Hitler"
    # and "Hitler"? No - different surfaces. Use the LLM's own surface to prove transfer.
    # (coref-sourced mentions are dropped by the main wrapper before this, so the
    # stub returns only GLiNER spans - what reconcile_spans actually receives.)
    def ner_fn(doc_id, text):
        return [
            EntityMention(text="Adolf Hitler", label="ORG", start_char=10, end_char=22,
                          chunk_id="c", doc_id="d", sources=["gliner"]),
            EntityMention(text="Munich", label="LOCATION", start_char=30, end_char=36,
                          chunk_id="c", doc_id="d", sources=["gliner"]),
        ]
    stats = reconcile_spans([ex], {"d": "x" * 50}, ner_fn, add_missed=True)
    check("one span transferred", stats["transferred"] == 1, str(stats))
    check("one ner-only added", stats["added"] == 1, str(stats))
    transferred = [m for m in ex.mentions if m.attributes.get("reconciled") == "ner_reconciled"]
    check("transfer relabeled to LLM type (PERSON not ORG)",
          len(transferred) == 1 and transferred[0].label == "PERSON", str(transferred))
    check("transfer carries a real span",
          transferred and transferred[0].start_char == 10 and transferred[0].end_char == 22, "")
    only = [m for m in ex.mentions if m.attributes.get("reconciled") == "ner_only"]
    check("ner-only is the unmatched Munich", len(only) == 1 and only[0].text == "Munich", str(only))

    # add_missed off: span transfer still happens, recall net suppressed.
    ex2 = DocumentExtraction("d", "p", mentions=[llm_m("Adolf Hitler", "PERSON")])
    s2 = reconcile_spans([ex2], {"d": "x" * 50}, ner_fn, add_missed=False)
    check("add_missed off -> no recall net", s2["added"] == 0 and s2["transferred"] == 1, str(s2))
    # A doc with no text is skipped (no NER run).
    ex3 = DocumentExtraction("d", "p", mentions=[llm_m("X", "PERSON")])
    s3 = reconcile_spans([ex3], {}, ner_fn, add_missed=True)
    check("missing text -> doc skipped", s3["docs"] == 0, str(s3))


def test_manual_batch() -> None:
    from intelligence.manual_batch import (build_batch_prompt, parse_batch_response,
                                           _coerce_doc_map)
    print("-- mode 4: manual batch (gemini_batch)")

    docs = [("doc_a", "Acme funded the Berger Fund with $5,000."),
            ("doc_b", "Jane chairs Acme.")]
    prompts = build_batch_prompt(docs, ["PERSON", "ORG"], relation_types=["funded", "chairs"],
                                 edge_qualifiers=["monetary_value"])
    check("single prompt under budget", len(prompts) == 1, str(len(prompts)))
    pr = prompts[0]
    check("both docs embedded", '<doc id="doc_a">' in pr and '<doc id="doc_b">' in pr, "")
    check("keyed-output instruction present", "SINGLE JSON object" in pr, "")
    check("qualifier slot in batch schema", '"monetary_value"' in pr, "")
    check("no narrator instruction without authors", "FIRST-PERSON" not in pr, "")

    # Narrator hint (Abel): author rides in the doc tag + the first-person rule.
    auth = build_batch_prompt(docs, ["PERSON", "ORG"], authors={"doc_a": "Hans Müller"})
    check("author stamped in doc tag", '<doc id="doc_a" author="Hans Müller">' in auth[0], "")
    check("first-person rule present with authors", "FIRST-PERSON" in auth[0], "")
    check("unauthored doc tag stays bare", '<doc id="doc_b">' in auth[0], "")
    # Splits when the corpus exceeds the budget; every file keeps the header.
    many = [(f"d{i}", "x" * 500) for i in range(6)]
    parts = build_batch_prompt(many, ["ORG"], char_budget=1000)
    check("splits past budget", len(parts) > 1, str(len(parts)))
    check("each split self-contained", all("BATCH MODE" in p for p in parts), "")
    # Doc-count cap is the anti-truncation knob: 10 small docs, 4 per file -> 3 files.
    ten = [(f"e{i}", "y") for i in range(10)]
    byn = build_batch_prompt(ten, ["ORG"], max_docs=4)
    check("splits by doc count", len(byn) == 3, str(len(byn)))
    # count real doc tags only (the header carries one literal <doc id="..."> example).
    check("each file at most max_docs", all(p.count('<doc id="e') <= 4 for p in byn),
          str([p.count('<doc id="e') for p in byn]))

    # Coercion accepts keyed dict, list form, and a bare single-doc object.
    check("coerce keyed", set(_coerce_doc_map({"d1": {"entities": []}})) == {"d1"}, "")
    check("coerce list", set(_coerce_doc_map([{"doc_id": "d2", "relationships": []}])) == {"d2"}, "")
    check("coerce bare single", set(_coerce_doc_map({"entities": [], "relationships": []})) == {""}, "")

    # Parse a reply (code-fence wrapped, like a chat model emits) -> extraction with
    # the qualifier carried and the evidence-verbatim flag set against the doc text.
    reply = '```json\n{"doc_a": {"entities": [{"name":"Acme","type":"ORG"}],' \
            '"relationships": [{"source":"Acme","target":"Berger Fund","type":"funded",' \
            '"evidence":"Acme funded the Berger Fund with $5,000.","monetary_value":"$5,000"}],' \
            '"timeline": []}}\n```'
    meta = {"doc_a": {"text": docs[0][1], "source_path": "a.txt", "author": ""}}
    exts = parse_batch_response(reply, meta, ["PERSON", "ORG"], edge_qualifiers=["monetary_value"])
    check("one extraction parsed", len(exts) == 1 and exts[0].doc_id == "doc_a", str(exts))
    rels = exts[0].relationships
    check("relationship parsed", len(rels) == 1 and rels[0].rel_type == "funded", str(rels))
    check("qualifier captured on import",
          rels[0].attributes.get("qual_monetary_value") == "$5,000", str(rels[0].attributes))
    check("verbatim evidence not flagged unverified",
          "evidence_unverified" not in rels[0].attributes, str(rels[0].attributes))
    check("backend tagged gemini_batch", exts[0].meta.get("backend") == "gemini_batch", "")

    # Narrator flag: the mention matching the doc author is marked is_author, which
    # the letter_id stamp + German metadata join key off (without it: 0 merges). A
    # close spelling variant still flags (model 'corrects' Vilwak -> Villwak from the
    # text); an unrelated person never does.
    nar = '{"doc_n": {"entities": [{"name":"August Villwak","type":"PERSON"},' \
          '{"name":"Adolf Hitler","type":"PERSON"}], "relationships": [], "timeline": []}}'
    meta_n = {"doc_n": {"text": "August Villwak joined.", "source_path": "v.rtf",
                        "author": "August Vilwak"}}  # filename spelling off by one 'l'
    ex_n = parse_batch_response(nar, meta_n, ["PERSON"])[0]
    flagged = {m.text for m in ex_n.mentions if m.attributes.get("is_author")}
    check("narrator flagged via spelling variant", flagged == {"August Villwak"}, str(flagged))
    # Exact match flags too; a different person stays unflagged.
    meta_x = {"doc_n": {"text": "x", "source_path": "v.rtf", "author": "August Villwak"}}
    ex_x = parse_batch_response(nar, meta_x, ["PERSON"])[0]
    fx = {m.text for m in ex_x.mentions if m.attributes.get("is_author")}
    check("narrator flagged on exact name", fx == {"August Villwak"}, str(fx))


def test_gemini_batch_resume() -> None:
    import tempfile
    from pathlib import Path
    from main import Pipeline
    print("-- gemini_batch --resume checkpoint predicate")
    # The saved reply file IS the checkpoint: a reply that RECOVERS (same repair ladder
    # as import) is done and skipped on resume; a truncated/empty/missing one gets
    # re-submitted. Must NOT use a strict parse - gemma prepends a prose preamble, so a
    # strict check re-POSTed every finished gemma batch into the flaky endpoint.
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        good = dd / "r1.json"; good.write_text('{"doc_a": {"entities": []}}', encoding="utf-8")
        trunc = dd / "r2.json"; trunc.write_text('{"doc_a": {"entities": [{"name":"X"', encoding="utf-8")
        empty = dd / "r3.json"; empty.write_text("", encoding="utf-8")
        # gemma's real shape: prose paraphrase (with stray brackets) then the JSON.
        preamble = dd / "r4.json"
        preamble.write_text("Here is the extraction. confidence float [0, 1].\n"
                            '{"doc_b": {"entities": [{"name": "Ude"}], "relationships": []}}',
                            encoding="utf-8")
        check("complete reply -> skip", Pipeline._reply_complete(good) is True, "")
        check("prose-preamble reply -> skip (not re-POSTed)",
              Pipeline._reply_complete(preamble) is True, "")
        check("truncated reply -> redo", Pipeline._reply_complete(trunc) is False, "")
        check("empty reply -> redo", Pipeline._reply_complete(empty) is False, "")
        check("missing reply -> redo", Pipeline._reply_complete(dd / "nope.json") is False, "")

    # batch_post_llm live backend: the post-extraction LLM steps run through Gemini's
    # OpenAI-compatible endpoint, so the api block is repointed there (no client built).
    from main import gemini_live_config
    from config import load_config
    api = gemini_live_config(load_config("domain/nazi_era/config_nazi_era.yaml")).intelligence.api
    check("live cfg provider is openai", api.provider == "openai", api.provider)
    check("live cfg points at gemini endpoint",
          "generativelanguage.googleapis.com" in api.base_url, api.base_url)
    check("live cfg forces json mode", api.json_mode is True, "")
    check("live cfg uses the batch key", api.api_key_env == "GEMINI_API_KEY", api.api_key_env)
    check("live cfg uses the batch model", api.model.startswith("gemini-"), api.model)

    # Proactive --submit pacing: hold request starts at batch_rpm/min so a free-tier
    # run doesn't trip 429 and burn backoff. 0 = unthrottled.
    from main import _rpm_delay
    check("rpm 0 -> no throttle", _rpm_delay(100.0, 100.0, 0) == 0.0, "")
    check("rpm 10 -> 6s spacing, full wait at t=0",
          abs(_rpm_delay(100.0, 100.0, 10) - 6.0) < 1e-9, str(_rpm_delay(100.0, 100.0, 10)))
    check("interval already elapsed -> no wait",
          _rpm_delay(100.0, 110.0, 10) == 0.0, str(_rpm_delay(100.0, 110.0, 10)))
    check("partway through interval -> remainder",
          abs(_rpm_delay(100.0, 104.0, 10) - 2.0) < 1e-9, str(_rpm_delay(100.0, 104.0, 10)))


def test_gemini_submit() -> None:
    from unittest.mock import MagicMock, patch
    from intelligence.manual_batch import submit_to_gemini
    print("-- gemini --submit API call (mocked)")

    seen = {}

    def ok_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, headers=headers or {}, body=json or {})
        m = MagicMock(); m.status_code = 200; m.raise_for_status = lambda: None
        m.json = lambda: {"candidates": [{"content": {"parts": [{"text": '{"doc_a":{}}'}]},
                                          "finishReason": "STOP"}]}
        return m

    with patch("requests.post", ok_post):
        out = submit_to_gemini("PROMPT", "KEY", model="gemini-2.5-flash")
    check("returns reply text", out == '{"doc_a":{}}', out)
    check("hits generateContent endpoint", "gemini-2.5-flash:generateContent" in seen["url"], seen["url"])
    check("sends api key header", seen["headers"].get("x-goog-api-key") == "KEY", str(seen["headers"]))
    check("forces JSON output",
          seen["body"]["generationConfig"]["responseMimeType"] == "application/json", "")
    check("sets high output cap",
          seen["body"]["generationConfig"]["maxOutputTokens"] == 65536, "")
    # Default disables thinking so the output budget isn't burned on reasoning tokens.
    check("disables thinking on flash",
          seen["body"]["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0, "")

    # 2.5-pro can't go to 0; a 0 request is bumped to its 128 floor (not omitted).
    with patch("requests.post", ok_post):
        submit_to_gemini("P", "KEY", model="gemini-2.5-pro", thinking_budget=0)
    check("pro floors thinking at 128",
          seen["body"]["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 128, "")

    # Negative budget keeps the model default on -> no thinkingConfig sent at all.
    with patch("requests.post", ok_post):
        submit_to_gemini("P", "KEY", model="gemini-2.5-flash", thinking_budget=-1)
    check("negative budget omits thinkingConfig",
          "thinkingConfig" not in seen["body"]["generationConfig"], "")

    # Gemma has no thinking mode and rejects thinkingConfig -> never send it, even at
    # the default budget (so --batch-model gemma-... works unchanged).
    with patch("requests.post", ok_post):
        submit_to_gemini("P", "KEY", model="gemma-4-31b-it", thinking_budget=0)
    check("gemma omits thinkingConfig",
          "thinkingConfig" not in seen["body"]["generationConfig"], "")

    # 429 -> backoff -> 200 (retry path), with sleep stubbed so the test is instant.
    calls = {"n": 0}

    def flaky_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        m = MagicMock()
        if calls["n"] == 1:
            m.status_code = 429
            return m
        m.status_code = 200; m.raise_for_status = lambda: None
        m.json = lambda: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        return m

    with patch("requests.post", flaky_post), patch("time.sleep", lambda *a: None):
        out2 = submit_to_gemini("P", "KEY", max_retries=3)
    check("retries past 429", out2 == "ok" and calls["n"] == 2, str(calls))

    # Server-requested retry delay is honored exactly (capped), not guessed.
    from intelligence.manual_batch import _retry_after_seconds
    h = MagicMock(); h.headers = {"Retry-After": "12"}; h.json = lambda: {}
    check("Retry-After header parsed", _retry_after_seconds(h) == 12.0, "")
    b = MagicMock(); b.headers = {}
    b.json = lambda: {"error": {"details": [{"@type": "RetryInfo", "retryDelay": "27s"}]}}
    check("retryDelay body parsed", _retry_after_seconds(b) == 27.0, "")
    n = MagicMock(); n.headers = {}; n.json = lambda: {"error": {}}
    check("no hint -> None (falls back to backoff)", _retry_after_seconds(n) is None, "")
    # The 429 retry actually waits the server-requested delay.
    waited = {}

    def post_with_delay(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        m = MagicMock()
        if calls["n"] == 1:
            m.status_code = 429; m.headers = {}
            m.json = lambda: {"error": {"details": [{"retryDelay": "9s"}]}}
            return m
        m.status_code = 200; m.raise_for_status = lambda: None
        m.json = lambda: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        return m
    calls["n"] = 0
    with patch("requests.post", post_with_delay), patch("time.sleep", lambda s: waited.update(s=s)):
        submit_to_gemini("P", "KEY", max_retries=3)
    check("429 waits the server-stated 9s", waited.get("s") == 9.0, str(waited))

    # A read timeout / dropped connection is the server stalling, not our ingest. The
    # POST raises before any reply exists, so retry it like a 5xx instead of killing
    # the batch. (Regression: gemma free endpoint timed out and lost the whole batch.)
    import requests as _rq
    calls["n"] = 0

    def timeout_then_ok(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _rq.exceptions.ReadTimeout("read timed out")
        m = MagicMock(); m.status_code = 200; m.raise_for_status = lambda: None
        m.json = lambda: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        return m
    with patch("requests.post", timeout_then_ok), patch("time.sleep", lambda *a: None):
        out3 = submit_to_gemini("P", "KEY", max_retries=3)
    check("read timeout retried, not fatal", out3 == "ok" and calls["n"] == 2, str(calls))

    # Persistent timeout re-raises after the retries so --resume re-queues the batch
    # (the reply file is never written, so resume sees it as undone).
    calls["n"] = 0

    def always_timeout(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        raise _rq.exceptions.ConnectionError("connection reset")
    raised = False
    try:
        with patch("requests.post", always_timeout), patch("time.sleep", lambda *a: None):
            submit_to_gemini("P", "KEY", max_retries=3)
    except _rq.exceptions.ConnectionError:
        raised = True
    check("persistent network failure re-raises after retries", raised and calls["n"] == 3,
          str(calls))


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


def test_documents_snapshot() -> None:
    import tempfile
    from pathlib import Path
    from core.preprocessor import read_documents_snapshot, write_documents_snapshot
    from core.schema import Document
    print("-- ingestion checkpoint (documents.jsonl snapshot)")
    docs = [Document(doc_id="url_a", source_path="http://x/y",
                     text="Müller joined the NSDAP.", meta={"source_type": "url", "n_chars": 5}),
            Document(doc_id="url_b", source_path="http://x/z", text="", meta={})]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "documents.jsonl"
        n = write_documents_snapshot(docs, p)
        back = read_documents_snapshot(p)
        check("snapshot count", n == 2 and len(back) == 2, str(n))
        check("doc_id preserved", back[0].doc_id == "url_a", back[0].doc_id)
        check("unicode text preserved", back[0].text == "Müller joined the NSDAP.", back[0].text)
        check("meta preserved", back[0].meta.get("source_type") == "url", str(back[0].meta))
        # bad line is skipped, not raised
        p.write_text('{"doc_id":"ok","source_path":"s","text":"t","meta":{}}\nNOT JSON\n',
                     encoding="utf-8")
        rec = read_documents_snapshot(p)
        check("bad line skipped", len(rec) == 1 and rec[0].doc_id == "ok", str(rec))


def test_script_parser() -> None:
    from core.script_parser import looks_like_script, parse_scenes, script_copresence
    from postprocess.evidence_tiers import tier_allows
    from postprocess.tie_classes import classify, is_symmetric
    print("-- script parser (scene co-presence)")
    script = ("FADE IN:\nINT. TAVERN - NIGHT\nGANDALF\nYou shall not pass.\nBILBO\n"
              "I never could refuse you.\nTHORIN (CONT'D)\nEnough talk.\n"
              "EXT. MOUNTAIN - DAY\nTHORIN\nWe march.\nBILBO\nMust we?\n"
              "CUT TO:\nINT. CAVE - NIGHT\nGOLLUM\nMy precious.\nBILBO\nWho are you?\n")
    check("looks_like_script true", looks_like_script(script) is True, "")
    check("prose not a script", looks_like_script("Just a normal paragraph. Nothing here.") is False, "")
    scenes = parse_scenes(script)
    check("speaker cues parsed", scenes and ["Gandalf", "Bilbo", "Thorin"] == scenes[0], str(scenes))
    edges = script_copresence(script, "d")
    pairs = {(e.source, e.target): e.attributes["scene_weight"] for e in edges}
    check("Newman weight sums across scenes", pairs.get(("Bilbo", "Thorin")) == 1.5, str(pairs))
    check("all edges co_present_in_scene", all(e.rel_type == "co_present_in_scene" for e in edges), "")
    check("script edge symmetric", is_symmetric("co_present_in_scene") is True, "")
    check("co-presence classed cooccurrence",
          classify("co_present_in_scene", "PERSON", "PERSON") == "cooccurrence", "")
    check("script_copresence is proximity tier",
          (not tier_allows("script_copresence", "conservative"))
          and tier_allows("script_copresence", "full"), "")
    check("non-script -> no edges", script_copresence("plain prose, no scenes", "d") == [], "")


def test_generic_relation_ontology() -> None:
    from domain.base_domain import load_domain
    from postprocess.tie_classes import classify, polarity
    print("-- generic default relation ontology")
    d = load_domain("generic")
    onto = d.relation_ontology()
    check("generic ontology nonempty", len(onto) >= 20, str(len(onto)))
    check("canonical names align with tie maps",
          classify("led", "PERSON", "ORG") == "affiliation"
          and polarity("supported") == "positive"
          and classify("born_in", "PERSON", "LOCATION") == "biographical", "")


def test_fiction_narrative_scheme() -> None:
    from postprocess.narrative import ELEMENT_SCHEMES, FICTION_ELEMENT_RULES, categorize
    print("-- fiction narrative scheme")
    check("schemes registered", set(ELEMENT_SCHEMES) >= {"life_course", "fiction"}, str(list(ELEMENT_SCHEMES)))
    check("fiction beats categorize",
          categorize("they kissed at the wedding", FICTION_ELEMENT_RULES) == "romance"
          and categorize("he set out on a long journey", FICTION_ELEMENT_RULES) == "journey", "")


def test_playwright_routing() -> None:
    from unittest.mock import MagicMock, patch
    import core.crawler as cr
    print("-- playwright fetcher raw-resource routing")
    f = cr.PlaywrightFetcher(timeout=5)
    seen = {}

    def fake_get(url, *a, **k):
        seen["url"] = url
        m = MagicMock(); m.ok = True; m.text = "User-agent: *"
        return m
    with patch.object(cr, "http_get", fake_get):
        f("https://example.com/robots.txt")
        f("https://example.com/sitemap.xml")
    check("raw resource bypasses browser", seen.get("url") is not None and f._ok is None, str(seen))


def test_social_connectors() -> None:
    import json as _json
    from core.social import fetch_social, parse_spec, posts_to_documents, social_structure
    from core.social import reddit as _reddit
    from core.social import hackernews as _hn
    from core.social.base import extract_mentions
    from postprocess.evidence_tiers import tier_allows
    from postprocess.tie_classes import classify
    print("-- social connectors (reddit/hn structure)")

    # spec parsing + facebook is refused with the sanctioned alternative.
    check("spec parsed", parse_spec("reddit:datascience") == ("reddit", "datascience"), "")
    fb_refused = False
    try:
        fetch_social("facebook:somegroup")
    except ValueError as e:
        fb_refused = "ToS" in str(e) or "not supported" in str(e)
    check("facebook scraping refused", fb_refused, "")
    check("mention extraction", extract_mentions("hi @carol and u/bob!") == ["carol", "bob"],
          str(extract_mentions("hi @carol and u/bob!")))

    # Reddit via injected fetch (no network): one submission + two comments.
    listing = _json.dumps({"data": {"children": [
        {"kind": "t3", "data": {"id": "p1", "author": "alice", "title": "Hello",
                                "selftext": "hey u/bob", "subreddit": "test",
                                "permalink": "/r/test/p1", "score": 5, "num_comments": 2}}]}})
    comments = _json.dumps([
        {"data": {"children": []}},
        {"data": {"children": [
            {"kind": "t1", "data": {"id": "c1", "author": "bob", "body": "hi @carol",
                                    "parent_id": "t3_p1", "subreddit": "test"}},
            {"kind": "t1", "data": {"id": "c2", "author": "carol", "body": "ok",
                                    "parent_id": "t1_c1", "subreddit": "test"}}]}}])

    def rfetch(url):
        return comments if "/comments/" in url else listing

    posts = _reddit.fetch("test", fetch=rfetch, delay=0)
    check("reddit posts parsed", len(posts) == 3, str(len(posts)))
    docs = posts_to_documents(posts)
    by_author = {p.author: p for p in posts}
    check("parent_author resolved", by_author["bob"].parent_author == "alice"
          and by_author["carol"].parent_author == "bob", "")
    # structure of bob's comment: replied_to alice, mentions carol, posted_in r/test
    bob_doc = next(d for d in docs if d.meta.get("author") == "bob")
    ments, edges = social_structure(bob_doc)
    et = {(e.source, e.target, e.rel_type) for e in edges}
    check("replied_to edge", ("bob", "alice", "replied_to") in et, str(et))
    check("mentions edge", ("bob", "carol", "mentions") in et, str(et))
    check("posted_in edge", ("bob", "r/test", "posted_in") in et, str(et))
    check("social edges are social_graph",
          all(e.attributes.get("edge_source") == "social_graph" for e in edges), "")
    check("user node is PERSON", any(m.label == "PERSON" and m.text == "bob" for m in ments), "")
    check("community node is ORG", any(m.label == "ORG" and m.text == "r/test" for m in ments), "")

    # tie-class + tier registration
    check("replied_to -> interaction", classify("replied_to", "PERSON", "PERSON") == "interaction", "")
    check("posted_in -> affiliation", classify("posted_in", "PERSON", "ORG") == "affiliation", "")
    check("social_graph is asserted tier",
          tier_allows("social_graph", "conservative") and tier_allows("social_graph", "full"), "")

    # Hacker News via injected fetch: a story + one comment.
    hn = {"https://hacker-news.firebaseio.com/v0/topstories.json": "[10]",
          "https://hacker-news.firebaseio.com/v0/item/10.json":
              _json.dumps({"id": 10, "type": "story", "by": "ann", "title": "T",
                           "kids": [11], "score": 9}),
          "https://hacker-news.firebaseio.com/v0/item/11.json":
              _json.dumps({"id": 11, "type": "comment", "by": "ben", "parent": 10,
                           "text": "nice"})}
    hposts = _hn.fetch("top", fetch=lambda u: hn[u])
    check("hn story+comment parsed", len(hposts) == 2 and hposts[0].author == "ann", str(hposts))
    check("hn comment shares story community",
          hposts[1].community == hposts[0].community and hposts[0].community.startswith("HN:"), "")


def test_social_fediverse() -> None:
    import json as _json
    from core.social import posts_to_documents, social_structure, supported_platforms
    from core.social import bluesky as _bsky
    from core.social import lemmy as _lemmy
    print("-- social: bluesky + lemmy + truthsocial")
    for p in ("bluesky", "bsky", "lemmy", "truthsocial", "truth"):
        check(f"{p} registered", p in supported_platforms(), str(supported_platforms()))

    # Bluesky search via public AppView (injected).
    bsky_resp = _json.dumps({"posts": [
        {"uri": "at://x/1", "author": {"handle": "alice.bsky.social"},
         "record": {"text": "hello @bob.bsky.social"}, "likeCount": 3},
        {"uri": "at://x/2", "author": {"handle": "bob.bsky.social"},
         "record": {"text": "hi", "reply": {"parent": {"uri": "at://x/1"},
                                            "root": {"uri": "at://x/1"}}}}]})
    bposts = _bsky.fetch("climate", fetch=lambda u: bsky_resp)
    check("bluesky posts parsed", len(bposts) == 2 and bposts[0].author == "alice.bsky.social",
          str(bposts))
    bdocs = posts_to_documents(bposts)
    reply_doc = next(d for d in bdocs if d.meta.get("author") == "bob.bsky.social")
    check("bluesky reply parent resolved", reply_doc.meta.get("parent_author") == "alice.bsky.social",
          str(reply_doc.meta))
    a_doc = next(d for d in bdocs if d.meta.get("author") == "alice.bsky.social")
    _, a_edges = social_structure(a_doc)
    check("bluesky dotted-handle mention",
          ("alice.bsky.social", "bob.bsky.social", "mentions") in
          {(e.source, e.target, e.rel_type) for e in a_edges}, str(a_edges))

    # Lemmy community + threaded comments (injected).
    plist = _json.dumps({"posts": [{"post": {"id": 1, "name": "T", "body": "hi @u@lemmy.world",
                                             "ap_id": "u1"}, "creator": {"name": "alice"},
                                    "community": {"name": "tech"}, "counts": {"score": 5}}]})
    clist = _json.dumps({"comments": [
        {"comment": {"id": 11, "content": "reply", "path": "0.11", "post_id": 1},
         "creator": {"name": "bob"}},
        {"comment": {"id": 12, "content": "nested", "path": "0.11.12", "post_id": 1},
         "creator": {"name": "carol"}}]})
    lposts = _lemmy.fetch("lemmy.world/tech",
                          fetch=lambda u: clist if "/comment/" in u else plist, delay=0)
    check("lemmy submission+comments", len(lposts) == 3, str(len(lposts)))
    ldocs = posts_to_documents(lposts)
    by_author = {d.meta.get("author"): d for d in ldocs}
    check("lemmy nested reply resolved",
          by_author["carol"].meta.get("parent_author") == "bob"
          and by_author["bob"].meta.get("parent_author") == "alice", "")
    _, bob_edges = social_structure(by_author["bob"])
    check("lemmy replied_to + posted_in",
          {("bob", "alice", "replied_to"), ("bob", "!tech", "posted_in")}
          <= {(e.source, e.target, e.rel_type) for e in bob_edges}, str(bob_edges))


def test_social_telegram() -> None:
    from core.social import posts_to_documents, social_structure, supported_platforms
    from core.social import telegram as _tg
    from postprocess.evidence_tiers import tier_allows
    from postprocess.tie_classes import classify
    print("-- social: telegram (channel preview, forwards)")
    for p in ("telegram", "tg"):
        check(f"{p} registered", p in supported_platforms(), str(supported_platforms()))

    # Minimal t.me/s/ preview: msg 100 is a forward from @telegram; msg 101 mentions
    # @bob and links t.me/alice. Owner/date hrefs and the permalink must NOT leak in.
    page = (
        '<div class="tgme_widget_message" data-post="durov/100">'
        '<a class="tgme_widget_message_owner_name" href="https://t.me/durov"><span>Durov</span></a>'
        '<a class="tgme_widget_message_forwarded_from_name" href="https://t.me/telegram">Telegram</a>'
        '<div class="tgme_widget_message_text js-message_text">big news</div>'
        '<a class="tgme_widget_message_date" href="https://t.me/durov/100">'
        '<time datetime="2024-01-01T00:00:00+00:00"></time></a>'
        '<span class="tgme_widget_message_views">1.2K</span></div>'
        '<div class="tgme_widget_message" data-post="durov/101">'
        '<a class="tgme_widget_message_owner_name" href="https://t.me/durov"><span>Durov</span></a>'
        '<div class="tgme_widget_message_text js-message_text">hi @bob see '
        '<a href="https://t.me/alice">alice</a></div>'
        '<a class="tgme_widget_message_date" href="https://t.me/durov/101">'
        '<time datetime="2024-01-02T00:00:00+00:00"></time></a></div>')

    posts = _tg.fetch("@durov", fetch=lambda u: page, delay=0)
    check("telegram messages parsed", len(posts) == 2 and posts[0].author == "durov", str(posts))
    check("telegram forward captured", posts[0].forwarded_from == "telegram", str(posts[0]))
    check("telegram mentions, no permalink/owner leak",
          set(posts[1].mentions) == {"alice", "bob"}, str(posts[1].mentions))

    docs = posts_to_documents(posts)
    by_pid = {d.meta.get("filename"): d for d in docs}
    fwd_doc = next(d for d in docs if d.meta.get("forwarded_from") == "telegram")
    _, fwd_edges = social_structure(fwd_doc)
    et = {(e.source, e.target, e.rel_type) for e in fwd_edges}
    check("telegram forwarded_from edge", ("durov", "telegram", "forwarded_from") in et, str(et))
    check("forwarded_from is social_graph asserted",
          all(e.attributes.get("edge_source") == "social_graph" for e in fwd_edges)
          and tier_allows("social_graph", "conservative"), "")
    check("forwarded_from -> interaction",
          classify("forwarded_from", "PERSON", "PERSON") == "interaction", "")

    mention_doc = next(d for d in docs if d.meta.get("author") == "durov"
                       and not d.meta.get("forwarded_from"))
    _, m_edges = social_structure(mention_doc)
    met = {(e.source, e.target, e.rel_type) for e in m_edges}
    check("telegram mention edges",
          {("durov", "alice", "mentions"), ("durov", "bob", "mentions")} <= met, str(met))


def test_input_adapters() -> None:
    import io as _io
    import tempfile
    import zipfile
    from pathlib import Path
    from core.preprocessor import strip_boilerplate, extract_text
    print("-- input adapters: boilerplate strip + epub")

    # Boilerplate strip: a nav fragment removed, real text kept, spacing collapsed.
    nav = "Or browse groups, people, or influence networks."
    txt = f"{nav} Jane Roe is on the board of the Acme Fund."
    out = strip_boilerplate(txt, [r"Or browse groups, people, or influence networks\.\s*"])
    check("boilerplate fragment stripped", nav not in out and "Jane Roe" in out, out)
    check("boilerplate no-op without patterns", strip_boilerplate(txt, []) == txt, "")

    # EPUB: a minimal zip of XHTML chapters; nav/toc entries skipped, prose extracted.
    d = Path(tempfile.mkdtemp(prefix="epub_"))
    epub = d / "book.epub"
    with zipfile.ZipFile(epub, "w") as zf:
        zf.writestr("ch1.xhtml", "<html><body><p>Frodo met Gandalf in the Shire.</p></body></html>")
        zf.writestr("ch2.xhtml", "<html><body><p>Aragorn fought Sauron at the gate.</p></body></html>")
        zf.writestr("nav.xhtml", "<html><body><p>Table of contents listing here.</p></body></html>")
    text = extract_text(epub)
    check("epub prose extracted", "Frodo met Gandalf" in text and "Aragorn fought Sauron" in text, text[:120])
    check("epub nav entry skipped", "Table of contents" not in text, text[:120])
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def test_wiki_connector() -> None:
    import json as _json
    from core.wiki import fetch_wiki, parse_spec
    print("-- wiki: mediawiki connector")
    check("wiki spec parsed",
          parse_spec("en.wikipedia.org:Category:Physicists") == ("en.wikipedia.org", "Category:Physicists"), "")

    members = _json.dumps({"query": {"categorymembers": [
        {"title": "Ada Lovelace"}, {"title": "Charles Babbage"}]}})
    ada = _json.dumps({"query": {"pages": {"1": {"title": "Ada Lovelace",
        "extract": "Ada Lovelace worked with Charles Babbage on the Analytical Engine.\n\n== References ==\nCited works here."}}}})
    bab = _json.dumps({"query": {"pages": {"2": {"title": "Charles Babbage",
        "extract": "Charles Babbage designed the Difference Engine."}}}})

    def wfetch(url):
        if "list=categorymembers" in url:
            return members
        if "Ada%20Lovelace" in url or "Ada_Lovelace" in url or "Ada Lovelace" in url:
            return ada
        return bab

    docs = fetch_wiki("en.wikipedia.org:Category:Computing pioneers", limit=10, fetch=wfetch)
    check("wiki category resolved to pages", len(docs) == 2, str(len(docs)))
    by = {d.meta.get("title"): d for d in docs}
    check("wiki page text extracted", "Analytical Engine" in by["Ada Lovelace"].text, "")
    check("wiki reference tail stripped",
          "Cited works here" not in by["Ada Lovelace"].text, by["Ada Lovelace"].text[-60:])
    check("wiki source_type tagged",
          all(d.meta.get("source_type") == "wiki" for d in docs), "")
    check("wiki doc_id stable + url source",
          docs[0].source_path.startswith("https://en.wikipedia.org/wiki/"), docs[0].source_path)

    # explicit pipe-separated titles (no category)
    docs2 = fetch_wiki("en.wikipedia.org:Ada Lovelace|Charles Babbage", limit=10, fetch=wfetch)
    check("wiki explicit titles fetched", len(docs2) == 2, str(len(docs2)))


def test_littlesis_connector() -> None:
    import json as _json
    from core.littlesis import fetch_littlesis, littlesis_structure, parse_spec, _endpoint
    from postprocess.evidence_tiers import tier_allows
    from postprocess.tie_classes import classify
    print("-- littlesis: curated relationship graph")

    check("ls spec search", parse_spec("search:Koch Industries") == ("search", "Koch Industries"), "")
    check("ls spec id", parse_spec("id:28220") == ("id", "28220"), "")
    check("ls spec bare == search", parse_spec("Koch") == ("search", "Koch"), "")
    # URL endpoint parse keeps internal hyphens in the name, splits id off the first.
    check("ls endpoint parse",
          _endpoint("https://littlesis.org/org/12-Coca-Cola_Co") == ("ORG", "12", "Coca-Cola Co"),
          str(_endpoint("https://littlesis.org/org/12-Coca-Cola_Co")))

    search = _json.dumps({"data": [{"type": "entities", "id": 28220, "attributes": {
        "id": 28220, "name": "Koch Industries, Inc.", "primary_ext": "Org",
        "blurb": "Private conglomerate.", "summary": "Owns many companies."}}]})
    rels = _json.dumps({"meta": {"currentPage": 1, "pageCount": 1}, "data": [
        {"attributes": {"category_id": 1, "amount": None, "currency": None,
                        "start_date": None, "description": "Rich Fink has a position at Koch Industries, Inc.",
                        "category_attributes": {"is_board": True, "is_executive": True}},
         "entity": "https://littlesis.org/person/41346-Rich_Fink",
         "related": "https://littlesis.org/org/28220-Koch_Industries,_Inc."},
        {"attributes": {"category_id": 5, "amount": 1500000, "currency": "usd",
                        "start_date": "2023-07-05", "description": "Koch gave money to Senate Leadership Fund"},
         "entity": "https://littlesis.org/org/28220-Koch_Industries,_Inc.",
         "related": "https://littlesis.org/org/244260-Senate_Leadership_Fund_(Super_PAC)"}]})

    def lsfetch(url):
        return rels if "/relationships" in url else search

    docs = fetch_littlesis("search:Koch", limit=5, fetch=lsfetch)
    check("ls one seed entity", len(docs) == 1 and docs[0].meta.get("name") == "Koch Industries, Inc.", "")
    check("ls license recorded", docs[0].meta.get("license") == "CC BY-SA 4.0", str(docs[0].meta.get("license")))
    check("ls edges captured", len(docs[0].meta.get("ls_edges") or []) == 2, "")

    ments, edges = littlesis_structure(docs[0])
    et = {(e.source, e.target, e.rel_type) for e in edges}
    check("ls position -> board_member_of",
          ("Rich Fink", "Koch Industries, Inc.", "board_member_of") in et, str(et))
    check("ls donation -> donated_to",
          ("Koch Industries, Inc.", "Senate Leadership Fund (Super PAC)", "donated_to") in et, str(et))
    donation = next(e for e in edges if e.rel_type == "donated_to")
    check("ls donation amount -> qual_monetary_value",
          donation.attributes.get("qual_monetary_value") == 1500000, str(donation.attributes))
    check("ls edges are littlesis source",
          all(e.attributes.get("edge_source") == "littlesis" for e in edges), "")
    check("ls person node is PERSON",
          any(m.text == "Rich Fink" and m.label == "PERSON" for m in ments), "")
    check("ls org node is ORG",
          any(m.text == "Senate Leadership Fund (Super PAC)" and m.label == "ORG" for m in ments), "")
    check("ls is asserted tier",
          tier_allows("littlesis", "conservative") and tier_allows("littlesis", "full"), "")
    check("ls donated_to tie-class affiliation",
          classify("donated_to", "ORG", "ORG") == "affiliation", "")


def test_littlesis_bulk() -> None:
    import gzip
    import json as _json
    import shutil
    import tempfile
    from pathlib import Path
    from core.littlesis import load_bulk, littlesis_structure, _iter_json_array
    print("-- littlesis: bulk dump import")

    # Streaming array parser must yield all objects across a tiny chunk boundary.
    import io as _io
    arr = '[{"a":1},{"a":2},{"a":3}]'
    got = list(_iter_json_array(_io.StringIO(arr), chunk_size=4))
    check("stream array parser crosses chunks",
          [o["a"] for o in got] == [1, 2, 3], str(got))

    recs = [
        {"attributes": {"category_id": 1, "amount": None,
                        "category_attributes": {"is_board": True},
                        "description": "Allen Questrom position at Walmart"},
         "entity": "https://littlesis.org/person/1006-Allen_Questrom",
         "related": "https://littlesis.org/org/1-Walmart_Inc."},
        {"attributes": {"category_id": 5, "amount": 50000, "currency": "usd",
                        "start_date": "2020-01-01", "description": "Walmart gave to Some PAC"},
         "entity": "https://littlesis.org/org/1-Walmart_Inc.",
         "related": "https://littlesis.org/org/999-Some_PAC"},
        {"attributes": {"category_id": 10, "amount": None,
                        "description": "Walmart owns Sam's Club"},
         "entity": "https://littlesis.org/org/1-Walmart_Inc.",
         "related": "https://littlesis.org/org/777-Sams_Club"},
    ]
    d = Path(tempfile.mkdtemp(prefix="lsbulk_"))
    gz = d / "relationships.json.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        fh.write(_json.dumps(recs))
    try:
        docs = load_bulk(gz)
        total = sum(len(x.meta["ls_edges"]) for x in docs)
        check("bulk imports all edges", total == 3, str(total))
        # category filter: only the donation
        only_donation = load_bulk(gz, categories={5})
        de = [e for doc in only_donation for e in doc.meta["ls_edges"]]
        check("bulk category filter", len(de) == 1 and de[0]["rel"] == "donated_to", str(de))
        # the donation maps with the amount on the asserted edge
        _, edges = littlesis_structure(only_donation[0])
        check("bulk donation amount carried",
              edges[0].attributes.get("qual_monetary_value") == 50000
              and edges[0].attributes.get("edge_source") == "littlesis", str(edges[0].attributes))
        # name filter keeps only edges touching the named entity (normalized, drops "Inc.")
        wal = load_bulk(gz, names={"Walmart"})
        check("bulk name filter (normalized) keeps Walmart edges",
              sum(len(x.meta["ls_edges"]) for x in wal) == 3, "")
        # id filter
        pac = load_bulk(gz, ids={"999"})
        check("bulk id filter", sum(len(x.meta["ls_edges"]) for x in pac) == 1, "")

        # induced: keep only edges where BOTH endpoints match (Walmart->Some PAC).
        ind = load_bulk(gz, names={"Walmart Inc.", "Some PAC"}, both_endpoints=True)
        ie = [e for doc in ind for e in doc.meta["ls_edges"]]
        check("bulk induced (both endpoints)",
              len(ie) == 1 and ie[0]["rel"] == "donated_to", str(ie))

        # entities.json enrichment: node carries blurb/types as attr_*; isolated node added.
        ents = [
            {"attributes": {"id": 1, "name": "Walmart Inc.", "primary_ext": "Org",
                            "blurb": "Retail giant", "types": ["Organization", "Business"]}},
            {"attributes": {"id": 999, "name": "Some PAC", "primary_ext": "Org"}},
            {"attributes": {"id": 777, "name": "Sam's Club", "primary_ext": "Org"}},
            {"attributes": {"id": 1006, "name": "Allen Questrom", "primary_ext": "Person"}},
            {"attributes": {"id": 555, "name": "Lonely Corp", "primary_ext": "Org"}},  # no edges
        ]
        egz = d / "entities.json.gz"
        with gzip.open(egz, "wt", encoding="utf-8") as fh:
            fh.write(_json.dumps(ents))
        enr = load_bulk(gz, entities_path=egz, names={"Walmart"})
        wal_doc = next(x for x in enr if x.meta["name"] == "Walmart Inc.")
        check("bulk enrich: node blurb/types in ls_attrs",
              wal_doc.meta["ls_attrs"].get("ls_blurb") == "Retail giant"
              and wal_doc.meta["ls_attrs"].get("ls_types") == "Organization;Business",
              str(wal_doc.meta.get("ls_attrs")))
        ments, _ = littlesis_structure(wal_doc)
        anchor = next(m for m in ments if m.text == "Walmart Inc.")
        check("bulk enrich: attrs ride on the node mention (-> attr_* columns)",
              anchor.attributes.get("ls_blurb") == "Retail giant"
              and anchor.attributes.get("littlesis") is True, str(anchor.attributes))
        # include_isolated brings in the edge-less entity 555 as a node.
        iso = load_bulk(gz, entities_path=egz, include_isolated=True)
        check("bulk include_isolated adds edge-less entity",
              any(x.meta["name"] == "Lonely Corp" for x in iso), "")
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
    test_relation_family_scoring()
    test_scorer_entity_one_to_one()
    test_newman_cooccurrence()
    test_affiliation_projection()
    test_org_actor_projection()
    test_new_domain_packages()
    test_disparity_backbone()
    test_signed_balance()
    test_unsupported_excluded_from_substantive()
    test_trust_verification_gates_metric_drop()
    test_polarity_conflicts()
    test_causal_tie_class()
    test_generic_ontology()
    test_org_name_cleanup()
    test_sparse_chunk_gate()
    test_edge_consolidation()
    test_relation_type_signatures()
    test_type_hint_prompt()
    test_biographical_inference()
    test_author_anchoring()
    test_relation_verify()
    test_functional_consistency()
    test_recall_pass()
    test_expansion_schema_load()
    test_quality_pillars()
    test_evidence_grounding()
    test_faithfulness_tags_exported()
    test_edge_qualifiers()
    test_run_meta_provenance()
    test_span_reconcile()
    test_manual_batch()
    test_gemini_submit()
    test_gemini_batch_resume()
    test_qid_consolidation()
    test_extraction_schema()
    test_llm_review_protects_high_degree()
    test_reference_stripping()
    test_narrative_transitions()
    test_bench_bio()
    test_coref_clusters()
    test_coref_prompt_hint()
    test_coref_observability()
    test_connection_type()
    test_html_extraction()
    test_crawl_url_norm()
    test_crawler()
    test_crawl_checkpoint()
    test_coref_service_warmup()
    test_documents_snapshot()
    test_script_parser()
    test_generic_relation_ontology()
    test_fiction_narrative_scheme()
    test_playwright_routing()
    test_social_connectors()
    test_social_fediverse()
    test_social_telegram()
    test_input_adapters()
    test_wiki_connector()
    test_littlesis_connector()
    test_littlesis_bulk()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("all tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
