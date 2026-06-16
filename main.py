# CLI + orchestrator. Stages: ingest -> extract (foundation+LLM, checkpointed)
# -> analyze (aggregate, dedup, review, infer, tag, graph, export).
# Run: python main.py --config config.yaml [--stage X] [--resume] [--mode X]

from __future__ import annotations

import os

# spaCy (thinc) and GLiNER2 (torch) each bundle an OpenMP runtime. With a
# transformer spaCy model (en_core_web_trf) and GLiNER2 sharing one CPU process
# the duplicate libiomp init aborts the process - an intermittent segfault while
# loading foundation models (hits the English ollama/crawl path: foundation runs
# on CPU there). Allow the duplicate runtime; do NOT pin thread count (that would
# throttle CPU inference). Must be set before torch/spacy are imported below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from config import Config, load_config
from core.foundation import FoundationLayer
from core.preprocessor import gather_documents
from core.schema import Document, DocumentExtraction, stable_id
from checkpoint.manager import CheckpointManager
from domain.base_domain import load_domain
from intelligence.base import IntelligenceBackend
from postprocess.aggregator import aggregate, normalize_name
from postprocess.canonical_inference import InferenceEngine
from postprocess.deduplicator import Deduplicator
from postprocess.exporter import Exporter
from postprocess.gephi_builder import GephiBuilder
from postprocess.quality_review import QualityReviewer
from postprocess.tagger import Tagger

console = Console()

# Relation types that are inherently asymmetric. Forced directed so the graph
# builder doesn't sort endpoints (which flips display, e.g. "<org> member_of <person>").


# Backend factory
def build_backend(config: Config, foundation: FoundationLayer, domain=None) -> IntelligenceBackend:
    """Instantiate the intelligence backend for the configured mode."""
    mode = config.mode
    if mode == "api":
        from intelligence.api_backend import ApiBackend
        return ApiBackend(config, domain=domain)
    if mode == "ollama":
        from intelligence.ollama_backend import OllamaBackend
        return OllamaBackend(config, domain=domain)
    if mode == "python_only":
        from intelligence.python_backend import PythonBackend
        # Reuse the foundation's loaded spaCy engine to avoid a second load.
        return PythonBackend(config, spacy_engine=foundation.spacy, domain=domain)
    if mode == "langextract":
        from intelligence.langextract_backend import LangExtractBackend
        return LangExtractBackend(config, domain=domain)
    raise ValueError(f"Unknown mode: {mode}")


# Pipeline
class Pipeline:
    """End-to-end orchestrator. Holds shared, lazily-built components."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.run_dir = config.run_output_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.domain = load_domain(config.domain.name)
        self._foundation: Optional[FoundationLayer] = None
        self._backend: Optional[IntelligenceBackend] = None

    # Lazy heavy components
    @property
    def foundation(self) -> FoundationLayer:
        if self._foundation is None:
            console.print("[cyan]Loading foundation models (spaCy + GLiNER)...[/cyan]")
            self._foundation = FoundationLayer(self.config, domain=self.domain)
        return self._foundation

    @property
    def backend(self) -> IntelligenceBackend:
        if self._backend is None:
            console.print(f"[cyan]Initializing intelligence backend: {self.config.mode}[/cyan]")
            self._backend = build_backend(self.config, self.foundation, domain=self.domain)
        return self._backend

    # Stage: extract
    def run_extract(self, resume: bool, limit: int = 0,
                    extra_urls: tuple[str, ...] = (), urls_file: str = "",
                    text: str = "", crawl_seeds: tuple[str, ...] = ()) -> list[DocumentExtraction]:
        """Run foundation + intelligence over all documents with checkpointing."""
        from tqdm import tqdm

        documents = self._gather(limit, extra_urls, urls_file, text, crawl_seeds)
        if not documents:
            console.print("[yellow]No documents found.[/yellow]")
            return []

        ckpt = CheckpointManager(
            self.run_dir / "checkpoints",
            self.config.run_name,
            enabled=self.config.checkpoint.enabled,
        )
        if not resume:
            # Fresh run: ignore prior checkpoint state for skipping.
            done_ids: set[str] = set()
        else:
            done_ids = ckpt.completed_ids

        flush_every = max(1, self.config.checkpoint.flush_every)
        processed = 0
        with ckpt:
            for doc in tqdm(documents, desc="Extracting", unit="doc"):
                if resume and doc.doc_id in done_ids:
                    continue
                foundation_results = self.foundation.process_document(doc)
                extraction = self.backend.extract_document(
                    doc.doc_id, doc.source_path, foundation_results
                )
                ckpt.save(extraction, flush=(processed % flush_every == 0))
                processed += 1

        # Return only the current run's documents (respect --limit / input set),
        # not the entire accumulated checkpoint.
        current_ids = {d.doc_id for d in documents}
        extractions = [ex for ex in ckpt.load_all() if ex.doc_id in current_ids]
        console.print(
            f"[green]Extract stage complete:[/green] {len(extractions)} documents "
            f"({processed} newly processed)."
        )
        return extractions

    # Document gathering (shared by extract + analyze)
    def _gather(self, limit: int = 0, extra_urls: tuple[str, ...] = (),
                urls_file: str = "", text: str = "", crawl_seeds: tuple[str, ...] = (),
                for_extract: bool = True):
        """Collect the current run's documents (files + URLs + crawl + text),
        honoring --limit. Dedups by doc_id so the same URL from two sources is one
        node."""
        io = self.config.io
        documents = gather_documents(
            input_path=io.input_path or None,
            glob=io.input_glob,
            encoding=io.encoding,
            urls=list(io.urls) + list(extra_urls),
            urls_file=urls_file or (io.urls_file or None),
            text=text or None,
            timeout=io.request_timeout,
            use_docling=io.use_docling,
        )
        documents.extend(self._gather_crawl(crawl_seeds, for_extract))

        # Dedup by doc_id (a crawled url may also be in io.urls / a file mirror).
        seen: set[str] = set()
        documents = [d for d in documents
                     if not (d.doc_id in seen or seen.add(d.doc_id))]

        if limit and limit > 0:
            documents = documents[:limit]
            console.print(f"[yellow]--limit active: {len(documents)} documents.[/yellow]")
        return documents

    def _gather_crawl(self, crawl_seeds: tuple[str, ...], for_extract: bool) -> list[Document]:
        """Expand seed URLs into subpage Documents via the crawler. On extract,
        crawl and cache the discovered URL list; on analyze, rebuild lightweight
        id-only stubs from that cache so re-analysis never re-crawls."""
        cc = self.config.io.crawl
        seeds = list(cc.seeds) + list(crawl_seeds)
        if not seeds or not (cc.enabled or crawl_seeds):
            return []

        cache = self.run_dir / "crawled_urls.txt"
        if not for_extract and cache.exists():
            urls = [u.strip() for u in cache.read_text(encoding="utf-8").splitlines() if u.strip()]
            console.print(f"[cyan]Crawl: reusing {len(urls)} cached urls (no re-crawl).[/cyan]")
            return [Document(doc_id=stable_id(u, prefix="url_", length=10),
                             source_path=u, text="",
                             meta={"filename": u, "source_type": "url"}) for u in urls]

        from core.crawler import Crawler
        console.print(f"[cyan]Crawling {len(seeds)} seed(s) "
                      f"(max_pages={cc.max_pages}, depth={cc.max_depth}, "
                      f"robots={'on' if cc.respect_robots else 'OFF'})...[/cyan]")
        docs = Crawler(self._crawl_opts()).crawl(seeds)
        console.print(f"[green]Crawl: {len(docs)} page(s) fetched.[/green]")
        try:
            cache.write_text("\n".join(d.source_path for d in docs), encoding="utf-8")
        except Exception:  # noqa: BLE001 - cache is a nicety, not load-bearing
            pass
        return docs

    def _crawl_opts(self):
        from core.crawler import CrawlOptions
        from core.preprocessor import _USER_AGENT
        cc = self.config.io.crawl
        return CrawlOptions(
            max_pages=cc.max_pages, max_depth=cc.max_depth,
            stay_on_host=cc.stay_on_host, stay_under_path=cc.stay_under_path,
            allow=tuple(cc.allow), deny=tuple(cc.deny), delay=cc.delay,
            respect_robots=cc.respect_robots, use_sitemap=cc.use_sitemap,
            user_agent=cc.user_agent or _USER_AGENT,
            timeout=cc.timeout or self.config.io.request_timeout,
            max_bytes=cc.max_bytes,
        )

    # Stage: analyze
    def run_analyze(self, extractions: list[DocumentExtraction]) -> dict[str, str]:
        """Aggregate, dedup, review, infer, tag, build graph, and export."""
        if not extractions:
            console.print("[yellow]No extractions to analyze.[/yellow]")
            return {}

        # doc_id -> letter_id / author, for provenance + metadata join.
        from postprocess.manifest import build_manifest
        manifest = build_manifest(extractions, self.domain)

        # 1. Aggregate.
        agg = aggregate(extractions)

        # 1a0. Resolve first-person narrator placeholders into the real person the
        # document names ("Narrator [doc] is Jane Doe"), merging the two nodes and
        # dropping the identity edge. Critical for corpora without author metadata.
        from postprocess.identity_resolution import resolve_narrator_identities
        agg.entities, agg.relationships = resolve_narrator_identities(
            agg.entities, agg.relationships
        )

        # 1a. Enforce configured entity types at analyze time too. This lets an
        # already-extracted checkpoint benefit from restrict_to_label_types
        # (dropping spaCy's off-target DATE/EVENT/etc.) without re-extraction.
        # Relations to dropped entities are removed later by the dedup remap.
        if self.config.foundation.restrict_to_label_types:
            allowed = set(self.config.foundation.label_map.values())
            allowed |= set(self.domain.gliner_label_map().values())
            before = len(agg.entities)
            agg.entities = [e for e in agg.entities if e.label in allowed]
            if before != len(agg.entities):
                console.print(f"[cyan]Type restriction: kept {len(agg.entities)}/"
                              f"{before} entities (types {sorted(allowed)}).[/cyan]")

        # 1b. Ontology alignment: normalize relation-type vocabulary (domain or
        # config supplied). No-op when no ontology is configured (generic path).
        if self.config.ontology.enabled:
            from postprocess.ontology import OntologyAligner, resolve_relation_ontology
            onto = resolve_relation_ontology(self.config, self.domain)
            if onto:
                aligner = OntologyAligner(onto, self.config.ontology.fuzzy_threshold,
                                          self.config.ontology.drop_unmapped)
                agg.relationships = aligner.apply(agg.relationships)

        # 2. Deduplicate (+ remap relationships onto entity ids).
        dedup = Deduplicator(self.config.dedup, domain_aliases=self.domain.aliases())
        entities, relationships, _name_to_id = dedup.resolve(
            agg.entities, agg.relationships
        )

        llm_capable = self.config.mode in ("api", "ollama")

        # 2a1. Drop alias_of leftovers - a dedup artifact, not a social edge.
        relationships = [r for r in relationships if r.rel_type != "alias_of"]

        # 2a2. Membership edges should point at an org/institution. A
        # "member_of <profession>" or reversed "<org> member_of <person>" is
        # suspect - TAG it (suspect_membership) so it's filterable in Gephi rather
        # than silently dropped. Only delete if drop_nonorg_membership is set.
        org_ids = {e.entity_id for e in entities if e.label in ("ORG", "INSTITUTION")}
        mem = {"member_of", "joined", "served_in"}
        suspect = 0
        for r in relationships:
            if r.rel_type in mem and r.target not in org_ids:
                r.attributes["suspect_membership"] = True
                suspect += 1
        if suspect:
            console.print(f"[cyan]Tagged {suspect} non-org membership edges (suspect_membership).[/cyan]")
        if self.config.inference.drop_nonorg_membership:
            before = len(relationships)
            relationships = [r for r in relationships
                             if not (r.rel_type in mem and r.target not in org_ids)]
            if before != len(relationships):
                console.print(f"[cyan]Dropped {before - len(relationships)} non-org membership edges.[/cyan]")

        # 2a3. Type-signature consistency (ASP-style, Tran et al. 2025): a
        # relation whose endpoint types contradict its signature ("led" into a
        # place, "born_in" into an org) is a likely misextraction. Tag
        # type_violation so it stays filterable; drop only if configured. Loose
        # relations carry no signature and are never flagged.
        if self.config.ontology.enabled:
            from postprocess.ontology import check_relation_types
            type_of = {e.entity_id: e.label for e in entities}
            relationships, n_typeviol = check_relation_types(
                relationships, type_of,
                drop=self.config.ontology.drop_type_violations)
            if n_typeviol:
                verb = "Dropped" if self.config.ontology.drop_type_violations else "Tagged"
                console.print(f"[cyan]{verb} {n_typeviol} type-signature violations.[/cyan]")

        # 2b. LLM-assisted dedup: merge same-entity nodes the rules missed.
        if self.config.dedup.llm_assist and llm_capable:
            from postprocess.llm_dedup import apply_llm_merges
            entities, relationships = apply_llm_merges(entities, relationships, self.backend)

        # 3. Quality review (rules always; LLM in api/ollama when enabled).
        reviewer = QualityReviewer(self.config.quality, stopwords=self.domain.entity_stopwords())
        review_backend = self.backend if llm_capable else None
        entities, relationships = reviewer.review(
            entities, relationships, self.config.mode, backend=review_backend
        )

        # 3b. Enrichment: subtype + attributes - after quality so we only spend
        # LLM calls on entities that survived.
        if self.config.enrichment.enabled and llm_capable:
            from postprocess.enricher import Enricher
            entities = Enricher(self.config.enrichment.batch_size,
                                subtypes=self.domain.entity_subtypes()).run(entities, self.backend)

        # 3c. Optional Wikidata linking (off by default; network, fail-soft).
        # When on, a shared QID is a high-precision cross-doc identity key: fold
        # same-QID nodes string dedup kept apart, before inference reads the ids.
        if self.config.linking.enabled:
            from postprocess.wikidata import consolidate_by_qid, link_entities
            entities = link_entities(entities, self.config.linking)
            if self.config.linking.consolidate_by_qid:
                entities, relationships, _name_to_id = consolidate_by_qid(
                    entities, relationships, _name_to_id)

        # 4. Inference (co-occurrence + proximity + canonical edges). Pass the
        # raw mentions + dedup name map so within-document window co-occurrence
        # can resolve surfaces to surviving entity ids.
        inference = InferenceEngine(self.config.inference, domain=self.domain)
        relationships = inference.run(entities, relationships,
                                      mentions=agg.mentions, name_to_id=_name_to_id)

        # 4b. Optionally drop degree-0 nodes (cleaner SNA graphs).
        if self.config.quality.drop_isolated_nodes:
            linked = {r.source for r in relationships} | {r.target for r in relationships}
            before = len(entities)
            entities = [e for e in entities if e.entity_id in linked]
            if before != len(entities):
                console.print(f"[cyan]Dropped {before - len(entities)} isolated nodes.[/cyan]")

        # 5a. Stamp author nodes with their letter_id (home doc where the narrator
        # was detected), even when the author is also mentioned in other letters.
        for e in entities:
            if e.attributes.get("is_author"):
                home = e.attributes.get("author_doc") or (e.doc_ids[0] if len(e.doc_ids) == 1 else None)
                info = manifest.get(home) if home else None
                if info and info["letter_id"]:
                    e.attributes["letter_id"] = info["letter_id"]

        # 5b. Merge spreadsheet metadata onto author nodes + materialize verified
        # edges from it (born_in / resided_in / member_of with membership data).
        if self.config.io.metadata_file:
            meta = self.domain.load_metadata(self.config.io.metadata_file)
            entities, relationships = self._apply_metadata(entities, relationships, meta)

        # 5c. Tagging (+ flag public/historical reference figures). Runs AFTER the
        # metadata merge so metadata-derived edges are also tagged and counted in
        # degree.
        tagger = Tagger()
        entities, relationships = tagger.tag(
            entities, relationships, reference_figures=self.domain.reference_figures()
        )

        # 5e. Directedness follows tie semantics: reciprocal ties (met_with,
        # family_of, co_occurs_with) undirected, the rest directed.
        from postprocess import tie_classes
        for r in relationships:
            r.directed = not tie_classes.is_symmetric(r.rel_type)

        # 6. Build graph + metrics.
        builder = GephiBuilder()
        tables = builder.build(entities, relationships, agg.timeline, manifest=manifest,
                               period_fn=self.domain.temporal_period)

        # 6b. Optional NetworkX SNA metrics Gephi can't compute (brokerage,
        # bridges, articulation) + graph-health QA. Fail-soft; opt-in.
        if self.config.export.graph_metrics:
            from postprocess import graph_metrics
            report = graph_metrics.enrich(tables)
            if report:
                report["quality_pillars"] = graph_metrics.quality_pillars(report, tables)
                import json as _json
                (self.run_dir / "graph_report.json").write_text(
                    _json.dumps(report, indent=2), encoding="utf-8")
                qa = report.get("qa_substantive", {})
                console.print(
                    f"[cyan]Graph QA (substantive): {qa.get('nodes',0)} nodes, "
                    f"{qa.get('components',0)} components, giant {qa.get('largest_cc_pct',0)}%, "
                    f"{report.get('bridges',0)} bridges, "
                    f"{report.get('articulation_points',0)} articulation points.[/cyan]")

        # 7. Export.
        exporter = Exporter(
            self.run_dir, self.config.export.formats, gephi=self.config.export.gephi
        )
        written = exporter.export(tables, entities, extractions, manifest=manifest)

        # 7a2. Narrative-sequence network (Bearman & Stovel 2000): element->element
        # transitions across the corpus timeline. Opt-in; fail-soft.
        if getattr(self.config.export, "narrative_network", False) and agg.timeline:
            try:
                from postprocess.narrative import _ELEMENT_RULES, write_narrative
                rules = self.domain.narrative_rules() or _ELEMENT_RULES
                written.update(write_narrative(self.run_dir, agg.timeline, rules=rules))
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Narrative network skipped: {exc}[/yellow]")

        # 7b. Codebook: variable definitions + this run's value inventories.
        if self.config.export.codebook:
            from postprocess.codebook import write_codebook
            mode_cfg = getattr(self.config.intelligence, self.config.mode, None)
            cb = write_codebook(self.run_dir, tables, self.config, domain=self.domain,
                                model=getattr(mode_cfg, "model", ""))
            if cb:
                written["codebook"] = str(cb)

        self._print_summary(entities, tables, written)
        return written

    def _apply_metadata(self, entities, relationships, meta):
        """Merge metadata onto author nodes and add verified edges from it."""
        from core.schema import Entity, Relationship, stable_id
        index = {normalize_name(e.canonical_name): e for e in entities}
        for e in entities:
            for a in e.aliases:
                index.setdefault(normalize_name(a), e)

        def get_or_make(name, label):
            key = normalize_name(name)
            if key in index:
                return index[key]
            ent = Entity(entity_id=f"meta_{stable_id(key, label)}", canonical_name=name,
                         label=label, mention_count=1, confidence=0.97,
                         attributes={"source": "metadata"},
                         tags={"entity_scope": "specific", "relevance_tier": "secondary"})
            entities.append(ent)
            index[key] = ent
            return ent

        n_nodes_before, n_edges = len(entities), 0
        merged = 0
        for e in list(entities):
            lid = e.attributes.get("letter_id")
            if not (lid and lid in meta):
                continue
            row = meta[lid]
            for k, v in row.items():
                e.attributes[k] = v
            mn = row.get("meta_name")
            if mn and normalize_name(mn) != normalize_name(e.canonical_name):
                if e.canonical_name not in e.aliases:
                    e.aliases.append(e.canonical_name)
                e.canonical_name = mn
            merged += 1
            for spec in self.domain.metadata_edges(row):
                tgt = get_or_make(spec["target"], spec.get("type", "ORG"))
                if tgt.entity_id == e.entity_id:
                    continue
                relationships.append(Relationship(
                    source=e.entity_id, target=tgt.entity_id, rel_type=spec["rel"],
                    doc_id=e.attributes.get("author_doc", ""), confidence=0.97, directed=True,
                    origin="canonical",
                    attributes={"edge_source": "metadata", **spec.get("attrs", {})}))
                n_edges += 1
        console.print(f"[cyan]Metadata: merged onto {merged} authors, "
                      f"added {len(entities) - n_nodes_before} nodes + {n_edges} verified edges.[/cyan]")
        return entities, relationships

    # Reporting
    def _print_summary(self, entities, tables, written) -> None:
        table = Table(title="Pipeline Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Mode", self.config.mode)
        table.add_row("Entities", str(len(entities)))
        table.add_row("Edges", str(len(tables.edges)))
        table.add_row("Timeline events", str(len(tables.timeline)))
        core = sum(1 for e in entities if e.tags.get("relevance_tier") == "core")
        table.add_row("Core entities", str(core))
        console.print(table)

        files = Table(title="Artifacts", header_style="bold cyan")
        files.add_column("Name")
        files.add_column("Path")
        for name, path in written.items():
            files.add_row(name, path)
        console.print(files)

    def _write_run_meta(self, stage: str, resume: bool, limit: int) -> None:
        """Snapshot the effective config into the run dir so every output is
        traceable to the exact model/mode/settings that produced it."""
        import json
        from datetime import datetime, timezone

        mode_cfg = getattr(self.config.intelligence, self.config.mode, None)
        meta = {
            "run_name": self.config.run_name,
            "mode": self.config.mode,
            "model": getattr(mode_cfg, "model", ""),
            "stage": stage,
            "resume": resume,
            "limit": limit,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": sys.platform,
            "config": self.config.model_dump(mode="json"),
        }
        (self.run_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Full run
    def run(self, stage: str, resume: bool, limit: int = 0,
            extra_urls: tuple[str, ...] = (), urls_file: str = "", text: str = "",
            crawl_seeds: tuple[str, ...] = ()) -> None:
        self._write_run_meta(stage, resume, limit)
        ckpt = CheckpointManager(
            self.run_dir / "checkpoints",
            self.config.run_name,
            enabled=self.config.checkpoint.enabled,
        )

        if stage in ("all", "ingest", "extract"):
            extractions = self.run_extract(resume=resume, limit=limit,
                                           extra_urls=extra_urls, urls_file=urls_file,
                                           text=text, crawl_seeds=crawl_seeds)
            if stage == "ingest":
                console.print("[green]Ingest/extract stage finished.[/green]")
                return
        else:
            # Analyze-only: score exactly the current input set (respecting
            # --limit), filtering the checkpoint rather than dumping all of it.
            documents = self._gather(limit, extra_urls, urls_file, text,
                                     crawl_seeds, for_extract=False)
            current_ids = {d.doc_id for d in documents}
            all_ckpt = ckpt.load_all()
            extractions = [ex for ex in all_ckpt if ex.doc_id in current_ids]
            console.print(
                f"[cyan]Checkpoint has {len(all_ckpt)} docs; analyzing "
                f"{len(extractions)} that match the current input set "
                f"(limit={limit or 'all'}).[/cyan]"
            )

        if stage in ("all", "analyze", "extract"):
            self.run_analyze(extractions)


# CLI
@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True),
              help="Path to the YAML config file.")
@click.option("--stage", type=click.Choice(["all", "ingest", "extract", "analyze"]),
              default="all", show_default=True, help="Which stage(s) to run.")
@click.option("--resume", is_flag=True, default=False,
              help="Resume from existing checkpoint, skipping completed documents.")
@click.option("--mode", type=click.Choice(["api", "python_only", "ollama", "langextract"]), default=None,
              help="Override the execution mode from the config.")
@click.option("--limit", type=int, default=0,
              help="Process only the first N documents (handy for quick test runs).")
@click.option("--url", "urls", multiple=True,
              help="Fetch and analyze a web page / PDF URL (repeatable).")
@click.option("--urls-file", default="",
              help="Path to a newline-delimited list of URLs to fetch.")
@click.option("--crawl", "crawl_seeds", multiple=True,
              help="Crawl a site from this seed URL and analyze its subpages "
                   "(repeatable). Enables crawling; tune depth/pages/scope in the "
                   "config's io.crawl block.")
@click.option("--crawl-max-pages", "crawl_max_pages", type=int, default=None,
              help="Override io.crawl.max_pages (page cap for --crawl).")
@click.option("--crawl-max-depth", "crawl_max_depth", type=int, default=None,
              help="Override io.crawl.max_depth (link hops for --crawl).")
@click.option("--text", "direct_text", default="",
              help="Analyze a raw text string directly (e.g. pasted input).")
@click.option("--min-entity-confidence", "min_entity_confidence", type=float, default=None,
              help="Override quality.min_entity_confidence (e.g. 0.5). Lets you A/B "
                   "the precision/recall trade at analyze time with no re-extraction.")
@click.option("--ollama-model", default="", help="Override intelligence.ollama.model.")
@click.option("--metadata", "metadata_file", default="", help="Override io.metadata_file (xlsx).")
@click.option("--run-name", "run_name", default="", help="Override run_name (output subdir), "
              "e.g. to A/B models without overwriting each other.")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose (DEBUG) logging.")
def cli(config_path: str, stage: str, resume: bool, mode: Optional[str],
        limit: int, urls: tuple[str, ...], urls_file: str,
        crawl_seeds: tuple[str, ...], crawl_max_pages: Optional[int],
        crawl_max_depth: Optional[int], direct_text: str,
        min_entity_confidence: Optional[float], ollama_model: str, metadata_file: str,
        run_name: str, verbose: bool) -> None:
    """Run the NER + SNA extraction pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # Quiet noisy third-party loggers unless verbose.
    if not verbose:
        for noisy in ("httpx", "urllib3", "sentence_transformers", "transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        config = load_config(config_path, overrides={"mode": mode} if mode else None)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Config error:[/red] {exc}")
        sys.exit(2)

    if min_entity_confidence is not None:
        config.quality.min_entity_confidence = min_entity_confidence
        console.print(f"[cyan]Override: quality.min_entity_confidence = {min_entity_confidence}[/cyan]")
    if ollama_model:
        config.intelligence.ollama.model = ollama_model
    if metadata_file:
        config.io.metadata_file = metadata_file
    if run_name:
        config.run_name = run_name
        console.print(f"[cyan]Override: run_name = {run_name}[/cyan]")
    if crawl_seeds:
        config.io.crawl.enabled = True
    if crawl_max_pages is not None:
        config.io.crawl.max_pages = crawl_max_pages
    if crawl_max_depth is not None:
        config.io.crawl.max_depth = crawl_max_depth

    console.rule(f"[bold]NER + SNA Pipeline - run '{config.run_name}' (mode={config.mode})")
    pipeline = Pipeline(config)
    try:
        pipeline.run(stage=stage, resume=resume, limit=limit,
                     extra_urls=urls, urls_file=urls_file, text=direct_text,
                     crawl_seeds=crawl_seeds)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Progress is saved in the checkpoint; "
                      "re-run with --resume to continue.[/yellow]")
        sys.exit(130)
    console.rule("[bold green]Done")


if __name__ == "__main__":
    cli()
