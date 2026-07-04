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
import threading
from pathlib import Path
from typing import Callable, Optional

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


def _rpm_delay(last_start: float, now: float, rpm: int) -> float:
    """Seconds to sleep before the next --submit POST to hold request starts at
    <= ``rpm`` per minute. 0 when rpm<=0 (unthrottled) or the interval has elapsed."""
    if rpm <= 0:
        return 0.0
    return max(0.0, 60.0 / rpm - (now - last_start))


# Backend factory
def gemini_live_config(config: Config) -> Config:
    """A config copy whose `api` block points at Gemini's OpenAI-compatible endpoint,
    so an ApiBackend can run the post-extraction LLM steps (dedup/review/enrich) in
    gemini_batch mode with the same --submit key. Pure (no I/O) so it's testable."""
    ic = config.intelligence
    cfg = config.model_copy(deep=True)
    api = cfg.intelligence.api
    api.provider = "openai"
    api.base_url = ic.batch_base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
    api.model = ic.batch_model
    api.api_key_env = ic.batch_api_key_env
    api.json_mode = True
    api.max_tokens = max(api.max_tokens, 8192)  # headroom for flash thinking + the JSON
    return cfg


def build_backend(config: Config, foundation: Callable[[], FoundationLayer],
                  domain=None) -> IntelligenceBackend:
    """Instantiate the intelligence backend for the configured mode.

    ``foundation`` is a thunk: only the branch that reads it pays the model
    load. Keeps the which-mode-needs-foundation knowledge here, not in callers.
    """
    mode = config.mode
    if mode == "api":
        from intelligence.api_backend import ApiBackend
        return ApiBackend(config, domain=domain)
    if mode == "gemini_batch":
        # Extraction came from the batch reply; this backend is only for the post-
        # extraction LLM steps (dedup/review/enrich) via Gemini's OpenAI endpoint.
        from intelligence.api_backend import ApiBackend
        return ApiBackend(gemini_live_config(config), domain=domain)
    if mode == "ollama":
        from intelligence.ollama_backend import OllamaBackend
        return OllamaBackend(config, domain=domain)
    if mode == "python_only":
        from intelligence.python_backend import PythonBackend
        # Reuse the foundation's loaded spaCy engine to avoid a second load.
        return PythonBackend(config, spacy_engine=foundation().spacy, domain=domain)
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
        # Serializes foundation construction + NER when parallel_docs > 1 (spaCy
        # pipelines are not thread-safe). Uncontended in the sequential path.
        self._foundation_lock = threading.Lock()

    # Lazy heavy components
    @property
    def foundation(self) -> FoundationLayer:
        if self._foundation is None:
            console.print("[cyan]Loading foundation (spaCy now; GLiNER on first NER use)...[/cyan]")
            self._foundation = FoundationLayer(self.config, domain=self.domain)
        return self._foundation

    @property
    def backend(self) -> IntelligenceBackend:
        if self._backend is None:
            console.print(f"[cyan]Initializing intelligence backend: {self.config.mode}[/cyan]")
            # Thunk, not the property: building an LLM backend must not force the
            # spaCy+GLiNER load (GLiNER can hang/segfault on load, and the LLM
            # analyze path never reads it - reconcile_ner loads it explicitly).
            self._backend = build_backend(self.config, lambda: self.foundation,
                                          domain=self.domain)
        return self._backend

    # Stage: extract
    def run_extract(self, resume: bool, limit: int = 0,
                    extra_urls: tuple[str, ...] = (), urls_file: str = "",
                    text: str = "", crawl_seeds: tuple[str, ...] = ()) -> list[DocumentExtraction]:
        """Run foundation + intelligence over all documents with checkpointing."""
        from tqdm import tqdm

        documents = self._gather(limit, extra_urls, urls_file, text, crawl_seeds, resume=resume)
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
        parallel = max(1, self.config.intelligence.parallel_docs)
        if parallel > 1 and self.config.mode not in ("ollama", "api"):
            console.print(f"[yellow]parallel_docs={parallel} ignored: mode "
                          f"'{self.config.mode}' extracts sequentially.[/yellow]")
            parallel = 1
        processed = 0
        with ckpt:
            if parallel > 1:
                todo = [d for d in documents
                        if not (resume and d.doc_id in done_ids)]
                processed = self._extract_parallel(todo, ckpt, parallel)
            else:
                for doc in tqdm(documents, desc="Extracting", unit="doc"):
                    if resume and doc.doc_id in done_ids:
                        continue
                    extraction = self._extract_one(doc, self.backend)
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

    def _extract_one(self, doc: Document, backend: IntelligenceBackend) -> DocumentExtraction:
        """Foundation + LLM + structure fold-ins for one document. Shared by the
        sequential loop and the parallel workers; the foundation lock also covers
        the lazy FoundationLayer construction on first use."""
        with self._foundation_lock:
            foundation_results = self.foundation.process_document(doc)
        extraction = backend.extract_document(
            doc.doc_id, doc.source_path, foundation_results
        )
        # Scripts: fold in scene co-presence edges (opt-in; a no-op on prose).
        if self.config.intelligence.parse_scripts:
            from core.script_parser import script_copresence
            extraction.relationships.extend(
                script_copresence(doc.text, doc.doc_id))
        # Social posts: fold in the platform-stated reply/mention/posted_in
        # structure (asserted social_graph edges + USER/COMMUNITY nodes).
        if (doc.meta or {}).get("source_type") == "social":
            from core.social import social_structure
            sm, se = social_structure(doc)
            extraction.mentions.extend(sm)
            extraction.relationships.extend(se)
        # LittleSis: fold in the curated relationship graph (asserted typed edges,
        # donations carry qual_monetary_value) + PERSON/ORG nodes.
        if (doc.meta or {}).get("source_type") == "littlesis":
            from core.littlesis import littlesis_structure
            lm, le = littlesis_structure(doc)
            extraction.mentions.extend(lm)
            extraction.relationships.extend(le)
        return extraction

    def _extract_parallel(self, todo: list[Document], ckpt: CheckpointManager,
                          parallel: int) -> int:
        """Worker-pool extraction over documents. Each worker leases its own
        backend from a queue (extract_document keeps per-call state like
        _chunk_failed on the instance, so backends must not be shared), the
        foundation and checkpoint writes sit behind locks, and only the LLM
        calls actually overlap. BackendUnavailable from any worker aborts the
        run; docs in flight at that moment are simply retried on --resume."""
        import queue
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from tqdm import tqdm

        if not todo:
            return 0
        n = min(parallel, len(todo))
        console.print(f"[cyan]Parallel extraction: {n} workers.[/cyan]")
        backends: queue.Queue = queue.Queue()
        for _ in range(n):
            backends.put(build_backend(self.config, lambda: self.foundation,
                                       domain=self.domain))
        save_lock = threading.Lock()
        processed = 0

        def work(doc: Document) -> DocumentExtraction:
            backend = backends.get()
            try:
                return self._extract_one(doc, backend)
            finally:
                backends.put(backend)

        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = [ex.submit(work, d) for d in todo]
            try:
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc="Extracting", unit="doc"):
                    extraction = fut.result()  # re-raises BackendUnavailable
                    with save_lock:
                        ckpt.save(extraction, flush=True)
                    processed += 1
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
        return processed

    # Mode 4: manual batch (gemini_batch). Export one self-contained prompt holding
    # whole documents, paste it into a long-context model, import the JSON reply.
    def run_batch_export(self, limit: int, extra_urls, urls_file, text, crawl_seeds,
                         submit: bool = False, resume: bool = False) -> bool:
        """Write the batch prompt file(s). With submit=True, POST each to the Gemini
        API and write the reply files too (no manual paste). With resume=True the
        submit step skips batches whose reply is already on disk and complete, so an
        interrupted/rate-limited run continues without re-paying for done batches.
        Returns True if any reply was written (so run() can continue to analyze)."""
        from intelligence.manual_batch import build_batch_prompt, extraction_spec
        documents = self._gather(limit, extra_urls, urls_file, text, crawl_seeds)
        if not documents:
            console.print("[yellow]No documents found.[/yellow]")
            return False
        spec = extraction_spec(self.config, self.domain)
        # First-person corpora (Abel): stamp each doc's author into its tag so the
        # model attributes 'I/we' to that narrator. Filename-based, no models.
        authors = self._batch_authors(documents)
        prompts = build_batch_prompt(
            [(d.doc_id, d.text) for d in documents], spec["label_types"],
            spec["relation_types"], spec["relation_guide"],
            spec["edge_qualifiers"], spec["type_signatures"],
            char_budget=self.config.intelligence.batch_char_budget,
            max_docs=self.config.intelligence.batch_max_docs,
            authors=authors,
        )
        multi = len(prompts) > 1

        def stem(i: int, kind: str, ext: str) -> str:
            return f"gemini_batch_{kind}.{i:03d}.{ext}" if multi \
                else f"gemini_batch_{kind}.{ext}"

        paths: list[Path] = []
        for i, pr in enumerate(prompts, 1):
            p = self.run_dir / stem(i, "prompt", "txt")
            p.write_text(pr, encoding="utf-8")
            paths.append(p)
        approx = sum(len(p) for p in prompts) // 4
        per = (len(documents) + len(paths) - 1) // max(1, len(paths))
        console.print(
            f"[green]Wrote {len(paths)} batch prompt file(s)[/green] "
            f"({len(documents)} docs, ~{per} docs/file, ~{approx:,} input tokens) to {self.run_dir}"
        )

        if submit:
            return self._submit_batches(prompts, stem, resume=resume)

        if per > 40:
            console.print(f"[yellow]~{per} docs/file may truncate the reply. Re-run with "
                          "--batch-docs 25 (or lower), or add --submit to call the API.[/yellow]")
        resp = "gemini_batch_response.json" if not multi \
            else "gemini_batch_response.NNN.json (one per prompt)"
        console.print(
            "[cyan]Next:[/cyan] in the model set max output tokens to the maximum, "
            "paste/upload each prompt, save the JSON reply to "
            f"[bold]{self.run_dir / resp}[/bold], then run the same command "
            "with [bold]--stage analyze[/bold]. (Or re-run with [bold]--submit[/bold] to "
            "skip the paste and call the API directly.)"
        )
        return False

    def _submit_batches(self, prompts, stem, resume: bool = False) -> bool:
        """POST each batch prompt to the Gemini API, writing the reply files. With
        resume, a batch whose reply file already exists and parses as complete JSON
        is skipped (the reply file IS the checkpoint), so an interrupted run picks up
        where it stopped. A truncated/partial reply does not parse -> it is re-POSTed.
        Resume assumes the same --batch-docs/--batch-budget (batch boundaries must
        line up with the saved files)."""
        import os
        import time

        from intelligence.manual_batch import submit_to_gemini
        ic = self.config.intelligence
        key = os.environ.get(ic.batch_api_key_env, "")
        if not key:
            console.print(f"[red]--submit needs an API key in ${ic.batch_api_key_env}. "
                          "Get a free one at aistudio.google.com/apikey and set it:\n"
                          f"  $env:{ic.batch_api_key_env} = \"...\"[/red]")
            sys.exit(2)
        from tqdm import tqdm
        ok = skipped = 0
        last_post = 0.0
        for i, pr in enumerate(tqdm(prompts, desc="Gemini", unit="batch"), 1):
            out_path = self.run_dir / stem(i, "response", "json")
            if resume and self._reply_complete(out_path):
                skipped += 1
                ok += 1
                continue
            wait = _rpm_delay(last_post, time.monotonic(), ic.batch_rpm)
            if wait > 0:
                time.sleep(wait)
            last_post = time.monotonic()
            try:
                reply = submit_to_gemini(
                    pr, key, model=ic.batch_model, base_url=ic.batch_base_url,
                    max_output_tokens=ic.batch_max_output_tokens,
                    thinking_budget=ic.batch_thinking_budget,
                    timeout=ic.batch_request_timeout)
            except Exception as exc:  # noqa: BLE001 - one batch failing must not kill the rest
                console.print(f"[red]Batch {i} failed: {exc}[/red]")
                continue
            out_path.write_text(reply, encoding="utf-8")
            ok += 1
        done = f"[green]Gemini: {ok}/{len(prompts)} batches done"
        if skipped:
            done += f" ({skipped} already on disk, skipped)"
        console.print(done + f" (model={ic.batch_model}).[/green]")
        if ok < len(prompts):
            console.print("[yellow]Some batches failed - re-run with --resume to retry "
                          "only the missing ones; analyze flags any uncovered docs.[/yellow]")
        return ok > 0

    @staticmethod
    def _reply_complete(path: Path) -> bool:
        """True if a saved reply exists and holds a complete JSON body. Strips a prose
        preamble first (_outermost_span) then STRICT-parses: gemma ignores the JSON
        mime type and prepends a prose paraphrase, so a bare strict parse rejected its
        complete replies and made --resume re-POST every finished gemma batch into the
        flaky endpoint. Not repair_json - that balances unclosed braces and would pass
        a TRUNCATED reply, which must be re-POSTed (smaller --batch-docs), not kept."""
        import json

        from intelligence.json_repair import _outermost_span
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            obj = json.loads(_outermost_span(path.read_text(encoding="utf-8")))
        except (ValueError, OSError, RecursionError):
            return False
        return bool(obj)

    def run_batch_import(self, ckpt: CheckpointManager,
                         import_json: tuple[str, ...] | str,
                         limit: int, extra_urls, urls_file, text, crawl_seeds) -> None:
        from intelligence.manual_batch import extraction_spec, parse_batch_response
        if isinstance(import_json, str):
            import_json = (import_json,) if import_json else ()
        if import_json:
            # Each value is a file, a directory (glob the standard reply name
            # inside it), or a glob pattern. Directory form avoids the Git Bash
            # glob-expansion trap (shell splits *.json into N positional args).
            files: list[Path] = []
            for pat in import_json:
                p = Path(pat)
                if p.is_dir():
                    files.extend(sorted(p.glob("gemini_batch_response*.json")))
                else:
                    expanded = sorted(Path().glob(pat))
                    files.extend(expanded or [p])
            files = sorted(set(files))
        else:
            files = sorted(self.run_dir.glob("gemini_batch_response*.json"))
        files = [f for f in files if f.exists()]
        if not files:
            console.print(f"[red]No batch reply JSON found in {self.run_dir} "
                          "(expected gemini_batch_response*.json). Run --stage extract "
                          "first, paste the prompt, and save the reply.[/red]")
            sys.exit(2)
        documents = self._gather(limit, extra_urls, urls_file, text, crawl_seeds,
                                 for_extract=True)
        spec = extraction_spec(self.config, self.domain)
        authors = self._batch_authors(documents)
        doc_meta = {
            d.doc_id: {"text": d.text, "source_path": d.source_path,
                       "author": authors.get(d.doc_id, "")}
            for d in documents
        }
        extractions = []
        for f in files:
            raw = f.read_text(encoding="utf-8", errors="replace")
            extractions.extend(parse_batch_response(
                raw, doc_meta, spec["label_types"], spec["edge_qualifiers"],
                spec["date_vocab"]))
        # A single-doc reply the model returned unkeyed lands under doc_id "".
        if len(documents) == 1 and len(extractions) == 1 and not extractions[0].doc_id:
            extractions[0].doc_id = documents[0].doc_id
            extractions[0].source_path = documents[0].source_path
        with ckpt:
            for ex in extractions:
                ckpt.save(ex, flush=True)
        # Coverage check across ALL reply files (a split corpus is normal).
        covered = {ex.doc_id for ex in extractions}
        missing = [d.doc_id for d in documents if d.doc_id not in covered]
        console.print(f"[green]Imported {len(extractions)} documents from "
                      f"{len(files)} reply file(s) into the checkpoint.[/green]")
        if missing:
            console.print(f"[yellow]{len(missing)} of {len(documents)} documents not "
                          f"covered by any reply (truncated output?). Re-export with a "
                          f"smaller --batch-budget and redo those. First few: "
                          f"{missing[:5]}[/yellow]")

    def _reconcile_ner(self, extractions, doc_texts: dict[str, str]) -> None:
        """gemini_batch: re-run local NER and fold spans onto the LLM entities.

        Loads the foundation (GLiNER + spaCy) - which gemini_batch otherwise never
        touches - so this pays a model-load + per-doc NER cost for the proximity
        floor + recall net. Coref-sourced mentions are dropped (the whole-doc model
        already handled the narrator)."""
        from core.schema import Document
        from postprocess.span_reconcile import reconcile_spans

        console.print("[cyan]Reconciling whole-doc entities against local GLiNER/spaCy "
                      "NER (loads foundation models)...[/cyan]")

        def ner_fn(doc_id: str, text: str):
            results = self.foundation.process_document(
                Document(doc_id=doc_id, source_path="", text=text))
            out = []
            for r in results:
                for m in r.mentions:
                    if any(str(s).startswith("coref") for s in (m.sources or [])):
                        continue
                    out.append(m)
            return out

        stats = reconcile_spans(
            extractions, doc_texts, ner_fn,
            add_missed=self.config.intelligence.reconcile_add_missed)
        console.print(
            f"[green]Reconciliation: {stats['transferred']} spans transferred, "
            f"{stats['added']} NER-only mentions added across {stats['docs']} docs.[/green]")

    def _batch_authors(self, documents) -> dict[str, str]:
        """doc_id -> narrator name for first-person corpora, when narrator
        resolution is on (the live pipeline's author_name equivalent, but
        filename-based so the batch export needs no models)."""
        if not self.config.coreference.narrator_resolution:
            return {}
        out: dict[str, str] = {}
        for d in documents:
            name = self.domain.narrator_name(Path(d.source_path).name, d.doc_id)
            if name:
                out[d.doc_id] = name
        return out

    # Document gathering (shared by extract + analyze)
    def _gather(self, limit: int = 0, extra_urls: tuple[str, ...] = (),
                urls_file: str = "", text: str = "", crawl_seeds: tuple[str, ...] = (),
                for_extract: bool = True, resume: bool = False):
        """Collect the current run's documents (files + URLs + crawl + text),
        honoring --limit. Dedups by doc_id so the same URL from two sources is one
        node."""
        io = self.config.io
        # Ingestion checkpoint: if a corpus snapshot is set, load it and skip all
        # live sources (crawl/fetch/file walk). The portable, mode-independent path -
        # scrape once, ship the jsonl, extract anywhere. doc_ids are preserved.
        if io.documents_file:
            from core.preprocessor import read_documents_snapshot
            docs = read_documents_snapshot(io.documents_file)
            console.print(f"[cyan]Loaded {len(docs)} documents from snapshot "
                          f"{io.documents_file} (no crawl/fetch).[/cyan]")
            if limit and limit > 0:
                docs = docs[:limit]
                console.print(f"[yellow]--limit active: {len(docs)} documents.[/yellow]")
            return docs
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
        documents.extend(self._gather_crawl(crawl_seeds, for_extract, resume=resume))
        documents.extend(self._gather_social(for_extract))
        documents.extend(self._gather_wiki(for_extract))
        documents.extend(self._gather_littlesis(for_extract))

        # Dedup by doc_id (a crawled url may also be in io.urls / a file mirror).
        seen: set[str] = set()
        documents = [d for d in documents
                     if not (d.doc_id in seen or seen.add(d.doc_id))]

        if limit and limit > 0:
            documents = documents[:limit]
            console.print(f"[yellow]--limit active: {len(documents)} documents.[/yellow]")

        # Freeze the ingestion checkpoint: a portable snapshot of the gathered corpus
        # (crawl + fetch + preprocess result). Only on a real gather with text - never
        # overwrite it with the text-less stubs the analyze path rebuilds. Ship this
        # file or feed it back with --ingest-from to re-extract in another mode.
        if for_extract and any(d.text for d in documents):
            try:
                from core.preprocessor import write_documents_snapshot
                snap = self.run_dir / "documents.jsonl"
                # Guard: never let a small/partial gather clobber a much larger existing
                # snapshot (an interrupted re-crawl over a finished scrape). Write a .new
                # sidecar and KEEP the old file; the user reconciles deliberately. A scrape
                # is expensive and unrecoverable, so the snapshot is treated as precious.
                if snap.exists():
                    existing = sum(1 for _ in snap.open("r", encoding="utf-8"))
                    if existing > max(20, 2 * len(documents)):
                        alt = self.run_dir / "documents.jsonl.new"
                        write_documents_snapshot(documents, alt)
                        console.print(
                            f"[red]Refusing to overwrite {snap} ({existing} docs) with only "
                            f"{len(documents)} - wrote {alt} instead. The existing scrape is "
                            f"kept. Delete/rename it yourself if the smaller gather is intended."
                            f"[/red]")
                        return documents
                write_documents_snapshot(documents, snap)
            except Exception:  # noqa: BLE001 - snapshot is a convenience, not load-bearing
                pass
        return documents

    def _gather_social(self, for_extract: bool) -> list[Document]:
        """Fetch social-media sources (io.social: 'platform:target' specs) into posts as
        Documents. The explicit reply/mention/posted_in structure is added per-post in
        run_extract. On extract, fetch + cache to social_docs.jsonl; on analyze, reuse
        the cache so re-analysis never re-hits the APIs."""
        specs = list(self.config.io.social)
        if not specs:
            return []
        from core.preprocessor import read_documents_snapshot, write_documents_snapshot
        cache = self.run_dir / "social_docs.jsonl"
        if not for_extract:
            if cache.exists():
                docs = read_documents_snapshot(cache)
                console.print(f"[cyan]Social: reusing {len(docs)} cached posts "
                              "(no re-fetch).[/cyan]")
                return docs
            return []
        from core.social import fetch_social, posts_to_documents
        io = self.config.io
        docs: list[Document] = []
        for spec in specs:
            try:
                posts = fetch_social(spec, limit=io.social_limit, depth=io.social_depth)
                docs.extend(posts_to_documents(posts))
                console.print(f"[green]Social {spec}: {len(posts)} post(s).[/green]")
            except Exception as exc:  # noqa: BLE001 - one source failing isn't fatal
                console.print(f"[red]Social {spec} failed: {exc}[/red]")
        try:
            write_documents_snapshot(docs, cache)
        except Exception:  # noqa: BLE001 - cache is a nicety
            pass
        return docs

    def _gather_wiki(self, for_extract: bool) -> list[Document]:
        """Fetch MediaWiki sources (io.wiki: 'host:Target' specs) as clean article
        Documents via the API. On extract, fetch + cache to wiki_docs.jsonl; on
        analyze, reuse the cache so re-analysis never re-hits the API."""
        specs = list(self.config.io.wiki)
        if not specs:
            return []
        from core.preprocessor import read_documents_snapshot, write_documents_snapshot
        cache = self.run_dir / "wiki_docs.jsonl"
        if not for_extract:
            if cache.exists():
                docs = read_documents_snapshot(cache)
                console.print(f"[cyan]Wiki: reusing {len(docs)} cached pages "
                              "(no re-fetch).[/cyan]")
                return docs
            return []
        from core.wiki import fetch_wiki
        limit = self.config.io.wiki_limit
        docs: list[Document] = []
        for spec in specs:
            try:
                wd = fetch_wiki(spec, limit=limit)
                docs.extend(wd)
                console.print(f"[green]Wiki {spec}: {len(wd)} page(s).[/green]")
            except Exception as exc:  # noqa: BLE001 - one source failing isn't fatal
                console.print(f"[red]Wiki {spec} failed: {exc}[/red]")
        try:
            write_documents_snapshot(docs, cache)
        except Exception:  # noqa: BLE001 - cache is a nicety
            pass
        return docs

    def _gather_littlesis(self, for_extract: bool) -> list[Document]:
        """Fetch LittleSis sources (io.littlesis: 'search:term' / 'id:N' specs) as entity
        Documents carrying their curated relationships (imported as asserted edges by the
        structure hook in run_extract). Cached to littlesis_docs.jsonl; reused on analyze."""
        specs = list(self.config.io.littlesis)
        if not specs:
            return []
        from core.preprocessor import read_documents_snapshot, write_documents_snapshot
        cache = self.run_dir / "littlesis_docs.jsonl"
        if not for_extract:
            if cache.exists():
                docs = read_documents_snapshot(cache)
                console.print(f"[cyan]LittleSis: reusing {len(docs)} cached entities "
                              "(no re-fetch).[/cyan]")
                return docs
            return []
        from core.littlesis import fetch_littlesis
        limit = self.config.io.littlesis_limit
        docs: list[Document] = []
        for spec in specs:
            try:
                ld = fetch_littlesis(spec, limit=limit)
                docs.extend(ld)
                n_edges = sum(len((d.meta or {}).get("ls_edges") or []) for d in ld)
                console.print(f"[green]LittleSis {spec}: {len(ld)} entit(ies), "
                              f"{n_edges} edge(s).[/green]")
            except Exception as exc:  # noqa: BLE001 - one source failing isn't fatal
                console.print(f"[red]LittleSis {spec} failed: {exc}[/red]")
        if docs:
            console.print("[dim]LittleSis data is CC BY-SA 4.0 - attribute "
                          "'LittleSis / Public Accountability Initiative' in any published network.[/dim]")
        try:
            write_documents_snapshot(docs, cache)
        except Exception:  # noqa: BLE001 - cache is a nicety
            pass
        return docs

    def _gather_crawl(self, crawl_seeds: tuple[str, ...], for_extract: bool,
                      resume: bool = False) -> list[Document]:
        """Expand seed URLs into subpage Documents via the crawler. On extract,
        crawl and cache the discovered URL list; on analyze, rebuild lightweight
        id-only stubs from that cache so re-analysis never re-crawls. Shows a live
        progress bar and checkpoints the frontier so a long crawl can be stopped
        (Ctrl-C) and continued with --stage fetch --resume."""
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
        # Optional JS rendering: a headless-Chromium fetcher injected into the same
        # crawler (scope/robots/rate-limit/caps unchanged). One browser for the run,
        # closed in finally. Falls back to plain GET if Playwright isn't installed.
        fetcher = None
        if getattr(cc, "render_js", False):
            from core.crawler import PlaywrightFetcher
            from core.preprocessor import _USER_AGENT
            fetcher = PlaywrightFetcher(user_agent=cc.user_agent or _USER_AGENT,
                                        timeout=cc.timeout or self.config.io.request_timeout,
                                        max_bytes=cc.max_bytes)
        console.print(f"[cyan]Crawling {len(seeds)} seed(s) "
                      f"(max_pages={cc.max_pages}, depth={cc.max_depth}, "
                      f"robots={'on' if cc.respect_robots else 'OFF'}, "
                      f"js={'on' if fetcher else 'off'}, "
                      f"resume={'on' if resume else 'off'})...[/cyan]")
        crawler = None
        try:
            from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                                       TaskProgressColumn, TextColumn, TimeElapsedColumn)
            with Progress(TextColumn("[cyan]{task.description}"), BarColumn(),
                          MofNCompleteColumn(), TaskProgressColumn(),
                          TimeElapsedColumn(), console=console, transient=False) as prog:
                task = prog.add_task("starting", total=max(cc.max_pages, 1))

                def on_progress(ev: dict) -> None:
                    e = ev.get("event")
                    if e == "start" and ev.get("resumed"):
                        prog.update(task, completed=ev["docs"],
                                    description=f"resumed: {ev['docs']} done, {ev['frontier']} queued")
                    elif e == "page":
                        u = ev["url"].split("://", 1)[-1]
                        u = u[:54] + "..." if len(u) > 57 else u
                        prog.update(task, completed=ev["docs"],
                                    description=f"queue {ev['frontier']:>5} · fetched {ev['fetched']:>5} · {u}")
                    elif e in ("done", "interrupted"):
                        prog.update(task, completed=ev["docs"])

                crawler = Crawler(self._crawl_opts(), fetch=fetcher,
                                  on_progress=on_progress,
                                  checkpoint_dir=str(self.run_dir / "crawl_state"),
                                  resume=resume)
                docs = crawler.crawl(seeds)
        finally:
            if fetcher is not None:
                fetcher.close()

        if crawler is not None and crawler.interrupted:
            console.print(f"[yellow]Crawl interrupted: {len(docs)} page(s) so far. "
                          f"Continue with the same command + --resume.[/yellow]")
        elif crawler is not None and crawler.complete:
            console.print(f"[green]Crawl complete: {len(docs)} page(s) (whole scope drained).[/green]")
        else:
            console.print(f"[green]Crawl: {len(docs)} page(s) (hit max_pages={cc.max_pages}). "
                          f"Raise the cap and --resume to fetch more.[/green]")
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
            checkpoint_every=getattr(cc, "checkpoint_every", 25),
            strip_patterns=tuple(getattr(cc, "boilerplate", ()) or ()),
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

        # 1a0. Network expansion: load the schema of an existing graph and lock this
        # run to it - the relation vocabulary ("strict edge formatting") and the
        # entity kinds. Grows a curated network from new documents without it
        # drifting into new relation types or off-target entity types. No-op unless
        # enabled. Runs in analyze too, so you can re-lock an existing checkpoint.
        expand_types: Optional[set[str]] = None
        expand_relations: Optional[set[str]] = None
        if self.config.expansion.enabled:
            from postprocess.expansion import load_network_schema
            ex = self.config.expansion
            schema = load_network_schema(ex.source)
            if schema.empty:
                console.print(f"[yellow]Expansion on but no schema loaded from "
                              f"'{ex.source}' - locks are no-ops.[/yellow]")
            else:
                if ex.entity_types:
                    expand_types = {t.upper() for t in ex.entity_types}
                elif ex.lock_entity_types:
                    expand_types = set(schema.entity_types)
                if ex.lock_relations:
                    expand_relations = set(schema.relation_types)
                console.print(
                    f"[cyan]Expansion: locking to {len(expand_relations or [])} "
                    f"relation types, {len(expand_types or [])} entity kinds "
                    f"from '{ex.source}'.[/cyan]")

        # 1a. Enforce entity types at analyze time: the configured label set, or
        # the expansion entity-kind lock when active. Lets an already-extracted
        # checkpoint be narrowed without re-extraction (drops spaCy's off-target
        # DATE/EVENT/...); relations to dropped entities go later in the dedup remap.
        if expand_types is not None or self.config.foundation.restrict_to_label_types:
            if expand_types is not None:
                allowed = expand_types
            else:
                allowed = set(self.config.foundation.label_map.values())
                allowed |= set(self.domain.gliner_label_map().values())
            before = len(agg.entities)
            agg.entities = [e for e in agg.entities if e.label in allowed]
            if before != len(agg.entities):
                console.print(f"[cyan]Type restriction: kept {len(agg.entities)}/"
                              f"{before} entities (types {sorted(allowed)}).[/cyan]")

        # 1b. Ontology alignment: normalize relation-type vocabulary (domain or
        # config supplied). With expansion, restrict that vocabulary to the
        # relation types already in the source network (keeping their synonyms so
        # surface forms still map), and drop anything off-vocabulary if configured.
        if self.config.ontology.enabled or expand_relations is not None:
            from postprocess.ontology import OntologyAligner, resolve_relation_ontology
            if expand_relations is not None:
                base = resolve_relation_ontology(self.config, self.domain)
                onto = {rt: base.get(rt, []) for rt in expand_relations}
                drop_unmapped = self.config.expansion.drop_unmapped_relations
            else:
                onto = resolve_relation_ontology(self.config, self.domain)
                drop_unmapped = self.config.ontology.drop_unmapped
            if onto:
                aligner = OntologyAligner(onto, self.config.ontology.fuzzy_threshold,
                                          drop_unmapped)
                agg.relationships = aligner.apply(agg.relationships)

        # 2. Deduplicate (+ remap relationships onto entity ids).
        dedup = Deduplicator(self.config.dedup, domain_aliases=self.domain.aliases())
        entities, relationships, _name_to_id = dedup.resolve(
            agg.entities, agg.relationships
        )

        # Cross-document author anchoring: fold a lone surname uniquely naming one
        # author into that author node (forges the cross-letter edge). After dedup so
        # endpoints are ids; zero-ambiguity + capped.
        if self.config.inference.link_known_authors:
            from postprocess.identity_resolution import link_known_authors
            entities, relationships = link_known_authors(
                entities, relationships,
                min_len=self.config.inference.link_known_authors_min_len)

        llm_capable = self.config.mode in ("api", "ollama")
        ic = self.config.intelligence
        # gemini_batch can run the post-extraction LLM steps through Gemini's OpenAI
        # endpoint when batch_post_llm is on AND the --submit key is set (else building
        # the backend would raise). Opt-in: it's extra API calls at analyze time.
        batch_llm_ready = (self.config.mode == "gemini_batch" and ic.batch_post_llm
                           and bool(os.environ.get(ic.batch_api_key_env)))
        if batch_llm_ready:
            llm_capable = True
            console.print("[cyan]gemini_batch: running dedup/review/enrichment through "
                          f"{ic.batch_model} (batch_post_llm).[/cyan]")

        # Surface the silent gap: without a live backend any LLM-assisted post-step the
        # config asks for is skipped. Tell the user which, so a domain run isn't quietly
        # weaker than its config implies. (gemini_batch's win is whole-doc extraction.)
        if not llm_capable:
            wants = []
            if self.config.dedup.llm_assist:
                wants.append("dedup.llm_assist")
            if self.config.quality.llm_review is True or self.config.quality.llm_review == "auto":
                wants.append("quality.llm_review")
            if self.config.enrichment.enabled:
                wants.append("enrichment")
            if wants:
                hint = ("set batch_post_llm + $%s" % ic.batch_api_key_env
                        if self.config.mode == "gemini_batch" else "run in ollama/api mode")
                console.print(f"[yellow]Mode '{self.config.mode}' has no live LLM backend - "
                              f"skipping {', '.join(wants)} (rule-based dedup/review still run). "
                              f"To use them, {hint}.[/yellow]")

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

            # 2a3b. Functional-property consistency: a subject with one functional
            # relation (born_in, ...) pointing at two targets is a contradiction.
            if self.config.ontology.check_functional_consistency:
                from postprocess.ontology import check_functional_consistency
                relationships, n_fc = check_functional_consistency(
                    relationships, drop=self.config.ontology.drop_functional_conflicts)
                if n_fc:
                    verb = "Dropped from" if self.config.ontology.drop_functional_conflicts else "Tagged"
                    console.print(f"[cyan]{verb} {n_fc} functional-property "
                                  "conflicts (functional_conflict).[/cyan]")

        # 2a4. Relation self-verification: re-check each LLM edge against its evidence
        # (does the sentence actually assert that tie?). The post-hoc half of accuracy;
        # tags verification=supported/unsupported (or drops). LLM modes only.
        if self.config.quality.verify_relations and llm_capable:
            from postprocess.relation_verify import verify_relations
            id_to_name = {e.entity_id: e.canonical_name for e in entities}
            relationships, n_unsup = verify_relations(
                relationships, self.backend, id_to_name,
                batch_size=self.config.quality.verify_batch_size,
                max_relations=self.config.quality.verify_max,
                drop=self.config.quality.verify_drop)
            if n_unsup:
                verb = "Dropped" if self.config.quality.verify_drop else "Tagged"
                console.print(f"[cyan]{verb} {n_unsup} evidence-unsupported relations "
                              "(verification).[/cyan]")

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
            entities = link_entities(entities, self.config.linking,
                                     cache_path=self.run_dir / "wikidata_cache.json")
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
                               period_fn=self.domain.temporal_period,
                               edge_qualifiers=self.config.intelligence.edge_qualifiers)

        # 6b. Optional NetworkX SNA metrics Gephi can't compute (brokerage,
        # bridges, articulation) + graph-health QA. Fail-soft; opt-in.
        if self.config.export.graph_metrics:
            from postprocess import graph_metrics
            report = graph_metrics.enrich(
                tables, trust_verification=self.config.quality.trust_verification)
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
                from postprocess.narrative import (ELEMENT_SCHEMES, _ELEMENT_RULES,
                                                    write_narrative)
                # Domain-supplied rules win; else the config scheme picks a built-in
                # ("fiction" for novels/scripts), defaulting to life_course.
                scheme = getattr(self.config.export, "narrative_scheme", "auto")
                rules = (self.domain.narrative_rules()
                         or ELEMENT_SCHEMES.get(scheme) or _ELEMENT_RULES)
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

    def _effective_model(self) -> str:
        """The model that does the work for the current mode. gemini_batch keeps it
        in batch_model (no `gemini_batch` sub-config), the others in <mode>.model."""
        if self.config.mode == "gemini_batch":
            return self.config.intelligence.batch_model
        mode_cfg = getattr(self.config.intelligence, self.config.mode, None)
        return getattr(mode_cfg, "model", "")

    def _write_run_meta(self, stage: str, resume: bool, limit: int) -> None:
        """Snapshot the effective config into the run dir so every output is
        traceable to the exact model/mode/settings that produced it.

        The model/config that produced the checkpoint is recorded at EXTRACT time;
        an analyze-only re-run carries default settings (the user rarely re-passes
        --batch-model on analyze), so it must PRESERVE the extraction provenance
        instead of clobbering it with defaults."""
        import json
        from datetime import datetime, timezone

        model = self._effective_model()
        config_snap = self.config.model_dump(mode="json")
        path = self.run_dir / "run_meta.json"
        did_extract = stage in ("all", "ingest", "extract")
        if not did_extract and path.exists():
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
                model = prior.get("model") or model
                config_snap = prior.get("config") or config_snap
            except Exception:  # noqa: BLE001 - a corrupt prior meta just gets replaced
                pass
        meta = {
            "run_name": self.config.run_name,
            "mode": self.config.mode,
            "model": model,
            "stage": stage,
            "resume": resume,
            "limit": limit,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": sys.platform,
            "config": config_snap,
        }
        path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Full run
    def run(self, stage: str, resume: bool, limit: int = 0,
            extra_urls: tuple[str, ...] = (), urls_file: str = "", text: str = "",
            crawl_seeds: tuple[str, ...] = (), import_json: tuple[str, ...] | str = "",
            submit: bool = False) -> None:
        self._write_run_meta(stage, resume, limit)

        # fetch: crawl/preprocess only, freeze the corpus to documents.jsonl, stop.
        # No models, no GPU - run it on a laptop, then ship the snapshot and extract
        # in any --mode elsewhere (--ingest-from). Skips the whole extract machinery.
        if stage == "fetch":
            documents = self._gather(limit, extra_urls, urls_file, text, crawl_seeds,
                                     for_extract=True, resume=resume)
            snap = self.run_dir / "documents.jsonl"
            console.print(f"[green]Fetch: froze {len(documents)} documents to {snap}.\n"
                          f"Re-run extraction anywhere with: --ingest-from {snap} "
                          f"--mode <ollama|api|...>[/green]")
            return

        ckpt = CheckpointManager(
            self.run_dir / "checkpoints",
            self.config.run_name,
            enabled=self.config.checkpoint.enabled,
        )

        # Manual batch mode: extract = write the prompt and stop (a human runs the
        # model); analyze = import the JSON reply into the checkpoint, then proceed.
        if self.config.mode == "gemini_batch":
            if stage in ("all", "ingest", "extract"):
                submitted = self.run_batch_export(limit, extra_urls, urls_file, text,
                                                  crawl_seeds, submit=submit, resume=resume)
                # Manual path (or ingest, or submit produced nothing): stop after
                # writing prompts. With --submit the replies are already on disk, so
                # fall through to import + analyze for a single end-to-end command.
                if stage == "ingest" or not (submit and submitted):
                    return
            self.run_batch_import(ckpt, import_json, limit, extra_urls, urls_file,
                                  text, crawl_seeds)
            documents = self._gather(limit, extra_urls, urls_file, text, crawl_seeds,
                                     for_extract=False)
            current_ids = {d.doc_id for d in documents}
            extractions = [ex for ex in ckpt.load_all() if ex.doc_id in current_ids]
            # Optional: fold local GLiNER/spaCy spans onto the span-less whole-doc
            # entities so the proximity co-occurrence floor + evidence grounding work.
            if self.config.intelligence.reconcile_ner:
                self._reconcile_ner(extractions, {d.doc_id: d.text for d in documents})
            self.run_analyze(extractions)
            return

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
@click.option("--stage", type=click.Choice(["all", "fetch", "ingest", "extract", "analyze"]),
              default="all", show_default=True,
              help="Which stage(s) to run. 'fetch' = crawl/preprocess only, write the "
                   "portable documents.jsonl snapshot and stop (no models loaded).")
@click.option("--ingest-from", "ingest_from", default="",
              help="Load the corpus from a documents.jsonl snapshot (override "
                   "io.documents_file) - no crawl/fetch/file walk. Pair with --stage "
                   "fetch on one machine to scrape, then run extraction in any --mode "
                   "(e.g. on a remote ollama box) from the shipped snapshot.")
@click.option("--resume", is_flag=True, default=False,
              help="Resume from existing checkpoint, skipping completed documents.")
@click.option("--mode", type=click.Choice(["api", "python_only", "ollama", "langextract", "gemini_batch"]), default=None,
              help="Override the execution mode from the config.")
@click.option("--import-json", "import_json", multiple=True,
              help="gemini_batch: reply JSON to import at --stage analyze. Pass a "
                   "directory (globs gemini_batch_response*.json inside it), a file, "
                   "or repeat the flag. Default: <run>/gemini_batch_response*.json.")
@click.option("--batch-budget", "batch_budget", type=int, default=None,
              help="gemini_batch: chars of document text per prompt file (override "
                   "intelligence.batch_char_budget). Lower it for more, smaller "
                   "batches if the model truncates its JSON reply.")
@click.option("--batch-docs", "batch_docs", type=int, default=None,
              help="gemini_batch: max DOCUMENTS per prompt file (override "
                   "intelligence.batch_max_docs). The reliable anti-truncation knob; "
                   "20-40 suits dense first-person sources.")
@click.option("--submit", "submit", is_flag=True, default=False,
              help="gemini_batch: POST each prompt to the Gemini API (free key in "
                   "$GEMINI_API_KEY) and continue straight to analyze - no manual paste.")
@click.option("--batch-model", "batch_model", default="",
              help="gemini_batch --submit: Gemini model (override "
                   "intelligence.batch_model), e.g. gemini-2.5-pro for higher quality.")
@click.option("--batch-thinking", "batch_thinking", type=int, default=None,
              help="gemini_batch --submit: thinking-token budget (override "
                   "intelligence.batch_thinking_budget). 0 = off (default; frees the "
                   "output budget for JSON so the reply doesn't truncate). <0 keeps "
                   "the model's default reasoning on.")
@click.option("--batch-rpm", "batch_rpm", type=int, default=None,
              help="gemini_batch --submit: cap request starts at this many per "
                   "minute (override intelligence.batch_rpm). Set to the free-tier "
                   "limit (e.g. 10 for gemini-2.5-flash) to avoid 429 backoff churn.")
@click.option("--reconcile-ner", "reconcile_ner", is_flag=True, default=False,
              help="gemini_batch: re-run local GLiNER/spaCy after the whole-doc reply "
                   "and fold spans onto the span-less LLM entities (turns on the "
                   "proximity co-occurrence floor + adds GLiNER-only entities as a "
                   "recall net). Loads foundation models at analyze time.")
@click.option("--verify-relations", "verify_relations", is_flag=True, default=False,
              help="Turn on quality.verify_relations: the LLM re-checks each edge "
                   "against its evidence (tags verification=supported/unsupported). "
                   "LLM modes (api/ollama, or gemini_batch with --batch-post-llm).")
@click.option("--recall-pass", "recall_pass", is_flag=True, default=False,
              help="Turn on intelligence.recall_pass: re-prompt over the whole doc for "
                   "cross-chunk relations the first pass missed (chunked LLM modes).")
@click.option("--parse-scripts", "parse_scripts", is_flag=True, default=False,
              help="Turn on intelligence.parse_scripts: add scene co-presence edges for "
                   "documents that parse as a screenplay/TV script (character network).")
@click.option("--link-authors", "link_authors", is_flag=True, default=False,
              help="Turn on inference.link_known_authors: fold a lone surname uniquely "
                   "naming one author into that author node (cross-document edges).")
@click.option("--batch-post-llm", "batch_post_llm", is_flag=True, default=False,
              help="gemini_batch: run dedup/review/enrich/verify through Gemini's "
                   "OpenAI endpoint with the --submit key (intelligence.batch_post_llm).")
@click.option("--structured-output", "structured_output", is_flag=True, default=False,
              help="Turn on intelligence.structured_output: schema-constrain the "
                   "extraction call (ollama format-schema / OpenAI json_schema) so a "
                   "weak model can't leak prose into the JSON. Recommended for ollama.")
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
@click.option("--render-js", "render_js", is_flag=True, default=False,
              help="Render JavaScript with headless Chromium during crawl (io.crawl."
                   "render_js). For SPA/JS sites that return an empty shell to a plain "
                   "GET. Needs: pip install playwright && playwright install chromium.")
@click.option("--social", "social", multiple=True,
              help="Social-media source 'platform:target' (repeatable): reddit:datascience, "
                   "hackernews:top, mastodon:mastodon.social/tag/ai, twitter:from:nasa. "
                   "Pulls posts + the reply/mention/posted_in graph. Twitter needs "
                   "$TWITTER_BEARER_TOKEN; facebook is not supported (Meta ToS).")
@click.option("--social-limit", "social_limit", type=int, default=None,
              help="Posts/items per --social source (override io.social_limit).")
@click.option("--social-depth", "social_depth", type=int, default=None,
              help="0 = top-level posts only; 1 = also pull comment/reply trees "
                   "(override io.social_depth).")
@click.option("--wiki", "wiki", multiple=True,
              help="MediaWiki source 'host:Target' (repeatable): "
                   "en.wikipedia.org:Ada Lovelace, en.wikipedia.org:Category:Physicists, "
                   "harrypotter.fandom.com:Category:Death Eaters. Pulls clean article "
                   "prose via the API (not the page HTML).")
@click.option("--wiki-limit", "wiki_limit", type=int, default=None,
              help="Pages per --wiki source / category cap (override io.wiki_limit).")
@click.option("--littlesis", "littlesis", multiple=True,
              help="LittleSis source 'search:term' or 'id:N' (repeatable): imports the "
                   "curated relationship graph as asserted edges (donations carry amounts). "
                   "CC BY-SA - attribute in any published network.")
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
        import_json: tuple[str, ...], batch_budget: Optional[int], batch_docs: Optional[int],
        submit: bool, batch_model: str, batch_thinking: Optional[int],
        batch_rpm: Optional[int], reconcile_ner: bool,
        verify_relations: bool, recall_pass: bool, parse_scripts: bool, link_authors: bool,
        batch_post_llm: bool, structured_output: bool,
        limit: int, urls: tuple[str, ...], urls_file: str,
        crawl_seeds: tuple[str, ...], crawl_max_pages: Optional[int],
        crawl_max_depth: Optional[int], render_js: bool,
        social: tuple[str, ...], social_limit: Optional[int], social_depth: Optional[int],
        wiki: tuple[str, ...], wiki_limit: Optional[int],
        littlesis: tuple[str, ...],
        direct_text: str,
        min_entity_confidence: Optional[float], ollama_model: str, metadata_file: str,
        ingest_from: str, run_name: str, verbose: bool) -> None:
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
    # Feature toggles (A/B testing) - all four default off in config.
    if verify_relations:
        config.quality.verify_relations = True
    if recall_pass:
        config.intelligence.recall_pass = True
    if parse_scripts:
        config.intelligence.parse_scripts = True
    if link_authors:
        config.inference.link_known_authors = True
    if batch_post_llm:
        config.intelligence.batch_post_llm = True
    if structured_output:
        config.intelligence.structured_output = True
    if ollama_model:
        config.intelligence.ollama.model = ollama_model
    if batch_budget is not None:
        config.intelligence.batch_char_budget = batch_budget
    if batch_docs is not None:
        config.intelligence.batch_max_docs = batch_docs
    if batch_model:
        config.intelligence.batch_model = batch_model
    if batch_thinking is not None:
        config.intelligence.batch_thinking_budget = batch_thinking
    if batch_rpm is not None:
        config.intelligence.batch_rpm = batch_rpm
    if reconcile_ner:
        config.intelligence.reconcile_ner = True
    if metadata_file:
        config.io.metadata_file = metadata_file
    if ingest_from:
        config.io.documents_file = ingest_from
        console.print(f"[cyan]Override: io.documents_file = {ingest_from}[/cyan]")
    if run_name:
        config.run_name = run_name
        console.print(f"[cyan]Override: run_name = {run_name}[/cyan]")
    if crawl_seeds:
        config.io.crawl.enabled = True
    if crawl_max_pages is not None:
        config.io.crawl.max_pages = crawl_max_pages
    if crawl_max_depth is not None:
        config.io.crawl.max_depth = crawl_max_depth
    if render_js:
        config.io.crawl.render_js = True
    if social:
        config.io.social = list(config.io.social) + list(social)
    if social_limit is not None:
        config.io.social_limit = social_limit
    if social_depth is not None:
        config.io.social_depth = social_depth
    if wiki:
        config.io.wiki = list(config.io.wiki) + list(wiki)
    if wiki_limit is not None:
        config.io.wiki_limit = wiki_limit
    if littlesis:
        config.io.littlesis = list(config.io.littlesis) + list(littlesis)

    console.rule(f"[bold]NER + SNA Pipeline - run '{config.run_name}' (mode={config.mode})")
    pipeline = Pipeline(config)
    try:
        pipeline.run(stage=stage, resume=resume, limit=limit,
                     extra_urls=urls, urls_file=urls_file, text=direct_text,
                     crawl_seeds=crawl_seeds, import_json=import_json, submit=submit)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Progress is saved in the checkpoint; "
                      "re-run with --resume to continue.[/yellow]")
        sys.exit(130)
    console.rule("[bold green]Done")


if __name__ == "__main__":
    cli()
