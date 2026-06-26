# Benchmark runner: dataset -> pipeline inputs + gold -> (run) -> (evaluate).

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import (ace2005, common, conll2003, dialogre, dwie, germeval, hipe,
               ontonotes5, redocred, tacred, uner, wnut17)

ADAPTERS = {
    "redocred": redocred,
    "dwie": dwie,
    "ace2005": ace2005,
    "tacred": tacred,
    "hipe": hipe,
    "dialogre": dialogre,
    "germeval": germeval,
    "conll2003": conll2003,
    "ontonotes5": ontonotes5,
    "wnut17": wnut17,
    "uner": uner,
}
DEFAULT_SPLIT = {"redocred": "test", "dwie": "train", "ace2005": "test",
                 "tacred": "test", "hipe": "dev", "dialogre": "dev",
                 "germeval": "validation", "conll2003": "test",
                 "ontonotes5": "test", "wnut17": "test", "uner": "test"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare/run/evaluate IE benchmarks.")
    ap.add_argument("--dataset", required=True, choices=list(ADAPTERS))
    ap.add_argument("--split", default=None, help="Dataset split (HF datasets).")
    ap.add_argument("--limit", type=int, default=50, help="Max documents (0 = all).")
    ap.add_argument("--path", default="", help="Local file path (ace2005 / tacred).")
    ap.add_argument("--workdir", default="data/bench", help="Where to write inputs/output.")
    ap.add_argument("--mode", default="python_only",
                    choices=["python_only", "api", "ollama", "gemini_batch"])
    ap.add_argument("--spacy-model", default="",
                    help="Default: adapter's preferred model (German for hipe), "
                         "else en_core_web_trf.")
    ap.add_argument("--gliner-model", default="",
                    help="Default: adapter's preferred model, else "
                         "fastino/gliner2-multi-v1 (same model the domains run; "
                         "gliner2-large-v1 segfaults on CPU load on some boxes).")
    ap.add_argument("--ollama-model", default="qwen3.5:9b")
    ap.add_argument("--batch-docs", type=int, default=10,
                    help="gemini_batch only: docs per prompt file (anti-truncation). "
                         "Benchmark docs are short, so 10 is safe; raise for speed.")
    ap.add_argument("--types", default="",
                    help="Comma-separated target types (default per dataset), "
                         "e.g. PERSON,ORG,LOCATION. Trims GLiNER labels + gold.")
    ap.add_argument("--min-entity-confidence", type=float, default=0.0,
                    help="Drop entities below this confidence (0 = keep all). Try "
                         "0.5 to A/B test precision vs recall; not a default.")
    ap.add_argument("--constrain-relations", action="store_true",
                    help="Inject the dataset's relation inventory as an ontology so "
                         "the LLM emits those labels (makes TYPED relation F1 "
                         "comparable; meaningful only with --mode ollama/api and "
                         "readable labels like DWIE).")
    ap.add_argument("--structured-output", action="store_true",
                    help="Schema-constrain the extraction call (ollama format / "
                         "OpenAI json_schema) so a weak local model can't leak prose "
                         "into the JSON. Tags the run variant _struct for A/B.")
    ap.add_argument("--coref", action="store_true",
                    help="Enable cross-sentence pronoun/alias coreference (off by "
                         "default; needs fastcoref + transformers<5, else it silently "
                         "uses a weak heuristic). Feeds a pronoun->name REFERENCE KEY "
                         "into the extraction prompt (the cross-sentence recall lever) "
                         "plus the co-occurrence layer. A/B it - the gain depends on "
                         "neural clusters firing. Tags variant _coref.")
    ap.add_argument("--coref-service", default="",
                    help="Route coref to the uvicorn microservice at this URL "
                         "(e.g. http://127.0.0.1:8000) instead of in-process "
                         "fastcoref. Implies --coref. Use when the pipeline venv has "
                         "transformers>=5 (in-process fastcoref breaks there).")
    ap.add_argument("--run", action="store_true", help="Invoke the pipeline after prep.")
    ap.add_argument("--resume", action="store_true",
                    help="Pass --resume to the pipeline: skip docs already in the "
                         "checkpoint (finish an interrupted run without re-extracting).")
    ap.add_argument("--eval", action="store_true", help="Run evaluation after the run.")
    args = ap.parse_args(argv)

    adapter = ADAPTERS[args.dataset]
    split = args.split or DEFAULT_SPLIT[args.dataset]
    spacy_model = (args.spacy_model
                   or getattr(adapter, "DEFAULT_SPACY_MODEL", "en_core_web_trf"))
    gliner_model = (args.gliner_model
                    or getattr(adapter, "DEFAULT_GLINER_MODEL", "fastino/gliner2-multi-v1"))

    # 1. Load + convert.
    print(f"Loading {args.dataset} (split={split}, limit={args.limit}) ...")
    if args.dataset in ("ace2005", "tacred"):
        if not args.path:
            ap.error(f"--path is required for {args.dataset} (local LDC-derived JSON).")
        docs = adapter.load(path=args.path, limit=args.limit)
    else:
        # path is optional here: HF-id override (conll2003/ontonotes5/wnut17) or
        # treebank config (uner); ignored by adapters that do not use it.
        docs = adapter.load(split=split, limit=args.limit, path=args.path)

    # Trim to target types (both gold and the GLiNER labels the pipeline gets).
    types = ([t.strip().upper() for t in args.types.split(",") if t.strip()]
             or getattr(adapter, "DEFAULT_TARGET_TYPES", ["PERSON", "ORG", "LOCATION"]))
    docs = common.filter_docs_to_types(docs, types)
    gliner_labels, label_map = common.labels_for_types(types)

    # Relation ontology for LLM constraint (dataset's distinct relation labels).
    ontology_relations = None
    if args.constrain_relations:
        ontology_relations = sorted({r.type for d in docs for r in d.relations if r.type})

    n_ent = sum(len(d.entities) for d in docs)
    n_rel = sum(len(d.relations) for d in docs)
    print(f"  {len(docs)} docs | {n_ent} gold entities | {n_rel} gold relations")
    print(f"  target types: {types}  -> gliner labels: {gliner_labels}")
    if ontology_relations:
        print(f"  constraining LLM to {len(ontology_relations)} relation labels")

    # 2. Write inputs + gold + config. A "variant" tag keeps mode / flag
    # combinations in separate output dirs, configs, and reports so A/B runs
    # never clobber each other.
    variant = args.mode
    if args.min_entity_confidence > 0:
        variant += f"_minconf{int(round(args.min_entity_confidence * 100))}"
    if ontology_relations:
        variant += "_constr"
    if args.structured_output:
        variant += "_struct"
    if args.coref or args.coref_service:
        variant += "_coref"
    work = Path(args.workdir)
    ds_dir = work / args.dataset
    input_dir = ds_dir / "inputs"
    output_dir = ds_dir / "output"
    gold_path = work / f"{args.dataset}.gold.json"
    config_path = work / f"{args.dataset}.{variant}.config.yaml"
    run_name = f"{args.dataset}_{split}_{variant}"

    common.write_inputs(docs, input_dir)
    common.write_gold(docs, gold_path)
    common.build_config(
        run_name=run_name, input_dir=input_dir, output_dir=output_dir,
        gliner_labels=gliner_labels, label_map=label_map,
        mode=args.mode, spacy_model=spacy_model, gliner_model=gliner_model,
        config_path=config_path, ollama_model=args.ollama_model,
        ontology_relations=ontology_relations,
        min_entity_confidence=args.min_entity_confidence,
        structured_output=args.structured_output,
        coref=args.coref,
        coref_service_url=args.coref_service,
    )
    run_dir = output_dir / run_name
    print(f"  inputs : {input_dir}")
    print(f"  gold   : {gold_path}")
    print(f"  config : {config_path}")

    py = sys.executable
    run_cmd = [py, "main.py", "--config", str(config_path)]
    if args.resume:
        run_cmd.append("--resume")
    # gemini_batch only completes with --submit (else it writes prompts and stops for
    # a manual paste). The whole-document path = no chunk-boundary recall loss; the
    # benchmark scores it through the same evaluate harness. Needs $GEMINI_API_KEY.
    if args.mode == "gemini_batch":
        run_cmd += ["--submit", "--batch-docs", str(args.batch_docs)]
    eval_cmds = [
        [py, "-m", "evaluation.evaluate", "--gold", str(gold_path),
         "--run-dir", str(run_dir), "--edge-sources", tier,
         "--out", str(work / f"{args.dataset}.{variant}.report.{tier}.json")]
        for tier in ("conservative", "moderate", "full")
    ]

    print("\nNext steps (also runnable directly):")
    print("  RUN :  " + " ".join(run_cmd))
    for c in eval_cmds:
        print("  EVAL:  " + " ".join(c))

    # 3. Optionally run + evaluate.
    if args.run:
        print("\n=== Running pipeline ===")
        rc = subprocess.call(run_cmd)
        if rc != 0:
            print(f"Pipeline exited with code {rc}; skipping eval.")
            return rc
    if args.eval:
        if not run_dir.exists():
            print(f"\nRun dir {run_dir} not found - run the pipeline first (--run).")
            return 1
        print("\n=== Evaluating ===")
        for c in eval_cmds:
            subprocess.call(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
