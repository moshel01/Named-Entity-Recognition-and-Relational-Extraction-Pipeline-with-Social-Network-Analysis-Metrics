# Evaluation harness

Scores a pipeline run against hand-annotated **gold** data and reports entity
and relation **precision / recall / F1**. Pure standard library + the pipeline's
name normalization - no ML dependencies, so it runs anywhere.

## 1. Create a gold file

Hand-annotate a representative sample (aim for **20-30 documents** to start).
Copy [gold_template.json](gold_template.json) and fill it in:

```json
{
  "documents": [
    {
      "doc_id": "abel_0001",
      "entities": [
        {"name": "Hans Müller", "type": "PERSON"},
        {"name": "SA", "type": "ORG"}
      ],
      "relations": [
        {"source": "Hans Müller", "target": "SA", "type": "joined"}
      ]
    }
  ]
}
```

- `type` uses canonical types: `PERSON, ORG, LOCATION, EVENT, RANK, INSTITUTION, DATE`.
- Relation `type` may be omitted (scored in the type-agnostic relation metric).
- Use the **same names you expect in the network**; aliases are handled
  automatically - if the pipeline merges "der Führer" -> "Adolf Hitler", a gold
  entry of either form still matches.
- Scoring is **corpus-level** (the union over documents), matching what
  `entities.json` and `gephi_edges.csv` represent.

## 2. Run a pipeline run

```bash
python main.py --config domain/nazi_era/config_nazi_era.yaml
```

## 3. Score it

```bash
# Score everything
python -m evaluation.evaluate --gold gold.json --run-dir output/abel_papers

# Score only text-supported edges (academic "conservative" network)
python -m evaluation.evaluate --gold gold.json --run-dir output/abel_papers \
    --edge-sources conservative

# Explicit paths + save a JSON report
python -m evaluation.evaluate --gold gold.json \
    --entities output/abel_papers/entities.json \
    --edges    output/abel_papers/gephi_edges.csv \
    --out report.json
```

## What you get

- **Entities** - P/R/F1 overall, a type-agnostic variant, and a per-type table.
- **Relations** - typed (endpoints + relation type must match) and untyped
  (endpoints only) P/R/F1. Endpoints are matched undirected and alias-resolved.
- Up to 50 false positives / false negatives per metric for error inspection
  (in the `--out` JSON report).

## Edge-source tiers

`--edge-sources` lets you score each evidentiary network separately:

| Tier | Includes |
|------|----------|
| `conservative` | `llm_extracted`, `gliner_extracted`, `rule_extracted` |
| `moderate` | + `sna_inferred`, `rule_cooccurrence` |
| `full` / `all` | + `pipeline_inferred`, `canonical_inferred` |

Report the conservative network's edge precision as your headline relation
number; report the others to show the effect of inference.

## Interpreting results

- **Entity recall low?** GLiNER labels/threshold or (for German) you're still on
  an English-only GLiNER model - switch to `fastino/gliner2-multi-v1`.
- **Relation recall low in `python_only`?** Expected - the rule backend is
  precision-oriented. Compare against `api`/`ollama` modes.
- **Relation precision low in `full` but fine in `conservative`?** The inference
  layer is adding speculative edges; report conservative for claims.
