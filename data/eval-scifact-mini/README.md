SciFact mini benchmark slice for ContextRAG.

Source:
- BEIR SciFact test split
- Download URL: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip

Contents:
- `documents/`: 220 selected scientific abstracts
- `queries.jsonl`: 40 queries with multi-relevance labels
- `provenance.json`: build/source metadata
- `annotations/`: dual-annotation rounds and agreement report

Rebuild:

```bash
python3 scripts/build_eval_scifact_mini.py
```

Validation:

```bash
uv run contextrag validate-dataset --dataset data/eval-scifact-mini
```
