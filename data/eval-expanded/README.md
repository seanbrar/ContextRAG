Expanded mixed-corpus benchmark for ContextRAG.

Contents:
- `documents/`: copied from `data/eval-mixed/documents`
- `queries.jsonl`: 88 total queries
  - 60 legacy single-label queries (`relevant_ids`)
  - 28 multi-label/graded queries (`relevant`)

Schema:
- Legacy line shape: `{"query": "...", "relevant_ids": ["doc1", ...]}`
- v2 line shape: `{"query": "...", "relevant": [{"id": "doc1", "score": 2.0}, ...]}`

Rebuild command:

```bash
python3 scripts/build_eval_expanded.py
```

Validation:

```bash
uv run contextrag validate-dataset --dataset data/eval-expanded
```
