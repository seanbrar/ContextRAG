This directory contains a small, public RFC-based demo dataset for offline evaluation.

Layout:

- documents/ (RFC text files, one per document)
- queries.jsonl (one JSON object per line)

Each line in queries.jsonl contains:
{"query": "...", "relevant_ids": ["rfcXXXX"]}

Quick demo:

```bash
uv run contextrag eval \
  --dataset data/demo \
  --baseline uniform \
  --k 5 \
  --embed-provider local \
  --output runs/demo_eval.json \
  --run-dir runs/demo_eval
```
