# Reproducibility

This document describes how to reproduce ContextRAG evaluations and interpret artifacts.

## Datasets

The evaluation runner expects:

```
dataset/
├── documents/      # one text file per document
└── queries.jsonl   # one JSON object per line
```

Each `queries.jsonl` line contains:

```
{"query": "...", "relevant_ids": ["doc_id_1", "doc_id_2"]}
```

Included datasets:

- `data/demo` — small RFC-based dataset for offline demo runs
- `data/eval-mixed` — mixed corpus used in the main evaluation (larger)

## Offline Demo (Deterministic)

This run uses local embeddings and produces reproducible artifacts:

```bash
uv run contextrag eval \
  --dataset data/demo \
  --baseline uniform \
  --k 5 \
  --embed-provider local \
  --output runs/demo_eval.json \
  --run-dir runs/demo_eval
```

The first run downloads the local embedding model
(`sentence-transformers/all-MiniLM-L6-v2`).

## Config-Driven Runs

For larger evals with saved artifacts:

```bash
uv run contextrag eval --config experiments/eval_rfc.yaml --run-dir runs/eval_rfc
```

## Artifacts

Each run directory contains:

```
runs/{run_name}/
├── summary.json     # aggregate metrics and timing
├── per_query.jsonl  # per-query metrics and hits
├── metadata.json    # dataset/config details
└── manifest.json    # config hash, dataset fingerprint, versions, system info
```

The top-level output JSON (e.g., `runs/demo_eval.json`) matches the
`summary.json` content and includes per-query records inline.

## Determinism Notes

- Local embeddings are deterministic for a fixed model version.
- Switching embedding providers requires rebuilding the index.
- API-based providers may introduce nondeterminism depending on model and
  service configuration.
