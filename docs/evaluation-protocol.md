# Evaluation Protocol

This document defines the evaluation procedure for ContextRAG runs.

## Inputs

Dataset layout:

```
dataset/
├── documents/
└── queries.jsonl
```

Each line in `queries.jsonl` is:

```
{"query": "...", "relevant_ids": ["doc_id_1", "doc_id_2"]}
```

## Procedure

1. Load documents and (optionally) apply routing-based chunking.
2. Build a vector index with the selected embedding provider.
3. For each query, retrieve top-k chunks.
4. Compute precision@k and recall@k from `relevant_ids`.
5. Log timing, token counts, and run metadata.

## Configuration

The CLI accepts:

- `--dataset` (required)
- `--baseline` (`uniform` or `router`)
- `--k` (top-k)
- `--embed-provider` (`auto`, `openai`, `openrouter`, `local`)
- `--embedding-model` (optional override)
- `--run-dir` (artifact output)

Configs can be supplied with `--config` (YAML), see `experiments/`.

## Determinism and Limits

- Local embeddings are deterministic for a fixed model version.
- API-based providers may change results over time.
- The evaluation does not use randomness; repeated runs on the same environment
  and model should match.

## Artifacts

Each run directory includes:

- `summary.json` — aggregate metrics and timing
- `per_query.jsonl` — per-query precision/recall and hits
- `metadata.json` — dataset and run configuration
- `manifest.json` — config hash, dataset fingerprint, dependency versions, system info
