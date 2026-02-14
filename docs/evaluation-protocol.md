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

Optional schema v2 shape with graded relevance:

```
{"query": "...", "relevant": [{"id": "doc_id_1", "score": 2.0}, {"id": "doc_id_2", "score": 1.0}]}
```

Validation rules:
- `query` must be non-empty text
- `relevant_ids` (or `relevant[].id`) must be non-empty, unique ids
- optional scores must be numeric and `> 0`

## Procedure

1. Load documents and (optionally) apply routing-based chunking.
2. Build a vector index with the selected embedding provider.
3. For each query, retrieve top-k chunks.
4. Compute precision@k and recall@k from `relevant_ids`.
5. Log timing, token counts, and run metadata.

## Annotation Protocol (v2)

The expanded benchmark (`data/eval-expanded`) uses a two-shape relevance schema:
- `relevant_ids` for binary relevance
- `relevant` for graded relevance (`id`, `score`)

Scoring rubric for `score`:
- `2.0`: Primary source for the query intent
- `1.0`: Supporting/secondary relevant source

Curation rules:
1. Keep query wording answerable from corpus text only.
2. Prefer at least 2 relevant documents for cross-document questions.
3. Avoid near-duplicate queries that differ only by punctuation or tense.
4. Require unique relevant ids per query.
5. When uncertain, default to binary (`relevant_ids`) instead of forcing a graded label.

## Configuration

The CLI accepts:

- `--dataset` (required)
- `--baseline` (`uniform` or `router`)
- `--k` (top-k)
- `--embed-provider` (`auto`, `openai`, `openrouter`, `local`)
- `--embedding-model` (optional override)
- `--run-dir` (artifact output)

Dataset schema can be validated explicitly:

```bash
uv run contextrag validate-dataset --dataset data/eval-mixed
```

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
