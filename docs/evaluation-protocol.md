# Evaluation Protocol

This document defines the evaluation procedure for ContextRAG runs.

Preregistered analysis plan: `docs/preregistration.md`.
Annotation process: `docs/annotation_protocol.md`.

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

External holdout split (`data/eval-external`) follows the same schema and is
used to test whether conclusions transfer beyond the original mixed corpus.

Scoring rubric for `score`:
- `2.0`: Primary source for the query intent
- `1.0`: Supporting/secondary relevant source

Curation rules:
1. Keep query wording answerable from corpus text only.
2. Prefer at least 2 relevant documents for cross-document questions.
3. Avoid near-duplicate queries that differ only by punctuation or tense.
4. Require unique relevant ids per query.
5. When uncertain, default to binary (`relevant_ids`) instead of forcing a graded label.

Hard-negative curation rules:
1. Include contrastive wording (`not X`, `rather than Y`) for a subset of queries.
2. Keep hard negatives topically close to positives (e.g., HTTP semantics vs framing RFCs).
3. Use low-but-positive graded relevance (`0 < score < 1`) only when a document is
   intentionally near-miss context, not a true answer source.
4. Preserve at least one clearly primary source (`score >= 1`) per query.

## Configuration

The CLI accepts:

- `--dataset` (required)
- `--baseline` (`uniform`, `router`, `adaptive`, `semantic`)
- `--retrieval-mode` (`dense`, `bm25`, `hybrid`, `dense-rerank`)
- `--k` (top-k)
- `--embed-provider` (`auto`, `openrouter`, `local`)
- `--embedding-model` (optional override)
- `--uniform-chunk-tokens` (optional chunk-size override)
- `--chunk-overlap-tokens` (optional overlap)
- `--retrieval-candidates` (candidate pool for hybrid/rerank)
- `--run-dir` (artifact output)

Note: direct OpenAI embedding provider selection is intentionally unsupported in
ContextRAG's embedding path. OpenAI embedding models can still be addressed via
OpenRouter model IDs (for example, `openai/text-embedding-3-small`).

Dataset schema can be validated explicitly:

```bash
uv run contextrag validate-dataset --dataset data/eval-mixed
uv run contextrag validate-dataset --dataset data/eval-expanded
uv run contextrag validate-dataset --dataset data/eval-external
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

## Preregistration Compliance

When reporting benchmark claims, include:
- primary endpoint outcome (`nDCG@k`)
- corrected p-values (Holm-adjusted)
- effect sizes and equivalence/non-inferiority outputs
- any explicit deviations from `docs/preregistration.md`
