# Reproducibility

This document describes how to reproduce ContextRAG evaluations and interpret artifacts.

Analysis expectations are preregistered in `docs/preregistration.md`.

Core claim boundary:
- Compare `uniform` vs `router`
- Use `retrieval_mode=dense`
- Use `data/eval-expanded` and `data/eval-external`

Other baselines/retrieval modes are exploratory extensions.

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
- `data/eval-expanded` — expanded mixed corpus with multi-relevance and hard negatives
- `data/eval-external` — external RFC holdout split for transfer checks
- `data/eval-scifact-mini` — public BEIR SciFact slice for non-RFC transfer checks

To rebuild the SciFact slice:

```bash
python3 scripts/build_eval_scifact_mini.py
```

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

Embedding-provider note:
- Embeddings support `openrouter` and `local` providers.
- OpenAI model families can be selected through OpenRouter model IDs
  (for example, `openai/text-embedding-3-small`).

## Matrix Runs (Recommended)

Run the claim-aligned matrix with local embeddings:

```bash
uv run contextrag core matrix \
  --dataset data/eval-expanded \
  --k-values 3,5,10 \
  --embed-provider local \
  --run-root runs/matrix_eval_expanded_local
```

This writes:
- one run directory per `(baseline, k)` pair
- `matrix_summary.json` and `matrix_summary.md`
- per-`k` uniform-vs-router comparisons under `comparisons/`

Optional dashboard rendering:

```bash
python3 scripts/render_matrix_report.py \
  --input runs/matrix_eval_expanded_local/matrix_summary.json \
  --output docs/matrix_eval_expanded_local.md
```

## Reviewer Bundle (Recommended)

Build all reviewer-facing local artifacts in one command:

```bash
make reviewer-bundle
```

This regenerates:
- expanded local matrix (`runs/reviewer_bundle/matrix_eval_expanded_local`)
- external holdout matrix (`runs/reviewer_bundle/matrix_eval_external_local`)
- markdown dashboards under `docs/`
- paper-ready tables at `docs/paper_tables.md`

## Exploratory Extensions

See `docs/exploratory.md` for baseline sweeps, alternate retrieval modes, and
public transfer-slice workflows.

## Artifacts

Each run directory contains:

```
runs/{run_name}/
├── summary.json     # aggregate metrics and timing
├── per_query.jsonl  # per-query metrics and hits
├── metadata.json    # dataset/config details
└── manifest.json    # config hash, dataset fingerprint, versions, system info
```

`manifest.json` includes:
- `manifest_schema_version`
- `git_commit` (when available)
- dataset fingerprint (`sha256`, file count, byte count)
- package/system versions

The top-level output JSON (e.g., `runs/demo_eval.json`) matches the
`summary.json` content and includes per-query records inline.

## Determinism Notes

- Local embeddings are deterministic for a fixed model version.
- Switching embedding providers requires rebuilding the index.
- API-based providers may introduce nondeterminism depending on model and
  service configuration.
