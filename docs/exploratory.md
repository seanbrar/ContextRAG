# Exploratory Extensions

This page collects non-canonical evaluation extensions.

Canonical claim runs should use:
- `contextrag core eval`
- `contextrag core matrix`

## Expanded Baseline Study

```bash
make baseline-study
```

Runs exploratory variants on `data/eval-expanded`:
- uniform chunk-size sweep (`512`, `1000`, `2000`)
- overlap/no-overlap variants
- semantic chunking baseline
- lexical (`bm25`), hybrid, and dense-rerank retrieval modes

Outputs:
- `runs/baseline_study/baseline_study_summary.json`
- `docs/baseline_study.md`

## Public Benchmark Slice

`data/eval-scifact-mini` is a non-RFC transfer slice built from BEIR SciFact
(40 queries, 220 docs).

```bash
uv run contextrag validate-dataset --dataset data/eval-scifact-mini
```

## Artifact Evaluation Mode

```bash
make artifact-eval
```

Rebuilds reviewer + baseline-study artifacts and verifies them against
`docs/artifact_checksums.json`.
