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

## Public Benchmark Variants

`data/eval-scifact-mini` is now part of the canonical core matrix scope.
Exploratory work on this slice should focus on non-core retrieval modes or
chunking variants.

## Artifact Evaluation Mode

```bash
make artifact-eval
```

Rebuilds reviewer + baseline-study artifacts and verifies them against
`docs/artifact_checksums.json`.
