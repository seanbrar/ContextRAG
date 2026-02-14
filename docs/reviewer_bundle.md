# Reviewer Bundle

Canonical claim: length-based routing does not outperform uniform chunking on committed benchmarks.

## Included Matrix Runs

- `data/eval-expanded`: `runs/reviewer_bundle/matrix_eval_expanded_local/matrix_summary.json`
  uniform@5: P=0.258, R=0.835, Hit@1=0.800, MRR=0.877, nDCG=0.808
  router@5:  P=0.248, R=0.785, Hit@1=0.730, MRR=0.818, nDCG=0.751
- `data/eval-external`: `runs/reviewer_bundle/matrix_eval_external_local/matrix_summary.json`
  uniform@5: P=0.289, R=0.818, Hit@1=0.667, MRR=0.799, nDCG=0.756
  router@5:  P=0.289, R=0.818, Hit@1=0.639, MRR=0.780, nDCG=0.739
- `data/eval-scifact-mini`: `runs/reviewer_bundle/matrix_eval_scifact_local/matrix_summary.json`
  uniform@5: P=0.210, R=0.950, Hit@1=0.850, MRR=0.890, nDCG=0.901
  router@5:  P=0.210, R=0.950, Hit@1=0.850, MRR=0.890, nDCG=0.901

## Generated Reports

- `docs/matrix_eval_expanded_local.md`
- `docs/matrix_eval_external_local.md`
- `docs/matrix_eval_scifact_local.md`
- `docs/paper_tables.md`
- `docs/preregistration_lock.json`

## Rebuild

```bash
make reviewer-bundle
```
