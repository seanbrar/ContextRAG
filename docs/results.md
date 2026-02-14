# Evaluation Results

This document summarizes canonical ContextRAG results from the reviewer bundle artifacts.

Canonical artifact source:
- `runs/reviewer_bundle/matrix_eval_expanded_local/matrix_summary.json`
- `runs/reviewer_bundle/matrix_eval_external_local/matrix_summary.json`
- `runs/reviewer_bundle/matrix_eval_scifact_local/matrix_summary.json`

## Primary Finding

**Length-based routing (`router`) does not outperform uniform chunking (`uniform`) in this benchmark setup.**

Observed across canonical local matrices (`k={3,5,10}`):
- `data/eval-expanded`: router underperforms uniform on all reported metrics
- `data/eval-external`: precision/recall ties at `k=5,10`; rank-sensitive metrics remain lower for router
- `data/eval-scifact-mini`: tie across all reported metrics

## Core Matrix Snapshot (k=5)

| Dataset | Baseline | Precision@5 | Recall@5 | Hit@1 | MRR@5 | nDCG@5 |
| --- | --- | --- | --- | --- | --- | --- |
| `data/eval-expanded` | uniform | 0.258 | 0.835 | 0.800 | 0.877 | 0.808 |
| `data/eval-expanded` | router | 0.248 | 0.785 | 0.730 | 0.818 | 0.751 |
| `data/eval-external` | uniform | 0.289 | 0.818 | 0.667 | 0.799 | 0.756 |
| `data/eval-external` | router | 0.289 | 0.818 | 0.639 | 0.780 | 0.739 |
| `data/eval-scifact-mini` | uniform | 0.210 | 0.950 | 0.850 | 0.890 | 0.901 |
| `data/eval-scifact-mini` | router | 0.210 | 0.950 | 0.850 | 0.890 | 0.901 |

## Inference and Preregistration Alignment

Preregistered source:
- `docs/preregistration.md`
- lock artifact: `docs/preregistration_lock.json`

Canonical comparison artifacts now include:
- paired per-query deltas
- bootstrap 95% CI
- paired randomization p-values
- Holm-adjusted p-values
- Cohen's d and Cliff's delta
- primary endpoint TOST outputs (`p(lower)`, `p(upper)`, equivalence/non-inferiority)

Primary endpoint:
- `nDCG@k`
- equivalence margin: `epsilon = 0.02`
- TOST alpha: `0.05`

## Annotation and Label Quality Status

Core datasets with retrospective dual-pass annotation artifacts:
- `data/eval-expanded/annotations/annotator_a.jsonl`
- `data/eval-expanded/annotations/annotator_b.jsonl`
- `data/eval-expanded/annotations/agreement.json`
- `data/eval-external/annotations/annotator_a.jsonl`
- `data/eval-external/annotations/annotator_b.jsonl`
- `data/eval-external/annotations/agreement.json`

Public slice with dual-round artifacts:
- `data/eval-scifact-mini/annotations/annotator_a.jsonl`
- `data/eval-scifact-mini/annotations/annotator_b.jsonl`
- `data/eval-scifact-mini/annotations/agreement.json`

## Scope Notes

- Mixed hosted runs (`data/eval-mixed`) remain useful context but are not the sole basis for canonical inference.
- SciFact ties indicate a null effect on this public transfer slice under current settings.
- This claim is evidence for this protocol and datasets, not as a universal statement about all adaptive chunking methods.

## Deviations

- None for the canonical reviewer bundle workflow.

## Related Artifacts

- Matrix dashboards:
  - `docs/matrix_eval_expanded_local.md`
  - `docs/matrix_eval_external_local.md`
  - `docs/matrix_eval_scifact_local.md`
- Aggregated paper tables: `docs/paper_tables.md`
- Reviewer index: `docs/reviewer_bundle.md`
- Baseline extensions: `docs/baseline_study.md`
