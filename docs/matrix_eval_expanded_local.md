# Matrix Report

- Dataset: `data/eval-expanded`
- Baselines: `uniform, router`
- k values: `3, 5, 10`
- Embed provider: `local`
- Embedding model override: `None`

## Aggregate Metrics

| Baseline | k | Precision@k | Recall@k | Hit@1 | MRR@k | nDCG@k |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 3 | 0.352 | 0.743 | 0.784 | 0.839 | 0.780 |
| uniform | 5 | 0.252 | 0.853 | 0.784 | 0.866 | 0.820 |
| uniform | 10 | 0.153 | 0.912 | 0.784 | 0.866 | 0.847 |
| router | 3 | 0.330 | 0.674 | 0.705 | 0.765 | 0.707 |
| router | 5 | 0.241 | 0.796 | 0.705 | 0.798 | 0.755 |
| router | 10 | 0.150 | 0.878 | 0.705 | 0.808 | 0.795 |

## Uniform vs Router (Per-k Summary Deltas)

| k | Δ Precision@k | Δ Recall@k | Δ Hit@1 | Δ MRR@k | Δ nDCG@k |
| --- | --- | --- | --- | --- | --- |
| 3 | -0.023 | -0.068 | -0.080 | -0.074 | -0.072 |
| 5 | -0.011 | -0.057 | -0.080 | -0.068 | -0.065 |
| 10 | -0.003 | -0.034 | -0.080 | -0.059 | -0.052 |
