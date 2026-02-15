# Evaluation Results

## Primary Finding

**Length-based routing (`router`) does not outperform uniform chunking (`uniform`).**

Observed across local matrices (`k={3,5,10}`):
- `data/eval-expanded`: router underperforms uniform on all reported metrics
- `data/eval-external`: precision/recall ties at `k=5,10`; rank-sensitive metrics remain lower for router
- `data/eval-scifact-mini`: tie across all reported metrics

## Matrix Snapshot (k=5)

| Dataset | Baseline | Precision@5 | Recall@5 | Hit@1 | MRR@5 | nDCG@5 |
| --- | --- | --- | --- | --- | --- | --- |
| `data/eval-expanded` | uniform | 0.258 | 0.835 | 0.800 | 0.877 | 0.808 |
| `data/eval-expanded` | router | 0.248 | 0.785 | 0.730 | 0.818 | 0.751 |
| `data/eval-external` | uniform | 0.289 | 0.818 | 0.667 | 0.799 | 0.756 |
| `data/eval-external` | router | 0.289 | 0.818 | 0.639 | 0.780 | 0.739 |
| `data/eval-scifact-mini` | uniform | 0.210 | 0.950 | 0.850 | 0.890 | 0.901 |
| `data/eval-scifact-mini` | router | 0.210 | 0.950 | 0.850 | 0.890 | 0.901 |

## Why No Improvement?

1. **Embedding robustness**: Modern models (text-embedding-3-small, MiniLM) appear robust to chunk boundary effects
2. **Query-document matching**: Similarity search finds relevant content regardless of chunking granularity
3. **Threshold arbitrariness**: The 3.5K/15K thresholds were based on old model context limits, not optimized for retrieval

## What This Means

- Uniform chunking matches or beats routing with less complexity. There's no reason to add the routing code.
- Better retrieval probably means better embeddings or reranking, not chunk routing.

## Limitations

- Primary corpus is standards-domain focused (RFCs + literary texts)
- Length thresholds are heuristic, not learned
- Hosted provider results may shift as upstream models update
- SciFact transfer slice confirms the null effect on a public benchmark, but remains a small sample

## Scope Notes

- Mixed hosted runs (`data/eval-mixed`) are worth reviewing but are not the primary evidence
- This finding applies to this protocol and these datasets, not as a universal statement about all adaptive chunking methods

## Related Artifacts

- Matrix dashboards:
  - `docs/matrix_eval_expanded_local.md`
  - `docs/matrix_eval_external_local.md`
  - `docs/matrix_eval_scifact_local.md`
