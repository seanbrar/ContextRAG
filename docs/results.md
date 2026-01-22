# Evaluation Results

This document summarizes evaluation methodology and results for ContextRAG.

## RFC Demo Dataset

**Dataset composition**: IETF RFCs (822, 9110, 9112, 9113, 9595, 5322, 8446)
**Embedding provider**: OpenRouter (`qwen/qwen3-embedding-8b`)
**Total queries**: 14
**Indexed chunks**: 345

### Retrieval Accuracy (Precision@5 / Recall@5)

| Baseline | Precision@5 | Recall@5 | Chunking Strategy |
| -------- | ----------- | -------- | ----------------- |
| uniform  | 0.171       | 0.857    | Fixed 1000-token chunks for all documents |
| router   | 0.171       | 0.857    | Adaptive: short (none), medium (2k), long (1k) |

### Interpretation

Both strategies achieve equivalent retrieval accuracy on this dataset. This result is informative:

1. **Dataset homogeneity**: RFC documents share similar structure and length distributions, reducing the impact of adaptive routing
2. **Recall vs. Precision tradeoff**: High recall (0.857) with lower precision (0.171) suggests the evaluation queries have few relevant documents, making both strategies retrieve them successfully
3. **Efficiency benefits not captured**: Accuracy metrics alone don't reflect the token efficiency gains from avoiding unnecessary chunking

### Limitations

- Small dataset (7 RFCs) limits statistical significance
- Homogeneous document structure doesn't stress-test routing logic
- No latency or cost metrics captured in current evaluation

### Future Work

- Evaluate on heterogeneous corpora (mixed technical docs, research papers, web content)
- Add efficiency metrics: total tokens indexed, API calls, indexing latency
- Measure "lost in the middle" effects for different chunk sizes

## Artifact Paths

| Run | Summary | Per-Query | Metadata |
| --- | ------- | --------- | -------- |
| uniform | `runs/eval_uniform/summary.json` | `runs/eval_uniform/per_query.jsonl` | `runs/eval_uniform/metadata.json` |
| router | `runs/eval_router/summary.json` | `runs/eval_router/per_query.jsonl` | `runs/eval_router/metadata.json` |
