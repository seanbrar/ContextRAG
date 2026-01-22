# Evaluation Results

This document summarizes evaluation methodology and results for ContextRAG.

## RFC Demo Dataset

**Dataset composition**: 7 IETF RFCs (822, 9110, 9112, 9113, 9595, 5322, 8446)
**Embedding provider**: OpenRouter (`qwen/qwen3-embedding-8b`)
**Total queries**: 14
**Total source tokens**: 341,613

### Retrieval Accuracy (Precision@5 / Recall@5)

| Baseline | Precision@5 | Recall@5 | Indexed Chunks |
| -------- | ----------- | -------- | -------------- |
| uniform  | 0.171       | 0.857    | 345 |
| router   | 0.171       | 0.857    | 345 |

### Efficiency Metrics

| Metric | Uniform | Router |
| ------ | ------- | ------ |
| Source documents | 7 | 7 |
| Total chunks | 345 | 345 |
| Total indexed tokens | 341,612 | 341,612 |
| Avg chunk size (tokens) | 990.2 | 990.2 |
| Index build time (sec) | 0.53 | 0.55 |
| Avg query latency (ms) | 1,625 | 2,153 |

**Router document classification**:
- Short (≤3.5k tokens): 0
- Medium (3.5k–15k tokens): 0
- Long (>15k tokens): 7

### Interpretation

Both strategies achieve identical results because **all 7 RFC documents exceed 15,000 tokens**, placing them in the "long" category. Both strategies apply 1000-token chunking to long documents, resulting in identical index construction.

This finding is itself valuable:

1. **The efficiency metrics expose the root cause** — without category distribution data, the identical accuracy results appeared mysterious
2. **RFC documents are not representative** — technical specifications are unusually long; real-world corpora include varied document lengths
3. **The routing strategy provides no benefit for uniformly-long corpora** — its value emerges with heterogeneous document collections

### Timing Analysis

Query latency differences (1.6s vs 2.2s average) are attributable to API variability rather than algorithmic differences, since both strategies produce identical indexes.

### Limitations

- All documents fall into a single length category, preventing routing differentiation
- Small dataset (7 documents) limits statistical significance
- API-based embeddings introduce latency variance

### Future Work

- **Heterogeneous corpus**: Evaluate on mixed document lengths (README files, API docs, research papers, code comments)
- **Synthetic dataset**: Generate documents at specific length thresholds to verify routing behavior
- **Local embeddings**: Reduce latency variance by using local embedding models

## Artifact Paths

| Run | Summary | Per-Query | Metadata |
| --- | ------- | --------- | -------- |
| uniform v2 | `runs/eval_uniform_v2/summary.json` | `runs/eval_uniform_v2/per_query.jsonl` | `runs/eval_uniform_v2/metadata.json` |
| router v2 | `runs/eval_router_v2/summary.json` | `runs/eval_router_v2/per_query.jsonl` | `runs/eval_router_v2/metadata.json` |
