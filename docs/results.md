# Evaluation Results

This document summarizes evaluation methodology and results for ContextRAG.

## Primary Finding

**Adaptive length-based chunking does not outperform uniform chunking in the evaluated setup.**

Across committed runs, the router strategy never beats the uniform baseline:
- mixed corpus (hosted `text-embedding-3-small`): identical precision/recall
- RFC-only corpus (hosted OpenRouter embeddings): identical precision/recall
- expanded local matrix (`k={3,5,10}`): router underperforms on all tested `k`
- external local holdout (`k={3,5,10}`): router ties or underperforms uniform

The router produces marginally fewer chunks (1.8% reduction on mixed corpus), but this efficiency gain does not translate to better retrieval quality.

For the mixed-corpus experiment committed in this repository, three repeated `k=5` runs produced identical aggregate precision/recall.

---

## Methods (Summary)

**Evaluation task**: For each query, retrieve top-k chunks and compute precision@k and recall@k against annotated `relevant_ids`.

**Chunking baselines**:
- Uniform: fixed 1,000-token chunks
- Router: length-based chunking (short/medium/long)

**Embedding providers**:
- Mixed corpus: hosted `text-embedding-3-small`
- RFC-only corpus: OpenRouter (`qwen/qwen3-embedding-8b`)
- Expanded/external matrices: local MiniLM (`sentence-transformers/all-MiniLM-L6-v2`)

**Determinism**: Local embeddings are deterministic for a fixed model version. Hosted API providers can change behavior over time; repeated runs here were stable for aggregate precision/recall.

**Artifacts**: Each run writes `summary.json`, `per_query.jsonl`, `metadata.json`, and `manifest.json` under `runs/{run_name}/`.

## Mixed Corpus Evaluation (Primary)

**Dataset composition**: 12 documents (3 short stories, 1 novella excerpt, 8 RFCs)
**Embedding provider**: hosted `text-embedding-3-small`
**Total queries**: 60
**Total source tokens**: 493,423
**Runs**: 3 repeated runs (identical aggregate precision/recall)

### Document Length Distribution

| Category | Token Range | Count | Examples |
| -------- | ----------- | ----- | -------- |
| Short | ≤3,500 | 3 | Gettysburg Address (468), Gift of the Magi (2,764) |
| Medium | 3,500–15,000 | 1 | The Yellow Wallpaper (7,714) |
| Long | >15,000 | 8 | RFC 9110 (117k), A Scandal in Bohemia (137k) |

### Retrieval Accuracy

| Baseline | Precision@5 | Recall@5 | Indexed Chunks | Aggregate Variance |
| -------- | ----------- | -------- | -------------- | -------- |
| Uniform  | 0.197       | 0.983    | 499            | 0 across 3 runs |
| Router   | 0.197       | 0.983    | 490            | 0 across 3 runs |

### Efficiency Comparison

| Metric | Uniform | Router | Delta |
| ------ | ------- | ------ | ----- |
| Total chunks | 499 | 490 | -1.8% |
| Avg chunk size (tokens) | 988.8 | 1,007.0 | +1.8% |
| Index build time (sec) | 0.71 | 0.79 | +11% |
| Avg query latency (ms) | 268 | 312 | +16% |

### Interpretation

1. **Accuracy equivalence**: Both strategies retrieve the same relevant documents with identical precision and recall
2. **Marginal efficiency gain**: Router produces 9 fewer chunks (1.8% reduction), insufficient to justify added complexity
3. **No latency benefit**: Router is actually slightly slower due to classification overhead
4. **Stable aggregate outcome**: Precision/recall were unchanged across 3 repeated runs

---

## RFC-Only Evaluation (Secondary)

**Dataset composition**: 7 IETF RFCs
**Embedding provider**: OpenRouter (`qwen/qwen3-embedding-8b`)
**Total queries**: 14
**Total source tokens**: 341,613

### Results

| Baseline | Precision@5 | Recall@5 | Indexed Chunks |
| -------- | ----------- | -------- | -------------- |
| Uniform  | 0.171       | 0.857    | 345 |
| Router   | 0.171       | 0.857    | 345 |

**Document classification**: All 7 documents exceed 15k tokens → all classified as "long" → identical chunking applied.

This evaluation demonstrates that with homogeneous document lengths, the router provides zero differentiation.

---

## Hypothesis and Conclusion

### Original Hypothesis

Length-based routing would improve retrieval by:
- Preserving semantic coherence in short documents (no chunking)
- Using larger chunks for medium documents (fewer boundaries)
- Applying fine-grained chunking only to very long documents

### Observed Result

**The hypothesis is not supported in this benchmark setup.** Across committed slices, routing never beats uniform and is often slightly worse on rank-sensitive metrics.

### Possible Explanations

1. **Embedding robustness**: Modern embedding models (text-embedding-3-small) may be robust to chunk boundary effects
2. **Query-document matching**: Similarity search finds relevant content regardless of how it's chunked
3. **Threshold arbitrariness**: The 3.5k/15k thresholds were based on old model context limits, not retrieval optimization

### Value of This Finding

This negative result is itself valuable:
- **Simplicity wins**: Uniform chunking is simpler and equal or better in committed runs
- **Methodology demonstration**: Rigorous comparison with efficiency metrics and multiple runs
- **Infrastructure reusability**: The evaluation framework can test other strategies

### Scope Notes

- Mixed-corpus queries currently use one relevant document id per query.
- At `k=5`, precision therefore has a practical ceiling near `0.2`, which limits sensitivity.
- This result should be interpreted as evidence for this dataset/protocol, not as a universal statement about all adaptive chunking methods.

---

## Expanded Matrix Evaluation (Local, v2 Dataset)

To stress-test the methodology with multi-relevance labels, we ran:
- Dataset: `data/eval-expanded` (100 queries, including graded relevance + hard negatives)
- Provider: local MiniLM (`sentence-transformers/all-MiniLM-L6-v2`)
- Matrix: `uniform` vs `router` across `k={3,5,10}`

### Aggregate Metrics

| Baseline | k | Precision@k | Recall@k | Hit@1 | MRR@k | nDCG@k |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform | 3 | 0.363 | 0.728 | 0.800 | 0.853 | 0.766 |
| Router | 3 | 0.343 | 0.668 | 0.730 | 0.788 | 0.702 |
| Uniform | 5 | 0.258 | 0.835 | 0.800 | 0.877 | 0.808 |
| Router | 5 | 0.248 | 0.785 | 0.730 | 0.818 | 0.751 |
| Uniform | 10 | 0.156 | 0.901 | 0.810 | 0.882 | 0.842 |
| Router | 10 | 0.152 | 0.866 | 0.730 | 0.826 | 0.789 |

### Interpretation

1. Router did not outperform uniform on any tested `k` value in this local matrix.
2. The direction of effect was consistently negative across precision, recall, Hit@1, MRR, and nDCG.
3. Per-k comparison artifacts include confidence intervals and paired randomization p-values under `runs/matrix_eval_expanded_local/comparisons/`.
4. This run demonstrates a broader, more sensitive evaluation setup than the original single-label `k=5` slice.

Full local dashboard: `docs/matrix_eval_expanded_local.md`

---

## External Holdout Matrix (Local)

To check transfer beyond the original mixed corpus, we added an external RFC holdout:
- Dataset: `data/eval-external` (10 held-out RFCs, 36 queries)
- Provider: local MiniLM (`sentence-transformers/all-MiniLM-L6-v2`)
- Matrix: `uniform` vs `router` across `k={3,5,10}`

### Aggregate Metrics

| Baseline | k | Precision@k | Recall@k | Hit@1 | MRR@k | nDCG@k |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform | 3 | 0.407 | 0.725 | 0.667 | 0.792 | 0.712 |
| Router | 3 | 0.398 | 0.716 | 0.639 | 0.773 | 0.689 |
| Uniform | 5 | 0.289 | 0.818 | 0.667 | 0.799 | 0.756 |
| Router | 5 | 0.289 | 0.818 | 0.639 | 0.780 | 0.739 |
| Uniform | 10 | 0.175 | 0.930 | 0.667 | 0.808 | 0.812 |
| Router | 10 | 0.175 | 0.926 | 0.639 | 0.789 | 0.794 |

### Interpretation

1. Router still does not outperform uniform on any tested `k`.
2. Precision/recall effects are near-zero on this split, while rank-sensitive metrics (Hit@1/MRR/nDCG) remain slightly worse for router.
3. Combined with expanded-matrix underperformance, this strengthens the canonical claim that length-based routing does not provide a quality win here.

Full local dashboard: `docs/matrix_eval_external_local.md`

---

## Embedding Cost Comparison

The evaluation framework tracks embedding costs to inform provider selection decisions.

### Methodology

Cost is calculated as:
- **Index cost** = (total indexed tokens / 1,000,000) × model cost per million tokens
- **Query cost** = (total query tokens / 1,000,000) × model cost per million tokens
- **Total cost** = Index cost + Query cost

### Results (Mixed Corpus, Uniform Baseline)

| Model | Cost/M tokens | Index Cost | Query Cost | Total Cost | Precision@5 | Recall@5 |
|-------|---------------|------------|------------|------------|-------------|----------|
| text-embedding-3-small | $0.02 | $0.009868 | $0.000017 | $0.009886 | 0.197 | 0.983 |
| text-embedding-3-large | $0.13 | $0.064145 | $0.000113 | $0.064257 | 0.200 | 1.000 |

### Interpretation

1. **6.5× cost difference**: text-embedding-3-large costs 6.5× more than text-embedding-3-small
2. **Marginal accuracy gain**: +0.3% precision, +1.7% recall for the more expensive model
3. **Index-dominated cost**: Query costs are negligible (~0.2% of total) for typical workloads
4. **Cost-quality tradeoff**: For most use cases, text-embedding-3-small provides sufficient quality at significantly lower cost

### Provider Cost Comparison (Theoretical)

Based on published pricing (as of January 2025):

| Provider/Model | Cost/M tokens | Relative Cost |
|----------------|---------------|---------------|
| thenlper/gte-base (OpenRouter) | $0.005 | 0.25× |
| qwen/qwen3-embedding-8b (OpenRouter) | $0.01 | 0.5× |
| text-embedding-3-small (OpenAI) | $0.02 | 1× (baseline) |
| text-embedding-3-large (OpenAI) | $0.13 | 6.5× |

The custom ChromaDB-OpenRouter integration enables 50–90% cost reduction by routing to cheaper embedding providers while maintaining the same evaluation infrastructure.

---

## Artifact Paths

| Dataset | Baseline | Summary | Per-Query |
| ------- | -------- | ------- | --------- |
| Mixed | Uniform | `runs/eval_mixed_uniform/summary.json` | `runs/eval_mixed_uniform/per_query.jsonl` |
| Mixed | Router | `runs/eval_mixed_router/summary.json` | `runs/eval_mixed_router/per_query.jsonl` |
| RFC | Uniform | `runs/eval_uniform_v2/summary.json` | `runs/eval_uniform_v2/per_query.jsonl` |
| RFC | Router | `runs/eval_router_v2/summary.json` | `runs/eval_router_v2/per_query.jsonl` |
| Expanded matrix | Uniform/Router | `runs/reviewer_bundle/matrix_eval_expanded_local/*/summary.json` | `runs/reviewer_bundle/matrix_eval_expanded_local/*/per_query.jsonl` |
| External matrix | Uniform/Router | `runs/reviewer_bundle/matrix_eval_external_local/*/summary.json` | `runs/reviewer_bundle/matrix_eval_external_local/*/per_query.jsonl` |
