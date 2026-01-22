# Evaluation Results

This document summarizes evaluation methodology and results for ContextRAG.

## Primary Finding

**Adaptive length-based chunking does not improve retrieval accuracy over uniform chunking.**

Across two datasets (homogeneous RFC corpus and heterogeneous mixed corpus), the router strategy achieves identical precision and recall to the uniform baseline. The router produces marginally fewer chunks (1.8% reduction on mixed corpus) but this efficiency gain does not translate to accuracy improvement.

This negative result is reproducible and deterministic across multiple runs.

---

## Mixed Corpus Evaluation (Primary)

**Dataset composition**: 12 documents (3 short stories, 1 novella excerpt, 8 RFCs)
**Embedding provider**: OpenAI (`text-embedding-3-small`)
**Total queries**: 60
**Total source tokens**: 493,423
**Runs**: 3 (deterministic results)

### Document Length Distribution

| Category | Token Range | Count | Examples |
| -------- | ----------- | ----- | -------- |
| Short | ≤3,500 | 3 | Gettysburg Address (468), Gift of the Magi (2,764) |
| Medium | 3,500–15,000 | 1 | The Yellow Wallpaper (7,714) |
| Long | >15,000 | 8 | RFC 9110 (117k), A Scandal in Bohemia (137k) |

### Retrieval Accuracy

| Baseline | Precision@5 | Recall@5 | Indexed Chunks | Variance |
| -------- | ----------- | -------- | -------------- | -------- |
| Uniform  | 0.197       | 0.983    | 499            | 0 (deterministic) |
| Router   | 0.197       | 0.983    | 490            | 0 (deterministic) |

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
4. **Deterministic results**: Zero variance across 3 runs confirms this is not sampling noise

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

**The hypothesis is not supported.** Retrieval accuracy is invariant to chunking strategy across both homogeneous and heterogeneous corpora.

### Possible Explanations

1. **Embedding robustness**: Modern embedding models (text-embedding-3-small) may be robust to chunk boundary effects
2. **Query-document matching**: Similarity search finds relevant content regardless of how it's chunked
3. **Threshold arbitrariness**: The 3.5k/15k thresholds were based on old model context limits, not retrieval optimization

### Value of This Finding

This negative result is itself valuable:
- **Simplicity wins**: Uniform chunking is simpler and equally effective
- **Methodology demonstration**: Rigorous comparison with efficiency metrics and multiple runs
- **Infrastructure reusability**: The evaluation framework can test other strategies

---

## Artifact Paths

| Dataset | Baseline | Summary | Per-Query |
| ------- | -------- | ------- | --------- |
| Mixed | Uniform | `runs/eval_mixed_uniform/summary.json` | `runs/eval_mixed_uniform/per_query.jsonl` |
| Mixed | Router | `runs/eval_mixed_router/summary.json` | `runs/eval_mixed_router/per_query.jsonl` |
| RFC | Uniform | `runs/eval_uniform_v2/summary.json` | `runs/eval_uniform_v2/per_query.jsonl` |
| RFC | Router | `runs/eval_router_v2/summary.json` | `runs/eval_router_v2/per_query.jsonl` |
