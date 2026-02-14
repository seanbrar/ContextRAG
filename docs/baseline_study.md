# Baseline Study

This study expands beyond `uniform` vs `router` and evaluates:
- uniform chunk-size sweep (`512`, `1000`, `2000`)
- overlap vs no-overlap (`chunk_overlap_tokens=200` vs `0`)
- semantic chunking (`baseline=semantic`)
- lexical retrieval (`retrieval_mode=bm25`)
- hybrid retrieval (`retrieval_mode=hybrid`)
- dense retrieval with lexical rerank (`retrieval_mode=dense-rerank`)

## Rebuild

```bash
make baseline-study
```

Artifacts:
- `runs/baseline_study/baseline_study_summary.json`
- `docs/baseline_study.md`

Config files used are under `experiments/`:
- `eval_expanded_uniform_512_local.yaml`
- `eval_expanded_uniform_local.yaml`
- `eval_expanded_uniform_2000_local.yaml`
- `eval_expanded_uniform_overlap_local.yaml`
- `eval_expanded_semantic_local.yaml`
- `eval_expanded_bm25_uniform.yaml`
- `eval_expanded_hybrid_uniform_local.yaml`
- `eval_expanded_dense_rerank_uniform_local.yaml`
