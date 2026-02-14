# Architecture

ContextRAG is an evaluation pipeline for retrieval chunking strategies:

1) Load dataset (`documents/` + `queries.jsonl`).
2) Apply chunking strategy (`uniform`, `adaptive`, or `router` alias).
3) Build embeddings via `chromaroute` (OpenRouter/local depending on config).
4) Index/query via ChromaDB.
5) Compute retrieval metrics and write run artifacts.

Run artifacts (`--run-dir`) currently include:

- `summary.json` (aggregate metrics, timing, efficiency, cost)
- `per_query.jsonl` (per-query retrieved ids and metrics)
- `metadata.json` (run configuration)
- `manifest.json` (config hash, dataset fingerprint, environment versions/system info)
