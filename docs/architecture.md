# Architecture

ContextRAG is a pipeline for adaptive retrieval:

1) Ingest: HTML/Markdown normalization and cleaning.
2) Route: length-based classification (short/medium/long).
3) Embed: provider-agnostic embeddings with optional chunking.
4) Index: Chroma vector store for similarity search.
5) Eval: reproducible benchmark runs with baselines and metadata.

Core artifacts:

- `manifest.jsonl` for ingest metadata
- `routing.jsonl` for routing decisions
- `embeddings.jsonl` for cached embeddings
- `runs/*` for eval output (summary, per-query, metadata)
