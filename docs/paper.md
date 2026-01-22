# ContextRAG: Adaptive Routing for Retrieval-Augmented Generation

## Abstract

ContextRAG is a retrieval-augmented generation system that adapts document
processing based on length. Instead of uniform chunking, it routes documents
into size-aware pipelines and applies different embedding strategies. This
reduces token waste on short documents and improves retrieval quality on long
ones. The system is designed to be reproducible with public datasets and
config-driven evaluation.

## Motivation

RAG systems are sensitive to context limits and chunking strategy. Uniform
chunking can degrade semantic coherence for long documents and waste capacity
on short ones. ContextRAG introduces adaptive routing to align processing
strategy with document length and structure while keeping the pipeline simple
to deploy.

## Method

### Routing

Documents are routed into length buckets based on token count:

- short: <= 3.5k tokens
- medium: 3.5k–15k tokens
- long: > 15k tokens

Routing can be configured in the CLI (`contextrag route`) and is recorded in
`routing.jsonl` for reproducibility.

### Embeddings

Embedding generation is provider-agnostic. OpenAI, OpenRouter, or local models
can be used depending on available credentials and configuration. For long
documents, chunking uses a fixed token budget per chunk to preserve coherence.

### Vector Store

Embeddings are stored in Chroma. The collection can be ephemeral or persisted,
and embedding functions are attached to the collection for consistent behavior.

## Evaluation

We evaluate retrieval performance using a small RFC-based dataset with
length-stratified queries. Metrics include precision@k and recall@k, with
baselines for uniform chunking versus length-aware routing.

Suggested eval command:

```bash
poetry run contextrag eval --config experiments/eval_rfc.yaml --run-dir runs/eval_rfc
```

## Limitations

- Token thresholds are static and not tuned per dataset.
- Context limits depend on the embedding model; chunking heuristics are fixed.
- The demo dataset is small and intended for reproducibility rather than scale.

## Future Work

- Dynamic routing based on structure and semantic density.
- Adaptive chunk size using model context limits and retrieval sensitivity.
- Larger benchmark suites and automated ablation reports.
