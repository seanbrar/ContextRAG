# ContextRAG: Evaluating Adaptive Chunking for Retrieval-Augmented Generation

## Abstract

ContextRAG explores whether adaptive chunking strategies—routing documents to different processing pipelines based on length—can improve retrieval quality in RAG systems. We implement a three-tier routing system and evaluate it against uniform chunking across committed benchmark slices. **Our evaluation finds that adaptive routing never outperforms uniform chunking**: it ties on the hosted mixed-corpus slice and underperforms on the expanded local matrix. This negative result suggests that modern embedding models are robust to simple length-based chunk routing, and simpler uniform strategies may be preferable. The project contributes a reproducible evaluation framework with provider-agnostic embeddings, statistical comparison utilities, and comprehensive efficiency metrics.

## Motivation

### Original Context (2022–2023)

This project began when GPT-3.5 (4K context) was significantly cheaper than GPT-3.5-16K. The original motivation was **cost-based model routing**: route short documents to cheaper models, reserving expensive long-context models only when necessary.

### Evolution

As context windows expanded dramatically (128K–2M tokens by 2024–2025) and pricing structures changed, the cost arbitrage diminished. The project evolved to test whether **adaptive chunking** could improve retrieval quality by:

1. Preserving semantic coherence in short documents (no chunking)
2. Using larger chunks for medium documents (fewer boundary artifacts)
3. Applying fine-grained chunking only to very long documents

## Method

### Routing Strategy

Documents are classified by token count into three categories:

| Category | Token Range | Chunking Strategy |
|----------|-------------|-------------------|
| Short | ≤3,500 | None (full document embedded) |
| Medium | 3,500–15,000 | 2,000-token chunks |
| Long | >15,000 | 1,000-token chunks |

The uniform baseline uses 1,000-token chunks for all documents regardless of length.

### Infrastructure

- **Provider-agnostic embeddings**: OpenAI, OpenRouter, or local models with automatic provider selection
- **Vector store**: ChromaDB with batched indexing for large corpora
- **Evaluation framework**: YAML-driven configs, efficiency metrics, artifact logging

## Evaluation

### Dataset

Mixed corpus of 12 documents spanning three length categories:
- **Short** (3): Gettysburg Address, The Gift of the Magi, The Cask of Amontillado
- **Medium** (1): The Yellow Wallpaper
- **Long** (8): IETF RFCs (822, 5322, 8446, 9110, 9112, 9113, 9595), A Scandal in Bohemia

Total: 493,423 tokens, 60 queries, 3 evaluation runs.

Additional benchmark slices:
- **Expanded mixed corpus** (`data/eval-expanded`): 100 queries with graded relevance and hard negatives
- **External holdout** (`data/eval-external`): 10 held-out RFCs, 36 queries with graded relevance

### Results

| Baseline | Precision@5 | Recall@5 | Chunks | Variance |
|----------|-------------|----------|--------|----------|
| Uniform | 0.197 | 0.983 | 499 | 0 |
| Router | 0.197 | 0.983 | 490 | 0 |

**Key finding (canonical)**: Length-based routing does not beat uniform chunking. On the hosted mixed-corpus slice, both strategies are identical on precision/recall. On the expanded local matrix, router underperforms uniform across tested `k` values.

## Discussion

### Why No Improvement?

1. **Embedding robustness**: Modern models (text-embedding-3-small) appear robust to chunk boundary effects
2. **Query-document matching**: Similarity search finds relevant content regardless of chunking granularity
3. **Threshold arbitrariness**: The 3.5K/15K thresholds were based on old model context limits, not optimized for retrieval

### Value of This Work

This negative result is itself informative:
- **Simplicity wins**: Uniform chunking is equal or better in committed runs with less complexity
- **Methodology contribution**: Reproducible framework for testing RAG strategies
- **Infrastructure reusability**: Provider-agnostic embeddings and evaluation tools

## Limitations

- Primary mixed-corpus result is still a narrow 12-document corpus
- Length thresholds are heuristic, not learned
- Hosted-provider reproducibility can drift over time
- Expanded local matrix (`data/eval-expanded`) improves metric sensitivity, but remains corpus-specific
- External holdout checks transfer but remains standards-domain focused

## Future Directions

- **Cost-aware routing**: Route to different providers based on cost/quality tradeoffs
- **Hybrid search**: Combine dense and sparse retrieval (BM25)
- **Alternative strategies**: Semantic chunking, overlapping windows, hierarchical embeddings
- **Broader benchmarks**: Evaluate on standard IR datasets (NQ, TriviaQA)
