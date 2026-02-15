# ContextRAG Evolution (2022-2025)

This document traces the project's journey from cost-based model routing to RAG evaluation tool.

## 2022-2023: The Cost Routing Era

### The Original Problem

When GPT-3.5 launched, there was a significant cost difference between context sizes:

| Model | Context | Cost (1K input) |
|-------|---------|-----------------|
| GPT-3.5-turbo (4K) | 4,096 | $0.0015 |
| GPT-3.5-turbo-16K | 16,384 | $0.003 |

For workloads with variable document sizes, this created an optimization opportunity: route short documents to the cheaper 4K model, and only use the 16K model when necessary.

### The Routing Implementation

The system classified documents into buckets:
- **Short** (< 500 tokens): Use GPT-3.5-turbo (cheapest)
- **Medium** (500-3500 tokens): Use GPT-3.5-turbo (still fits)
- **Long** (> 3500 tokens): Use GPT-3.5-turbo-16K (required)

This achieved measurable cost savings on mixed workloads.

## 2024: The Context Window Expansion

### The Landscape Changed

By late 2023/early 2024, context windows grew fast:

| Model | Context | Released |
|-------|---------|----------|
| GPT-4-turbo | 128K | Nov 2023 |
| Claude 3 | 200K | Mar 2024 |
| Gemini 1.5 | 1M+ | Feb 2024 |

The original cost-routing motivation became less compelling. With 128K+ contexts, even "long" documents fit easily.

### Pivot to Chunking

The research question shifted:

> If we're chunking for retrieval anyway, does adaptive chunk sizing improve retrieval quality?

The hypothesis was that document length should inform chunk size:
- Short documents (<1K tokens): Keep whole
- Medium documents (1K-4K): Standard chunks (2K)
- Long documents (>4K): Smaller chunks (1K)

### Evaluation Infrastructure

This pivot required proper evaluation infrastructure:
- YAML-driven experiment configs
- Reproducible benchmarking
- Comprehensive metrics (P@k, R@k, efficiency, cost)

## 2025: The Null Result

### Testing the Hypothesis

We tested adaptive chunking with reproducible CLI configs and local matrix comparisons across multiple datasets.

### The Finding

**No win for routing.** Precision@5 and Recall@5 were identical on the hosted mixed-corpus slice, and later expanded local-matrix runs showed uniform outperforming router.

This held across:
- Mixed-corpus runs with OpenAI-family embedding model `text-embedding-3-small`
- RFC-only runs with OpenRouter `qwen/qwen3-embedding-8b`
- Multiple repeated runs on the mixed corpus

### Interpretation

In this project, routing by document length did not produce better retrieval than uniform chunking. The embedding models we tested were indifferent to chunk size differences, so the routing logic just added complexity for nothing.

## The chromaroute Extraction

### Clean Separation

The embedding layer turned out to be useful on its own, so we extracted it into [chromaroute](https://github.com/seanbrar/chromaroute):

- Provider-agnostic ChromaDB embedding functions
- OpenRouter -> Local fallback chain
- Production-ready error handling

ContextRAG now depends on chromaroute for embeddings, keeping only the evaluation infrastructure.

## Lessons Learned

1. **Negative results are valuable.** The null finding simplifies production RAG: just use uniform chunking.

2. **Build evaluation early.** Without the benchmarking framework, this result would have stayed invisible.

3. **Extract reusable components.** chromaroute emerged from ContextRAG's embedding abstraction and now stands on its own.

## Current State

ContextRAG is a focused evaluation CLI:

```bash
# Primary commands
contextrag eval      # Full evaluation with configurable strategies
contextrag demo      # Offline evaluation with local embeddings
contextrag doctor    # Check configuration health
contextrag matrix    # Run baseline-by-k experiment matrix
contextrag compare   # Compare two evaluation runs

# Database operations
contextrag db index  # Build vector index
contextrag db query  # Query index
```

The routing and ingest commands were removed -- they represented historical complexity, not current value. Embedding functionality is now delegated to [chromaroute](https://github.com/seanbrar/chromaroute).
