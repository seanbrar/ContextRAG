# Design Decisions

This document explains key engineering decisions in ContextRAG and their rationale.

## Length-Based Routing Thresholds

**Decision**: Use fixed thresholds (3,500 / 15,000 tokens) for routing documents.

**Rationale**:
- 3,500 tokens fits comfortably within GPT-3.5-turbo's 4K context window with room for system prompts
- 15,000 tokens is below GPT-3.5-turbo-16K's limit, allowing single-pass processing
- Simple heuristics are interpretable and debuggable vs. learned routing

**Tradeoffs**:
- Fixed thresholds don't adapt to embedding model context limits
- May not be optimal for all document types (code vs. prose)

**Future consideration**: Parameterize thresholds or learn them from data.

## Provider-Agnostic Embedding Layer

**Decision**: Support multiple embedding providers (OpenAI, OpenRouter, local/HuggingFace) with automatic fallback.

**Rationale**:
- OpenAI embeddings are high quality but require API key and incur costs
- OpenRouter provides free-tier access to various models
- Local embeddings enable offline operation and zero marginal cost
- Fallback chain ensures the system works in degraded conditions

**Implementation**:
```
OpenAI (if OPENAI_API_KEY) → OpenRouter (if OPENROUTER_API_KEY) → Local (always available)
```

**Tradeoffs**:
- Different providers produce incompatible embedding spaces
- Must rebuild index when switching providers

## Chunking Strategy

**Decision**: Adaptive chunking based on document length category.

| Category | Token Range | Chunk Size | Rationale |
|----------|-------------|------------|-----------|
| Short | ≤3,500 | None (full doc) | Preserves semantic coherence |
| Medium | 3,500–15,000 | 2,000 tokens | Balances context with chunk count |
| Long | >15,000 | 1,000 tokens | Aggressive chunking for very long docs |

**Rationale**:
- Short documents lose context when chunked unnecessarily
- Medium documents benefit from larger chunks that preserve paragraph-level semantics
- Long documents require chunking regardless; smaller chunks improve retrieval granularity

**Tradeoffs**:
- Chunk boundaries may split semantic units
- No overlap implemented (could improve retrieval at boundary regions)

## ChromaDB as Vector Store

**Decision**: Use ChromaDB for vector storage and retrieval.

**Rationale**:
- Lightweight, embeddable (no separate server process)
- Native Python integration
- Supports persistent and ephemeral collections
- HNSW index provides good recall/speed tradeoff

**Tradeoffs**:
- Not designed for massive scale (millions of vectors)
- Limited query capabilities vs. specialized vector databases

**Future consideration**: Add Pinecone/Weaviate support for production scale.

## Evaluation Framework Design

**Decision**: YAML-driven configuration with artifact logging.

**Rationale**:
- Reproducibility: configs are version-controlled
- Comparability: standardized output format across runs
- Debuggability: per-query results enable error analysis

**Artifact structure**:
```
runs/{run_name}/
├── summary.json      # Aggregate metrics
├── per_query.jsonl   # Individual query results
└── metadata.json     # Run configuration and environment
```

## Model Selection (Historical Context)

**Decision**: Default to GPT-3.5 Turbo variants (4K and 16K context).

**Context**: Initial development (2022–2023) predates widespread availability of 128K+ context models.

**Rationale at time**:
- GPT-3.5 offered best cost/performance for text understanding tasks
- 16K variant handled medium-length documents without chunking
- GPT-4 was significantly more expensive with minimal benefit for embedding tasks

**Current status**: Model selection is now configurable. The routing architecture remains relevant even with larger context windows due to:
- Cost scaling with context length
- "Lost in the middle" attention degradation
- Latency increases with longer prompts

## Testing Strategy

**Decision**: Unit tests for core functions; integration tests skip without API keys.

**Rationale**:
- Core text processing (markdown normalization, tokenization) is deterministic and testable
- API-dependent tests require credentials and incur costs
- Skipped tests clearly indicate missing dependencies

**Coverage priorities**:
1. Text normalization functions (high coverage)
2. Metric calculations (high coverage)
3. Provider integrations (mocked HTTP calls)
4. End-to-end pipeline (manual verification)
