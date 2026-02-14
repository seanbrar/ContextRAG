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

**Decision**: Use `chromaroute` for provider-agnostic embeddings with OpenRouter + local fallback.

**Rationale**:
- OpenRouter provides access to multiple hosted embedding models with routing support
- Local embeddings enable offline operation and zero marginal cost
- Provider selection ensures a working default with minimal setup

**Implementation**:
```
Auto provider selection with explicit overrides:
- OpenRouter when `OPENROUTER_API_KEY` is present
- Local fallback for offline/default operation
```
`chromaroute` handles provider routing and failure fallback.

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

**Evaluation outcome**: Testing showed no accuracy improvement over uniform chunking. On hosted mixed-corpus runs the strategies tied on precision/recall, while on the expanded local matrix router underperformed uniform across tested `k` values. This suggests modern embedding models are robust to simple length-based routing and that extra routing complexity is not justified here. See `docs/results.md` for full analysis.

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

## Configuration Architecture

**Decision**: Use composition where `Config` contains `chromaroute.EmbedConfig`.

**Rationale**:
- Single source of truth for embedding configuration
- Clean delegation to chromaroute for all embedding operations
- Avoids field duplication between ContextRAG and chromaroute configs

**Structure**:
```python
@dataclass(frozen=True)
class Config:
    # Embeddings (delegated to chromaroute)
    embed: EmbedConfig

    # Chat-specific (ContextRAG)
    openai_api_key: str | None
    chat_provider: str
    openai_chat_model: str
    openrouter_chat_model: str
```

**Tradeoffs**:
- Requires chromaroute as a dependency (acceptable since it's our extracted library)

## Chat Providers (Future Work)

**Decision**: Preserve chat provider abstractions for planned semantic chunking research.

**Context**: The `providers/` module contains `ChatProvider` abstractions for OpenAI and OpenRouter. These were originally used for dataset creation and document categorization. Current evaluation uses embedding-only retrieval.

**Future direction**: Semantic chunking research.

**Hypothesis**: LLM-guided semantic boundaries may improve retrieval quality compared to fixed-token chunking.

**Approach**:
```
Document → LLM identifies semantic breaks → Chunk at boundaries → Embed → Evaluate
```

This extends the current evaluation framework to test whether intelligent segmentation outperforms the null result found with adaptive token-based chunking. The chat providers enable this research path without requiring new infrastructure.

**Status**: Preserved, not currently active in CLI. Will be integrated when semantic chunking experiments begin.
