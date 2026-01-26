"""Pluggable chunking strategies for document processing.

This module provides a registry of chunking strategies that can be easily extended.
Each strategy takes document content and returns chunks with token counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tiktoken import Encoding

from contextrag.core.constants import (LONG_CHUNK_TOKENS, MEDIUM_CHUNK_TOKENS,
                                       MEDIUM_MAX_TOKENS, SHORT_MAX_TOKENS,
                                       TOKENIZER_NAME, UNIFORM_CHUNK_TOKENS)
from contextrag.core.tokenizer import get_encoding


@dataclass
class ChunkResult:
    """Result of chunking a document.

    Attributes:
        chunks: List of (chunk_text, token_count) tuples.
        source_tokens: Total tokens in original document (avoids re-encoding).
        category: Document category for adaptive strategies (short/medium/long).
    """

    chunks: list[tuple[str, int]]
    source_tokens: int
    category: str | None = None


# Strategy function signature
ChunkingStrategy = Callable[[str, Encoding], ChunkResult]


def _chunk_tokens(
    tokens: list[int], chunk_size: int, encoding: Encoding
) -> list[tuple[str, int]]:
    """Split tokens into chunks of specified size.

    Args:
        tokens: List of token IDs.
        chunk_size: Maximum tokens per chunk.
        encoding: Tiktoken encoding for decoding.

    Returns:
        List of (chunk_text, token_count) tuples.
    """
    if not tokens:
        return []
    chunks: list[tuple[str, int]] = []
    for idx in range(0, len(tokens), chunk_size):
        chunk_slice = tokens[idx : idx + chunk_size]
        chunks.append((encoding.decode(chunk_slice), len(chunk_slice)))
    return chunks


def uniform(content: str, encoding: Encoding) -> ChunkResult:
    """Uniform chunking strategy.

    All documents are split into fixed-size chunks regardless of length.
    Uses UNIFORM_CHUNK_TOKENS (default: 1000 tokens).

    Args:
        content: Document text.
        encoding: Tiktoken encoding.

    Returns:
        ChunkResult with uniform chunks.
    """
    tokens = encoding.encode(content)
    chunks = _chunk_tokens(tokens, UNIFORM_CHUNK_TOKENS, encoding)
    return ChunkResult(chunks=chunks, source_tokens=len(tokens), category=None)


def adaptive(content: str, encoding: Encoding) -> ChunkResult:
    """Adaptive chunking strategy based on document length.

    Routes documents to different chunk sizes:
    - Short (<=3500 tokens): No chunking, embed full document
    - Medium (3501-15000 tokens): 2000-token chunks
    - Long (>15000 tokens): 1000-token chunks

    Args:
        content: Document text.
        encoding: Tiktoken encoding.

    Returns:
        ChunkResult with category-appropriate chunks.
    """
    tokens = encoding.encode(content)
    token_count = len(tokens)

    if token_count <= SHORT_MAX_TOKENS:
        # Short documents: keep whole
        return ChunkResult(
            chunks=[(content, token_count)],
            source_tokens=token_count,
            category="short",
        )
    elif token_count <= MEDIUM_MAX_TOKENS:
        # Medium documents: larger chunks
        chunks = _chunk_tokens(tokens, MEDIUM_CHUNK_TOKENS, encoding)
        return ChunkResult(chunks=chunks, source_tokens=token_count, category="medium")
    else:
        # Long documents: smaller chunks
        chunks = _chunk_tokens(tokens, LONG_CHUNK_TOKENS, encoding)
        return ChunkResult(chunks=chunks, source_tokens=token_count, category="long")


# Strategy registry - easy to extend with new strategies
_STRATEGIES: dict[str, ChunkingStrategy] = {
    "uniform": uniform,
    "adaptive": adaptive,
    "router": adaptive,  # Alias for backwards compatibility
}


def get_strategy(name: str) -> ChunkingStrategy:
    """Get a chunking strategy by name.

    Args:
        name: Strategy name ("uniform", "adaptive", or "router").

    Returns:
        The chunking strategy function.

    Raises:
        ValueError: If strategy name is unknown.
    """
    if name not in _STRATEGIES:
        available = ", ".join(_STRATEGIES)
        raise ValueError(f"Unknown chunking strategy: {name!r}. Available: {available}")
    return _STRATEGIES[name]


def list_strategies() -> list[str]:
    """List available strategy names."""
    return list(_STRATEGIES.keys())


def chunk_document(
    content: str,
    strategy: str = "uniform",
    tokenizer_name: str = TOKENIZER_NAME,
) -> ChunkResult:
    """Chunk a document using the specified strategy.

    Convenience function that handles encoding creation.

    Args:
        content: Document text.
        strategy: Strategy name ("uniform", "adaptive", or "router").
        tokenizer_name: Tiktoken encoding name.

    Returns:
        ChunkResult with chunks and optional category.
    """
    encoding = get_encoding(tokenizer_name)
    strategy_fn = get_strategy(strategy)
    return strategy_fn(content, encoding)
