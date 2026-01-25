"""Embedding function factory with provider selection support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chromaroute import build_embedding_function as chromaroute_build

if TYPE_CHECKING:
    from chromadb.api.types import EmbeddingFunction

    from contextrag.config import AppConfig


def build_embedding_function(
    config: AppConfig | None = None,
    embedding_model: str | None = None,
    embed_provider: str | None = None,
) -> EmbeddingFunction[Any]:
    """Build a ChromaDB-compatible embedding function via chromaroute.

    Args:
        config: Optional ContextRAG AppConfig. If None, uses chromaroute defaults.
        embedding_model: Optional model name override.
        embed_provider: Optional provider override ("openrouter", "local", "auto").

    Returns:
        A ChromaDB-compatible EmbeddingFunction instance.
    """
    if config is None:
        return chromaroute_build(
            embedding_model=embedding_model,
            embed_provider=embed_provider,
        )

    return chromaroute_build(
        config=config.embed_config,
        embedding_model=embedding_model,
        embed_provider=embed_provider,
    )
