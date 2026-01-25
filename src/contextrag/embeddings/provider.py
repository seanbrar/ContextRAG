"""Embedding function factory with provider selection support."""

from __future__ import annotations

from chromadb.api.types import EmbeddingFunction
from chromaroute import EmbedConfig, build_embedding_function as chromaroute_build

from contextrag.config import AppConfig, require_embedding_provider


def build_embedding_function(
    config: AppConfig | None = None,
    embedding_model: str | None = None,
    embed_provider: str | None = None,
) -> EmbeddingFunction:
    """Build a ChromaDB-compatible embedding function via chromaroute.

    Args:
        config: Optional ContextRAG AppConfig. If None, uses chromaroute defaults.
        embedding_model: Optional model name override.
        embed_provider: Optional provider override ("openrouter", "local", "auto").

    Returns:
        A ChromaDB-compatible EmbeddingFunction instance.

    Raises:
        ValueError: If a provider requires an API key that isn't configured.
    """
    if config is None:
        return chromaroute_build(
            embedding_model=embedding_model,
            embed_provider=embed_provider,
        )

    resolved_provider = config.resolve_embed_provider(embed_provider)
    require_embedding_provider(
        config,
        resolved_provider,
        explicit_provider=embed_provider,
    )

    # Validate provider JSON if present
    if config.openrouter_embed_provider_json:
        config.openrouter_embed_provider_config()

    # Build chromaroute config directly (thread-safe, no env mutation)
    embed_config = EmbedConfig(
        openrouter_api_key=config.openrouter_api_key,
        openrouter_base_url=config.openrouter_base_url,
        openrouter_embeddings_model=config.openrouter_embeddings_model,
        openrouter_referer=config.openrouter_referer,
        openrouter_title=config.openrouter_title,
        openrouter_provider_json=config.openrouter_embed_provider_json,
        local_embeddings_model=config.local_embeddings_model,
        embed_provider=resolved_provider,
    )

    return chromaroute_build(
        config=embed_config,
        embedding_model=embedding_model,
        embed_provider=embed_provider,
    )
