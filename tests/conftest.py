"""Test utilities for creating AppConfig instances."""

from contextrag.config import AppConfig


def make_test_config(**overrides) -> AppConfig:
    """Create an AppConfig with sensible test defaults.
    
    Override any field by passing it as a keyword argument.
    """
    defaults = dict(
        openai_api_key="test-openai-key",
        openai_chat_model="gpt-4o-mini",
        openrouter_api_key="test-openrouter-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        chat_provider="auto",
        embed_provider="auto",
        openrouter_embeddings_model="openai/text-embedding-3-small",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
    )
    defaults.update(overrides)
    return AppConfig(**defaults)
