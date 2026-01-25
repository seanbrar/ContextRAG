from typing import Any

from chromaroute import EmbedConfig
from contextrag.config import AppConfig


def make_test_config(**overrides: Any) -> AppConfig:
    """Create an AppConfig with sensible test defaults.
    
    Override any field by passing it as a keyword argument.
    """
    embed_defaults: dict[str, Any] = dict(
        openrouter_api_key="test-openrouter-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_embeddings_model="openai/text-embedding-3-small",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_provider_json=None,
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        embed_provider="auto",
    )
    
    # Extract embed related overrides
    embed_keys = list(embed_defaults.keys())
    embed_overrides = {k: overrides.pop(k) for k in embed_keys if k in overrides}
    embed_defaults.update(embed_overrides)
    embed_config = EmbedConfig(**embed_defaults)
    
    app_defaults: dict[str, Any] = dict(
        openai_api_key="test-openai-key",
        openai_chat_model="gpt-4o-mini",
        openrouter_chat_model="mistralai/devstral-2512:free",
        chat_provider="auto",
        embed_config=embed_config,
    )
    app_defaults.update(overrides)
    return AppConfig(**app_defaults)
