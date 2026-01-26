from typing import Any

from chromaroute import EmbedConfig

from contextrag.config import Config


def make_test_config(**overrides: Any) -> Config:
    """Create a Config with sensible test defaults.
    
    Override any field by passing it as a keyword argument.
    Embedding-related fields are automatically routed to the nested EmbedConfig.
    """
    # Embedding fields handled by chromaroute.EmbedConfig
    embed_fields = {
        "openrouter_api_key", "openrouter_base_url", "openrouter_embeddings_model",
        "openrouter_referer", "openrouter_title", "openrouter_provider_json",
        "local_embeddings_model", "embed_provider"
    }

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

    chat_defaults: dict[str, Any] = dict(
        openai_api_key="test-openai-key",
        chat_provider="auto",
        openai_chat_model="gpt-4o-mini",
        openrouter_chat_model="mistralai/devstral-2512:free",
    )

    # Separate overrides
    for key, value in overrides.items():
        if key in embed_fields:
            embed_defaults[key] = value
        elif key in chat_defaults:
            chat_defaults[key] = value
        elif key == "embed" and isinstance(value, EmbedConfig):
            # Special case for passing a pre-constructed EmbedConfig
            embed_defaults = {} # Not used if embed is passed directly below
        else:
            # Fallback for unexpected keys, try to put them in chat for now
            chat_defaults[key] = value

    embed_config = overrides.get("embed")
    if not isinstance(embed_config, EmbedConfig):
        embed_config = EmbedConfig(**embed_defaults)

    return Config(
        embed=embed_config,
        **chat_defaults
    )
