import os

import pytest

from contextrag.config import (
    AppConfig,
    require_embedding_provider,
    resolve_embed_provider,
)


def _config(
    *,
    openai_key: str | None = None,
    openrouter_key: str | None = None,
    embed_provider: str = "auto",
) -> AppConfig:
    return AppConfig(
        openai_api_key=openai_key,
        openai_embeddings_model="text-embedding-3-small",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key=openrouter_key,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        contextrag_chat_provider="openai",
        contextrag_embed_provider=embed_provider,
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    )


def test_resolve_embed_provider_auto_prefers_openai():
    config = _config(openai_key="ok", openrouter_key="ok")
    assert resolve_embed_provider(config, None) == "openai"


def test_resolve_embed_provider_auto_falls_back_to_openrouter():
    config = _config(openai_key=None, openrouter_key="ok")
    assert resolve_embed_provider(config, None) == "openrouter"


def test_resolve_embed_provider_auto_falls_back_to_local():
    config = _config(openai_key=None, openrouter_key=None)
    assert resolve_embed_provider(config, None) == "local"


def test_resolve_embed_provider_explicit_override():
    config = _config(openai_key="ok", openrouter_key="ok")
    assert resolve_embed_provider(config, "openrouter") == "openrouter"


def test_resolve_embedding_model_prefers_explicit():
    config = _config(openai_key="ok", openrouter_key="ok")
    assert (
        config.resolve_embedding_model(provider="openai", explicit_model="custom")
        == "custom"
    )


def test_resolve_embedding_model_by_provider():
    config = _config(openai_key="ok", openrouter_key="ok")
    assert config.resolve_embedding_model(provider="openai") == "text-embedding-3-small"
    assert config.resolve_embedding_model(provider="openrouter") == "qwen/qwen3-embedding-8b"
    assert (
        config.resolve_embedding_model(provider="local")
        == "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_openrouter_embed_provider_config_none():
    config = _config(openai_key="ok", openrouter_key="ok")
    assert config.openrouter_embed_provider_config() is None


def test_openrouter_embed_provider_config_order_and_fallback():
    config = _config(openai_key="ok", openrouter_key="ok")
    config = config.__class__(**{**config.__dict__,
        "openrouter_embed_provider_order": "a,b",
        "openrouter_embed_allow_fallbacks": "true",
    })
    assert config.openrouter_embed_provider_config() == {
        "order": ["a", "b"],
        "allow_fallbacks": True,
    }


def test_openrouter_embed_provider_config_invalid_json():
    config = _config(openai_key="ok", openrouter_key="ok")
    config = config.__class__(**{**config.__dict__,
        "openrouter_embed_provider_json": "{invalid}",
    })
    with pytest.raises(ValueError, match="OPENROUTER_EMBED_PROVIDER_JSON"):
        config.openrouter_embed_provider_config()


def test_require_embedding_provider_openai_errors():
    config = _config(openai_key=None, openrouter_key="ok")
    with pytest.raises(ValueError, match="--embed-provider openai"):
        require_embedding_provider(
            config,
            resolved_provider="openai",
            explicit_provider="openai",
        )


def test_require_embedding_provider_auto_openai_errors():
    config = _config(openai_key=None, openrouter_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY or OPENROUTER_API_KEY"):
        require_embedding_provider(
            config,
            resolved_provider="openai",
            explicit_provider=None,
        )


def test_require_embedding_provider_openrouter_errors():
    config = _config(openai_key="ok", openrouter_key=None)
    with pytest.raises(ValueError, match="--embed-provider openrouter"):
        require_embedding_provider(
            config,
            resolved_provider="openrouter",
            explicit_provider="openrouter",
        )
