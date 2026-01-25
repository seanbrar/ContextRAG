import pytest

from chromaroute import EmbedConfig
from contextrag.config import (AppConfig, require_embedding_provider,
                               resolve_embed_provider)


def _config(
    *,
    openai_key: str | None = None,
    openrouter_key: str | None = None,
    embed_provider: str = "auto",
    chat_provider: str = "openai",
) -> AppConfig:
    embed_config = EmbedConfig(
        openrouter_api_key=openrouter_key,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_provider_json=None,
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        embed_provider=embed_provider,
    )
    return AppConfig(
        openai_api_key=openai_key,
        openai_chat_model="gpt-4o-mini",
        openrouter_chat_model="mistralai/devstral-2512:free",
        chat_provider=chat_provider,
        embed_config=embed_config,
    )


def test_resolve_embed_provider_auto_falls_back_to_openrouter():
    config = _config(openrouter_key="ok")
    assert resolve_embed_provider(config, None) == "openrouter"


def test_resolve_embed_provider_auto_falls_back_to_local():
    config = _config(openai_key=None, openrouter_key=None)
    assert resolve_embed_provider(config, None) == "local"


def test_resolve_embed_provider_explicit_override():
    config = _config(openrouter_key="ok")
    assert resolve_embed_provider(config, "openrouter") == "openrouter"




def test_openrouter_embed_provider_config_none():
    config = _config(openrouter_key="ok")
    assert config.embed_config.openrouter_provider_config() is None


def test_openrouter_embed_provider_config_invalid_json():
    config = _config(openrouter_key="ok")
    # Manually recreate config with invalid JSON string
    embed_config = config.embed_config
    new_embed_config = EmbedConfig(
        openrouter_api_key=embed_config.openrouter_api_key,
        openrouter_base_url=embed_config.openrouter_base_url,
        openrouter_embeddings_model=embed_config.openrouter_embeddings_model,
        openrouter_referer=embed_config.openrouter_referer,
        openrouter_title=embed_config.openrouter_title,
        openrouter_provider_json="{invalid}",
        local_embeddings_model=embed_config.local_embeddings_model,
        embed_provider=embed_config.embed_provider,
    )
    config = AppConfig(
        openai_api_key=config.openai_api_key,
        openai_chat_model=config.openai_chat_model,
        openrouter_chat_model=config.openrouter_chat_model,
        chat_provider=config.chat_provider,
        embed_config=new_embed_config,
    )
    with pytest.raises(ValueError, match="OPENROUTER_EMBED_PROVIDER_JSON"):
        config.embed_config.openrouter_provider_config()




def test_require_embedding_provider_openrouter_errors():
    config = _config(openai_key=None, openrouter_key=None)
    with pytest.raises(ValueError, match="--embed-provider openrouter"):
        require_embedding_provider(
            config,
            resolved_provider="openrouter",
            explicit_provider="openrouter",
        )


def test_resolve_chat_provider_auto_prefers_openai_key():
    config = _config(openai_key="ok", openrouter_key="alt", chat_provider="auto")
    assert config.resolve_chat_provider() == "openai"


def test_resolve_chat_provider_auto_falls_back_to_openrouter():
    config = _config(openai_key=None, openrouter_key="ok", chat_provider="auto")
    assert config.resolve_chat_provider() == "openrouter"


def test_resolve_chat_provider_auto_requires_key():
    config = _config(openai_key=None, openrouter_key=None, chat_provider="auto")
    with pytest.raises(ValueError, match="No API key available for chat"):
        config.resolve_chat_provider()


def test_require_embedding_provider_openrouter_implicit_error():
    config = _config(openai_key=None, openrouter_key=None)
    with pytest.raises(ValueError, match="OpenRouter embeddings"):
        require_embedding_provider(config, resolved_provider="openrouter")
