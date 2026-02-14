import pytest
from chromaroute import EmbedConfig

from contextrag.config import Config


def _config(
    *,
    openai_key: str | None = None,
    openrouter_key: str | None = None,
    embed_provider: str = "auto",
    chat_provider: str = "openai",
    openrouter_provider_json: str | None = None,
) -> Config:
    embed = EmbedConfig(
        openrouter_api_key=openrouter_key,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_provider_json=openrouter_provider_json,
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        embed_provider=embed_provider,
    )
    return Config(
        embed=embed,
        openai_api_key=openai_key,
        chat_provider=chat_provider,
        openai_chat_model="gpt-4o-mini",
        openrouter_chat_model="mistralai/devstral-2512:free",
    )


def test_resolve_embed_provider_auto_falls_back_to_openrouter():
    config = _config(openrouter_key="ok")
    assert config.resolve_embed_provider() == "openrouter"


def test_resolve_embed_provider_auto_falls_back_to_local():
    config = _config(openai_key=None, openrouter_key=None)
    assert config.resolve_embed_provider() == "local"


def test_resolve_embed_provider_explicit_override():
    config = _config(openrouter_key="ok")
    assert config.resolve_embed_provider("openrouter") == "openrouter"




def test_openrouter_embed_provider_config_none():
    config = _config(openrouter_key="ok")
    assert config.embed.openrouter_provider_config() is None


def test_openrouter_embed_provider_config_invalid_json():
    config = _config(openrouter_key="ok", openrouter_provider_json="{invalid}")
    with pytest.raises(ValueError, match="OPENROUTER_EMBED_PROVIDER_JSON"):
        config.embed.openrouter_provider_config()




def test_require_embedding_provider_openrouter_errors():
    config = _config(openai_key=None, openrouter_key=None)
    with pytest.raises(ValueError, match="--embed-provider openrouter"):
        config.require_embed_provider(
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
        config.require_embed_provider(resolved_provider="openrouter")


def test_require_embedding_provider_openai_errors_when_explicit():
    config = _config(openai_key=None, openrouter_key="ok")
    with pytest.raises(ValueError, match="--embed-provider openai"):
        config.require_embed_provider(
            resolved_provider="openai",
            explicit_provider="openai",
        )


def test_require_embedding_provider_openai_implicit_error():
    config = _config(openai_key=None, openrouter_key="ok")
    with pytest.raises(ValueError, match="OpenAI embeddings"):
        config.require_embed_provider(resolved_provider="openai")
