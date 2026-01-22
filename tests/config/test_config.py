import os

from contextrag.config import AppConfig, resolve_embed_provider


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
