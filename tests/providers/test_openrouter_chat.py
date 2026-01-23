import pytest

from contextrag.config import AppConfig
from contextrag.providers.openrouter_chat import OpenRouterChatProvider


def test_openrouter_chat_provider_requires_key(monkeypatch):
    config = AppConfig(
        openai_api_key=None,
        openai_embeddings_model="text-embedding-3-small",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="openai",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    )
    monkeypatch.setattr("contextrag.providers.openrouter_chat.load_config", lambda: config)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterChatProvider()


def test_openrouter_chat_provider_complete(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = self
            self.completions = self

        def create(self, model, messages, temperature):
            class Choice:
                message = type("Message", (), {"content": "ok"})()

            return type("Response", (), {"choices": [Choice()]})()

    config = AppConfig(
        openai_api_key=None,
        openai_embeddings_model="text-embedding-3-small",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key="key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="openai",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    )
    monkeypatch.setattr("contextrag.providers.openrouter_chat.load_config", lambda: config)
    provider = OpenRouterChatProvider(client=FakeClient())
    content = provider.complete([{"role": "user", "content": "hi"}])
    assert content == "ok"
