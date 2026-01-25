import pytest

from chromaroute import EmbedConfig
from contextrag.config import AppConfig
from contextrag.providers.openrouter_chat import OpenRouterChatProvider


def test_openrouter_chat_provider_requires_key(monkeypatch):
    embed_config = EmbedConfig(
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_provider_json=None,
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        embed_provider="auto",
    )
    config = AppConfig(
        openai_api_key=None,
        openai_chat_model="gpt-4o-mini",
        openrouter_chat_model="mistralai/devstral-2512:free",
        chat_provider="openai",
        embed_config=embed_config,
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

    embed_config = EmbedConfig(
        openrouter_api_key="key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_provider_json=None,
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        embed_provider="auto",
    )
    config = AppConfig(
        openai_api_key=None,
        openai_chat_model="gpt-4o-mini",
        openrouter_chat_model="mistralai/devstral-2512:free",
        chat_provider="openai",
        embed_config=embed_config,
    )
    monkeypatch.setattr("contextrag.providers.openrouter_chat.load_config", lambda: config)
    provider = OpenRouterChatProvider(client=FakeClient())
    content = provider.complete([{"role": "user", "content": "hi"}])
    assert content == "ok"
