import pytest

from contextrag.config import Config
from contextrag.providers.factory import (OpenRouterChatProvider,
                                          build_chat_provider)


def test_openrouter_chat_provider_requires_key():
    config = Config(
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openai_api_key=None,
        chat_provider="openrouter",
        openai_chat_model="gpt-4o-mini",
        openrouter_chat_model="mistralai/devstral-2512:free",
        embed_provider="auto",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_provider_json=None,
    )
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build_chat_provider(config=config, provider="openrouter")


def test_openrouter_chat_provider_complete(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = self
            self.completions = self

        def create(self, model, messages, temperature):
            class Choice:
                message = type("Message", (), {"content": "ok"})()

            return type("Response", (), {"choices": [Choice()]})()

    monkeypatch.setattr("contextrag.providers.factory.OpenAI", lambda **_: FakeClient())
    provider = OpenRouterChatProvider(
        model="mistralai/devstral-2512:free",
        api_key="key",
        base_url="https://openrouter.ai/api/v1",
    )
    content = provider.complete([{"role": "user", "content": "hi"}])
    assert content == "ok"
