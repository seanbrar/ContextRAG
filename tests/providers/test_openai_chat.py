from chromaroute import EmbedConfig

from contextrag.config import Config
from contextrag.providers.factory import (OpenAIChatProvider,
                                          build_chat_provider)


def test_openai_chat_provider_complete(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = self
            self.completions = self

        def create(self, model, messages, temperature):
            class Choice:
                message = type("Message", (), {"content": "ok"})()

            return type("Response", (), {"choices": [Choice()]})()

    monkeypatch.setattr("contextrag.providers.factory.OpenAI", lambda **_: FakeClient())
    provider = OpenAIChatProvider(model="gpt-4o-mini", api_key="key")
    content = provider.complete([{"role": "user", "content": "hi"}])
    assert content == "ok"


def test_build_chat_provider_auto_selects_openai():
    config = Config(
        embed=EmbedConfig(
            openrouter_api_key=None,
            openrouter_base_url="https://openrouter.ai/api/v1",
            embed_provider="auto",
            openrouter_embeddings_model="qwen/qwen3-embedding-8b",
            local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
            openrouter_referer=None,
            openrouter_title=None,
            openrouter_provider_json=None,
        ),
        openai_api_key="key",
        chat_provider="auto",
        openai_chat_model="gpt-4o-mini",
        openrouter_chat_model="mistralai/devstral-2512:free",
    )
    provider = build_chat_provider(config=config)
    assert isinstance(provider, OpenAIChatProvider)
    assert provider.model == config.openai_chat_model
