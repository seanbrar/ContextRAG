import pytest

from contextrag.providers.base import ChatProvider, EmbeddingsProvider


class DummyEmbeddingsProvider(EmbeddingsProvider):
    def embed(self, texts, model=None):
        return EmbeddingsProvider.embed(self, texts, model)


class DummyChatProvider(ChatProvider):
    def complete(self, messages, model=None, temperature=0):
        return ChatProvider.complete(self, messages, model, temperature)


def test_embeddings_provider_base_raises():
    provider = DummyEmbeddingsProvider()
    with pytest.raises(NotImplementedError):
        provider.embed(["a"])


def test_chat_provider_base_raises():
    provider = DummyChatProvider()
    with pytest.raises(NotImplementedError):
        provider.complete([{"role": "user", "content": "hi"}])
