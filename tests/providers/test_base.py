import pytest

from contextrag.providers.base import ChatProvider


class DummyChatProvider(ChatProvider):
    def complete(self, messages, model=None, temperature=0):
        return ChatProvider.complete(self, messages, model, temperature)


def test_chat_provider_base_raises():
    provider = DummyChatProvider()
    with pytest.raises(NotImplementedError):
        provider.complete([{"role": "user", "content": "hi"}])
