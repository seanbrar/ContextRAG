import pytest

from contextrag.core import tokenizer


class DummyEncoding:
    def __init__(self, tokens):
        self._tokens = tokens

    def encode(self, text):
        return self._tokens


def test_count_tokens_uses_tiktoken(monkeypatch):
    dummy = DummyEncoding(tokens=[1, 2, 3, 4])

    def fake_get_encoding(name):
        assert name == "cl100k_base"
        return dummy

    monkeypatch.setattr(tokenizer.tiktoken, "get_encoding", fake_get_encoding)
    assert tokenizer.count_tokens("hello") == 4


def test_count_tokens_rejects_non_string():
    with pytest.raises(TypeError, match="Expected a string"):
        tokenizer.count_tokens(123)
