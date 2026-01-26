import pytest

from contextrag.chunking.strategies import (_chunk_tokens, adaptive,
                                            get_strategy, list_strategies,
                                            uniform)
from contextrag.core.tokenizer import get_encoding


@pytest.fixture
def encoding():
    return get_encoding("cl100k_base")


def test_chunk_tokens_empty(encoding):
    assert _chunk_tokens([], 100, encoding) == []


def test_get_strategy_valid():
    assert get_strategy("uniform") == uniform
    assert get_strategy("adaptive") == adaptive
    assert get_strategy("router") == adaptive


def test_get_strategy_invalid():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_strategy("nonexistent")


def test_list_strategies():
    strategies = list_strategies()
    assert "uniform" in strategies
    assert "adaptive" in strategies
    assert "router" in strategies
    assert len(strategies) == 3


def test_uniform_strategy(encoding):
    # Short text
    text = "one two three"
    result = uniform(text, encoding)
    assert len(result.chunks) == 1
    assert result.chunks[0][0] == text
    assert result.category is None

    # Long text doesn't change category in uniform, just chunks it
    # We won't simulate a massive text here, just verify structure
    
    
def test_adaptive_categories(encoding):
    # Short
    short_text = "word " * 10
    result = adaptive(short_text, encoding)
    assert result.category == "short"
    assert len(result.chunks) == 1
    
    # We can mock the token counts effectively by just checking logic paths, 
    # but since we rely on tiktoken, we can force specific lengths if needed.
    # For now, this covers the basic wiring.
