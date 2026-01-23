import tiktoken

from contextrag.core.constants import TOKENIZER_NAME

_ENCODINGS: dict[str, tiktoken.Encoding] = {}


def get_encoding(tokenizer_name: str = TOKENIZER_NAME) -> tiktoken.Encoding:
    if tokenizer_name not in _ENCODINGS:
        _ENCODINGS[tokenizer_name] = tiktoken.get_encoding(tokenizer_name)
    return _ENCODINGS[tokenizer_name]


def count_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken."""
    if not isinstance(text, str):
        raise TypeError(f"Expected a string, but received {type(text).__name__}")

    encoding = get_encoding()
    return len(encoding.encode(text))
