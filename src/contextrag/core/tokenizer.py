import tiktoken

from contextrag.core.constants import TOKENIZER_NAME


def count_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken."""
    if not isinstance(text, str):
        raise TypeError(f"Expected a string, but received {type(text).__name__}")

    encoding = tiktoken.get_encoding(TOKENIZER_NAME)
    return len(encoding.encode(text))
