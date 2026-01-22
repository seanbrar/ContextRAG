import tiktoken


def count_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken."""
    if not isinstance(text, str):
        raise TypeError(f"Expected a string, but received {type(text).__name__}")

    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
