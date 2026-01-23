from contextrag.core.constants import TOKENIZER_NAME
from contextrag.core.tokenizer import get_encoding


def chunk_text_by_words(text: str, chunk_words: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_words <= 0:
        return [" ".join(words)]
    chunks: list[str] = []
    step = max(chunk_words - overlap, 1)
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
    return chunks


def chunk_text_by_tokens(
    text: str,
    chunk_tokens: int,
    tokenizer_name: str = TOKENIZER_NAME,
) -> list[str]:
    encoding = get_encoding(tokenizer_name)
    tokens = encoding.encode(text)
    if not tokens:
        return []
    chunks: list[str] = []
    for idx in range(0, len(tokens), chunk_tokens):
        chunk_tokens_slice = tokens[idx : idx + chunk_tokens]
        chunks.append(encoding.decode(chunk_tokens_slice))
    return chunks
