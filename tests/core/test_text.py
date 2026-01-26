from contextrag.core.text import chunk_text_by_tokens, chunk_text_by_words


def test_chunk_text_by_words_empty():
    assert chunk_text_by_words("", 100, 10) == []


def test_chunk_text_by_words_basic():
    text = "one two three four five six"
    # ample chunk size
    assert chunk_text_by_words(text, 10, 0) == ["one two three four five six"]
    
    # exact split
    assert chunk_text_by_words(text, 3, 0) == ["one two three", "four five six"]
    
    # with overlap
    assert chunk_text_by_words(text, 3, 1) == [
        "one two three",
        "three four five",
        "five six",
    ]


def test_chunk_text_by_words_invalid_args():
    # 0 or negative chunk words -> return whole text as one chunk (per implementation logic)
    text = "one two"
    assert chunk_text_by_words(text, 0, 0) == ["one two"]
    assert chunk_text_by_words(text, -5, 0) == ["one two"]


def test_chunk_text_by_tokens_empty():
    assert chunk_text_by_tokens("", 10) == []


def test_chunk_text_by_tokens_basic():
    # This relies on the default cl100k_base encoding roughly
    text = "hello world"
    # Just ensure it runs and returns something non-empty for a standard string
    chunks = chunk_text_by_tokens(text, 100)
    assert len(chunks) == 1
    assert chunks[0] == text

    # Force a split
    # "hello world" is 2 tokens
    chunks = chunk_text_by_tokens(text, 1)
    assert len(chunks) == 2
    assert chunks == ["hello", " world"]
