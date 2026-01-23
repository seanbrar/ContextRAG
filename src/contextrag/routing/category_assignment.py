from __future__ import annotations

from contextrag.legacy import category_assignment as _legacy

ChatManager = _legacy.ChatManager
ChatModels = _legacy.ChatModels
count_tokens = _legacy.count_tokens
datetime = _legacy.datetime
preprocess_similarity_text = _legacy.preprocess_similarity_text
read_markdown_files = _legacy.read_markdown_files

__all__ = [
    "main",
    "read_markdown_files",
    "ChatManager",
    "ChatModels",
    "count_tokens",
    "datetime",
    "preprocess_similarity_text",
]


def main():
    _legacy.read_markdown_files = read_markdown_files
    _legacy.count_tokens = count_tokens
    _legacy.preprocess_similarity_text = preprocess_similarity_text
    _legacy.ChatManager = ChatManager
    _legacy.ChatModels = ChatModels
    _legacy.datetime = datetime
    return _legacy.main()


if __name__ == "__main__":
    main()
