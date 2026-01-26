import json

from contextrag.core import io, text


def test_iter_files_filters_and_sorts(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "c.json").write_text("c", encoding="utf-8")
    files = io.iter_files(tmp_path, [".md", ".txt"])
    assert [path.name for path in files] == ["a.md", "b.txt"]


def test_write_jsonl_roundtrip(tmp_path):
    output = tmp_path / "rows.jsonl"
    rows = [{"a": 1}, {"b": 2}]
    io.write_jsonl(output, rows)
    parsed = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert parsed == rows


def test_checksum_stable():
    assert io.checksum("hello") == io.checksum("hello")
    assert io.checksum("hello") != io.checksum("hello!")


def test_chunk_words_handles_overlap_and_empty():
    assert text.chunk_text_by_words("", chunk_words=3, overlap=1) == []
    assert text.chunk_text_by_words("one two three", chunk_words=0, overlap=2) == [
        "one two three"
    ]
    chunks = text.chunk_text_by_words("one two three four", chunk_words=2, overlap=1)
    assert chunks == ["one two", "two three", "three four"]
