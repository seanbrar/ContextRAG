import sys

import numpy as np
import pytest

from contextrag.core import cache, io
from contextrag.embeddings import similarity


def test_preprocess_text_strips_attachments_and_formatting():
    raw = (
        "# Header\n"
        "Some text\n"
        "![img](attachments/foo.png)\n"
        "\n"
        "## Attachments:\n"
        "extra\n"
        "[link](https://example.com)\n"
    )
    cleaned = similarity.preprocess_text(raw)
    assert "Attachments" not in cleaned
    assert "attachments/foo.png" not in cleaned
    assert "Header" not in cleaned
    assert "link" not in cleaned


def test_compute_similarity_uses_cache_and_openai(monkeypatch):
    files = {"a.md": "alpha", "b.md": "beta"}
    checksums = {
        "a.md": io.checksum("alpha"),
        "b.md": io.checksum("beta"),
    }
    cache = {checksums["a.md"]: [1.0, 0.0]}
    calls = {"count": 0}

    class FakeEmbeddings:
        def create(self, model, input, encoding_format, dimensions=None):
            calls["count"] += 1

            class Response:
                data = [type("Item", (), {"embedding": [0.0, 1.0]})()]

            return Response()

    class FakeClient:
        def __init__(self):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(similarity, "OpenAI", lambda: FakeClient())
    monkeypatch.setattr(similarity, "count_tokens", lambda text: 1)
    matrix = similarity.compute_similarity(files, checksums, cache)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)
    assert calls["count"] == 1


def test_load_embeddings_cache_missing(tmp_path):
    loaded = cache.load_json_cache(tmp_path / "missing.json")
    assert loaded == {}


def test_update_embeddings_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "cache.json"
    data = {"a": [1.0, 2.0]}
    cache.save_json_cache(cache_path, data)
    loaded = cache.load_json_cache(cache_path)
    assert loaded == data


def test_read_markdown_files_and_checksums(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    files, checksums = similarity.read_markdown_files(tmp_path)
    assert list(files.keys()) == ["a.md"]
    assert checksums["a.md"] == io.checksum("alpha")


def test_group_similar_files_threshold():
    matrix = np.array(
        [
            [1.0, 0.7, 0.2],
            [0.7, 1.0, 0.8],
            [0.2, 0.8, 1.0],
        ]
    )
    groups = similarity.group_similar_files(matrix, threshold=0.75)
    assert groups == {1: [2]}


def test_print_file_groupings_writes_file(tmp_path):
    files = {"a.md": "a", "b.md": "b"}
    groups = {0: [1]}
    output_path = tmp_path / "out.txt"
    similarity.print_file_groupings(files, groups, output_file=output_path)
    content = output_path.read_text(encoding="utf-8")
    assert "a.md" in content
    assert "b.md" in content


def test_parse_arguments_reads_cli_args(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--debug", "--output", "out.txt", "docs"],
    )
    args = similarity.parse_arguments()
    assert args.debug is True
    assert args.output == "out.txt"
    assert args.folder_path == "docs"


def test_count_tokens_rejects_non_string():
    with pytest.raises(TypeError, match="Expected a string"):
        similarity.count_tokens(123)


def test_compute_similarity_skips_large_documents(monkeypatch, capsys):
    files = {"a.md": "alpha", "b.md": "beta"}
    checksums = {
        "a.md": io.checksum("alpha"),
        "b.md": io.checksum("beta"),
    }
    cache = {checksums["a.md"]: [1.0, 0.0]}

    class FakeEmbeddings:
        def create(self, *args, **kwargs):
            raise AssertionError("Should not be called for large docs")

    class FakeClient:
        def __init__(self):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(similarity, "OpenAI", lambda: FakeClient())
    monkeypatch.setattr(similarity, "count_tokens", lambda text: 9001)

    matrix = similarity.compute_similarity(files, checksums, cache)
    assert matrix.shape == (1, 1)
    assert "Skipped b.md" in capsys.readouterr().out


def test_print_file_groupings_stdout(capsys):
    files = {"a.md": "a", "b.md": "b"}
    groups = {0: [1]}
    similarity.print_file_groupings(files, groups, output_file=None)
    stdout = capsys.readouterr().out
    assert "a.md" in stdout
    assert "b.md" in stdout


def test_main_flow_uses_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "embeddings_cache.json"
    cache_path.write_text("{}", encoding="utf-8")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("alpha", encoding="utf-8")

    monkeypatch.setattr(similarity, "load_json_cache", lambda _: {})
    monkeypatch.setattr(similarity, "save_json_cache", lambda *_: None)
    monkeypatch.setattr(similarity, "compute_similarity", lambda *_: np.eye(1))

    similarity.main(str(docs_dir), debug=False, output_file=str(tmp_path / "out.txt"))
