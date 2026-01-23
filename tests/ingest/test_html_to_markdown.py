from pathlib import Path

from contextrag.ingest.html_to_markdown import HTMLToMarkdownConverter


def test_remove_html_footer():
    html = "<html><body><div id='footer' role='contentinfo'>bye</div></body></html>"
    converter = HTMLToMarkdownConverter(".")
    cleaned = converter._remove_html_footer(html)
    assert "footer" not in cleaned


def test_get_target_folder_uses_token_counts(monkeypatch, tmp_path):
    converter = HTMLToMarkdownConverter(str(tmp_path))
    monkeypatch.setattr(
        "contextrag.ingest.html_to_markdown.count_tokens", lambda text: 100
    )
    assert converter._get_target_folder("x") == tmp_path / "short"

    monkeypatch.setattr(
        "contextrag.ingest.html_to_markdown.count_tokens", lambda text: 4000
    )
    assert converter._get_target_folder("x") == tmp_path / "medium"

    monkeypatch.setattr(
        "contextrag.ingest.html_to_markdown.count_tokens", lambda text: 20000
    )
    assert converter._get_target_folder("x") == tmp_path / "long"


def test_convert_all_files_writes_markdown(monkeypatch, tmp_path):
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    html_path = html_dir / "doc.html"
    html_path.write_text("<p>Hello</p>", encoding="utf-8")

    monkeypatch.setattr(
        "contextrag.ingest.html_to_markdown.count_tokens", lambda text: 100
    )

    converter = HTMLToMarkdownConverter(str(html_dir))
    converter.convert_all_files(use_target_folder=True)
    output_path = html_dir / "short" / "doc.md"
    assert output_path.exists()


def test_read_html_file_handles_errors(monkeypatch, tmp_path):
    converter = HTMLToMarkdownConverter(str(tmp_path))
    target = tmp_path / "missing.html"

    def fake_read_text(self, encoding="utf-8"):
        raise FileNotFoundError("nope")

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    assert converter._read_html_file(target) is None


def test_write_markdown_file_handles_errors(monkeypatch, tmp_path):
    converter = HTMLToMarkdownConverter(str(tmp_path))
    target = tmp_path / "out.md"

    def fake_write_text(self, content, encoding="utf-8"):
        raise PermissionError("nope")

    monkeypatch.setattr(Path, "write_text", fake_write_text)
    converter._write_markdown_file(target, "content")
    assert not target.exists()
