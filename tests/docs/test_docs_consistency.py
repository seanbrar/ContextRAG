from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_architecture_doc_has_current_artifacts():
    content = _read("docs/architecture.md")
    assert "manifest.jsonl" not in content
    assert "routing.jsonl" not in content
    assert "embeddings.jsonl" not in content
    assert "manifest.json" in content


def test_results_doc_scopes_claim():
    content = _read("docs/results.md")
    assert "in this benchmark setup" in content
    assert "not as a universal statement" in content
