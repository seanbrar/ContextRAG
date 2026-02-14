import json
from pathlib import Path

from contextrag.experiments.run_logger import write_run_artifacts


def test_write_run_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    documents_path = dataset_path / "documents"
    documents_path.mkdir(parents=True)
    (documents_path / "doc1.md").write_text("Hello world", encoding="utf-8")
    (dataset_path / "queries.jsonl").write_text(
        '{"query": "hello", "relevant_documents": ["doc1"]}\n',
        encoding="utf-8",
    )

    results = {
        "summary": {"k": 5, "precision_at_k": 0.5},
        "per_query": [{"query": "q1", "precision": 0.5}],
    }
    metadata = {"dataset": str(dataset_path), "baseline": "uniform", "k": 5}

    run_dir = tmp_path / "run"
    write_run_artifacts(
        run_dir=run_dir,
        results=results,
        metadata=metadata,
        dataset_path=dataset_path,
    )

    # Verify summary.json
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["precision_at_k"] == 0.5

    # Verify per_query.jsonl
    per_query_lines = (run_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(per_query_lines[0])["query"] == "q1"

    # Verify metadata.json
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["dataset"] == str(dataset_path)
    assert meta["baseline"] == "uniform"

    # Verify manifest.json
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_schema_version"] == 1
    assert "git_commit" in manifest
    assert "config_hash" in manifest
    assert "dataset" in manifest
    assert manifest["dataset"]["status"] == "ok"
    assert "versions" in manifest
    assert "system" in manifest
