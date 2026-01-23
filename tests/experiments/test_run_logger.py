import json

from contextrag.experiments.run_logger import write_run_artifacts


def test_write_run_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    results = {
        "summary": {"precision_at_k": 0.5},
        "per_query": [{"query": "q1"}],
    }
    metadata = {"dataset": "x"}

    write_run_artifacts(run_dir, results, metadata)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["precision_at_k"] == 0.5
    per_query_lines = (run_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(per_query_lines[0])["query"] == "q1"
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["dataset"] == "x"
