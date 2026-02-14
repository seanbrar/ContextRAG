import json
import subprocess
from pathlib import Path


def test_compute_annotation_agreement_script(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "doc1.txt").write_text("a", encoding="utf-8")
    (docs / "doc2.txt").write_text("b", encoding="utf-8")

    round_a = tmp_path / "a.jsonl"
    round_b = tmp_path / "b.jsonl"
    round_a.write_text(
        json.dumps({"query": "q", "relevant_ids": ["doc1"]}) + "\n",
        encoding="utf-8",
    )
    round_b.write_text(
        json.dumps({"query": "q", "relevant_ids": ["doc1", "doc2"]}) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "agreement.json"
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "compute_annotation_agreement.py"
    subprocess.run(
        [
            "python3",
            str(script),
            "--round-a",
            str(round_a),
            "--round-b",
            str(round_b),
            "--documents-dir",
            str(docs),
            "--output",
            str(output),
        ],
        check=True,
        cwd=repo_root,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    assert metrics["queries_compared"] == 1
    assert 0.0 <= metrics["mean_jaccard_relevant_ids"] <= 1.0
    assert "cohen_kappa_binary_relevance" in metrics
