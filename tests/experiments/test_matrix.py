from pathlib import Path

from contextrag.experiments import matrix as matrix_module


def test_run_matrix_writes_summary_and_comparisons(monkeypatch, tmp_path: Path):
    def fake_run_eval(
        dataset_path,
        baseline,
        k,
        persist_path=None,
        embed_provider=None,
        embedding_model=None,
        config=None,
    ):
        precision = 0.1 if baseline == "uniform" else 0.2
        return {
            "summary": {
                "baseline": baseline,
                "k": k,
                "precision_at_k": precision,
                "recall_at_k": 0.9,
                "hit_at_1": precision,
                "mrr_at_k": precision,
                "ndcg_at_k": precision,
                "embedding_provider": "local",
                "embedding_model": "fake-model",
            },
            "per_query": [
                {
                    "query": "q1",
                    "relevant_ids": ["doc1"],
                    "retrieved_ids": ["doc1"] if baseline == "router" else ["doc2"],
                    "retrieved_ids_unique": ["doc1"] if baseline == "router" else ["doc2"],
                    "precision_at_k": precision,
                    "recall_at_k": 1.0 if baseline == "router" else 0.0,
                    "hit_at_k": 1.0 if baseline == "router" else 0.0,
                    "hit_at_1": 1.0 if baseline == "router" else 0.0,
                    "reciprocal_rank_at_k": 1.0 if baseline == "router" else 0.0,
                    "ndcg_at_k": 1.0 if baseline == "router" else 0.0,
                    "unique_doc_ratio_at_k": 1.0,
                }
            ],
        }

    monkeypatch.setattr(matrix_module, "run_eval", fake_run_eval)

    summary = matrix_module.run_matrix(
        dataset_path=tmp_path,
        baselines=["uniform", "router"],
        k_values=[3, 5],
        run_root=tmp_path / "runs",
        persist_root=tmp_path / "persist",
        embed_provider="local",
        embedding_model="fake-model",
    )

    assert len(summary["rows"]) == 4
    assert len(summary["comparisons"]) == 2
    assert (tmp_path / "runs" / "matrix_summary.json").exists()
    assert (tmp_path / "runs" / "matrix_summary.md").exists()
