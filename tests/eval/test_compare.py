import json

from contextrag.eval.compare import compare_runs
from contextrag.core.io import write_jsonl


def test_compare_runs_detects_retrieval_changes(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()

    (run_a / "summary.json").write_text(
        json.dumps(
            {
                "baseline": "uniform",
                "k": 5,
                "embedding_provider": "local",
                "embedding_model": "m",
                "precision_at_k": 0.5,
                "recall_at_k": 1.0,
                "hit_at_k": 1.0,
                "hit_at_1": 0.5,
                "mrr_at_k": 0.75,
                "ndcg_at_k": 0.8,
                "unique_doc_ratio_at_k": 0.6,
            }
        ),
        encoding="utf-8",
    )
    (run_b / "summary.json").write_text(
        json.dumps(
            {
                "baseline": "router",
                "k": 5,
                "embedding_provider": "local",
                "embedding_model": "m",
                "precision_at_k": 0.5,
                "recall_at_k": 1.0,
                "hit_at_k": 1.0,
                "hit_at_1": 1.0,
                "mrr_at_k": 1.0,
                "ndcg_at_k": 1.0,
                "unique_doc_ratio_at_k": 0.8,
            }
        ),
        encoding="utf-8",
    )

    write_jsonl(
        run_a / "per_query.jsonl",
        [
            {
                "query": "q1",
                "relevant_ids": ["doc1"],
                "retrieved_ids": ["doc2", "doc1"],
                "retrieved_ids_unique": ["doc2", "doc1"],
                "precision_at_k": 0.5,
                "recall_at_k": 1.0,
                "hit_at_k": 1.0,
                "hit_at_1": 0.0,
                "reciprocal_rank_at_k": 0.5,
                "ndcg_at_k": 0.63,
                "unique_doc_ratio_at_k": 0.4,
            }
        ],
    )
    write_jsonl(
        run_b / "per_query.jsonl",
        [
            {
                "query": "q1",
                "relevant_ids": ["doc1"],
                "retrieved_ids": ["doc1", "doc2"],
                "retrieved_ids_unique": ["doc1", "doc2"],
                "precision_at_k": 0.5,
                "recall_at_k": 1.0,
                "hit_at_k": 1.0,
                "hit_at_1": 1.0,
                "reciprocal_rank_at_k": 1.0,
                "ndcg_at_k": 1.0,
                "unique_doc_ratio_at_k": 0.4,
            }
        ],
    )

    comparison = compare_runs(run_a, run_b)
    assert comparison["counts"]["queries_compared"] == 1
    assert comparison["counts"]["retrieved_ids_changed"] == 1
    assert comparison["counts"]["metric_values_changed"] == 1
    assert comparison["summary_metric_deltas"]["hit_at_1"] == 0.5
    assert "inference" in comparison
    assert "precision_at_k" in comparison["inference"]
    assert comparison["inference"]["precision_at_k"]["n_pairs"] == 1
