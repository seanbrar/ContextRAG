import pytest

from contextrag.eval.metrics import (hit_at_k, ndcg_at_k, precision_at_k,
                                     recall_at_k, reciprocal_rank_at_k,
                                     unique_doc_ratio_at_k,
                                     unique_preserve_order)


def test_precision_at_k_basic():
    retrieved = ["a", "b", "c"]
    relevant = ["b"]
    assert precision_at_k(retrieved, relevant, 2) == 0.5


def test_recall_at_k_basic():
    retrieved = ["a", "b", "c"]
    relevant = ["b", "d"]
    assert recall_at_k(retrieved, relevant, 3) == 0.5


def test_precision_at_k_zero():
    assert precision_at_k(["a"], ["a"], 0) == 0.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k(["a"], [], 3) == 0.0


def test_unique_preserve_order():
    assert unique_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert unique_preserve_order(["a", "b", "a", "c"], k=3) == ["a", "b"]


def test_hit_at_k():
    assert hit_at_k(["a", "b"], ["z"], 2) == 0.0
    assert hit_at_k(["a", "b"], ["b"], 2) == 1.0
    assert hit_at_k(["a", "b"], ["b"], 0) == 0.0


def test_reciprocal_rank_at_k():
    assert reciprocal_rank_at_k(["x", "y", "z"], ["y"], 3) == 0.5
    assert reciprocal_rank_at_k(["x", "y", "z"], ["z"], 2) == 0.0


def test_ndcg_at_k_binary():
    score = ndcg_at_k(["a", "b", "c"], ["b", "d"], 3)
    assert score == pytest.approx(0.38685, rel=1e-4)
    assert ndcg_at_k(["a"], [], 1) == 0.0


def test_ndcg_at_k_graded():
    score = ndcg_at_k(
        ["doc_a", "doc_b"],
        ["doc_a", "doc_b"],
        2,
        {"doc_a": 1.0, "doc_b": 3.0},
    )
    assert score < 1.0


def test_unique_doc_ratio_at_k():
    assert unique_doc_ratio_at_k(["a", "a", "a", "b"], 4) == 0.5
    assert unique_doc_ratio_at_k(["a"], 0) == 0.0
