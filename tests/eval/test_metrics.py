from contextrag.eval.metrics import precision_at_k, recall_at_k


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
