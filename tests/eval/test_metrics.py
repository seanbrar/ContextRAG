from contextrag.eval.metrics import precision_at_k, recall_at_k


def test_precision_at_k_basic():
    retrieved = ["a", "b", "c"]
    relevant = ["b"]
    assert precision_at_k(retrieved, relevant, 2) == 0.5


def test_recall_at_k_basic():
    retrieved = ["a", "b", "c"]
    relevant = ["b", "d"]
    assert recall_at_k(retrieved, relevant, 3) == 0.5
