from __future__ import annotations

import math


def unique_preserve_order(values: list[str], k: int | None = None) -> list[str]:
    """Return unique values in original order, optionally truncated to k inputs."""
    limited = values if k is None else values[:k]
    seen: set[str] = set()
    unique: list[str] = []
    for value in limited:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k <= 0 or not relevant_ids:
        return 0.0
    return 1.0 if set(retrieved_ids[:k]) & set(relevant_ids) else 0.0


def reciprocal_rank_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k <= 0 or not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Compute nDCG@k with binary relevance labels."""
    if k <= 0 or not relevant_ids:
        return 0.0

    relevant = set(relevant_ids)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(k, len(relevant))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def unique_doc_ratio_at_k(retrieved_ids: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    unique_count = len(unique_preserve_order(retrieved_ids, k))
    return unique_count / k
