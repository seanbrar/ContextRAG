"""Retrieval helpers for dense, lexical, hybrid, and rerank modes."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize_for_lexical(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class LexicalIndex:
    ids: list[str]
    term_frequencies: list[dict[str, int]]
    doc_lengths: list[int]
    document_frequency: dict[str, int]
    avg_doc_length: float
    postings: dict[str, list[int]]
    token_sets: dict[str, set[str]]


def build_lexical_index(
    documents: list[str],
    ids: list[str],
    include_token_sets: bool = True,
) -> LexicalIndex:
    term_frequencies: list[dict[str, int]] = []
    doc_lengths: list[int] = []
    document_frequency: dict[str, int] = {}
    postings: dict[str, list[int]] = {}
    token_sets: dict[str, set[str]] = {} if include_token_sets else {}

    for doc_idx, (doc_id, content) in enumerate(zip(ids, documents, strict=True)):
        tokens = tokenize_for_lexical(content)
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        term_frequencies.append(tf)
        doc_lengths.append(len(tokens))
        unique_tokens = set(tokens)
        if include_token_sets:
            token_sets[doc_id] = unique_tokens
        for token in unique_tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1
            postings.setdefault(token, []).append(doc_idx)

    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    return LexicalIndex(
        ids=ids,
        term_frequencies=term_frequencies,
        doc_lengths=doc_lengths,
        document_frequency=document_frequency,
        avg_doc_length=avg_doc_length,
        postings=postings,
        token_sets=token_sets,
    )


def _bm25_score(
    *,
    query_tokens: list[str],
    term_frequency: dict[str, int],
    doc_length: int,
    num_documents: int,
    document_frequency: dict[str, int],
    avg_doc_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not query_tokens or doc_length == 0 or num_documents == 0:
        return 0.0

    score = 0.0
    avg_len = avg_doc_length if avg_doc_length > 0 else 1.0
    for token in query_tokens:
        df = document_frequency.get(token, 0)
        if df == 0:
            continue
        tf = term_frequency.get(token, 0)
        if tf == 0:
            continue
        idf = math.log(1 + (num_documents - df + 0.5) / (df + 0.5))
        denom = tf + k1 * (1 - b + b * doc_length / avg_len)
        score += idf * ((tf * (k1 + 1)) / denom)
    return score


def rank_bm25(index: LexicalIndex, query: str, n_results: int) -> list[str]:
    query_tokens = tokenize_for_lexical(query)
    if n_results <= 0:
        return []
    if not query_tokens:
        return index.ids[:n_results]

    candidate_indices: set[int] = set()
    for token in set(query_tokens):
        candidate_indices.update(index.postings.get(token, []))

    if not candidate_indices:
        return index.ids[:n_results]

    scored: list[tuple[int, float]] = []
    n_docs = len(index.ids)
    for doc_idx in candidate_indices:
        tf = index.term_frequencies[doc_idx]
        doc_len = index.doc_lengths[doc_idx]
        score = _bm25_score(
            query_tokens=query_tokens,
            term_frequency=tf,
            doc_length=doc_len,
            num_documents=n_docs,
            document_frequency=index.document_frequency,
            avg_doc_length=index.avg_doc_length,
        )
        scored.append((doc_idx, score))
    scored.sort(key=lambda item: item[1], reverse=True)

    ranked_doc_indices = [doc_idx for doc_idx, _ in scored]
    if len(ranked_doc_indices) < n_results:
        seen = set(ranked_doc_indices)
        for doc_idx in range(n_docs):
            if doc_idx in seen:
                continue
            ranked_doc_indices.append(doc_idx)
            if len(ranked_doc_indices) >= n_results:
                break

    return [index.ids[doc_idx] for doc_idx in ranked_doc_indices[:n_results]]


def rank_hybrid_rrf(
    dense_ids: list[str],
    lexical_ids: list[str],
    n_results: int,
    rrf_k: int = 60,
) -> list[str]:
    if n_results <= 0:
        return []
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, doc_id in enumerate(lexical_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:n_results]]


def rerank_by_token_overlap(
    *,
    query: str,
    candidate_ids: list[str],
    token_sets: dict[str, set[str]],
    n_results: int,
) -> list[str]:
    query_tokens = set(tokenize_for_lexical(query))
    if not query_tokens:
        return candidate_ids[:n_results]
    scored: list[tuple[str, float, int]] = []
    for rank, doc_id in enumerate(candidate_ids):
        doc_tokens = token_sets.get(doc_id, set())
        overlap = len(query_tokens & doc_tokens)
        norm = math.sqrt(len(doc_tokens) + 1)
        score = overlap / norm if norm > 0 else 0.0
        scored.append((doc_id, score, rank))
    scored.sort(key=lambda item: (item[1], -item[2]), reverse=True)
    return [doc_id for doc_id, _, _ in scored[:n_results]]
