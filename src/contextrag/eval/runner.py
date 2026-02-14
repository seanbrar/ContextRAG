"""Evaluation runner for RAG retrieval experiments."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from chromaroute import VectorStore, build_embedding_function

from contextrag.chunking import get_strategy
from contextrag.config import Config, load_config
from contextrag.core.constants import (LONG_CHUNK_TOKENS, MEDIUM_CHUNK_TOKENS,
                                       TOKENIZER_NAME, UNIFORM_CHUNK_TOKENS)
from contextrag.core.costs import get_embedding_cost_per_million
from contextrag.core.tokenizer import get_encoding
from contextrag.eval.metrics import (hit_at_k, ndcg_at_k, precision_at_k,
                                     recall_at_k, reciprocal_rank_at_k,
                                     unique_doc_ratio_at_k,
                                     unique_preserve_order)
from contextrag.eval.query_schema import load_and_validate_queries
from contextrag.eval.retrieval import (build_lexical_index, rank_bm25,
                                       rank_hybrid_rrf,
                                       rerank_by_token_overlap)

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _load_queries(path: Path) -> list[dict[str, Any]]:
    """Load queries from JSONL file."""
    return load_and_validate_queries(path)


def _chunk_tokens_with_overlap(
    tokens: list[int],
    chunk_size: int,
    overlap_tokens: int,
    encoding: Any,
) -> list[tuple[str, int]]:
    if not tokens:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    overlap = max(0, overlap_tokens)
    step = max(1, chunk_size - overlap)

    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk_tokens = tokens[start:end]
        chunks.append((encoding.decode(chunk_tokens), len(chunk_tokens)))
        if end == len(tokens):
            break
        start += step
    return chunks


def _semantic_chunks(
    content: str,
    encoding: Any,
    chunk_size: int,
    overlap_tokens: int,
) -> list[tuple[str, int]]:
    if not content.strip():
        return []

    sentences = [part.strip() for part in SENTENCE_PATTERN.split(content) if part.strip()]
    if len(sentences) <= 1:
        tokens = encoding.encode(content)
        return _chunk_tokens_with_overlap(tokens, chunk_size, overlap_tokens, encoding)

    chunks: list[str] = []
    current_sentences: list[str] = []

    def _flush_current() -> None:
        if current_sentences:
            chunks.append(" ".join(current_sentences).strip())

    for sentence in sentences:
        candidate = " ".join([*current_sentences, sentence]).strip()
        if not candidate:
            continue
        candidate_tokens = len(encoding.encode(candidate))
        if current_sentences and candidate_tokens > chunk_size:
            _flush_current()
            current_sentences = [sentence]
        else:
            current_sentences.append(sentence)

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    if overlap_tokens <= 0:
        return [(chunk, len(encoding.encode(chunk))) for chunk in chunks]

    # Apply token-level overlap over the semantic chunks by re-chunking merged text.
    tokens = encoding.encode("\n\n".join(chunks))
    return _chunk_tokens_with_overlap(tokens, chunk_size, overlap_tokens, encoding)


def _build_index_inputs(
    documents_dir: Path,
    strategy_name: str,
    uniform_chunk_tokens: int | None = None,
    chunk_overlap_tokens: int = 0,
) -> tuple[list[str], list[str], dict[str, str], dict[str, Any]]:
    """Build index inputs with efficiency tracking.

    Args:
        documents_dir: Directory containing documents.
        strategy_name: Chunking strategy name.
        uniform_chunk_tokens: Optional chunk-size override for uniform/semantic.
        chunk_overlap_tokens: Overlap tokens between adjacent chunks.

    Returns:
        Tuple of (documents, ids, chunk_to_doc, efficiency_stats).
    """
    documents: list[str] = []
    ids: list[str] = []
    chunk_to_doc: dict[str, str] = {}

    encoding = get_encoding(TOKENIZER_NAME)
    strategy = get_strategy(strategy_name) if strategy_name not in {"semantic"} else None
    effective_uniform_chunk = uniform_chunk_tokens or UNIFORM_CHUNK_TOKENS

    source_doc_count = 0
    total_source_tokens = 0
    total_indexed_tokens = 0
    docs_by_category: dict[str, int] = {"short": 0, "medium": 0, "long": 0}

    for doc_path in sorted(documents_dir.glob("*")):
        if not doc_path.is_file():
            continue

        content = doc_path.read_text(encoding="utf-8")
        doc_id = doc_path.stem
        source_doc_count += 1

        tokens = encoding.encode(content)
        token_count = len(tokens)
        total_source_tokens += token_count

        category: str | None = None
        chunks: list[tuple[str, int]]

        if strategy_name == "uniform":
            chunks = _chunk_tokens_with_overlap(
                tokens,
                effective_uniform_chunk,
                chunk_overlap_tokens,
                encoding,
            )
        elif strategy_name == "semantic":
            chunks = _semantic_chunks(
                content,
                encoding,
                effective_uniform_chunk,
                chunk_overlap_tokens,
            )
        else:
            assert strategy is not None
            result = strategy(content, encoding)
            category = result.category
            if category:
                docs_by_category[category] += 1

            if chunk_overlap_tokens > 0 and category in {"medium", "long"}:
                chunk_size = MEDIUM_CHUNK_TOKENS if category == "medium" else LONG_CHUNK_TOKENS
                chunks = _chunk_tokens_with_overlap(tokens, chunk_size, chunk_overlap_tokens, encoding)
            else:
                chunks = result.chunks

        if len(chunks) == 1 and category == "short":
            chunk_text, chunk_tokens = chunks[0]
            documents.append(chunk_text)
            ids.append(doc_id)
            chunk_to_doc[doc_id] = doc_id
            total_indexed_tokens += chunk_tokens
            continue

        for idx, (chunk_text, chunk_tokens) in enumerate(chunks):
            chunk_id = f"{doc_id}::chunk{idx}"
            documents.append(chunk_text)
            ids.append(chunk_id)
            chunk_to_doc[chunk_id] = doc_id
            total_indexed_tokens += chunk_tokens

    efficiency_stats: dict[str, Any] = {
        "source_documents": source_doc_count,
        "total_chunks": len(documents),
        "total_source_tokens": total_source_tokens,
        "total_indexed_tokens": total_indexed_tokens,
        "avg_chunk_size_tokens": total_indexed_tokens / len(documents) if documents else 0,
        "chunking_overhead": (total_indexed_tokens / total_source_tokens - 1) * 100
        if total_source_tokens > 0
        else 0,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "uniform_chunk_tokens": effective_uniform_chunk,
    }

    if strategy_name in ("adaptive", "router"):
        efficiency_stats["documents_by_category"] = docs_by_category

    return documents, ids, chunk_to_doc, efficiency_stats


def _calculate_cost_metrics(
    model: str,
    indexed_tokens: int,
    query_tokens: int,
) -> dict[str, Any]:
    """Calculate embedding cost metrics."""
    total_tokens = indexed_tokens + query_tokens
    cost_per_million = get_embedding_cost_per_million(model)

    if cost_per_million is not None:
        index_cost = (indexed_tokens / 1_000_000) * cost_per_million
        query_cost = (query_tokens / 1_000_000) * cost_per_million
        return {
            "model_cost_per_million_tokens": cost_per_million,
            "total_query_tokens": query_tokens,
            "total_embedding_tokens": total_tokens,
            "index_cost_usd": round(index_cost, 6),
            "query_cost_usd": round(query_cost, 6),
            "total_cost_usd": round(index_cost + query_cost, 6),
        }

    return {
        "model_cost_per_million_tokens": None,
        "total_query_tokens": query_tokens,
        "total_embedding_tokens": total_tokens,
        "index_cost_usd": None,
        "query_cost_usd": None,
        "total_cost_usd": None,
        "note": f"Unknown pricing for model '{model}'",
    }


def run_eval(
    dataset_path: Path,
    baseline: str,
    k: int,
    persist_path: str | None = None,
    embed_provider: str | None = None,
    embedding_model: str | None = None,
    config: Config | None = None,
    retrieval_mode: str = "dense",
    uniform_chunk_tokens: int | None = None,
    chunk_overlap_tokens: int = 0,
    retrieval_candidates: int = 50,
) -> dict[str, Any]:
    """Run retrieval evaluation.

    Args:
        dataset_path: Path to dataset directory containing documents/ and queries.jsonl.
        baseline: Chunking strategy.
        k: Number of results to retrieve per query.
        persist_path: Optional path for persistent ChromaDB storage.
        embed_provider: Optional embedding provider override.
        embedding_model: Optional embedding model override.
        config: Optional Config. If None, loads from environment.
        retrieval_mode: dense | bm25 | hybrid | dense-rerank.
        uniform_chunk_tokens: Optional chunk-size override for uniform/semantic.
        chunk_overlap_tokens: Overlap tokens between chunks.
        retrieval_candidates: Candidate pool size for hybrid/rerank.

    Returns:
        Evaluation results dictionary with summary and per-query metrics.
    """
    documents_dir = dataset_path / "documents"
    queries_path = dataset_path / "queries.jsonl"

    if not documents_dir.exists():
        raise FileNotFoundError(f"Missing documents directory: {documents_dir}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    supported_retrieval_modes = {"dense", "bm25", "hybrid", "dense-rerank"}
    if retrieval_mode not in supported_retrieval_modes:
        raise ValueError(
            f"Unknown retrieval_mode '{retrieval_mode}'. "
            f"Expected one of: {', '.join(sorted(supported_retrieval_modes))}."
        )

    build_start = time.time()
    documents, ids, chunk_to_doc, efficiency_stats = _build_index_inputs(
        documents_dir,
        baseline,
        uniform_chunk_tokens=uniform_chunk_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    build_duration = time.time() - build_start

    lexical_index = None
    lexical_index_duration = 0.0
    if retrieval_mode in {"bm25", "hybrid", "dense-rerank"}:
        lexical_index_start = time.time()
        lexical_index = build_lexical_index(
            documents=documents,
            ids=ids,
            include_token_sets=retrieval_mode == "dense-rerank",
        )
        lexical_index_duration = time.time() - lexical_index_start

    vector_store: VectorStore | None = None
    cfg: Config | None = None
    resolved_provider = "none"
    resolved_model = "none"
    index_duration = 0.0

    if retrieval_mode != "bm25":
        cfg = config or load_config()
        index_start = time.time()
        embedding_fn = build_embedding_function(
            config=cfg.embed,
            embedding_model=embedding_model,
            embed_provider=embed_provider,
        )
        vector_store = VectorStore(
            collection_name=f"eval-{int(time.time())}",
            persist_path=persist_path,
            embedding_function=embedding_fn,
        )
        vector_store.add_documents(documents=documents, ids=ids)
        index_duration = time.time() - index_start

        resolved_provider = cfg.resolve_embed_provider(embed_provider)
        resolved_model = embedding_model or cfg.embed.resolve_model(resolved_provider)

    queries = _load_queries(queries_path)
    encoding = get_encoding(TOKENIZER_NAME)
    total_query_tokens = 0

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    hit_scores: list[float] = []
    hit_at_1_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    unique_doc_ratios: list[float] = []
    query_latencies: list[float] = []
    per_query: list[dict[str, Any]] = []

    candidate_count = max(k, retrieval_candidates)

    for entry in queries:
        query_text = entry["query"]
        relevant_ids = entry.get("relevant_ids", [])
        relevant_scores = entry.get("relevant_scores")
        total_query_tokens += len(encoding.encode(query_text))

        query_start = time.time()

        if retrieval_mode == "dense":
            assert vector_store is not None
            dense_results = vector_store.query(query_texts=[query_text], n_results=k)
            retrieved_chunk_ids = dense_results.get("ids", [[]])[0]
        elif retrieval_mode == "bm25":
            assert lexical_index is not None
            retrieved_chunk_ids = rank_bm25(lexical_index, query_text, k)
        elif retrieval_mode == "hybrid":
            assert vector_store is not None
            assert lexical_index is not None
            dense_results = vector_store.query(
                query_texts=[query_text],
                n_results=candidate_count,
            )
            dense_ids = dense_results.get("ids", [[]])[0]
            lexical_ids = rank_bm25(lexical_index, query_text, candidate_count)
            retrieved_chunk_ids = rank_hybrid_rrf(dense_ids, lexical_ids, k)
        else:
            assert vector_store is not None
            assert lexical_index is not None
            dense_results = vector_store.query(
                query_texts=[query_text],
                n_results=candidate_count,
            )
            dense_ids = dense_results.get("ids", [[]])[0]
            retrieved_chunk_ids = rerank_by_token_overlap(
                query=query_text,
                candidate_ids=dense_ids,
                token_sets=lexical_index.token_sets,
                n_results=k,
            )

        query_latency = time.time() - query_start
        query_latencies.append(query_latency)

        retrieved_ids = [chunk_to_doc.get(item, item) for item in retrieved_chunk_ids]
        unique_retrieved_ids = unique_preserve_order(retrieved_ids, k)

        precision = precision_at_k(retrieved_ids, relevant_ids, k)
        recall = recall_at_k(retrieved_ids, relevant_ids, k)
        hit = hit_at_k(retrieved_ids, relevant_ids, k)
        hit_at_1 = hit_at_k(retrieved_ids, relevant_ids, 1)
        reciprocal_rank = reciprocal_rank_at_k(unique_retrieved_ids, relevant_ids, k)
        ndcg = ndcg_at_k(unique_retrieved_ids, relevant_ids, k, relevant_scores)
        unique_doc_ratio = unique_doc_ratio_at_k(retrieved_ids, k)
        precision_scores.append(precision)
        recall_scores.append(recall)
        hit_scores.append(hit)
        hit_at_1_scores.append(hit_at_1)
        reciprocal_ranks.append(reciprocal_rank)
        ndcg_scores.append(ndcg)
        unique_doc_ratios.append(unique_doc_ratio)

        per_query.append(
            {
                "query": query_text,
                "relevant_ids": relevant_ids,
                **({"relevant_scores": relevant_scores} if relevant_scores else {}),
                "retrieved_ids": retrieved_ids,
                "retrieved_ids_unique": unique_retrieved_ids,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "hit_at_k": hit,
                "hit_at_1": hit_at_1,
                "reciprocal_rank_at_k": reciprocal_rank,
                "ndcg_at_k": ndcg,
                "unique_doc_ratio_at_k": unique_doc_ratio,
                "latency_ms": round(query_latency * 1000, 2),
            }
        )

    avg_query_latency = sum(query_latencies) / len(query_latencies) if query_latencies else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_hit_at_k = sum(hit_scores) / len(hit_scores) if hit_scores else 0.0
    avg_hit_at_1 = sum(hit_at_1_scores) / len(hit_at_1_scores) if hit_at_1_scores else 0.0
    avg_mrr_at_k = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    avg_ndcg_at_k = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    avg_unique_doc_ratio = (
        sum(unique_doc_ratios) / len(unique_doc_ratios) if unique_doc_ratios else 0.0
    )

    if retrieval_mode == "bm25":
        cost_metrics: dict[str, Any] = {
            "model_cost_per_million_tokens": 0.0,
            "total_query_tokens": total_query_tokens,
            "total_embedding_tokens": 0,
            "index_cost_usd": 0.0,
            "query_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "note": "No embedding model used in bm25 mode.",
        }
    else:
        cost_metrics = _calculate_cost_metrics(
            resolved_model,
            efficiency_stats["total_indexed_tokens"],
            total_query_tokens,
        )

    summary = {
        "timestamp": int(time.time()),
        "baseline": baseline,
        "retrieval_mode": retrieval_mode,
        "k": k,
        "total_queries": len(queries),
        "indexed_chunks": len(documents),
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "hit_at_k": avg_hit_at_k,
        "hit_at_1": avg_hit_at_1,
        "mrr_at_k": avg_mrr_at_k,
        "ndcg_at_k": avg_ndcg_at_k,
        "unique_doc_ratio_at_k": avg_unique_doc_ratio,
        "embedding_provider": resolved_provider,
        "embedding_model": resolved_model,
        "chunking": {
            "uniform_chunk_tokens": efficiency_stats["uniform_chunk_tokens"],
            "medium_chunk_tokens": MEDIUM_CHUNK_TOKENS,
            "long_chunk_tokens": LONG_CHUNK_TOKENS,
            "chunk_overlap_tokens": chunk_overlap_tokens,
        },
        "retrieval": {
            "mode": retrieval_mode,
            "candidate_pool": candidate_count,
        },
        "efficiency": {
            "source_documents": efficiency_stats["source_documents"],
            "total_chunks": efficiency_stats["total_chunks"],
            "total_source_tokens": efficiency_stats["total_source_tokens"],
            "total_indexed_tokens": efficiency_stats["total_indexed_tokens"],
            "avg_chunk_size_tokens": round(efficiency_stats["avg_chunk_size_tokens"], 1),
            "chunking_overhead_pct": round(efficiency_stats["chunking_overhead"], 2),
            **(
                {"documents_by_category": efficiency_stats["documents_by_category"]}
                if "documents_by_category" in efficiency_stats
                else {}
            ),
        },
        "timing": {
            "build_duration_sec": round(build_duration, 3),
            "index_duration_sec": round(index_duration, 3),
            "lexical_index_duration_sec": round(lexical_index_duration, 3),
            "avg_query_latency_ms": round(avg_query_latency * 1000, 2),
            "total_query_duration_sec": round(sum(query_latencies), 3),
        },
        "cost": cost_metrics,
    }

    return {"summary": summary, "per_query": per_query}
