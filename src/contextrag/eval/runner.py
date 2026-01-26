"""Evaluation runner for RAG retrieval experiments."""

from __future__ import annotations

import json
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
from contextrag.eval.metrics import precision_at_k, recall_at_k


def _load_queries(path: Path) -> list[dict[str, Any]]:
    """Load queries from JSONL file."""
    queries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def _build_index_inputs(
    documents_dir: Path, strategy_name: str
) -> tuple[list[str], list[str], dict[str, str], dict[str, Any]]:
    """Build index inputs with efficiency tracking.

    Args:
        documents_dir: Directory containing documents.
        strategy_name: Chunking strategy name ("uniform", "adaptive", "router").

    Returns:
        Tuple of (documents, ids, chunk_to_doc, efficiency_stats).
    """
    documents: list[str] = []
    ids: list[str] = []
    chunk_to_doc: dict[str, str] = {}

    # Get chunking strategy and encoding
    encoding = get_encoding(TOKENIZER_NAME)
    strategy = get_strategy(strategy_name)

    # Efficiency tracking
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

        # Apply chunking strategy (includes source token count)
        result = strategy(content, encoding)
        total_source_tokens += result.source_tokens

        # Track category if present (adaptive strategy)
        if result.category:
            docs_by_category[result.category] += 1

        # Add chunks to index
        if len(result.chunks) == 1 and result.category == "short":
            # Short document: use doc_id directly
            chunk_text, token_count = result.chunks[0]
            documents.append(chunk_text)
            ids.append(doc_id)
            chunk_to_doc[doc_id] = doc_id
            total_indexed_tokens += token_count
        else:
            # Multiple chunks: add index suffix
            for idx, (chunk_text, token_count) in enumerate(result.chunks):
                chunk_id = f"{doc_id}::chunk{idx}"
                documents.append(chunk_text)
                ids.append(chunk_id)
                chunk_to_doc[chunk_id] = doc_id
                total_indexed_tokens += token_count

    efficiency_stats: dict[str, Any] = {
        "source_documents": source_doc_count,
        "total_chunks": len(documents),
        "total_source_tokens": total_source_tokens,
        "total_indexed_tokens": total_indexed_tokens,
        "avg_chunk_size_tokens": total_indexed_tokens / len(documents) if documents else 0,
        "chunking_overhead": (total_indexed_tokens / total_source_tokens - 1) * 100
        if total_source_tokens > 0
        else 0,
    }

    # Include category breakdown for adaptive strategies
    if strategy_name in ("adaptive", "router"):
        efficiency_stats["documents_by_category"] = docs_by_category

    return documents, ids, chunk_to_doc, efficiency_stats


def _calculate_cost_metrics(
    model: str,
    indexed_tokens: int,
    query_tokens: int,
) -> dict[str, Any]:
    """Calculate embedding cost metrics.

    Args:
        model: Embedding model name.
        indexed_tokens: Total tokens indexed.
        query_tokens: Total tokens in queries.

    Returns:
        Cost metrics dictionary.
    """
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
) -> dict[str, Any]:
    """Run retrieval evaluation.

    Args:
        dataset_path: Path to dataset directory containing documents/ and queries.jsonl.
        baseline: Chunking strategy ("uniform", "adaptive", or "router").
        k: Number of results to retrieve per query.
        persist_path: Optional path for persistent ChromaDB storage.
        embed_provider: Optional embedding provider override.
        embedding_model: Optional embedding model override.
        config: Optional Config. If None, loads from environment.

    Returns:
        Evaluation results dictionary with summary and per-query metrics.
    """
    documents_dir = dataset_path / "documents"
    queries_path = dataset_path / "queries.jsonl"

    if not documents_dir.exists():
        raise FileNotFoundError(f"Missing documents directory: {documents_dir}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    # Build index with efficiency tracking
    build_start = time.time()
    documents, ids, chunk_to_doc, efficiency_stats = _build_index_inputs(
        documents_dir, baseline
    )
    build_duration = time.time() - build_start

    # Load config and create embedding function
    cfg = config or load_config()
    embed_config = cfg.to_embed_config()

    index_start = time.time()
    embedding_fn = build_embedding_function(
        config=embed_config,
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

    # Load and run queries
    queries = _load_queries(queries_path)
    encoding = get_encoding(TOKENIZER_NAME)
    total_query_tokens = 0

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    query_latencies: list[float] = []
    per_query: list[dict[str, Any]] = []

    for entry in queries:
        query_text = entry["query"]
        relevant_ids = entry.get("relevant_ids", [])
        total_query_tokens += len(encoding.encode(query_text))

        query_start = time.time()
        results = vector_store.query(query_texts=[query_text], n_results=k)
        query_latency = time.time() - query_start
        query_latencies.append(query_latency)

        retrieved_chunk_ids = results.get("ids", [[]])[0]
        retrieved_ids = [chunk_to_doc.get(item, item) for item in retrieved_chunk_ids]

        precision = precision_at_k(retrieved_ids, relevant_ids, k)
        recall = recall_at_k(retrieved_ids, relevant_ids, k)
        precision_scores.append(precision)
        recall_scores.append(recall)

        per_query.append({
            "query": query_text,
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "latency_ms": round(query_latency * 1000, 2),
        })

    # Resolve final provider and model
    resolved_provider = cfg.resolve_embed_provider(embed_provider)
    resolved_model = embedding_model or embed_config.resolve_model(resolved_provider)

    # Calculate aggregate metrics
    avg_query_latency = sum(query_latencies) / len(query_latencies) if query_latencies else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    summary = {
        "timestamp": int(time.time()),
        "baseline": baseline,
        "k": k,
        "total_queries": len(queries),
        "indexed_chunks": len(documents),
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "embedding_provider": resolved_provider,
        "embedding_model": resolved_model,
        "chunking": {
            "uniform_chunk_tokens": UNIFORM_CHUNK_TOKENS,
            "medium_chunk_tokens": MEDIUM_CHUNK_TOKENS,
            "long_chunk_tokens": LONG_CHUNK_TOKENS,
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
            "avg_query_latency_ms": round(avg_query_latency * 1000, 2),
            "total_query_duration_sec": round(sum(query_latencies), 3),
        },
        "cost": _calculate_cost_metrics(
            resolved_model,
            efficiency_stats["total_indexed_tokens"],
            total_query_tokens,
        ),
    }

    return {"summary": summary, "per_query": per_query}
