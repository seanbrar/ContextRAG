from __future__ import annotations

import json
import time
from pathlib import Path

from contextrag.config import load_config, resolve_embed_provider
from contextrag.core.chunking import chunk_text_by_tokens
from contextrag.core.constants import (
    LONG_CHUNK_TOKENS,
    MEDIUM_CHUNK_TOKENS,
    MEDIUM_MAX_TOKENS,
    SHORT_MAX_TOKENS,
    TOKENIZER_NAME,
    UNIFORM_CHUNK_TOKENS,
)
from contextrag.core.costs import get_embedding_cost_per_million
from contextrag.core.tokenizer import get_encoding
from contextrag.eval.metrics import precision_at_k, recall_at_k
from contextrag.index.vector_store import VectorDB

def _get_embedding_cost_per_million(model: str) -> float | None:
    return get_embedding_cost_per_million(model)


def _load_queries(path: Path) -> list[dict]:
    queries: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def _chunk_text(text: str, chunk_tokens: int) -> list[str]:
    return chunk_text_by_tokens(text, chunk_tokens, TOKENIZER_NAME)


def _chunk_tokens(
    tokens: list[int], chunk_tokens: int, encoding
) -> list[tuple[str, int]]:
    if not tokens:
        return []
    chunks: list[tuple[str, int]] = []
    for idx in range(0, len(tokens), chunk_tokens):
        chunk_slice = tokens[idx : idx + chunk_tokens]
        chunks.append((encoding.decode(chunk_slice), len(chunk_slice)))
    return chunks


def _build_index_inputs(
    documents_dir: Path, baseline: str
) -> tuple[list[str], list[str], dict[str, str], dict]:
    """Build index inputs with efficiency tracking.

    Returns:
        Tuple of (documents, ids, chunk_to_doc, efficiency_stats)
    """
    documents: list[str] = []
    ids: list[str] = []
    chunk_to_doc: dict[str, str] = {}

    # Efficiency tracking
    encoding = get_encoding(TOKENIZER_NAME)
    source_doc_count = 0
    total_source_tokens = 0
    total_indexed_tokens = 0
    docs_by_category = {"short": 0, "medium": 0, "long": 0}

    for doc_path in sorted(documents_dir.glob("*")):
        if not doc_path.is_file():
            continue
        content = doc_path.read_text(encoding="utf-8")
        doc_id = doc_path.stem
        source_doc_count += 1
        token_list = encoding.encode(content)
        doc_tokens = len(token_list)
        total_source_tokens += doc_tokens

        if baseline == "uniform":
            chunks = _chunk_tokens(token_list, UNIFORM_CHUNK_TOKENS, encoding)
            if not chunks:
                continue
            for idx, (chunk, chunk_tokens) in enumerate(chunks):
                chunk_id = f"{doc_id}::chunk{idx}"
                documents.append(chunk)
                ids.append(chunk_id)
                chunk_to_doc[chunk_id] = doc_id
                total_indexed_tokens += chunk_tokens
        else:
            if doc_tokens <= SHORT_MAX_TOKENS:
                docs_by_category["short"] += 1
                documents.append(content)
                ids.append(doc_id)
                chunk_to_doc[doc_id] = doc_id
                total_indexed_tokens += doc_tokens
            elif doc_tokens <= MEDIUM_MAX_TOKENS:
                docs_by_category["medium"] += 1
                chunks = _chunk_tokens(token_list, MEDIUM_CHUNK_TOKENS, encoding)
                for idx, (chunk, chunk_tokens) in enumerate(chunks):
                    chunk_id = f"{doc_id}::chunk{idx}"
                    documents.append(chunk)
                    ids.append(chunk_id)
                    chunk_to_doc[chunk_id] = doc_id
                    total_indexed_tokens += chunk_tokens
            else:
                docs_by_category["long"] += 1
                chunks = _chunk_tokens(token_list, LONG_CHUNK_TOKENS, encoding)
                for idx, (chunk, chunk_tokens) in enumerate(chunks):
                    chunk_id = f"{doc_id}::chunk{idx}"
                    documents.append(chunk)
                    ids.append(chunk_id)
                    chunk_to_doc[chunk_id] = doc_id
                    total_indexed_tokens += chunk_tokens

    efficiency_stats = {
        "source_documents": source_doc_count,
        "total_chunks": len(documents),
        "total_source_tokens": total_source_tokens,
        "total_indexed_tokens": total_indexed_tokens,
        "avg_chunk_size_tokens": total_indexed_tokens / len(documents) if documents else 0,
        "chunking_overhead": (total_indexed_tokens / total_source_tokens - 1) * 100
            if total_source_tokens > 0 else 0,
    }

    if baseline == "router":
        efficiency_stats["documents_by_category"] = docs_by_category

    return documents, ids, chunk_to_doc, efficiency_stats


def run_eval(
    dataset_path: Path,
    baseline: str,
    k: int,
    persist_path: str | None = None,
    embed_provider: str | None = None,
    embedding_model: str | None = None,
) -> dict:
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

    # Create and populate vector index
    index_start = time.time()
    vector_db = VectorDB(
        collection_name=f"eval-{int(time.time())}",
        persist_path=persist_path,
        embed_provider=embed_provider,
        embedding_model=embedding_model,
    )
    vector_db.add_documents(documents=documents, ids=ids)
    index_duration = time.time() - index_start

    queries = _load_queries(queries_path)
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    query_latencies: list[float] = []
    per_query: list[dict] = []

    # Track query tokens for cost calculation
    encoding = get_encoding(TOKENIZER_NAME)
    total_query_tokens = 0

    for entry in queries:
        query_text = entry["query"]
        relevant_ids = entry.get("relevant_ids", [])
        total_query_tokens += len(encoding.encode(query_text))

        query_start = time.time()
        results = vector_db.query(query_texts=[query_text], n_results=k)
        query_latency = time.time() - query_start
        query_latencies.append(query_latency)

        retrieved_chunk_ids = results.get("ids", [[]])[0]
        retrieved_ids = [chunk_to_doc.get(item, item) for item in retrieved_chunk_ids]
        precision = precision_at_k(retrieved_ids, relevant_ids, k)
        recall = recall_at_k(retrieved_ids, relevant_ids, k)
        precision_scores.append(precision)
        recall_scores.append(recall)
        per_query.append(
            {
                "query": query_text,
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "latency_ms": round(query_latency * 1000, 2),
            }
        )

    config = load_config()
    resolved_provider = resolve_embed_provider(config, embed_provider)
    if embedding_model:
        resolved_model = embedding_model
    elif resolved_provider == "openrouter":
        resolved_model = config.openrouter_embeddings_model
    else:
        resolved_model = config.local_embeddings_model

    avg_query_latency = (
        sum(query_latencies) / len(query_latencies) if query_latencies else 0.0
    )

    # Calculate embedding costs
    cost_per_million = _get_embedding_cost_per_million(resolved_model)
    total_embedding_tokens = efficiency_stats["total_indexed_tokens"] + total_query_tokens
    if cost_per_million is not None:
        index_cost_usd = (efficiency_stats["total_indexed_tokens"] / 1_000_000) * cost_per_million
        query_cost_usd = (total_query_tokens / 1_000_000) * cost_per_million
        total_cost_usd = index_cost_usd + query_cost_usd
        cost_metrics = {
            "model_cost_per_million_tokens": cost_per_million,
            "total_query_tokens": total_query_tokens,
            "total_embedding_tokens": total_embedding_tokens,
            "index_cost_usd": round(index_cost_usd, 6),
            "query_cost_usd": round(query_cost_usd, 6),
            "total_cost_usd": round(total_cost_usd, 6),
        }
    else:
        cost_metrics = {
            "model_cost_per_million_tokens": None,
            "total_query_tokens": total_query_tokens,
            "total_embedding_tokens": total_embedding_tokens,
            "index_cost_usd": None,
            "query_cost_usd": None,
            "total_cost_usd": None,
            "note": f"Unknown pricing for model '{resolved_model}'",
        }

    summary = {
        "timestamp": int(time.time()),
        "baseline": baseline,
        "k": k,
        "total_queries": len(queries),
        "indexed_chunks": len(documents),
        "precision_at_k": sum(precision_scores) / len(precision_scores)
        if precision_scores
        else 0.0,
        "recall_at_k": sum(recall_scores) / len(recall_scores)
        if recall_scores
        else 0.0,
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
        "cost": cost_metrics,
    }
    return {"summary": summary, "per_query": per_query}
