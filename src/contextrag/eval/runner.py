from __future__ import annotations

import json
import time
from pathlib import Path

import tiktoken

from contextrag.config import load_config, resolve_embed_provider
from contextrag.eval.metrics import precision_at_k, recall_at_k
from contextrag.index.vector_store import VectorDB

SHORT_MAX = 3500
MEDIUM_MAX = 15000
UNIFORM_CHUNK_TOKENS = 1000
MEDIUM_CHUNK_TOKENS = 2000
LONG_CHUNK_TOKENS = 1000
TOKENIZER_NAME = "cl100k_base"


def _load_queries(path: Path) -> list[dict]:
    queries: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def _chunk_text(text: str, chunk_tokens: int) -> list[str]:
    encoding = tiktoken.get_encoding(TOKENIZER_NAME)
    tokens = encoding.encode(text)
    if not tokens:
        return []
    chunks = []
    for idx in range(0, len(tokens), chunk_tokens):
        chunk_tokens_slice = tokens[idx : idx + chunk_tokens]
        chunks.append(encoding.decode(chunk_tokens_slice))
    return chunks


def _build_index_inputs(
    documents_dir: Path, baseline: str
) -> tuple[list[str], list[str], dict[str, str]]:
    documents: list[str] = []
    ids: list[str] = []
    chunk_to_doc: dict[str, str] = {}

    for doc_path in sorted(documents_dir.glob("*")):
        if not doc_path.is_file():
            continue
        content = doc_path.read_text(encoding="utf-8")
        doc_id = doc_path.stem
        if baseline == "uniform":
            chunks = _chunk_text(content, UNIFORM_CHUNK_TOKENS)
            if not chunks:
                continue
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}::chunk{idx}"
                documents.append(chunk)
                ids.append(chunk_id)
                chunk_to_doc[chunk_id] = doc_id
        else:
            token_count = len(tiktoken.get_encoding(TOKENIZER_NAME).encode(content))
            if token_count <= SHORT_MAX:
                documents.append(content)
                ids.append(doc_id)
                chunk_to_doc[doc_id] = doc_id
            elif token_count <= MEDIUM_MAX:
                chunks = _chunk_text(content, MEDIUM_CHUNK_TOKENS)
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}::chunk{idx}"
                    documents.append(chunk)
                    ids.append(chunk_id)
                    chunk_to_doc[chunk_id] = doc_id
            else:
                chunks = _chunk_text(content, LONG_CHUNK_TOKENS)
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}::chunk{idx}"
                    documents.append(chunk)
                    ids.append(chunk_id)
                    chunk_to_doc[chunk_id] = doc_id

    return documents, ids, chunk_to_doc


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

    documents, ids, chunk_to_doc = _build_index_inputs(documents_dir, baseline)

    vector_db = VectorDB(
        collection_name=f"eval-{int(time.time())}",
        persist_path=persist_path,
        embed_provider=embed_provider,
        embedding_model=embedding_model,
    )
    vector_db.add_documents(documents=documents, ids=ids)

    queries = _load_queries(queries_path)
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    per_query: list[dict] = []

    for entry in queries:
        query_text = entry["query"]
        relevant_ids = entry.get("relevant_ids", [])
        results = vector_db.query(query_texts=[query_text], n_results=k)
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
            }
        )

    config = load_config()
    resolved_provider = resolve_embed_provider(config, embed_provider)
    if resolved_provider == "openai":
        resolved_model = embedding_model or config.openai_embeddings_model
    elif resolved_provider == "openrouter":
        resolved_model = embedding_model or config.openrouter_embeddings_model
    elif resolved_provider == "local":
        resolved_model = embedding_model or config.local_embeddings_model
    else:
        resolved_model = embedding_model or "default"

    summary = {
        "timestamp": int(time.time()),
        "baseline": baseline,
        "k": k,
        "total_queries": len(queries),
        "indexed_documents": len(documents),
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
    }
    return {"summary": summary, "per_query": per_query}
