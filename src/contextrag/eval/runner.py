from __future__ import annotations

import json
import time
from pathlib import Path

from contextrag.eval.metrics import precision_at_k, recall_at_k
from contextrag.index.vector_store import VectorDB


def _load_queries(path: Path) -> list[dict]:
    queries: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def run_eval(
    dataset_path: Path,
    baseline: str,
    k: int,
    persist_path: str | None = None,
) -> dict:
    documents_dir = dataset_path / "documents"
    queries_path = dataset_path / "queries.jsonl"

    if not documents_dir.exists():
        raise FileNotFoundError(f"Missing documents directory: {documents_dir}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    documents: list[str] = []
    ids: list[str] = []
    for doc_path in sorted(documents_dir.glob("*")):
        if not doc_path.is_file():
            continue
        documents.append(doc_path.read_text(encoding="utf-8"))
        ids.append(doc_path.stem)

    vector_db = VectorDB(
        collection_name=f"eval-{int(time.time())}",
        persist_path=persist_path,
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
        retrieved_ids = results.get("ids", [[]])[0]
        precision = precision_at_k(retrieved_ids, relevant_ids, k)
        recall = recall_at_k(retrieved_ids, relevant_ids, k)
        precision_scores.append(precision)
        recall_scores.append(recall)
        per_query.append(
            {
                "query": query_text,
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "precision_at_k": precision,
                "recall_at_k": recall,
            }
        )

    summary = {
        "baseline": baseline,
        "k": k,
        "total_queries": len(queries),
        "precision_at_k": sum(precision_scores) / len(precision_scores)
        if precision_scores
        else 0.0,
        "recall_at_k": sum(recall_scores) / len(recall_scores)
        if recall_scores
        else 0.0,
    }
    return {"summary": summary, "per_query": per_query}
