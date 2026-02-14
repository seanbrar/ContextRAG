#!/usr/bin/env python3
"""Build a compact SciFact benchmark slice for ContextRAG."""

from __future__ import annotations

import csv
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "eval-scifact-mini"
CACHE_DIR = ROOT / ".cache" / "datasets"
ZIP_PATH = CACHE_DIR / "scifact.zip"
EXTRACT_DIR = CACHE_DIR / "scifact"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

QUERY_LIMIT = 40
NEGATIVE_DOCS = 180
SEED = 42


def _download_scifact() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        return
    with urllib.request.urlopen(SCIFACT_URL, timeout=120) as response:
        ZIP_PATH.write_bytes(response.read())


def _extract_scifact() -> Path:
    _download_scifact()
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(EXTRACT_DIR)
    return EXTRACT_DIR / "scifact"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    source_dir = _extract_scifact()
    corpus_rows = _load_jsonl(source_dir / "corpus.jsonl")
    query_rows = _load_jsonl(source_dir / "queries.jsonl")

    queries_by_id = {str(row["_id"]): str(row["text"]) for row in query_rows}
    corpus_by_id = {str(row["_id"]): row for row in corpus_rows}

    qrels_path = source_dir / "qrels" / "test.tsv"
    positives_by_query: dict[str, set[str]] = {}
    with qrels_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])
            score = int(row["score"])
            if score <= 0:
                continue
            positives_by_query.setdefault(query_id, set()).add(corpus_id)

    selected_query_ids = [
        query_id
        for query_id in sorted(positives_by_query)
        if query_id in queries_by_id
    ][:QUERY_LIMIT]

    positive_doc_ids: set[str] = set()
    query_records: list[dict[str, Any]] = []
    for query_id in selected_query_ids:
        relevant_ids = sorted(
            doc_id
            for doc_id in positives_by_query[query_id]
            if doc_id in corpus_by_id
        )
        if not relevant_ids:
            continue
        positive_doc_ids.update(relevant_ids)
        query_records.append(
            {
                "query": queries_by_id[query_id],
                "relevant_ids": relevant_ids,
            }
        )

    rng = random.Random(SEED)
    all_doc_ids = sorted(corpus_by_id)
    negative_candidates = [doc_id for doc_id in all_doc_ids if doc_id not in positive_doc_ids]
    rng.shuffle(negative_candidates)
    sampled_negatives = negative_candidates[:NEGATIVE_DOCS]
    final_doc_ids = sorted(positive_doc_ids | set(sampled_negatives))

    if TARGET.exists():
        shutil.rmtree(TARGET)
    docs_dir = TARGET / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    for doc_id in final_doc_ids:
        row = corpus_by_id[doc_id]
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()
        body = f"{title}\n\n{text}".strip()
        (docs_dir / f"{doc_id}.txt").write_text(body, encoding="utf-8")

    queries_path = TARGET / "queries.jsonl"
    with queries_path.open("w", encoding="utf-8") as handle:
        for row in query_records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    provenance = {
        "source": "BEIR SciFact",
        "source_url": SCIFACT_URL,
        "query_split": "test",
        "query_limit": QUERY_LIMIT,
        "negative_docs": NEGATIVE_DOCS,
        "seed": SEED,
        "selected_queries": len(query_records),
        "selected_documents": len(final_doc_ids),
    }
    (TARGET / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    readme = (
        "SciFact mini benchmark slice for ContextRAG.\n\n"
        "Source:\n"
        "- BEIR SciFact test split\n"
        "- Download URL: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip\n\n"
        "Contents:\n"
        f"- `documents/`: {len(final_doc_ids)} selected scientific abstracts\n"
        f"- `queries.jsonl`: {len(query_records)} queries with multi-relevance labels\n"
        "- `provenance.json`: build/source metadata\n"
        "- `annotations/`: dual-annotation rounds and agreement report\n\n"
        "Rebuild:\n\n"
        "```bash\n"
        "python3 scripts/build_eval_scifact_mini.py\n"
        "```\n\n"
        "Validation:\n\n"
        "```bash\n"
        "uv run contextrag validate-dataset --dataset data/eval-scifact-mini\n"
        "```\n"
    )
    (TARGET / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote {len(query_records)} queries and {len(final_doc_ids)} docs to {TARGET}")


if __name__ == "__main__":
    main()
