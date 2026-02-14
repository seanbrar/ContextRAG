"""Validation utilities for query dataset schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _validate_query_text(value: Any, line_no: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_no}: 'query' must be a non-empty string.")
    return value


def _validate_relevant_ids(value: Any, line_no: int) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Line {line_no}: 'relevant_ids' must be a non-empty list.")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Line {line_no}: 'relevant_ids' must contain non-empty strings.")
    if len(set(value)) != len(value):
        raise ValueError(f"Line {line_no}: 'relevant_ids' must not contain duplicates.")
    return value


def _validate_relevant_scores(value: Any, line_no: int) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Line {line_no}: 'relevant_scores' must be a non-empty object.")

    normalized: dict[str, float] = {}
    for key, raw_score in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"Line {line_no}: 'relevant_scores' keys must be non-empty strings.")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError(
                f"Line {line_no}: relevance score for '{key}' must be a number."
            )
        score = float(raw_score)
        if score <= 0:
            raise ValueError(f"Line {line_no}: relevance score for '{key}' must be > 0.")
        normalized[key] = score
    return normalized


def _parse_relevant_array(value: Any, line_no: int) -> tuple[list[str], dict[str, float] | None]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Line {line_no}: 'relevant' must be a non-empty list.")

    relevant_ids: list[str] = []
    relevance_scores: dict[str, float] = {}
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(
                f"Line {line_no}: each item in 'relevant' must be an object."
            )
        doc_id = item.get("id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(
                f"Line {line_no}: each item in 'relevant' must include non-empty 'id'."
            )
        if doc_id in relevance_scores:
            raise ValueError(f"Line {line_no}: duplicate relevant id '{doc_id}'.")
        raw_score = item.get("score", 1.0)
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError(
                f"Line {line_no}: relevance score for '{doc_id}' must be a number."
            )
        score = float(raw_score)
        if score <= 0:
            raise ValueError(f"Line {line_no}: relevance score for '{doc_id}' must be > 0.")
        relevant_ids.append(doc_id)
        relevance_scores[doc_id] = score

    return relevant_ids, relevance_scores


def validate_query_record(payload: dict[str, Any], line_no: int) -> dict[str, Any]:
    """Validate one query record and normalize supported schema variants.

    Supported formats:
    - Legacy: {"query": "...", "relevant_ids": ["doc1", ...]}
    - v2: {"query": "...", "relevant": [{"id": "doc1", "score": 2.0}, ...]}
      Optional: "relevant_scores" map for legacy shape.
    """
    query = _validate_query_text(payload.get("query"), line_no)

    relevant_ids: list[str]
    relevance_scores: dict[str, float] | None = None

    if "relevant" in payload:
        relevant_ids, relevance_scores = _parse_relevant_array(payload["relevant"], line_no)
    else:
        relevant_ids = _validate_relevant_ids(payload.get("relevant_ids"), line_no)
        if "relevant_scores" in payload:
            relevance_scores = _validate_relevant_scores(payload["relevant_scores"], line_no)
            invalid_keys = sorted(set(relevance_scores) - set(relevant_ids))
            if invalid_keys:
                joined = ", ".join(invalid_keys)
                raise ValueError(
                    f"Line {line_no}: 'relevant_scores' keys must also appear in "
                    f"'relevant_ids' (invalid: {joined})."
                )

    normalized: dict[str, Any] = {
        "query": query,
        "relevant_ids": relevant_ids,
    }
    if relevance_scores:
        normalized["relevant_scores"] = relevance_scores
    return normalized


def load_and_validate_queries(path: Path) -> list[dict[str, Any]]:
    """Load and validate query records from JSONL."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_no}: invalid JSON ({exc.msg}).") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Line {line_no}: query record must be a JSON object.")
            rows.append(validate_query_record(parsed, line_no))
    return rows
