"""Run-to-run comparison utilities for retrieval evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from contextrag.eval.stats import (bootstrap_mean_ci, mean,
                                   paired_randomization_p_value)

PER_QUERY_METRIC_FIELDS = (
    "precision_at_k",
    "recall_at_k",
    "hit_at_k",
    "hit_at_1",
    "reciprocal_rank_at_k",
    "ndcg_at_k",
    "unique_doc_ratio_at_k",
)

SUMMARY_METRIC_FIELDS = (
    "precision_at_k",
    "recall_at_k",
    "hit_at_k",
    "hit_at_1",
    "mrr_at_k",
    "ndcg_at_k",
    "unique_doc_ratio_at_k",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast(dict[str, Any], payload)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_run_paths(run: Path) -> tuple[Path, Path]:
    if run.is_dir():
        summary = run / "summary.json"
        per_query = run / "per_query.jsonl"
        if not summary.exists():
            raise FileNotFoundError(f"Missing summary.json in run directory: {run}")
        if not per_query.exists():
            raise FileNotFoundError(f"Missing per_query.jsonl in run directory: {run}")
        return summary, per_query

    if run.name == "summary.json":
        per_query = run.with_name("per_query.jsonl")
        if not per_query.exists():
            raise FileNotFoundError(f"Missing per_query.jsonl next to summary: {run}")
        return run, per_query

    if run.name == "per_query.jsonl":
        summary = run.with_name("summary.json")
        if not summary.exists():
            raise FileNotFoundError(f"Missing summary.json next to per_query file: {run}")
        return summary, run

    raise ValueError(
        "Run path must be a run directory or one of summary.json/per_query.jsonl."
    )


def _key_by_query_occurrence(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Key rows by query text + occurrence order to handle duplicate query strings."""
    counts: dict[str, int] = {}
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        query = str(row.get("query", ""))
        occ = counts.get(query, 0)
        counts[query] = occ + 1
        key = f"{query}##{occ}"
        keyed[key] = row
    return keyed


def compare_runs(run_a: Path, run_b: Path) -> dict[str, Any]:
    """Compare two evaluation runs and return aggregate + per-query deltas."""
    summary_path_a, per_query_path_a = _resolve_run_paths(run_a)
    summary_path_b, per_query_path_b = _resolve_run_paths(run_b)

    summary_a = _load_json(summary_path_a)
    summary_b = _load_json(summary_path_b)
    per_query_a = _load_jsonl(per_query_path_a)
    per_query_b = _load_jsonl(per_query_path_b)

    keyed_a = _key_by_query_occurrence(per_query_a)
    keyed_b = _key_by_query_occurrence(per_query_b)
    common_keys = sorted(set(keyed_a) & set(keyed_b))
    only_in_a = sorted(set(keyed_a) - set(keyed_b))
    only_in_b = sorted(set(keyed_b) - set(keyed_a))

    per_query_deltas: list[dict[str, Any]] = []
    changed_retrieved = 0
    changed_unique_retrieved = 0
    changed_metrics = 0
    metric_delta_totals = {metric: 0.0 for metric in PER_QUERY_METRIC_FIELDS}
    per_metric_deltas: dict[str, list[float]] = {
        metric: [] for metric in PER_QUERY_METRIC_FIELDS
    }

    for key in common_keys:
        row_a = keyed_a[key]
        row_b = keyed_b[key]

        retrieved_ids_a = row_a.get("retrieved_ids", [])
        retrieved_ids_b = row_b.get("retrieved_ids", [])
        retrieved_unique_a = row_a.get("retrieved_ids_unique", retrieved_ids_a)
        retrieved_unique_b = row_b.get("retrieved_ids_unique", retrieved_ids_b)

        retrieved_changed = retrieved_ids_a != retrieved_ids_b
        retrieved_unique_changed = retrieved_unique_a != retrieved_unique_b
        if retrieved_changed:
            changed_retrieved += 1
        if retrieved_unique_changed:
            changed_unique_retrieved += 1

        deltas = {
            metric: float(row_b.get(metric, 0.0)) - float(row_a.get(metric, 0.0))
            for metric in PER_QUERY_METRIC_FIELDS
        }
        for metric, delta in deltas.items():
            metric_delta_totals[metric] += delta
            per_metric_deltas[metric].append(delta)
        metric_changed = any(delta != 0 for delta in deltas.values())
        if metric_changed:
            changed_metrics += 1

        per_query_deltas.append(
            {
                "key": key,
                "query": row_a.get("query"),
                "relevant_ids": row_a.get("relevant_ids", []),
                "retrieved_ids_changed": retrieved_changed,
                "retrieved_ids_unique_changed": retrieved_unique_changed,
                "metric_deltas": deltas,
            }
        )

    compared = len(common_keys)
    avg_metric_deltas = {
        metric: (metric_delta_totals[metric] / compared if compared else 0.0)
        for metric in PER_QUERY_METRIC_FIELDS
    }
    summary_metric_deltas = {
        metric: float(summary_b.get(metric, 0.0)) - float(summary_a.get(metric, 0.0))
        for metric in SUMMARY_METRIC_FIELDS
    }
    inference = {
        metric: {
            "mean_delta": mean(per_metric_deltas[metric]),
            "ci95": list(bootstrap_mean_ci(per_metric_deltas[metric], confidence=0.95)),
            "paired_randomization_p_value": paired_randomization_p_value(
                per_metric_deltas[metric]
            ),
            "n_pairs": len(per_metric_deltas[metric]),
        }
        for metric in PER_QUERY_METRIC_FIELDS
    }

    return {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "summary_a": {
            "baseline": summary_a.get("baseline"),
            "k": summary_a.get("k"),
            "embedding_provider": summary_a.get("embedding_provider"),
            "embedding_model": summary_a.get("embedding_model"),
        },
        "summary_b": {
            "baseline": summary_b.get("baseline"),
            "k": summary_b.get("k"),
            "embedding_provider": summary_b.get("embedding_provider"),
            "embedding_model": summary_b.get("embedding_model"),
        },
        "counts": {
            "queries_compared": compared,
            "queries_only_in_run_a": len(only_in_a),
            "queries_only_in_run_b": len(only_in_b),
            "retrieved_ids_changed": changed_retrieved,
            "retrieved_ids_unique_changed": changed_unique_retrieved,
            "metric_values_changed": changed_metrics,
        },
        "summary_metric_deltas": summary_metric_deltas,
        "average_per_query_metric_deltas": avg_metric_deltas,
        "inference": inference,
        "queries_only_in_run_a": only_in_a,
        "queries_only_in_run_b": only_in_b,
        "per_query_deltas": per_query_deltas,
    }
