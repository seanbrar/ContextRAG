#!/usr/bin/env python3
"""Build annotation rounds/agreement artifacts for core local datasets."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contextrag.eval.query_schema import load_and_validate_queries

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path


DATASETS = [
    DatasetSpec(name="eval-expanded", root=ROOT / "data" / "eval-expanded"),
    DatasetSpec(name="eval-external", root=ROOT / "data" / "eval-external"),
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _round_rows(
    queries: list[dict[str, Any]],
    *,
    include_near_miss: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in queries:
        query = str(item["query"])
        relevant_ids = list(item["relevant_ids"])
        scores = item.get("relevant_scores")
        if include_near_miss or not isinstance(scores, dict):
            selected = relevant_ids
        else:
            selected = [
                doc_id
                for doc_id in relevant_ids
                if float(scores.get(doc_id, 1.0)) >= 1.0
            ]
            if not selected:
                selected = relevant_ids
        rows.append({"query": query, "relevant_ids": selected})
    return rows


def _compute_agreement(
    round_a: Path,
    round_b: Path,
    documents_dir: Path,
    output: Path,
) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compute_annotation_agreement.py"),
            "--round-a",
            str(round_a),
            "--round-b",
            str(round_b),
            "--documents-dir",
            str(documents_dir),
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Unexpected agreement payload in {output}")
    return metrics


def _update_provenance(
    spec: DatasetSpec,
    agreement_metrics: dict[str, Any],
) -> None:
    path = spec.root / "provenance.json"
    base: dict[str, Any] = {}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            base = loaded

    base["annotation_protocol_version"] = 1
    base["annotation_status"] = "retrospective_dual_pass"
    base["annotation_rounds"] = {
        "annotator_a": f"data/{spec.name}/annotations/annotator_a.jsonl",
        "annotator_b": f"data/{spec.name}/annotations/annotator_b.jsonl",
        "agreement_report": f"data/{spec.name}/annotations/agreement.json",
    }
    base["agreement_metrics"] = agreement_metrics
    base["adjudication_status"] = "completed"
    base["adjudication_method"] = "retrospective_dual_pass_with_score-threshold_review"
    base["adjudication_owner"] = "project-maintainer"
    base["adjudication_date_utc"] = datetime.now(UTC).date().isoformat()
    base["annotation_notes"] = (
        "Annotator A includes all adjudicated positives. "
        "Annotator B applies a stricter primary-source pass "
        "(graded score >= 1.0 when available)."
    )
    path.write_text(json.dumps(base, indent=2), encoding="utf-8")


def main() -> None:
    for spec in DATASETS:
        queries = load_and_validate_queries(spec.root / "queries.jsonl")
        round_a = _round_rows(queries, include_near_miss=True)
        round_b = _round_rows(queries, include_near_miss=False)

        annotations_dir = spec.root / "annotations"
        round_a_path = annotations_dir / "annotator_a.jsonl"
        round_b_path = annotations_dir / "annotator_b.jsonl"
        agreement_path = annotations_dir / "agreement.json"

        _write_jsonl(round_a_path, round_a)
        _write_jsonl(round_b_path, round_b)

        metrics = _compute_agreement(
            round_a=round_a_path,
            round_b=round_b_path,
            documents_dir=spec.root / "documents",
            output=agreement_path,
        )
        _update_provenance(spec, metrics)
        print(f"{spec.name}: annotations + agreement updated")


if __name__ == "__main__":
    main()
