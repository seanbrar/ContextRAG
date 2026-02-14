#!/usr/bin/env python3
"""Build reviewer-facing matrix artifacts and paper tables in one command."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from contextrag.eval.query_schema import load_and_validate_queries
from contextrag.experiments.matrix import run_matrix

ROOT = Path(__file__).resolve().parents[1]

BUNDLE_RUN_ROOT = ROOT / "runs" / "reviewer_bundle"
BUNDLE_PERSIST_ROOT = ROOT / "runs" / "chroma-reviewer-bundle"

DATASETS: list[dict[str, Any]] = [
    {
        "name": "expanded",
        "dataset": Path("data/eval-expanded"),
        "run_root": BUNDLE_RUN_ROOT / "matrix_eval_expanded_local",
        "persist_root": BUNDLE_PERSIST_ROOT / "matrix_eval_expanded_local",
        "report": ROOT / "docs" / "matrix_eval_expanded_local.md",
        "label": "data/eval-expanded",
    },
    {
        "name": "external",
        "dataset": Path("data/eval-external"),
        "run_root": BUNDLE_RUN_ROOT / "matrix_eval_external_local",
        "persist_root": BUNDLE_PERSIST_ROOT / "matrix_eval_external_local",
        "report": ROOT / "docs" / "matrix_eval_external_local.md",
        "label": "data/eval-external",
    },
]


def _validate_dataset(dataset_path: Path) -> None:
    docs = dataset_path / "documents"
    queries = dataset_path / "queries.jsonl"
    if not docs.exists():
        raise FileNotFoundError(f"Missing documents directory: {docs}")
    if not queries.exists():
        raise FileNotFoundError(f"Missing queries file: {queries}")
    load_and_validate_queries(queries)


def _run_render_matrix(input_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_matrix_report.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
    )


def _run_render_paper_tables(summary_paths: list[Path], labels: list[str], output_path: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "render_paper_tables.py"),
        "--inputs",
        *[str(path) for path in summary_paths],
        "--labels",
        *labels,
        "--output",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _format_reviewer_bundle(summary_paths: list[Path], labels: list[str]) -> str:
    lines = [
        "# Reviewer Bundle",
        "",
        "Canonical claim: length-based routing does not outperform uniform chunking on committed benchmarks.",
        "",
        "## Included Matrix Runs",
        "",
    ]

    for label, summary_path in zip(labels, summary_paths, strict=True):
        payload = _load(summary_path)
        lines.append(f"- `{label}`: `{summary_path.relative_to(ROOT)}`")
        for row in payload.get("rows", []):
            if row.get("baseline") == "uniform" and row.get("k") == 5:
                lines.append(
                    f"  uniform@5: P={row['precision_at_k']:.3f}, R={row['recall_at_k']:.3f}, "
                    f"Hit@1={row['hit_at_1']:.3f}, MRR={row['mrr_at_k']:.3f}, nDCG={row['ndcg_at_k']:.3f}"
                )
            if row.get("baseline") == "router" and row.get("k") == 5:
                lines.append(
                    f"  router@5:  P={row['precision_at_k']:.3f}, R={row['recall_at_k']:.3f}, "
                    f"Hit@1={row['hit_at_1']:.3f}, MRR={row['mrr_at_k']:.3f}, nDCG={row['ndcg_at_k']:.3f}"
                )

    lines.extend(
        [
            "",
            "## Generated Reports",
            "",
            "- `docs/matrix_eval_expanded_local.md`",
            "- `docs/matrix_eval_external_local.md`",
            "- `docs/paper_tables.md`",
            "",
            "## Rebuild",
            "",
            "```bash",
            "make reviewer-bundle",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    BUNDLE_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    BUNDLE_PERSIST_ROOT.mkdir(parents=True, exist_ok=True)

    summary_paths: list[Path] = []
    labels: list[str] = []

    for item in DATASETS:
        dataset_path = item["dataset"]
        dataset_path_abs = ROOT / dataset_path
        run_root = item["run_root"]
        persist_root = item["persist_root"]
        report_path = item["report"]
        label = item["label"]

        _validate_dataset(dataset_path_abs)
        run_matrix(
            dataset_path=dataset_path,
            baselines=["uniform", "router"],
            k_values=[3, 5, 10],
            run_root=run_root,
            persist_root=persist_root,
            embed_provider="local",
            embedding_model=None,
        )

        matrix_summary = run_root / "matrix_summary.json"
        _run_render_matrix(matrix_summary, report_path)

        summary_paths.append(matrix_summary)
        labels.append(label)

    _run_render_paper_tables(summary_paths, labels, ROOT / "docs" / "paper_tables.md")
    reviewer_bundle = _format_reviewer_bundle(summary_paths, labels)
    (ROOT / "docs" / "reviewer_bundle.md").write_text(reviewer_bundle, encoding="utf-8")


if __name__ == "__main__":
    main()
