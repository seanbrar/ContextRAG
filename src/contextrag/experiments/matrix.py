"""Experiment matrix execution and reporting."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from contextrag.eval.compare import compare_runs
from contextrag.eval.runner import run_eval
from contextrag.experiments.run_logger import write_run_artifacts


def _write_markdown_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Matrix Summary",
        "",
        "| Baseline | k | Precision@k | Recall@k | Hit@1 | MRR@k | nDCG@k |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['baseline']} | {row['k']} | "
            f"{row['precision_at_k']:.3f} | {row['recall_at_k']:.3f} | "
            f"{row['hit_at_1']:.3f} | {row['mrr_at_k']:.3f} | {row['ndcg_at_k']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_matrix(
    *,
    dataset_path: Path,
    baselines: list[str],
    k_values: list[int],
    run_root: Path,
    persist_root: Path | None = None,
    embed_provider: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Run a baseline/k matrix and write aggregate artifacts."""
    run_root.mkdir(parents=True, exist_ok=True)
    if persist_root:
        persist_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    combo_run_dirs: dict[tuple[str, int], Path] = {}

    for baseline in baselines:
        for k in k_values:
            combo = f"{baseline}_k{k}"
            run_dir = run_root / combo
            output_path = run_root / f"{combo}.json"
            persist_path: str | None = str(persist_root / combo) if persist_root else None

            results = run_eval(
                dataset_path=dataset_path,
                baseline=baseline,
                k=k,
                persist_path=persist_path,
                embed_provider=embed_provider,
                embedding_model=embedding_model,
            )

            output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            write_run_artifacts(
                run_dir=run_dir,
                results=results,
                metadata={
                    "dataset": str(dataset_path),
                    "baseline": baseline,
                    "k": k,
                    "embed_provider": embed_provider,
                    "embedding_model": embedding_model,
                    "output": str(output_path),
                    "persist": persist_path,
                    "matrix_run_root": str(run_root),
                },
                dataset_path=dataset_path,
            )

            summary = results["summary"]
            rows.append(
                {
                    "baseline": baseline,
                    "k": k,
                    "precision_at_k": float(summary["precision_at_k"]),
                    "recall_at_k": float(summary["recall_at_k"]),
                    "hit_at_1": float(summary.get("hit_at_1", 0.0)),
                    "mrr_at_k": float(summary.get("mrr_at_k", 0.0)),
                    "ndcg_at_k": float(summary.get("ndcg_at_k", 0.0)),
                    "run_dir": str(run_dir),
                }
            )
            combo_run_dirs[(baseline, k)] = run_dir

    comparisons_dir = run_root / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    comparisons: list[dict[str, Any]] = []

    for k in k_values:
        key_uniform = ("uniform", k)
        key_router = ("router", k)
        if key_uniform not in combo_run_dirs or key_router not in combo_run_dirs:
            continue
        comparison = compare_runs(combo_run_dirs[key_uniform], combo_run_dirs[key_router])
        comparison_path = comparisons_dir / f"uniform_vs_router_k{k}.json"
        comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        comparisons.append(
            {
                "k": k,
                "path": str(comparison_path),
                "summary_metric_deltas": comparison["summary_metric_deltas"],
                "inference": comparison["inference"],
            }
        )

    matrix_summary = {
        "generated_at": int(time.time()),
        "dataset": str(dataset_path),
        "baselines": baselines,
        "k_values": k_values,
        "embed_provider": embed_provider,
        "embedding_model": embedding_model,
        "rows": rows,
        "comparisons": comparisons,
    }

    (run_root / "matrix_summary.json").write_text(
        json.dumps(matrix_summary, indent=2),
        encoding="utf-8",
    )
    _write_markdown_summary(run_root / "matrix_summary.md", rows)
    return matrix_summary
