#!/usr/bin/env python3
"""Render paper-ready metric and inference tables from matrix summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _render(inputs: list[tuple[str, dict[str, Any]]]) -> str:
    lines: list[str] = [
        "# Paper Tables",
        "",
        "Generated from matrix artifacts. Deltas are router-uniform.",
    ]

    for label, payload in inputs:
        rows = payload.get("rows", [])
        comparisons = payload.get("comparisons", [])
        lines.extend(
            [
                "",
                f"## Dataset: {label}",
                "",
                "### Aggregate Metrics",
                "",
                "| Baseline | k | Precision@k | Recall@k | Hit@1 | MRR@k | nDCG@k |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                f"{row['baseline']} | {row['k']} | "
                f"{row['precision_at_k']:.3f} | {row['recall_at_k']:.3f} | "
                f"{row['hit_at_1']:.3f} | {row['mrr_at_k']:.3f} | {row['ndcg_at_k']:.3f} |"
            )

        lines.extend(
            [
                "",
                "### Inference (Uniform vs Router)",
                "",
                "| k | Metric | Mean Delta | 95% CI | p-value |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        metrics = [
            ("precision_at_k", "Precision@k"),
            ("recall_at_k", "Recall@k"),
            ("hit_at_1", "Hit@1"),
            ("reciprocal_rank_at_k", "MRR@k"),
            ("ndcg_at_k", "nDCG@k"),
        ]
        for comparison in comparisons:
            inference = comparison.get("inference", {})
            for metric_key, metric_label in metrics:
                if metric_key not in inference:
                    continue
                item = inference[metric_key]
                ci = item.get("ci95", [0.0, 0.0])
                lines.append(
                    "| "
                    f"{comparison['k']} | {metric_label} | {item.get('mean_delta', 0.0):.3f} | "
                    f"[{ci[0]:.3f}, {ci[1]:.3f}] | {item.get('paired_randomization_p_value', 1.0):.4f} |"
                )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = args.labels
    if labels and len(labels) != len(args.inputs):
        raise ValueError("--labels length must match --inputs length when provided.")

    loaded: list[tuple[str, dict[str, Any]]] = []
    for idx, path in enumerate(args.inputs):
        payload = _load(path)
        label = labels[idx] if labels else str(payload.get("dataset", path))
        loaded.append((label, payload))

    args.output.write_text(_render(loaded), encoding="utf-8")


if __name__ == "__main__":
    main()
