#!/usr/bin/env python3
"""Render a markdown report from a matrix_summary.json artifact."""

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


def _render(payload: dict[str, Any]) -> str:
    rows = payload.get("rows", [])
    comparisons = payload.get("comparisons", [])
    lines = [
        "# Matrix Report",
        "",
        f"- Dataset: `{payload.get('dataset')}`",
        f"- Baselines: `{', '.join(payload.get('baselines', []))}`",
        f"- k values: `{', '.join(str(k) for k in payload.get('k_values', []))}`",
        f"- Embed provider: `{payload.get('embed_provider')}`",
        f"- Embedding model override: `{payload.get('embedding_model')}`",
        "",
        "## Aggregate Metrics",
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

    lines.extend(
        [
            "",
            "## Uniform vs Router (Per-k Summary Deltas)",
            "",
            "| k | Δ Precision@k | Δ Recall@k | Δ Hit@1 | Δ MRR@k | Δ nDCG@k |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for comparison in comparisons:
        delta = comparison.get("summary_metric_deltas", {})
        lines.append(
            "| "
            f"{comparison['k']} | "
            f"{delta.get('precision_at_k', 0.0):.3f} | "
            f"{delta.get('recall_at_k', 0.0):.3f} | "
            f"{delta.get('hit_at_1', 0.0):.3f} | "
            f"{delta.get('mrr_at_k', 0.0):.3f} | "
            f"{delta.get('ndcg_at_k', 0.0):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Uniform vs Router (Inference)",
            "",
            "| k | Metric | Mean Delta (router-uniform) | 95% CI | p-value |",
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
        for metric_key, label in metrics:
            if metric_key not in inference:
                continue
            item = inference[metric_key]
            ci = item.get("ci95", [0.0, 0.0])
            lines.append(
                "| "
                f"{comparison['k']} | {label} | "
                f"{item.get('mean_delta', 0.0):.3f} | "
                f"[{ci[0]:.3f}, {ci[1]:.3f}] | "
                f"{item.get('paired_randomization_p_value', 1.0):.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = _load(args.input)
    report = _render(payload)
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
