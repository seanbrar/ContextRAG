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
        "| Baseline | Retrieval | Chunk | Overlap | k | Precision@k | Recall@k | Hit@1 | MRR@k | nDCG@k |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['baseline']} | {row.get('retrieval_mode', payload.get('retrieval_mode', 'dense'))} | "
            f"{row.get('uniform_chunk_tokens', payload.get('uniform_chunk_tokens', 0))} | "
            f"{row.get('chunk_overlap_tokens', payload.get('chunk_overlap_tokens', 0))} | "
            f"{row['k']} | "
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
            "| k | Metric | Mean Delta (router-uniform) | 95% CI | p (raw) | p (Holm) | Cohen's d | Cliff's delta |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
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
                f"{item.get('paired_randomization_p_value', 1.0):.4f} | "
                f"{item.get('holm_adjusted_p_value', 1.0):.4f} | "
                f"{item.get('cohen_d', 0.0):.3f} | "
                f"{item.get('cliffs_delta', 0.0):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Primary Endpoint (nDCG@k)",
            "",
            "| k | Margin | alpha | p(lower) | p(upper) | Equivalent | Non-inferior | Superior |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for comparison in comparisons:
        endpoint = comparison.get("inference", {}).get("ndcg_at_k", {})
        margin = endpoint.get("equivalence_margin", payload.get("equivalence_margin", 0.02))
        tost = endpoint.get("tost", {})
        alpha = float(tost.get("alpha", 0.05))
        p_lower = float(tost.get("p_value_lower", 1.0))
        p_upper = float(tost.get("p_value_upper", 1.0))
        lines.append(
            "| "
            f"{comparison['k']} | {margin:.3f} | "
            f"{alpha:.3f} | "
            f"{p_lower:.4f} | "
            f"{p_upper:.4f} | "
            f"{str(endpoint.get('equivalent_within_margin', False)).lower()} | "
            f"{str(endpoint.get('non_inferior_within_margin', False)).lower()} | "
            f"{str(endpoint.get('superior_to_zero', False)).lower()} |"
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
