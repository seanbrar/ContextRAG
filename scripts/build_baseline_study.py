#!/usr/bin/env python3
"""Run an expanded baseline study and render a compact markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contextrag.eval.runner import run_eval
from contextrag.experiments.eval_config import load_eval_config
from contextrag.experiments.run_logger import write_run_artifacts

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "baseline_study"
REPORT_PATH = ROOT / "docs" / "baseline_study.md"

SCENARIOS = [
    ("uniform-512-dense", Path("experiments/eval_expanded_uniform_512_local.yaml")),
    ("uniform-1000-dense", Path("experiments/eval_expanded_uniform_local.yaml")),
    ("uniform-2000-dense", Path("experiments/eval_expanded_uniform_2000_local.yaml")),
    ("uniform-overlap-dense", Path("experiments/eval_expanded_uniform_overlap_local.yaml")),
    ("semantic-dense", Path("experiments/eval_expanded_semantic_local.yaml")),
    ("uniform-bm25", Path("experiments/eval_expanded_bm25_uniform.yaml")),
    ("uniform-hybrid", Path("experiments/eval_expanded_hybrid_uniform_local.yaml")),
    (
        "uniform-dense-rerank",
        Path("experiments/eval_expanded_dense_rerank_uniform_local.yaml"),
    ),
]


def _render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Baseline Study",
        "",
        "Expanded baseline comparison on `data/eval-expanded`.",
        "",
        "| Scenario | Baseline | Retrieval | Chunk | Overlap | Precision@k | Recall@k | Hit@1 | MRR@k | nDCG@k |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['scenario']} | {row['baseline']} | {row['retrieval_mode']} | "
            f"{row['uniform_chunk_tokens']} | {row['chunk_overlap_tokens']} | "
            f"{row['precision_at_k']:.3f} | {row['recall_at_k']:.3f} | "
            f"{row['hit_at_1']:.3f} | {row['mrr_at_k']:.3f} | {row['ndcg_at_k']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(reuse_existing: bool = False) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for scenario, config_path in SCENARIOS:
        config = load_eval_config(ROOT / config_path)
        output_path = ROOT / config.output
        run_dir = ROOT / (config.run_dir or f"runs/{scenario}")

        if output_path.exists() and reuse_existing:
            results = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            results = run_eval(
                dataset_path=ROOT / config.dataset,
                baseline=config.baseline,
                k=config.k,
                persist_path=config.persist,
                embed_provider=config.embed_provider,
                embedding_model=config.embedding_model,
                retrieval_mode=config.retrieval_mode,
                uniform_chunk_tokens=config.uniform_chunk_tokens,
                chunk_overlap_tokens=config.chunk_overlap_tokens,
                retrieval_candidates=config.retrieval_candidates,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        write_run_artifacts(
            run_dir=run_dir,
            results=results,
            metadata={
                "scenario": scenario,
                "config_path": str(config_path),
                "dataset": config.dataset,
                "baseline": config.baseline,
                "retrieval_mode": config.retrieval_mode,
                "uniform_chunk_tokens": config.uniform_chunk_tokens,
                "chunk_overlap_tokens": config.chunk_overlap_tokens,
                "retrieval_candidates": config.retrieval_candidates,
                "k": config.k,
                "embed_provider": config.embed_provider,
                "embedding_model": config.embedding_model,
                "persist": config.persist,
                "output": config.output,
            },
            dataset_path=ROOT / config.dataset,
        )

        summary = results["summary"]
        rows.append(
            {
                "scenario": scenario,
                "baseline": summary["baseline"],
                "retrieval_mode": summary["retrieval_mode"],
                "uniform_chunk_tokens": summary["chunking"]["uniform_chunk_tokens"],
                "chunk_overlap_tokens": summary["chunking"]["chunk_overlap_tokens"],
                "precision_at_k": float(summary["precision_at_k"]),
                "recall_at_k": float(summary["recall_at_k"]),
                "hit_at_1": float(summary["hit_at_1"]),
                "mrr_at_k": float(summary["mrr_at_k"]),
                "ndcg_at_k": float(summary["ndcg_at_k"]),
            }
        )

    summary_path = RUN_ROOT / "baseline_study_summary.json"
    summary_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(_render(rows), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing run JSON outputs when present.",
    )
    args = parser.parse_args()
    main(reuse_existing=args.reuse_existing)
