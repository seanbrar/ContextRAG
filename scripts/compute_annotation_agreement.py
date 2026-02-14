#!/usr/bin/env python3
"""Compute agreement statistics between two relevance annotation rounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_annotations(path: Path) -> dict[str, set[str]]:
    annotations: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query = str(row["query"]).strip()
            relevant_ids = {str(doc_id) for doc_id in row.get("relevant_ids", [])}
            if not query:
                raise ValueError(f"Empty query found in {path}")
            annotations[query] = relevant_ids
    return annotations


def _load_doc_ids(documents_dir: Path) -> set[str]:
    return {path.stem for path in documents_dir.glob("*") if path.is_file()}


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _cohen_kappa_binary(
    ann_a: dict[str, set[str]],
    ann_b: dict[str, set[str]],
    doc_ids: set[str],
) -> float:
    if not doc_ids:
        return 0.0

    tp = fp = fn = tn = 0
    for query in sorted(set(ann_a) | set(ann_b)):
        relevant_a = ann_a.get(query, set())
        relevant_b = ann_b.get(query, set())
        for doc_id in doc_ids:
            a_label = doc_id in relevant_a
            b_label = doc_id in relevant_b
            if a_label and b_label:
                tp += 1
            elif a_label and not b_label:
                fn += 1
            elif not a_label and b_label:
                fp += 1
            else:
                tn += 1

    total = tp + fp + fn + tn
    if total == 0:
        return 0.0
    po = (tp + tn) / total
    p_a_pos = (tp + fn) / total
    p_a_neg = (fp + tn) / total
    p_b_pos = (tp + fp) / total
    p_b_neg = (fn + tn) / total
    pe = p_a_pos * p_b_pos + p_a_neg * p_b_neg
    if pe >= 1:
        return 1.0
    return (po - pe) / (1 - pe)


def compute_agreement(
    ann_a: dict[str, set[str]],
    ann_b: dict[str, set[str]],
    doc_ids: set[str],
) -> dict[str, Any]:
    common_queries = sorted(set(ann_a) & set(ann_b))
    if not common_queries:
        raise ValueError("No overlapping queries found between annotation rounds.")

    jaccards: list[float] = []
    macro_precision: list[float] = []
    macro_recall: list[float] = []
    macro_f1: list[float] = []

    for query in common_queries:
        a = ann_a[query]
        b = ann_b[query]
        union = a | b
        intersection = a & b
        jaccard = len(intersection) / len(union) if union else 1.0
        jaccards.append(jaccard)

        precision = len(intersection) / len(b) if b else 1.0
        recall = len(intersection) / len(a) if a else 1.0
        macro_precision.append(precision)
        macro_recall.append(recall)
        macro_f1.append(_f1(precision, recall))

    kappa = _cohen_kappa_binary(ann_a, ann_b, doc_ids)
    return {
        "queries_compared": len(common_queries),
        "mean_jaccard_relevant_ids": round(sum(jaccards) / len(jaccards), 4),
        "macro_precision": round(sum(macro_precision) / len(macro_precision), 4),
        "macro_recall": round(sum(macro_recall) / len(macro_recall), 4),
        "macro_f1": round(sum(macro_f1) / len(macro_f1), 4),
        "cohen_kappa_binary_relevance": round(kappa, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-a", type=Path, required=True)
    parser.add_argument("--round-b", type=Path, required=True)
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ann_a = _load_annotations(args.round_a)
    ann_b = _load_annotations(args.round_b)
    doc_ids = _load_doc_ids(args.documents_dir)
    metrics = compute_agreement(ann_a, ann_b, doc_ids)
    payload = {
        "round_a": str(args.round_a),
        "round_b": str(args.round_b),
        "documents_dir": str(args.documents_dir),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
