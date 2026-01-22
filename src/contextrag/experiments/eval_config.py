from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvalConfig:
    dataset: str
    baseline: str = "router"
    k: int = 5
    embed_provider: str | None = None
    embedding_model: str | None = None
    output: str = "runs/eval.json"
    persist: str | None = None


def _validate_baseline(value: str) -> str:
    if value not in {"uniform", "router"}:
        raise ValueError(f"Unsupported baseline '{value}'.")
    return value


def load_eval_config(path: Path) -> EvalConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Eval config must be a mapping.")

    dataset = payload.get("dataset")
    if not dataset:
        raise ValueError("Eval config must include 'dataset'.")

    baseline = _validate_baseline(payload.get("baseline", "router"))
    k = int(payload.get("k", 5))
    return EvalConfig(
        dataset=dataset,
        baseline=baseline,
        k=k,
        embed_provider=payload.get("embed_provider"),
        embedding_model=payload.get("embedding_model"),
        output=payload.get("output", "runs/eval.json"),
        persist=payload.get("persist"),
    )
