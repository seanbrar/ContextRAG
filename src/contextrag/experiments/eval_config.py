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
    run_dir: str | None = None


ALLOWED_KEYS = {
    "schema_version",
    "dataset",
    "baseline",
    "k",
    "embed_provider",
    "embedding_model",
    "output",
    "persist",
    "run_dir",
}

ALLOWED_BASELINES = {"uniform", "router"}
ALLOWED_EMBED_PROVIDERS = {"auto", "openai", "openrouter", "local"}
CURRENT_SCHEMA_VERSION = 1


def _validate_baseline(value: str) -> str:
    if value not in {"uniform", "router"}:
        raise ValueError(f"Unsupported baseline '{value}'.")
    return value


def load_eval_config(path: Path) -> EvalConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Eval config must be a mapping.")

    errors: list[str] = []
    unknown_keys = sorted(set(payload.keys()) - ALLOWED_KEYS)
    if unknown_keys:
        errors.append(f"Unknown keys: {', '.join(unknown_keys)}.")

    schema_version = payload.get("schema_version")
    if schema_version is not None:
        if not isinstance(schema_version, int):
            errors.append("'schema_version' must be an integer.")
        elif schema_version != CURRENT_SCHEMA_VERSION:
            errors.append(
                f"'schema_version' must be {CURRENT_SCHEMA_VERSION} (got {schema_version})."
            )

    dataset = payload.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        errors.append("'dataset' must be a non-empty string.")

    baseline = payload.get("baseline", "router")
    if not isinstance(baseline, str):
        errors.append("'baseline' must be a string.")
    elif baseline not in ALLOWED_BASELINES:
        errors.append(
            f"'baseline' must be one of: {', '.join(sorted(ALLOWED_BASELINES))}."
        )

    k = payload.get("k", 5)
    if isinstance(k, bool):
        errors.append("'k' must be an integer greater than 0.")
    else:
        try:
            k = int(k)
        except (TypeError, ValueError):
            errors.append("'k' must be an integer greater than 0.")
        else:
            if k <= 0:
                errors.append("'k' must be an integer greater than 0.")

    embed_provider = payload.get("embed_provider")
    if embed_provider is not None:
        if not isinstance(embed_provider, str):
            errors.append("'embed_provider' must be a string.")
        elif embed_provider not in ALLOWED_EMBED_PROVIDERS:
            errors.append(
                "'embed_provider' must be one of: "
                f"{', '.join(sorted(ALLOWED_EMBED_PROVIDERS))}."
            )

    embedding_model = payload.get("embedding_model")
    if embedding_model is not None and not isinstance(embedding_model, str):
        errors.append("'embedding_model' must be a string.")

    output = payload.get("output", "runs/eval.json")
    if output is not None and not isinstance(output, str):
        errors.append("'output' must be a string.")

    persist = payload.get("persist")
    if persist is not None and not isinstance(persist, str):
        errors.append("'persist' must be a string.")

    run_dir = payload.get("run_dir")
    if run_dir is not None and not isinstance(run_dir, str):
        errors.append("'run_dir' must be a string.")

    if errors:
        raise ValueError("Invalid eval config:\n- " + "\n- ".join(errors))

    baseline = _validate_baseline(baseline)
    return EvalConfig(
        dataset=dataset,
        baseline=baseline,
        k=k,
        embed_provider=embed_provider,
        embedding_model=embedding_model,
        output=output,
        persist=persist,
        run_dir=run_dir,
    )
