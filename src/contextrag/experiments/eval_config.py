from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EvalConfig:
    dataset: str
    baseline: str = "uniform"
    k: int = 5
    retrieval_mode: str = "dense"
    embed_provider: str | None = None
    embedding_model: str | None = None
    uniform_chunk_tokens: int | None = None
    chunk_overlap_tokens: int = 0
    retrieval_candidates: int = 50
    output: str = "runs/eval.json"
    persist: str | None = None
    run_dir: str | None = None


ALLOWED_KEYS = {
    "schema_version",
    "dataset",
    "baseline",
    "k",
    "retrieval_mode",
    "embed_provider",
    "embedding_model",
    "uniform_chunk_tokens",
    "chunk_overlap_tokens",
    "retrieval_candidates",
    "output",
    "persist",
    "run_dir",
}

ALLOWED_BASELINES = {"uniform", "adaptive", "router", "semantic"}
ALLOWED_RETRIEVAL_MODES = {"dense", "bm25", "hybrid", "dense-rerank"}
ALLOWED_EMBED_PROVIDERS = {"auto", "openrouter", "local"}
CURRENT_SCHEMA_VERSION = 1


def _validate_baseline(value: str) -> str:
    if value not in ALLOWED_BASELINES:
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

    baseline = payload.get("baseline", "uniform")
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

    retrieval_mode = payload.get("retrieval_mode", "dense")
    if not isinstance(retrieval_mode, str):
        errors.append("'retrieval_mode' must be a string.")
    elif retrieval_mode not in ALLOWED_RETRIEVAL_MODES:
        errors.append(
            "'retrieval_mode' must be one of: "
            f"{', '.join(sorted(ALLOWED_RETRIEVAL_MODES))}."
        )

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

    uniform_chunk_tokens = payload.get("uniform_chunk_tokens")
    if uniform_chunk_tokens is not None:
        if isinstance(uniform_chunk_tokens, bool):
            errors.append("'uniform_chunk_tokens' must be an integer greater than 0.")
        else:
            try:
                uniform_chunk_tokens = int(uniform_chunk_tokens)
            except (TypeError, ValueError):
                errors.append("'uniform_chunk_tokens' must be an integer greater than 0.")
            else:
                if uniform_chunk_tokens <= 0:
                    errors.append("'uniform_chunk_tokens' must be an integer greater than 0.")

    chunk_overlap_tokens = payload.get("chunk_overlap_tokens", 0)
    if isinstance(chunk_overlap_tokens, bool):
        errors.append("'chunk_overlap_tokens' must be an integer >= 0.")
    else:
        try:
            chunk_overlap_tokens = int(chunk_overlap_tokens)
        except (TypeError, ValueError):
            errors.append("'chunk_overlap_tokens' must be an integer >= 0.")
        else:
            if chunk_overlap_tokens < 0:
                errors.append("'chunk_overlap_tokens' must be an integer >= 0.")

    retrieval_candidates = payload.get("retrieval_candidates", 50)
    if isinstance(retrieval_candidates, bool):
        errors.append("'retrieval_candidates' must be an integer greater than 0.")
    else:
        try:
            retrieval_candidates = int(retrieval_candidates)
        except (TypeError, ValueError):
            errors.append("'retrieval_candidates' must be an integer greater than 0.")
        else:
            if retrieval_candidates <= 0:
                errors.append("'retrieval_candidates' must be an integer greater than 0.")

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
    assert isinstance(dataset, str)
    return EvalConfig(
        dataset=dataset,
        baseline=baseline,
        k=k,
        retrieval_mode=retrieval_mode,
        embed_provider=embed_provider,
        embedding_model=embedding_model,
        uniform_chunk_tokens=uniform_chunk_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        retrieval_candidates=retrieval_candidates,
        output=output,
        persist=persist,
        run_dir=run_dir,
    )
