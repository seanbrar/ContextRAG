from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from contextrag.core.io import write_jsonl


def _hash_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_fingerprint(dataset_path: Path | None) -> dict[str, Any]:
    if dataset_path is None:
        return {"status": "missing", "reason": "dataset_path not provided"}
    if not dataset_path.exists():
        return {"status": "missing", "reason": f"{dataset_path} does not exist"}

    files = [path for path in sorted(dataset_path.rglob("*")) if path.is_file()]
    hasher = hashlib.sha256()
    total_bytes = 0
    for file_path in files:
        rel_path = file_path.relative_to(dataset_path).as_posix()
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(b"\0")
        data = file_path.read_bytes()
        total_bytes += len(data)
        hasher.update(data)

    return {
        "status": "ok",
        "path": str(dataset_path),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": hasher.hexdigest(),
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _collect_versions() -> dict[str, str | None]:
    return {
        "contextrag": _package_version("ContextRAG") or _package_version("contextrag"),
        "python": platform.python_version(),
        "chromadb": _package_version("chromadb"),
        "openai": _package_version("openai"),
        "tiktoken": _package_version("tiktoken"),
        "sentence_transformers": _package_version("sentence-transformers"),
    }


def _system_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python_executable": sys.executable,
    }


def write_run_artifacts(
    run_dir: Path,
    results: dict[str, Any],
    metadata: dict[str, Any],
    dataset_path: Path | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    per_query_path = run_dir / "per_query.jsonl"
    meta_path = run_dir / "metadata.json"
    manifest_path = run_dir / "manifest.json"

    summary_path.write_text(
        json.dumps(results.get("summary", {}), indent=2),
        encoding="utf-8",
    )
    write_jsonl(per_query_path, results.get("per_query", []))
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    manifest = {
        "created_at": int(time.time()),
        "config": metadata,
        "config_hash": _hash_json(metadata),
        "dataset": _dataset_fingerprint(dataset_path),
        "versions": _collect_versions(),
        "system": _system_info(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
