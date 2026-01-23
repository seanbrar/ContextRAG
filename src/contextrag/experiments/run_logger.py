from __future__ import annotations

import json
from pathlib import Path

from contextrag.core.io import write_jsonl


def write_run_artifacts(run_dir: Path, results: dict, metadata: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    per_query_path = run_dir / "per_query.jsonl"
    meta_path = run_dir / "metadata.json"

    summary_path.write_text(
        json.dumps(results.get("summary", {}), indent=2),
        encoding="utf-8",
    )
    write_jsonl(per_query_path, results.get("per_query", []))
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
