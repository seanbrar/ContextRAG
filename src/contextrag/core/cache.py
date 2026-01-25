import json
from pathlib import Path
from typing import Any


def load_json_cache(cache_file: str | Path) -> dict[str, Any]:
    path = Path(cache_file)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Cache file must contain a JSON object: {path}")
    return payload


def save_json_cache(cache_file: str | Path, cache_data: dict[str, Any]) -> None:
    path = Path(cache_file)
    path.write_text(json.dumps(cache_data), encoding="utf-8")
