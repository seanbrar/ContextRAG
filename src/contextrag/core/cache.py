from pathlib import Path
import json


def load_json_cache(cache_file: str | Path) -> dict:
    path = Path(cache_file)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def save_json_cache(cache_file: str | Path, cache_data: dict) -> None:
    path = Path(cache_file)
    path.write_text(json.dumps(cache_data), encoding="utf-8")
