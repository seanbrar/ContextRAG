import json

from contextrag.core.cache import load_json_cache, save_json_cache


def test_load_json_cache_returns_empty_when_missing(tmp_path):
    missing = tmp_path / "missing.json"
    assert load_json_cache(missing) == {}


def test_save_and_load_json_cache_round_trip(tmp_path):
    cache_file = tmp_path / "cache.json"
    payload = {"a": 1, "b": ["x", "y"]}

    save_json_cache(cache_file, payload)

    loaded = load_json_cache(cache_file)
    assert loaded == payload
    assert json.loads(cache_file.read_text(encoding="utf-8")) == payload
