import json

from chromaroute import EmbedConfig, build_embedding_function
from chromaroute import embedding as embedding_module

from contextrag.config import Config


class DummyEmbedding:
    pass

def _config(**overrides: str | None) -> Config:
    def _get(key: str, default: str | None) -> str | None:
        return overrides.get(key, default)

    embed = EmbedConfig(
        openrouter_api_key=_get("openrouter_api_key", "ok"),
        openrouter_base_url=_get(
            "openrouter_base_url", "https://openrouter.ai/api/v1"
        ) or "https://openrouter.ai/api/v1",
        openrouter_embeddings_model=_get(
            "openrouter_embeddings_model", "qwen/qwen3-embedding-8b"
        ) or "qwen/qwen3-embedding-8b",
        openrouter_referer=_get("openrouter_referer", None),
        openrouter_title=_get("openrouter_title", None),
        openrouter_provider_json=_get("openrouter_provider_json", None),
        local_embeddings_model=_get(
            "local_embeddings_model", "sentence-transformers/all-MiniLM-L6-v2"
        ) or "sentence-transformers/all-MiniLM-L6-v2",
        embed_provider=_get("embed_provider", "auto") or "auto",
    )
    return Config(embed=embed)

def test_build_embedding_function_openrouter_json(monkeypatch):
    config = _config(
        openrouter_provider_json=json.dumps({"order": ["x"]}),
        embed_provider="openrouter",
    )
    captured = {}

    def fake_build(config, model):
        captured["config"] = config
        captured["model"] = model
        return DummyEmbedding()

    monkeypatch.setitem(embedding_module._PROVIDERS, "openrouter", fake_build)

    ef = build_embedding_function(
        config=config.embed,
        embedding_model=None,
        embed_provider="openrouter",
    )

    assert isinstance(ef, DummyEmbedding)
    assert captured["config"].openrouter_provider_json == config.embed.openrouter_provider_json

def test_build_embedding_function_openrouter_env(monkeypatch):
    config = _config(embed_provider="openrouter")
    captured = {}

    def fake_build(config, model):
        captured["config"] = config
        captured["model"] = model
        return DummyEmbedding()

    monkeypatch.setitem(embedding_module._PROVIDERS, "openrouter", fake_build)

    build_embedding_function(
        config=config.embed,
        embedding_model=None,
        embed_provider="openrouter",
    )
    assert captured["config"].openrouter_embeddings_model == config.embed.openrouter_embeddings_model

def test_build_embedding_function_local(monkeypatch):
    # Setup config with no OR key but local model
    config = _config(
        openrouter_api_key=None,
        local_embeddings_model="local-model",
        embed_provider="local",
    )

    captured = {}

    def fake_build(config, model):
        captured["config"] = config
        captured["model"] = model
        return DummyEmbedding()

    monkeypatch.setitem(embedding_module._PROVIDERS, "local", fake_build)

    ef = build_embedding_function(
        config=config.embed,
        embedding_model=None,
        embed_provider="local",
    )
    assert isinstance(ef, DummyEmbedding)
    assert captured["config"].local_embeddings_model == "local-model"

