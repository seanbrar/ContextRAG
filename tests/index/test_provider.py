import json

import pytest

from chromaroute import EmbedConfig
from contextrag.config import AppConfig
from contextrag.embeddings import provider


class DummyEmbedding:
    pass

def _config(**overrides):
    embed_data = dict(
        openrouter_api_key="ok",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_provider_json=None,
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        embed_provider="auto",
    )
    
    # Extract embed related overrides
    embed_overrides = {k: v for k, v in overrides.items() if k in embed_data or k == "openrouter_provider_json"}
    if "openrouter_provider_json" in embed_overrides:
        embed_overrides["openrouter_provider_json"] = embed_overrides.pop("openrouter_provider_json")
    
    embed_data.update(embed_overrides)
    embed_config = EmbedConfig(**embed_data)
    
    data = dict(
        openai_api_key="ok",
        openai_chat_model="gpt-4o-mini",
        openrouter_chat_model="mistralai/devstral-2512:free",
        chat_provider="openai",
        embed_config=embed_config,
    )
    # Remaining overrides for AppConfig
    app_overrides = {k: v for k, v in overrides.items() if k in data}
    data.update(app_overrides)
    
    return AppConfig(**data)

def test_build_embedding_function_openrouter_json(monkeypatch):
    config = _config(
        openai_api_key=None,
        openrouter_provider_json=json.dumps({"order": ["x"]}),
    )
    captured = {}

    def fake_build(config=None, embedding_model=None, embed_provider=None):
        captured["config"] = config
        return DummyEmbedding()

    monkeypatch.setattr(provider, "chromaroute_build", fake_build)
    
    ef = provider.build_embedding_function(config, None, "openrouter")
    
    assert isinstance(ef, DummyEmbedding)
    assert captured["config"].openrouter_provider_json == config.embed_config.openrouter_provider_json

def test_build_embedding_function_openrouter_env(monkeypatch):
    config = _config(openai_api_key=None)
    captured = {}

    def fake_build(config=None, embedding_model=None, embed_provider=None):
        captured["config"] = config
        return DummyEmbedding()

    monkeypatch.setattr(provider, "chromaroute_build", fake_build)
    
    provider.build_embedding_function(config, None, "openrouter")
    assert captured["config"].openrouter_embeddings_model == config.embed_config.openrouter_embeddings_model

def test_build_embedding_function_local(monkeypatch):
    # Setup config with no OR key but local model
    embed_config = EmbedConfig(
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_provider_json=None,
        local_embeddings_model="local-model",
        embed_provider="local"
    )
    config = AppConfig(
        openai_api_key=None,
        openai_chat_model="gpt-4o-mini",
        openrouter_chat_model="mistralai/devstral-2512:free",
        chat_provider="openai",
        embed_config=embed_config
    )
    
    captured = {}

    def fake_build(config=None, embedding_model=None, embed_provider=None):
        captured["config"] = config
        return DummyEmbedding()

    monkeypatch.setattr(provider, "chromaroute_build", fake_build)
    
    ef = provider.build_embedding_function(config, None, "local")
    assert isinstance(ef, DummyEmbedding)
    assert captured["config"].local_embeddings_model == "local-model"
