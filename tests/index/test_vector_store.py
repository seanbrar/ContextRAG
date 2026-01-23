import json

import pytest

from contextrag.config import AppConfig
from contextrag.index import vector_store


class DummyEmbedding:
    pass


def _config(**overrides):
    data = dict(
        openai_api_key="ok",
        openai_embeddings_model="text-embedding-3-small",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key="ok",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="auto",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    )
    data.update(overrides)
    return AppConfig(**data)


def test_build_embedding_function_openrouter_json(monkeypatch):
    config = _config(
        openai_api_key=None,
        openrouter_embed_provider_json=json.dumps({"order": ["x"]}),
    )

    monkeypatch.setattr(
        vector_store, "OpenRouterEmbeddingFunction", lambda **kwargs: DummyEmbedding()
    )
    db = vector_store.VectorDB.__new__(vector_store.VectorDB)
    ef = vector_store.VectorDB._build_embedding_function(
        db, config, None, "openrouter"
    )
    assert isinstance(ef, DummyEmbedding)


def test_build_embedding_function_openrouter_invalid_json(monkeypatch):
    config = _config(
        openai_api_key=None,
        openrouter_embed_provider_json="{bad json",
    )
    db = vector_store.VectorDB.__new__(vector_store.VectorDB)
    with pytest.raises(ValueError, match="OPENROUTER_EMBED_PROVIDER_JSON"):
        vector_store.VectorDB._build_embedding_function(db, config, None, "openrouter")


def test_build_embedding_function_openrouter_provider_order(monkeypatch):
    config = _config(
        openai_api_key=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order="alpha, beta",
        openrouter_embed_allow_fallbacks="true",
    )

    captured = {}

    def fake_openrouter(**kwargs):
        captured.update(kwargs)
        return DummyEmbedding()

    monkeypatch.setattr(vector_store, "OpenRouterEmbeddingFunction", fake_openrouter)
    db = vector_store.VectorDB.__new__(vector_store.VectorDB)
    vector_store.VectorDB._build_embedding_function(db, config, None, "openrouter")
    assert captured["provider"] == {"order": ["alpha", "beta"], "allow_fallbacks": True}


def test_build_embedding_function_local(monkeypatch):
    config = _config(openai_api_key=None, openrouter_api_key=None)

    monkeypatch.setattr(
        vector_store.embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        lambda model_name: DummyEmbedding(),
    )
    db = vector_store.VectorDB.__new__(vector_store.VectorDB)
    ef = vector_store.VectorDB._build_embedding_function(db, config, None, "local")
    assert isinstance(ef, DummyEmbedding)


def test_build_embedding_function_default(monkeypatch):
    config = _config()
    monkeypatch.setattr(
        vector_store.embedding_functions,
        "DefaultEmbeddingFunction",
        lambda: DummyEmbedding(),
    )
    db = vector_store.VectorDB.__new__(vector_store.VectorDB)
    ef = vector_store.VectorDB._build_embedding_function(db, config, None, "unknown")
    assert isinstance(ef, DummyEmbedding)


def test_build_embedding_function_openai_requires_key():
    config = _config(openai_api_key=None, openrouter_api_key=None)
    db = vector_store.VectorDB.__new__(vector_store.VectorDB)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        vector_store.VectorDB._build_embedding_function(db, config, None, "openai")


def test_build_embedding_function_openrouter_requires_key():
    config = _config(openai_api_key=None, openrouter_api_key=None)
    db = vector_store.VectorDB.__new__(vector_store.VectorDB)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        vector_store.VectorDB._build_embedding_function(db, config, None, "openrouter")


def test_get_or_create_collection_creates_when_missing():
    created = {}

    class FakeClient:
        def get_collection(self, name, embedding_function):
            raise vector_store.NotFoundError("missing")

        def create_collection(self, name, embedding_function, metadata):
            created["name"] = name
            created["metadata"] = metadata
            return "collection"

    db = vector_store.VectorDB.__new__(vector_store.VectorDB)
    db.collection_name = "test"
    db.client = FakeClient()
    db.openai_ef = DummyEmbedding()
    collection = vector_store.VectorDB.get_or_create_collection(db)
    assert collection == "collection"
    assert created["name"] == "test"
    assert created["metadata"]["hnsw:space"] == "cosine"


def test_get_or_create_collection_loads_existing():
    class FakeClient:
        def get_collection(self, name, embedding_function):
            return "existing"

    db = vector_store.VectorDB.__new__(vector_store.VectorDB)
    db.collection_name = "test"
    db.client = FakeClient()
    db.openai_ef = DummyEmbedding()
    collection = vector_store.VectorDB.get_or_create_collection(db)
    assert collection == "existing"


def test_add_documents_batches_and_generates_ids():
    calls = []

    class FakeCollection:
        def add(self, documents, ids):
            calls.append((documents, ids))

    db = vector_store.VectorDB.__new__(vector_store.VectorDB)
    db.collection = FakeCollection()
    db.add_documents(["a", "b", "c"], ids=None, batch_size=2)
    assert calls == [
        (["a", "b"], ["doc1", "doc2"]),
        (["c"], ["doc3"]),
    ]


def test_query_passes_through_results():
    class FakeCollection:
        def query(self, query_texts, n_results, include):
            return {"documents": [["a"]], "distances": [[0.1]]}

    db = vector_store.VectorDB.__new__(vector_store.VectorDB)
    db.collection = FakeCollection()
    results = db.query(query_texts=["q"], n_results=1)
    assert results["documents"][0][0] == "a"


def test_init_uses_provided_embedding_function(monkeypatch):
    class FakeClient:
        def get_collection(self, name, embedding_function):
            return "collection"

    monkeypatch.setattr(vector_store, "load_config", lambda: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(vector_store.chromadb, "Client", lambda: FakeClient())

    db = vector_store.VectorDB(
        collection_name="test",
        embedding_function=DummyEmbedding(),
    )
    assert db.collection == "collection"


def test_init_builds_embedding_function(monkeypatch):
    class FakeClient:
        def get_collection(self, name, embedding_function):
            return "collection"

    monkeypatch.setattr(vector_store.chromadb, "Client", lambda: FakeClient())
    monkeypatch.setattr(vector_store, "load_config", lambda: _config())
    monkeypatch.setattr(
        vector_store.VectorDB,
        "_build_embedding_function",
        lambda self, config, embedding_model, embed_provider: DummyEmbedding(),
    )

    db = vector_store.VectorDB(collection_name="test")
    assert isinstance(db.openai_ef, DummyEmbedding)
