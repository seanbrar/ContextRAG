import json
from pathlib import Path

import pytest

from contextrag.config import AppConfig
from contextrag.eval import runner


class DummyEncoding:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


def test_get_embedding_cost_per_million():
    assert runner._get_embedding_cost_per_million("text-embedding-3-small") == 0.02
    assert runner._get_embedding_cost_per_million("openai/text-embedding-3-small") == 0.02
    assert runner._get_embedding_cost_per_million("unknown") is None


def test_chunk_text(monkeypatch):
    monkeypatch.setattr(runner.tiktoken, "get_encoding", lambda name: DummyEncoding())
    chunks = runner._chunk_text("one two three four", chunk_tokens=2)
    assert chunks == ["one two", "three four"]


def test_chunk_text_empty_returns_empty(monkeypatch):
    class EmptyEncoding(DummyEncoding):
        def encode(self, text):
            return []

    monkeypatch.setattr(runner.tiktoken, "get_encoding", lambda name: EmptyEncoding())
    assert runner._chunk_text("", chunk_tokens=2) == []


def test_load_queries_skips_blank_lines(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text("\n" + json.dumps({"query": "q"}) + "\n", encoding="utf-8")
    assert runner._load_queries(path) == [{"query": "q"}]


def test_build_index_inputs_uniform(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.tiktoken, "get_encoding", lambda name: DummyEncoding())
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("one two three four five", encoding="utf-8")

    documents, ids, mapping, stats = runner._build_index_inputs(tmp_path, "uniform")
    assert documents
    assert ids[0].startswith("doc::chunk")
    assert mapping[ids[0]] == "doc"
    assert stats["source_documents"] == 1


def test_build_index_inputs_router_categories(monkeypatch, tmp_path):
    class FakeEncoding:
        def encode(self, text):
            if text == "short":
                return [0] * 10
            if text == "medium":
                return [0] * 4000
            if text == "long":
                return [0] * 20000
            return []

        def decode(self, tokens):
            return "chunk"

    monkeypatch.setattr(runner.tiktoken, "get_encoding", lambda name: FakeEncoding())
    (tmp_path / "short.md").write_text("short", encoding="utf-8")
    (tmp_path / "medium.md").write_text("medium", encoding="utf-8")
    (tmp_path / "long.md").write_text("long", encoding="utf-8")

    documents, ids, mapping, stats = runner._build_index_inputs(tmp_path, "router")
    assert mapping["short"] == "short"
    assert stats["documents_by_category"] == {"short": 1, "medium": 1, "long": 1}
    assert documents and ids


def test_run_eval_with_fake_vector_db(monkeypatch, tmp_path):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "doc1.md").write_text("alpha beta", encoding="utf-8")
    (documents_dir / "doc2.md").write_text("gamma delta", encoding="utf-8")

    queries = [
        {"query": "alpha", "relevant_ids": ["doc1"]},
        {"query": "gamma", "relevant_ids": ["doc2"]},
    ]
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        "\n".join(json.dumps(row) for row in queries),
        encoding="utf-8",
    )

    class FakeVectorDB:
        def __init__(self, *args, **kwargs):
            self.documents = []
            self.ids = []

        def add_documents(self, documents, ids):
            self.documents = documents
            self.ids = ids

        def query(self, query_texts, n_results):
            if "alpha" in query_texts[0]:
                return {"ids": [["doc1"]]}
            return {"ids": [["doc2"]]}

    config = AppConfig(
        openai_api_key=None,
        openai_embeddings_model="text-embedding-3-small",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="local",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    )

    monkeypatch.setattr(runner, "VectorDB", FakeVectorDB)
    monkeypatch.setattr(runner.tiktoken, "get_encoding", lambda name: DummyEncoding())
    monkeypatch.setattr(runner, "load_config", lambda: config)

    results = runner.run_eval(
        dataset_path=tmp_path,
        baseline="router",
        k=1,
        persist_path=None,
        embed_provider=None,
        embedding_model=None,
    )
    summary = results["summary"]
    assert summary["total_queries"] == 2
    assert summary["precision_at_k"] == 1.0
    assert summary["recall_at_k"] == 1.0


def test_run_eval_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="documents"):
        runner.run_eval(tmp_path, baseline="router", k=1)

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="queries"):
        runner.run_eval(tmp_path, baseline="router", k=1)


def test_run_eval_costs_with_openai_provider(monkeypatch, tmp_path):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "doc1.md").write_text("alpha beta", encoding="utf-8")
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        json.dumps({"query": "alpha", "relevant_ids": ["doc1"]}),
        encoding="utf-8",
    )

    class FakeVectorDB:
        def __init__(self, *args, **kwargs):
            pass

        def add_documents(self, documents, ids):
            pass

        def query(self, query_texts, n_results):
            return {"ids": [["doc1"]]}

    config = AppConfig(
        openai_api_key="key",
        openai_embeddings_model="text-embedding-3-small",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="openai",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    )

    monkeypatch.setattr(runner, "VectorDB", FakeVectorDB)
    monkeypatch.setattr(runner, "load_config", lambda: config)
    monkeypatch.setattr(runner.tiktoken, "get_encoding", lambda name: DummyEncoding())
    monkeypatch.setattr(runner, "resolve_embed_provider", lambda *_: "openai")

    results = runner.run_eval(
        dataset_path=tmp_path,
        baseline="router",
        k=1,
        persist_path=None,
        embed_provider=None,
        embedding_model=None,
    )
    cost = results["summary"]["cost"]
    assert cost["model_cost_per_million_tokens"] == 0.02
    assert cost["total_cost_usd"] is not None
