import json

import pytest

from contextrag.config import AppConfig
from contextrag.core import chunking
from contextrag.eval import runner


class DummyEncoding:
    name = "test"
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


def test_get_embedding_cost_per_million():
    from contextrag.core.costs import get_embedding_cost_per_million
    assert get_embedding_cost_per_million("text-embedding-3-small") == 0.02
    assert get_embedding_cost_per_million("openai/text-embedding-3-small") == 0.02
    assert get_embedding_cost_per_million("unknown") is None


def test_chunk_document(monkeypatch):
    from contextrag.chunking import chunk_document

    encoding = DummyEncoding()
    
    def uniform_strategy(text, enc):
        tokens = enc.encode(text)
        chunks = []
        for i in range(0, len(tokens), 2):
            chunk_tokens = tokens[i:i+2]
            chunks.append((enc.decode(chunk_tokens), len(chunk_tokens)))
        from contextrag.chunking.strategies import ChunkResult
        return ChunkResult(chunks=chunks, source_tokens=len(tokens), category="uniform")

    monkeypatch.setattr("contextrag.chunking.strategies.get_strategy", lambda name: uniform_strategy)
    monkeypatch.setattr("contextrag.chunking.strategies.get_encoding", lambda name: encoding)
    
    result = chunk_document("one two three four", strategy="uniform")
    assert [c[0] for c in result.chunks] == ["one two", "three four"]


def test_load_queries_skips_blank_lines(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text("\n" + json.dumps({"query": "q"}) + "\n", encoding="utf-8")
    assert runner._load_queries(path) == [{"query": "q"}]


def test_build_index_inputs_uniform(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "get_encoding", lambda name=None: DummyEncoding())
    monkeypatch.setattr(chunking, "get_encoding", lambda name=None: DummyEncoding())
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

    monkeypatch.setattr(runner, "get_encoding", lambda name=None: FakeEncoding())
    monkeypatch.setattr(chunking, "get_encoding", lambda name=None: FakeEncoding())
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
        openai_chat_model="gpt-4o-mini",
        openrouter_chat_model="mistralai/devstral-2512:free",
        chat_provider="openai",
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_provider_json=None,
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        embed_provider="local",
    )

    monkeypatch.setattr(runner, "VectorStore", FakeVectorDB)
    monkeypatch.setattr(runner, "get_encoding", lambda name=None: DummyEncoding())
    monkeypatch.setattr(chunking, "get_encoding", lambda name=None: DummyEncoding())
    monkeypatch.setattr(chunking, "get_encoding", lambda name=None: DummyEncoding())
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


def test_run_eval_costs_with_openrouter_provider(monkeypatch, tmp_path):
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
        openai_chat_model="gpt-4o-mini",
        openrouter_chat_model="mistralai/devstral-2512:free",
        chat_provider="openai",
        openrouter_api_key="ok",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_provider_json=None,
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        embed_provider="openrouter",
    )

    monkeypatch.setattr(runner, "VectorStore", FakeVectorDB)
    monkeypatch.setattr(runner, "load_config", lambda: config)
    monkeypatch.setattr(runner, "get_encoding", lambda name=None: DummyEncoding())
    monkeypatch.setattr(chunking, "get_encoding", lambda name=None: DummyEncoding())

    results = runner.run_eval(
        dataset_path=tmp_path,
        baseline="router",
        k=1,
        persist_path=None,
        embed_provider=None,
        embedding_model=None,
    )
    cost = results["summary"]["cost"]
    assert cost["model_cost_per_million_tokens"] == 0.01
    assert cost["total_cost_usd"] is not None
