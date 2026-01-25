from __future__ import annotations

import pytest

from contextrag.config import AppConfig
from contextrag.index.vector_store import VectorDB


class FakeEmbeddingFunction:
    def __call__(self, input):
        inputs = list(input)
        embeddings = []
        for text in inputs:
            length = float(len(text))
            checksum = float(sum(ord(ch) for ch in text) % 997)
            embeddings.append([length, checksum])
        return embeddings

    @staticmethod
    def name() -> str:
        return "default"

    @staticmethod
    def is_legacy() -> bool:
        return False

    def embed_query(self, input):
        return self.__call__(input)

    @staticmethod
    def supported_spaces() -> list[str]:
        return ["cosine"]

    @staticmethod
    def default_space() -> str:
        return "cosine"

    def get_config(self):
        return {"type": "fake"}

    @staticmethod
    def build_from_config(config):
        return FakeEmbeddingFunction()


class TestVectorDB:
    @pytest.fixture
    def vector_db(self):
        return VectorDB(
            collection_name="test_collection",
            embedding_function=FakeEmbeddingFunction(),
        )

    def test_get_or_create_collection(self, vector_db):
        collection = vector_db.get_or_create_collection()
        assert collection.name == "test_collection"

    def test_add_documents(self, vector_db):
        documents = ["document 1", "document 2", "document 3"]
        vector_db.add_documents(documents)
        stored = vector_db.collection.get()
        assert stored["ids"] is not None
        assert len(stored["ids"]) == len(documents)

    def test_query(self, vector_db):
        documents = ["document 1", "document 2", "document 3"]
        vector_db.add_documents(documents)
        query_texts = ["query 1", "query 2", "query 3"]
        results = vector_db.query(query_texts)
        assert len(results["documents"]) == len(query_texts)
        assert len(results["distances"]) == len(query_texts)
        for docs, distances in zip(results["documents"], results["distances"]):
            assert len(docs) == len(distances)

    def test_openrouter_provider_requires_key(self, vector_db):
        config = AppConfig(
            openai_api_key=None,
            openai_chat_model="gpt-4o-mini",
            
            
            openrouter_api_key=None,
            openrouter_base_url="https://openrouter.ai/api/v1",
            openrouter_chat_model="mistralai/devstral-2512:free",
            openrouter_embeddings_model="qwen/qwen3-embedding-8b",
            local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
            chat_provider="openai",
            embed_provider="openrouter",
            openrouter_referer=None,
            openrouter_title=None,
            openrouter_embed_provider_json=None,
        )
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
            vector_db._build_embedding_function(
                config=config,
                embedding_model=None,
                embed_provider="openrouter",
            )
