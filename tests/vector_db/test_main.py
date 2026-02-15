from __future__ import annotations

import pytest
from chromaroute import EmbedConfig, VectorStore

from contextrag.config import Config


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


class TestVectorStore:
    @pytest.fixture
    def vector_db(self):
        return VectorStore(
            collection_name="test_collection",
            embedding_function=FakeEmbeddingFunction(),
        )

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

    def test_openrouter_provider_requires_key(self):
        from chromaroute import build_embedding_function
        embed = EmbedConfig(
            openrouter_api_key=None,
            openrouter_base_url="https://openrouter.ai/api/v1",
            openrouter_embeddings_model="qwen/qwen3-embedding-8b",
            openrouter_referer=None,
            openrouter_title=None,
            openrouter_provider_json=None,
            local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
            embed_provider="openrouter",
        )
        config = Config(embed=embed)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
            build_embedding_function(
                config=config.embed,
                embed_provider="openrouter",
            )

