import os
import pytest
from contextrag.index.vector_store import VectorDB


pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set for OpenAI embeddings.",
)


class TestVectorDB:
    @pytest.fixture
    def vector_db(self):
        return VectorDB(collection_name="test_collection")

    def test_get_or_create_collection(self, vector_db):
        collection = vector_db.get_or_create_collection()
        assert collection.name == "test_collection"

    def test_add_documents(self, vector_db):
        documents = ["document 1", "document 2", "document 3"]
        vector_db.add_documents(documents)
        # Assert that the documents were added successfully
        assert len(vector_db.collection.documents) == len(documents)

    def test_query(self, vector_db):
        query_texts = ["query 1", "query 2", "query 3"]
        results = vector_db.query(query_texts)
        # Assert that the number of results matches the expected number
        assert len(results) == len(query_texts)

        # Assert that each result contains the required fields
        for result in results:
            assert "documents" in result
            assert "distances" in result
