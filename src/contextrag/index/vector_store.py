import chromadb
from chromadb.errors import NotFoundError
from contextrag.config import load_config
from contextrag.core.logging import get_logger
from contextrag.embeddings.provider import build_embedding_function


class VectorDB:
    """A class to manage vector database operations using ChromaDB.

    This class provides functionality to create and manage collections in ChromaDB,
    add documents, and perform similarity searches using OpenAI embeddings.

    Attributes:
        collection_name (str): Name of the ChromaDB collection.
        embedding_function: Embedding function instance.
        collection: ChromaDB collection instance.
    """

    def __init__(
        self,
        collection_name: str,
        persist_path: str | None = None,
        embedding_model: str | None = None,
        embed_provider: str | None = None,
        embedding_function=None,
        client=None,
    ):
        """Initialize VectorDB with a collection name.

        Args:
            collection_name (str): Name of the collection to create or load.
        """
        self.collection_name = collection_name
        self._logger = get_logger(__name__)
        self.client = client or (
            chromadb.PersistentClient(path=persist_path)
            if persist_path
            else chromadb.Client()
        )
        if embedding_function is not None:
            self.embedding_function = embedding_function
        else:
            config = load_config()
            self.embedding_function = self._build_embedding_function(
                config=config,
                embedding_model=embedding_model,
                embed_provider=embed_provider,
            )
        self.collection = self.get_or_create_collection()

    def _build_embedding_function(
        self,
        config,
        embedding_model: str | None,
        embed_provider: str | None,
    ):
        return build_embedding_function(
            config=config,
            embedding_model=embedding_model,
            embed_provider=embed_provider,
        )

    def get_or_create_collection(self):
        """Get an existing collection or create a new one if it doesn't exist.

        Returns:
            Collection: ChromaDB collection instance.
        """
        logger = getattr(self, "_logger", get_logger(__name__))
        try:
            collection = self.client.get_collection(
                self.collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info("Loaded existing collection: %s", self.collection_name)
        except (ValueError, NotFoundError):
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Created new collection: %s", self.collection_name)
        return collection

    def add_documents(self, documents, ids=None, batch_size: int = 100):
        """Add documents to the collection with batching.

        Args:
            documents (list): List of document strings to add.
            ids (list, optional): List of unique IDs for the documents.
                If not provided, sequential IDs will be generated.
            batch_size (int): Number of documents per batch to avoid API limits.
                Defaults to 100.
        """
        if ids is None:
            ids = [f"doc{i+1}" for i in range(len(documents))]

        total = len(documents)
        for i in range(0, total, batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            self.collection.add(documents=batch_docs, ids=batch_ids)

        logger = getattr(self, "_logger", get_logger(__name__))
        logger.info("Added %s documents to the collection", total)

    def query(self, query_texts, n_results=3):
        """Query the collection for similar documents.

        Args:
            query_texts (list): List of query strings.
            n_results (int, optional): Number of results to return. Defaults to 3.

        Returns:
            dict: Query results containing documents and their distances.
        """
        results = self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            include=["documents", "distances"],
        )
        return results
