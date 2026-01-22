import chromadb
from chromadb.errors import NotFoundError
import json

from chromadb.utils import embedding_functions
from contextrag.config import load_config, resolve_embed_provider
from contextrag.embeddings.openrouter_embedding import OpenRouterEmbeddingFunction


class VectorDB:
    """A class to manage vector database operations using ChromaDB.

    This class provides functionality to create and manage collections in ChromaDB,
    add documents, and perform similarity searches using OpenAI embeddings.

    Attributes:
        collection_name (str): Name of the ChromaDB collection.
        openai_ef: OpenAI embedding function instance.
        collection: ChromaDB collection instance.
    """

    def __init__(
        self,
        collection_name: str,
        persist_path: str | None = None,
        embedding_model: str | None = None,
        embed_provider: str | None = None,
    ):
        """Initialize VectorDB with a collection name.

        Args:
            collection_name (str): Name of the collection to create or load.
        """
        self.collection_name = collection_name
        self.client = (
            chromadb.PersistentClient(path=persist_path)
            if persist_path
            else chromadb.Client()
        )
        config = load_config()
        self.openai_ef = self._build_embedding_function(
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
        provider = resolve_embed_provider(config, embed_provider)

        if provider == "openai":
            model_name = embedding_model or config.openai_embeddings_model
            if not config.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
            return embedding_functions.OpenAIEmbeddingFunction(
                model_name=model_name,
            )
        if provider == "openrouter":
            model_name = embedding_model or config.openrouter_embeddings_model
            if not config.openrouter_api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is required for OpenRouter embeddings."
                )
            provider_config = None
            if config.openrouter_embed_provider_json:
                try:
                    provider_config = json.loads(config.openrouter_embed_provider_json)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "OPENROUTER_EMBED_PROVIDER_JSON must be valid JSON."
                    ) from exc
            else:
                order = None
                if config.openrouter_embed_provider_order:
                    order = [
                        item.strip()
                        for item in config.openrouter_embed_provider_order.split(",")
                        if item.strip()
                    ]
                allow_fallbacks = None
                if config.openrouter_embed_allow_fallbacks is not None:
                    allow_fallbacks = (
                        config.openrouter_embed_allow_fallbacks.lower() == "true"
                    )
                if order or allow_fallbacks is not None:
                    provider_config = {}
                    if order:
                        provider_config["order"] = order
                    if allow_fallbacks is not None:
                        provider_config["allow_fallbacks"] = allow_fallbacks
            return OpenRouterEmbeddingFunction(
                api_key=config.openrouter_api_key,
                model=model_name,
                base_url=config.openrouter_base_url,
                referer=config.openrouter_referer,
                title=config.openrouter_title,
                provider=provider_config,
            )
        if provider == "local":
            model_name = embedding_model or config.local_embeddings_model
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name,
            )
        return embedding_functions.DefaultEmbeddingFunction()

    def get_or_create_collection(self):
        """Get an existing collection or create a new one if it doesn't exist.

        Returns:
            Collection: ChromaDB collection instance.
        """
        try:
            collection = self.client.get_collection(
                self.collection_name,
                embedding_function=self.openai_ef,
            )
            print(f"Loaded existing collection: {self.collection_name}")
        except (ValueError, NotFoundError):
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.openai_ef,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"Created new collection: {self.collection_name}")
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

        print(f"Added {total} documents to the collection")

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


if __name__ == "__main__":
    # Usage example
    vector_db = VectorDB(collection_name="my_collection")

    documents = [
        "The capital of California is Sacramento.",
        "Sacramento is known for its vibrant farm-to-fork dining scene, with numerous restaurants sourcing ingredients from the surrounding agricultural region.",
        "The capital of Texas is Austin.",
        'Austin is famous for its live music scene, earning it the nickname "Live Music Capital of the World."',
        "The capital of Florida is Tallahassee.",
        "Tallahassee is home to the National High Magnetic Field Laboratory, which houses the world's most powerful magnets.",
        "The capital of New York is Albany.",
        "Albany is known for its rich history and architecture, including the Empire State Plaza and the New York State Capitol.",
        "The capital of Illinois is Springfield.",
        "Springfield is famous for its association with Abraham Lincoln, who lived there before becoming the 16th President of the United States.",
        "The capital of Georgia is Atlanta.",
        "Atlanta played a crucial role in the civil rights movement and is home to the Martin Luther King Jr. National Historical Park.",
        "The capital of Ohio is Columbus.",
        "Columbus is known for its vibrant arts scene, including the Columbus Museum of Art and the Wexner Center for the Arts.",
        "The capital of Virginia is Richmond.",
        "Richmond is famous for its Civil War history, with numerous battlefields and historical sites in the area.",
        "The capital of Colorado is Denver.",
        "Denver is known for its proximity to the Rocky Mountains, offering outdoor recreation opportunities such as skiing and hiking.",
        "The capital of Washington is Olympia.",
        "Olympia is home to the Hands On Children's Museum, which features interactive exhibits for children to learn through play.",
    ]

    vector_db.add_documents(documents)

    query_text = "What is the capital of California?"
    results = vector_db.query(query_texts=[query_text])

    print(f"Query: {query_text}")
    for i, result in enumerate(results["documents"][0]):
        print(f"Result {i+1}: {result}")
        print(f"Distance: {results['distances'][0][i]}\n")
