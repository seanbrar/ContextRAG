from chromadb.utils import embedding_functions

from contextrag.config import AppConfig, resolve_embed_provider
from contextrag.embeddings.openrouter_embedding import OpenRouterEmbeddingFunction


def build_embedding_function(
    config: AppConfig,
    embedding_model: str | None = None,
    embed_provider: str | None = None,
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
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter embeddings.")
        return OpenRouterEmbeddingFunction(
            api_key=config.openrouter_api_key,
            model=model_name,
            base_url=config.openrouter_base_url,
            referer=config.openrouter_referer,
            title=config.openrouter_title,
            provider=config.openrouter_embed_provider_config(),
        )
    if provider == "local":
        model_name = embedding_model or config.local_embeddings_model
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
        )
    return embedding_functions.DefaultEmbeddingFunction()
