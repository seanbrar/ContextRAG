from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None
    openai_embeddings_model: str
    openai_chat_model_short: str
    openai_chat_model_medium: str
    openrouter_api_key: str | None
    openrouter_base_url: str
    openrouter_chat_model: str
    openrouter_embeddings_model: str
    local_embeddings_model: str
    contextrag_chat_provider: str
    contextrag_embed_provider: str
    openrouter_referer: str | None
    openrouter_title: str | None
    openrouter_embed_provider_json: str | None
    openrouter_embed_provider_order: str | None
    openrouter_embed_allow_fallbacks: str | None


def load_config() -> AppConfig:
    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_embeddings_model=os.getenv(
            "OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"
        ),
        openai_chat_model_short=os.getenv(
            "OPENAI_CHAT_MODEL_SHORT", "gpt-3.5-turbo-1106"
        ),
        openai_chat_model_medium=os.getenv(
            "OPENAI_CHAT_MODEL_MEDIUM", "gpt-3.5-turbo-16k"
        ),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        openrouter_chat_model=os.getenv(
            "OPENROUTER_CHAT_MODEL", "mistralai/devstral-2512:free"
        ),
        openrouter_embeddings_model=os.getenv(
            "OPENROUTER_EMBEDDINGS_MODEL", "qwen/qwen3-embedding-8b"
        ),
        local_embeddings_model=os.getenv(
            "LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        contextrag_chat_provider=os.getenv("CONTEXTRAG_CHAT_PROVIDER", "openai"),
        contextrag_embed_provider=os.getenv("CONTEXTRAG_EMBED_PROVIDER", "auto"),
        openrouter_referer=os.getenv("OPENROUTER_REFERER"),
        openrouter_title=os.getenv("OPENROUTER_TITLE"),
        openrouter_embed_provider_json=os.getenv("OPENROUTER_EMBED_PROVIDER_JSON"),
        openrouter_embed_provider_order=os.getenv("OPENROUTER_EMBED_PROVIDER_ORDER"),
        openrouter_embed_allow_fallbacks=os.getenv(
            "OPENROUTER_EMBED_ALLOW_FALLBACKS"
        ),
    )


def resolve_embed_provider(
    config: AppConfig, explicit_provider: str | None = None
) -> str:
    provider = (explicit_provider or config.contextrag_embed_provider or "auto").lower()
    if provider == "auto":
        if config.openai_api_key:
            provider = "openai"
        elif config.openrouter_api_key:
            provider = "openrouter"
        else:
            provider = "local"
    return provider
