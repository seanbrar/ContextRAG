"""Application configuration with environment variable loading."""

from dataclasses import dataclass
import json
import os

from dotenv import load_dotenv


def _env(key: str, default: str | None = None) -> str | None:
    """Get environment variable with optional default."""
    return os.getenv(key, default)


@dataclass(frozen=True)
class AppConfig:
    """ContextRAG configuration.

    Chat providers are managed by ContextRAG.
    Embedding configuration is passed through to chromaroute.
    """

    # Chat providers (ContextRAG-managed)
    openai_api_key: str | None
    openai_chat_model: str
    openrouter_api_key: str | None
    openrouter_base_url: str
    openrouter_chat_model: str
    chat_provider: str  # "openai" | "openrouter" | "auto"

    # Embeddings (passed to chromaroute)
    embed_provider: str  # "openrouter" | "local" | "auto"
    openrouter_embeddings_model: str
    local_embeddings_model: str
    openrouter_referer: str | None
    openrouter_title: str | None
    openrouter_embed_provider_json: str | None

    def resolve_chat_provider(self, explicit_provider: str | None = None) -> str:
        """Resolve which chat provider to use.

        Priority: explicit > config > auto-detect based on API keys.
        """
        provider = (explicit_provider or self.chat_provider or "auto").lower()
        if provider == "auto":
            if self.openai_api_key:
                provider = "openai"
            elif self.openrouter_api_key:
                provider = "openrouter"
            else:
                raise ValueError("No API key available for chat. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")
        return provider

    def resolve_embed_provider(self, explicit_provider: str | None = None) -> str:
        """Resolve which embedding provider to use.

        Priority: explicit > config > auto-detect (openrouter if key, else local).
        """
        provider = (explicit_provider or self.embed_provider or "auto").lower()
        if provider == "auto":
            provider = "openrouter" if self.openrouter_api_key else "local"
        return provider

    def openrouter_embed_provider_config(self) -> dict | None:
        """Parse OpenRouter provider routing JSON config."""
        if not self.openrouter_embed_provider_json:
            return None
        try:
            return json.loads(self.openrouter_embed_provider_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "OPENROUTER_EMBED_PROVIDER_JSON must be valid JSON."
            ) from exc


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    load_dotenv()
    return AppConfig(
        # Chat
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_chat_model=_env("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        openrouter_base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openrouter_chat_model=_env("OPENROUTER_CHAT_MODEL", "mistralai/devstral-2512:free"),
        chat_provider=_env("CONTEXTRAG_CHAT_PROVIDER", "auto"),
        # Embeddings
        embed_provider=_env("EMBED_PROVIDER", "auto"),
        openrouter_embeddings_model=_env("OPENROUTER_EMBEDDINGS_MODEL", "openai/text-embedding-3-small"),
        local_embeddings_model=_env("LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        openrouter_referer=_env("OPENROUTER_REFERER"),
        openrouter_title=_env("OPENROUTER_TITLE"),
        openrouter_embed_provider_json=_env("OPENROUTER_EMBED_PROVIDER_JSON"),
    )


# Backwards-compatible helper functions
def resolve_embed_provider(config: AppConfig, explicit_provider: str | None = None) -> str:
    """Resolve embedding provider. Delegates to config.resolve_embed_provider()."""
    return config.resolve_embed_provider(explicit_provider)


def require_embedding_provider(
    config: AppConfig,
    resolved_provider: str,
    explicit_provider: str | None = None,
    error_cls: type[Exception] = ValueError,
) -> None:
    """Validate that required API keys are present for the selected provider."""
    if resolved_provider == "openrouter" and not config.openrouter_api_key:
        if explicit_provider == "openrouter":
            raise error_cls(
                "OPENROUTER_API_KEY is required when --embed-provider openrouter is selected."
            )
        raise error_cls("OPENROUTER_API_KEY is required for OpenRouter embeddings.")
