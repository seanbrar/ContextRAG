"""Application configuration with environment variable loading."""

from __future__ import annotations

import os
from dataclasses import dataclass

from chromaroute import EmbedConfig
from dotenv import load_dotenv


def _env(key: str, default: str | None = None) -> str | None:
    """Get environment variable with optional default."""
    return os.getenv(key, default)


# NOTE: Config duplicates EmbedConfig fields (flat) rather than composing it.
# Alternative: `embed: EmbedConfig` field. Current approach trades duplication
# for simpler environment loading and a single source of truth. See to_embed_config().
@dataclass(frozen=True)
class Config:
    """ContextRAG configuration.

    Single configuration object for the entire application.
    Embedding configuration is converted to chromaroute.EmbedConfig as needed.

    Attributes:
        openrouter_api_key: Shared OpenRouter API key (embeddings and chat).
        openrouter_base_url: OpenRouter API base URL.
        openai_api_key: OpenAI API key for chat.
        chat_provider: Chat provider selection ("openai", "openrouter", "auto").
        openai_chat_model: Model name for OpenAI chat.
        openrouter_chat_model: Model name for OpenRouter chat.
        embed_provider: Embedding provider selection ("openrouter", "local", "auto").
        openrouter_embeddings_model: Model name for OpenRouter embeddings.
        local_embeddings_model: Model name for local embeddings.
        openrouter_referer: Optional referer header for OpenRouter analytics.
        openrouter_title: Optional title for OpenRouter analytics.
        openrouter_provider_json: Optional provider routing JSON for OpenRouter.
    """

    # Shared credentials
    openrouter_api_key: str | None
    openrouter_base_url: str

    # OpenAI credentials
    openai_api_key: str | None

    # Chat configuration
    chat_provider: str  # "openai" | "openrouter" | "auto"
    openai_chat_model: str
    openrouter_chat_model: str

    # Embedding configuration
    embed_provider: str  # "openrouter" | "local" | "auto"
    openrouter_embeddings_model: str
    local_embeddings_model: str

    # Optional OpenRouter settings
    openrouter_referer: str | None
    openrouter_title: str | None
    openrouter_provider_json: str | None

    def resolve_chat_provider(self, explicit_provider: str | None = None) -> str:
        """Resolve which chat provider to use.

        Priority: explicit > config > auto-detect based on API keys.
        """
        provider = (explicit_provider or self.chat_provider or "auto").lower()
        if provider == "auto":
            if self.openai_api_key:
                return "openai"
            if self.openrouter_api_key:
                return "openrouter"
            raise ValueError(
                "No API key available for chat. "
                "Set OPENAI_API_KEY or OPENROUTER_API_KEY."
            )
        return provider

    def resolve_embed_provider(self, explicit_provider: str | None = None) -> str:
        """Resolve which embedding provider to use.

        Priority: explicit > config > auto-detect (openrouter if key, else local).
        """
        provider = (explicit_provider or self.embed_provider or "auto").lower()
        if provider == "auto":
            if self.openrouter_api_key:
                return "openrouter"
            return "local"
        return provider

    def to_embed_config(self) -> EmbedConfig:
        """Convert to chromaroute EmbedConfig for embedding operations."""
        return EmbedConfig(
            openrouter_api_key=self.openrouter_api_key,
            openrouter_base_url=self.openrouter_base_url,
            openrouter_embeddings_model=self.openrouter_embeddings_model,
            openrouter_referer=self.openrouter_referer,
            openrouter_title=self.openrouter_title,
            openrouter_provider_json=self.openrouter_provider_json,
            local_embeddings_model=self.local_embeddings_model,
            embed_provider=self.embed_provider,
        )

    def require_embed_provider(
        self,
        resolved_provider: str,
        explicit_provider: str | None = None,
        error_cls: type[Exception] = ValueError,
    ) -> None:
        """Validate that required API keys are present for the selected provider.

        Raises:
            error_cls: If required API key is missing.
        """
        if resolved_provider == "openrouter" and not self.openrouter_api_key:
            if explicit_provider == "openrouter":
                raise error_cls(
                    "OPENROUTER_API_KEY is required when "
                    "--embed-provider openrouter is selected."
                )
            raise error_cls("OPENROUTER_API_KEY is required for OpenRouter embeddings.")


def load_config() -> Config:
    """Load configuration from environment variables."""
    load_dotenv()

    return Config(
        # Shared credentials
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        openrouter_base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        or "https://openrouter.ai/api/v1",
        # OpenAI credentials
        openai_api_key=_env("OPENAI_API_KEY"),
        # Chat configuration
        chat_provider=_env("CONTEXTRAG_CHAT_PROVIDER", "auto") or "auto",
        openai_chat_model=_env("OPENAI_CHAT_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        openrouter_chat_model=_env("OPENROUTER_CHAT_MODEL", "mistralai/devstral-2512:free")
        or "mistralai/devstral-2512:free",
        # Embedding configuration
        embed_provider=_env("EMBED_PROVIDER", "auto") or "auto",
        openrouter_embeddings_model=_env(
            "OPENROUTER_EMBEDDINGS_MODEL", "openai/text-embedding-3-small"
        )
        or "openai/text-embedding-3-small",
        local_embeddings_model=_env(
            "LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        or "sentence-transformers/all-MiniLM-L6-v2",
        # Optional OpenRouter settings
        openrouter_referer=_env("OPENROUTER_REFERER"),
        openrouter_title=_env("OPENROUTER_TITLE"),
        openrouter_provider_json=_env("OPENROUTER_EMBED_PROVIDER_JSON"),
    )


# Backwards compatibility alias (deprecated)
AppConfig = Config
