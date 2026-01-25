"""Application configuration with environment variable loading."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from chromaroute import EmbedConfig


def _env(key: str, default: str | None = None) -> str | None:
    """Get environment variable with optional default."""
    return os.getenv(key, default)


@dataclass(frozen=True)
class AppConfig:
    """ContextRAG configuration.

    Chat providers are managed by ContextRAG.
    Embedding configuration is delegated to chromaroute via EmbedConfig.

    Attributes:
        openai_api_key: OpenAI API key for chat.
        openai_chat_model: Model name for OpenAI chat.
        openrouter_chat_model: Model name for OpenRouter chat.
        chat_provider: Chat provider selection ("openai", "openrouter", "auto").
        embed_config: Embedding configuration (passed to chromaroute).
    """

    # Chat providers (ContextRAG-specific)
    openai_api_key: str | None
    openai_chat_model: str
    openrouter_chat_model: str
    chat_provider: str  # "openai" | "openrouter" | "auto"

    # Embeddings (delegated to chromaroute)
    embed_config: EmbedConfig

    @property
    def openrouter_api_key(self) -> str | None:
        """OpenRouter API key (from embed_config, shared with chat)."""
        return self.embed_config.openrouter_api_key

    @property
    def openrouter_base_url(self) -> str:
        """OpenRouter base URL (from embed_config, shared with chat)."""
        return self.embed_config.openrouter_base_url

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
                raise ValueError(
                    "No API key available for chat. "
                    "Set OPENAI_API_KEY or OPENROUTER_API_KEY."
                )
        return provider

    def resolve_embed_provider(self, explicit_provider: str | None = None) -> str:
        """Resolve which embedding provider to use.

        Delegates to embed_config.resolve_provider().
        """
        return self.embed_config.resolve_provider(explicit_provider)


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    load_dotenv()

    embed_config = EmbedConfig(
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        openrouter_base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        or "https://openrouter.ai/api/v1",
        openrouter_embeddings_model=_env(
            "OPENROUTER_EMBEDDINGS_MODEL", "openai/text-embedding-3-small"
        )
        or "openai/text-embedding-3-small",
        openrouter_referer=_env("OPENROUTER_REFERER"),
        openrouter_title=_env("OPENROUTER_TITLE"),
        openrouter_provider_json=_env("OPENROUTER_EMBED_PROVIDER_JSON"),
        local_embeddings_model=_env(
            "LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        or "sentence-transformers/all-MiniLM-L6-v2",
        embed_provider=_env("EMBED_PROVIDER", "auto") or "auto",
    )

    return AppConfig(
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_chat_model=_env("OPENAI_CHAT_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        openrouter_chat_model=_env("OPENROUTER_CHAT_MODEL", "mistralai/devstral-2512:free")
        or "mistralai/devstral-2512:free",
        chat_provider=_env("CONTEXTRAG_CHAT_PROVIDER", "auto") or "auto",
        embed_config=embed_config,
    )


def resolve_embed_provider(
    config: AppConfig, explicit_provider: str | None = None
) -> str:
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
                "OPENROUTER_API_KEY is required when "
                "--embed-provider openrouter is selected."
            )
        raise error_cls("OPENROUTER_API_KEY is required for OpenRouter embeddings.")
