"""Application configuration with environment variable loading."""

from __future__ import annotations

import os
from dataclasses import dataclass

from chromaroute import EmbedConfig
from chromaroute import load_config as load_embed_config


def _env(key: str, default: str | None = None) -> str | None:
    """Get environment variable with optional default."""
    return os.getenv(key, default)


@dataclass(frozen=True)
class Config:
    """ContextRAG configuration.

    Uses composition: embedding config is delegated to chromaroute.EmbedConfig.
    Chat-specific configuration is managed here.

    Attributes:
        embed: Embedding configuration (chromaroute.EmbedConfig).
        openai_api_key: OpenAI API key for chat.
        chat_provider: Chat provider selection ("openai", "openrouter", "auto").
        openai_chat_model: Model name for OpenAI chat.
        openrouter_chat_model: Model name for OpenRouter chat.
    """

    # Embedding configuration (delegated to chromaroute)
    embed: EmbedConfig

    # OpenAI credentials (chat only - OpenRouter key is in embed)
    openai_api_key: str | None

    # Chat configuration
    chat_provider: str  # "openai" | "openrouter" | "auto"
    openai_chat_model: str
    openrouter_chat_model: str

    def resolve_chat_provider(self, explicit_provider: str | None = None) -> str:
        """Resolve which chat provider to use.

        Priority: explicit > config > auto-detect based on API keys.
        """
        provider = (explicit_provider or self.chat_provider or "auto").lower()
        if provider == "auto":
            if self.openai_api_key:
                return "openai"
            if self.embed.openrouter_api_key:
                return "openrouter"
            raise ValueError(
                "No API key available for chat. "
                "Set OPENAI_API_KEY or OPENROUTER_API_KEY."
            )
        return provider

    def resolve_embed_provider(self, explicit_provider: str | None = None) -> str:
        """Resolve which embedding provider to use.

        Delegates to chromaroute.EmbedConfig.resolve_provider().
        """
        return self.embed.resolve_provider(explicit_provider)

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
        if resolved_provider == "openrouter" and not self.embed.openrouter_api_key:
            if explicit_provider == "openrouter":
                raise error_cls(
                    "OPENROUTER_API_KEY is required when "
                    "--embed-provider openrouter is selected."
                )
            raise error_cls("OPENROUTER_API_KEY is required for OpenRouter embeddings.")

        if resolved_provider == "openai" and not self.openai_api_key:
            if explicit_provider == "openai":
                raise error_cls(
                    "OPENAI_API_KEY is required when "
                    "--embed-provider openai is selected."
                )
            raise error_cls("OPENAI_API_KEY is required for OpenAI embeddings.")


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        # Embedding configuration (delegated to chromaroute)
        embed=load_embed_config(),
        # OpenAI credentials
        openai_api_key=_env("OPENAI_API_KEY"),
        # Chat configuration
        chat_provider=_env("CONTEXTRAG_CHAT_PROVIDER", "auto") or "auto",
        openai_chat_model=_env("OPENAI_CHAT_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        openrouter_chat_model=_env("OPENROUTER_CHAT_MODEL", "mistralai/devstral-2512:free")
        or "mistralai/devstral-2512:free",
    )
