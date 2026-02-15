"""Application configuration with environment variable loading."""

from __future__ import annotations

from dataclasses import dataclass

from chromaroute import EmbedConfig
from chromaroute import load_config as load_embed_config


@dataclass(frozen=True)
class Config:
    """ContextRAG configuration.

    Uses composition: embedding config is delegated to chromaroute.EmbedConfig.

    Attributes:
        embed: Embedding configuration (chromaroute.EmbedConfig).
    """

    embed: EmbedConfig

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


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(embed=load_embed_config())
