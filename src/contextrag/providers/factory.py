"""Chat provider factory with provider selection support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from openai import OpenAI

from contextrag.config import Config, load_config
from contextrag.providers.base import ChatProvider, MessageParam

if TYPE_CHECKING:
    pass


class OpenAIChatProvider(ChatProvider):
    """OpenAI chat provider using the OpenAI SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
    ) -> None:
        """Initialize the OpenAI chat provider.

        Args:
            model: Model name (e.g., "gpt-4o-mini").
            api_key: OpenAI API key.
        """
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def complete(
        self,
        messages: list[MessageParam],
        model: str | None = None,
        temperature: float = 0,
    ) -> str:
        """Generate a chat completion via OpenAI API."""
        response = self._client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""


class OpenRouterChatProvider(ChatProvider):
    """OpenRouter chat provider using the OpenAI SDK with custom base URL."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        """Initialize the OpenRouter chat provider.

        Args:
            model: Model name (e.g., "mistralai/devstral-2512:free").
            api_key: OpenRouter API key.
            base_url: OpenRouter API base URL.
        """
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        messages: list[MessageParam],
        model: str | None = None,
        temperature: float = 0,
    ) -> str:
        """Generate a chat completion via OpenRouter API."""
        response = self._client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""


# Provider factory type
ProviderFactory = Callable[[Config, str | None], ChatProvider]


def _create_openai(config: Config, model: str | None) -> ChatProvider:
    """Factory function for OpenAI provider."""
    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI chat.")
    return OpenAIChatProvider(
        model=model or config.openai_chat_model,
        api_key=config.openai_api_key,
    )


def _create_openrouter(config: Config, model: str | None) -> ChatProvider:
    """Factory function for OpenRouter provider."""
    if not config.embed.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is required for OpenRouter chat.")
    return OpenRouterChatProvider(
        model=model or config.openrouter_chat_model,
        api_key=config.embed.openrouter_api_key,
        base_url=config.embed.openrouter_base_url,
    )


# Provider registry - easy to extend with new providers
_PROVIDERS: dict[str, ProviderFactory] = {
    "openai": _create_openai,
    "openrouter": _create_openrouter,
}


def build_chat_provider(
    config: Config | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> ChatProvider:
    """Build a chat provider with automatic provider selection.

    Args:
        config: Optional Config. If None, loads from environment.
        provider: Optional provider override ("openai", "openrouter", "auto").
        model: Optional model override.

    Returns:
        A ChatProvider instance.

    Raises:
        ValueError: If the provider is unknown or required API key is missing.

    Example:
        >>> provider = build_chat_provider()  # Auto-detect
        >>> provider = build_chat_provider(provider="openai")  # Explicit
        >>> response = provider.complete([{"role": "user", "content": "Hello"}])
    """
    cfg = config or load_config()
    provider_name = cfg.resolve_chat_provider(provider)

    if provider_name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS)
        raise ValueError(
            f"Unknown chat provider: {provider_name!r}. Available: {available}"
        )

    return _PROVIDERS[provider_name](cfg, model)
