from dataclasses import dataclass
import json
import os

from dotenv import load_dotenv


def _env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


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

    def resolve_embed_provider(self, explicit_provider: str | None = None) -> str:
        provider = (
            explicit_provider or self.contextrag_embed_provider or "auto"
        ).lower()
        if provider == "auto":
            if self.openai_api_key:
                provider = "openai"
            elif self.openrouter_api_key:
                provider = "openrouter"
            else:
                provider = "local"
        return provider

    def resolve_embedding_model(
        self,
        provider: str | None = None,
        explicit_model: str | None = None,
    ) -> str:
        if explicit_model:
            return explicit_model
        provider = (provider or self.resolve_embed_provider()).lower()
        if provider == "openai":
            return self.openai_embeddings_model
        if provider == "openrouter":
            return self.openrouter_embeddings_model
        if provider == "local":
            return self.local_embeddings_model
        return "default"

    def openrouter_embed_provider_config(self) -> dict | None:
        if self.openrouter_embed_provider_json:
            try:
                return json.loads(self.openrouter_embed_provider_json)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "OPENROUTER_EMBED_PROVIDER_JSON must be valid JSON."
                ) from exc
        order = None
        if self.openrouter_embed_provider_order:
            order = [
                item.strip()
                for item in self.openrouter_embed_provider_order.split(",")
                if item.strip()
            ]
        allow_fallbacks = None
        if self.openrouter_embed_allow_fallbacks is not None:
            allow_fallbacks = self.openrouter_embed_allow_fallbacks.lower() == "true"
        if not order and allow_fallbacks is None:
            return None
        provider_config: dict[str, object] = {}
        if order:
            provider_config["order"] = order
        if allow_fallbacks is not None:
            provider_config["allow_fallbacks"] = allow_fallbacks
        return provider_config


def load_config() -> AppConfig:
    load_dotenv()
    return AppConfig(
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_embeddings_model=_env(
            "OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"
        ),
        openai_chat_model_short=_env(
            "OPENAI_CHAT_MODEL_SHORT", "gpt-3.5-turbo-1106"
        ),
        openai_chat_model_medium=_env(
            "OPENAI_CHAT_MODEL_MEDIUM", "gpt-3.5-turbo-16k"
        ),
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        openrouter_base_url=_env(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        openrouter_chat_model=_env(
            "OPENROUTER_CHAT_MODEL", "mistralai/devstral-2512:free"
        ),
        openrouter_embeddings_model=_env(
            "OPENROUTER_EMBEDDINGS_MODEL", "qwen/qwen3-embedding-8b"
        ),
        local_embeddings_model=_env(
            "LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        contextrag_chat_provider=_env("CONTEXTRAG_CHAT_PROVIDER", "openai"),
        contextrag_embed_provider=_env("CONTEXTRAG_EMBED_PROVIDER", "auto"),
        openrouter_referer=_env("OPENROUTER_REFERER"),
        openrouter_title=_env("OPENROUTER_TITLE"),
        openrouter_embed_provider_json=_env("OPENROUTER_EMBED_PROVIDER_JSON"),
        openrouter_embed_provider_order=_env("OPENROUTER_EMBED_PROVIDER_ORDER"),
        openrouter_embed_allow_fallbacks=_env(
            "OPENROUTER_EMBED_ALLOW_FALLBACKS"
        ),
    )


def resolve_embed_provider(
    config: AppConfig, explicit_provider: str | None = None
) -> str:
    return config.resolve_embed_provider(explicit_provider)


def require_embedding_provider(
    config: AppConfig,
    resolved_provider: str,
    explicit_provider: str | None = None,
    error_cls: type[Exception] = ValueError,
) -> None:
    if resolved_provider == "openai" and not config.openai_api_key:
        if explicit_provider == "openai":
            raise error_cls(
                "OPENAI_API_KEY is required when --embed-provider openai is selected."
            )
        raise error_cls("OPENAI_API_KEY or OPENROUTER_API_KEY is required for embeddings.")
    if resolved_provider == "openrouter" and not config.openrouter_api_key:
        if explicit_provider == "openrouter":
            raise error_cls(
                "OPENROUTER_API_KEY is required when --embed-provider openrouter is selected."
            )
        raise error_cls("OPENROUTER_API_KEY is required for OpenRouter embeddings.")
