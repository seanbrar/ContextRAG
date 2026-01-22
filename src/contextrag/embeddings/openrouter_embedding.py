from __future__ import annotations

import os
from typing import Any, Dict, Iterable

import requests
from requests import exceptions as requests_exceptions

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function


@register_embedding_function
class OpenRouterEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        referer: str | None = None,
        title: str | None = None,
        provider: dict | None = None,
        timeout_s: int = 60,
        api_key_env_var: str = "OPENROUTER_API_KEY",
    ) -> None:
        self.api_key_env_var = api_key_env_var
        self.api_key = api_key or os.getenv(self.api_key_env_var)
        if not self.api_key:
            raise ValueError(
                f"The {self.api_key_env_var} environment variable is not set."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.referer = referer
        self.title = title
        self.provider = provider
        self.timeout_s = timeout_s

    def __call__(self, input: Documents) -> Embeddings:
        inputs = list(input)
        if not inputs:
            return []

        payload = {
            "model": self.model,
            "input": inputs,
            "encoding_format": "float",
        }
        if self.provider:
            payload["provider"] = self.provider
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title

        url = f"{self.base_url}/embeddings"
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
            )
        except requests_exceptions.RequestException as exc:
            message = (
                "OpenRouter embeddings request failed. "
                f"URL: {url}. "
                "Check network/DNS connectivity and OPENROUTER_BASE_URL."
            )
            raise ValueError(message) from exc
        if response.status_code != 200:
            status = response.status_code
            hint = ""
            if status == 401:
                hint = " Verify OPENROUTER_API_KEY."
            elif status == 402:
                hint = " Check OpenRouter credits."
            elif status == 404:
                hint = " Verify OPENROUTER_EMBEDDINGS_MODEL."
            elif status == 429:
                hint = " Rate limit exceeded; retry later."
            elif status == 529:
                hint = " Provider overloaded; consider allow_fallbacks."
            raise ValueError(
                "OpenRouter embeddings failed: "
                f"HTTP {status} {response.text}.{hint}"
            )
        data = response.json()
        if "data" not in data or not data["data"]:
            raise ValueError(f"No embedding data received: {data}")
        embeddings: list[list[float]] = []
        for item in data["data"]:
            embedding = item.get("embedding")
            if embedding is None:
                raise ValueError(f"Missing embedding in response: {item}")
            embeddings.append(embedding)
        return embeddings

    @staticmethod
    def name() -> str:
        return "openrouter"

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "OpenRouterEmbeddingFunction":
        api_key_env_var = config.get("api_key_env_var")
        model = config.get("model")
        base_url = config.get("base_url")
        referer = config.get("referer")
        title = config.get("title")
        provider = config.get("provider")
        timeout_s = config.get("timeout_s")
        if api_key_env_var is None or model is None:
            assert False, "This code should not be reached"
        return OpenRouterEmbeddingFunction(
            api_key_env_var=api_key_env_var,
            model=model,
            base_url=base_url,
            referer=referer,
            title=title,
            provider=provider,
            timeout_s=timeout_s,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "api_key_env_var": self.api_key_env_var,
            "model": self.model,
            "base_url": self.base_url,
            "referer": self.referer,
            "title": self.title,
            "provider": self.provider,
            "timeout_s": self.timeout_s,
        }

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]
