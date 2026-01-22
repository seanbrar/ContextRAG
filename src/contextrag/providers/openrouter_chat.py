from __future__ import annotations

from openai import OpenAI

from contextrag.config import load_config
from contextrag.providers.base import ChatProvider


class OpenRouterChatProvider(ChatProvider):
    def __init__(
        self,
        model: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        config = load_config()
        if not config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter.")

        self.model = model or config.openrouter_chat_model
        self.client = client or OpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
