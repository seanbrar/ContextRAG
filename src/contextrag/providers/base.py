from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

from openai.types.chat import ChatCompletionMessageParam

MessageParam: TypeAlias = ChatCompletionMessageParam


class ChatProvider(ABC):
    """Abstract base class for chat providers."""

    @abstractmethod
    def complete(
        self,
        messages: list[MessageParam],
        model: str | None = None,
        temperature: float = 0,
    ) -> str:
        """Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Optional model override.
            temperature: Sampling temperature (0-2).

        Returns:
            The model's response content.
        """
        raise NotImplementedError
