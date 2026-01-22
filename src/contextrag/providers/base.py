from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class EmbeddingsProvider(ABC):
    @abstractmethod
    def embed(
        self, texts: Iterable[str], model: str | None = None
    ) -> list[list[float]]:
        raise NotImplementedError


class ChatProvider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0,
    ) -> str:
        raise NotImplementedError
