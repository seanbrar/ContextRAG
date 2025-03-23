from typing import List, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletion


class ChatManager:
    """A class to manage chat interactions with OpenAI models.

    This class provides functionality to interact with OpenAI's chat models,
    maintaining conversation history and handling message completions.

    Attributes:
        client: OpenAI client instance for API interactions.
        conversation_history: List of conversation messages.
    """

    def __init__(self):
        """Initialize ChatManager with OpenAI client and empty conversation history."""
        self.client = OpenAI()
        self.conversation_history: List[dict] = []

    def complete(
        self,
        model: str,
        user_message: str,
        system_message: Optional[str] = None,
        temperature: float = 0,
    ) -> ChatCompletion:
        """Generate a chat completion using the specified model.

        Args:
            model: The OpenAI model identifier to use for completion.
            user_message: The message content from the user.
            system_message: Optional system message to set context.
            temperature: Sampling temperature (0-2). Lower means more focused.

        Returns:
            ChatCompletion: The model's response.
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": user_message})

        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

    def reset(self):
        """Reset conversation history to empty state."""
        self.conversation_history = []


# Model constants
class ChatModels:
    """Constants for available OpenAI chat model identifiers."""

    GPT_3_5_TURBO_1106 = "gpt-3.5-turbo-1106"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
