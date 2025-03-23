from typing import List, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletion


class ChatManager:
    def __init__(self):
        self.client = OpenAI()
        self.conversation_history: List[dict] = []

    def complete(
        self,
        model: str,
        user_message: str,
        system_message: Optional[str] = None,
        temperature: float = 0,
    ) -> ChatCompletion:
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
        """Reset conversation history."""
        self.conversation_history = []


# Model constants
class ChatModels:
    GPT_3_5_TURBO_1106 = "gpt-3.5-turbo-1106"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
