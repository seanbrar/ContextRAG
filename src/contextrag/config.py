from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None
    openai_embeddings_model: str
    openai_chat_model_short: str
    openai_chat_model_medium: str
    openrouter_api_key: str | None
    openrouter_base_url: str
    openrouter_chat_model: str
    contextrag_chat_provider: str


def load_config() -> AppConfig:
    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_embeddings_model=os.getenv(
            "OPENAI_EMBEDDINGS_MODEL", "thenlper/gte-base"
        ),
        openai_chat_model_short=os.getenv(
            "OPENAI_CHAT_MODEL_SHORT", "gpt-3.5-turbo-1106"
        ),
        openai_chat_model_medium=os.getenv(
            "OPENAI_CHAT_MODEL_MEDIUM", "gpt-3.5-turbo-16k"
        ),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        openrouter_chat_model=os.getenv(
            "OPENROUTER_CHAT_MODEL", "mistralai/devstral-2512:free"
        ),
        contextrag_chat_provider=os.getenv("CONTEXTRAG_CHAT_PROVIDER", "openai"),
    )
