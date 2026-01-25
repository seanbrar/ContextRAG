from contextrag.providers.openai_chat import ChatManager, OpenAIChatProvider
from contextrag.config import AppConfig


def test_openai_chat_provider_complete(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = self
            self.completions = self

        def create(self, model, messages, temperature):
            class Choice:
                message = type("Message", (), {"content": "ok"})()

            return type("Response", (), {"choices": [Choice()]})()

    config = AppConfig(
        openai_api_key="key",
        openai_chat_model="gpt-4o-mini",
        
        
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        chat_provider="openai",
        embed_provider="auto",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
    )

    monkeypatch.setattr("contextrag.providers.openai_chat.load_config", lambda: config)
    provider = OpenAIChatProvider(client=FakeClient())
    content = provider.complete([{"role": "user", "content": "hi"}])
    assert content == "ok"


def test_chat_manager_complete_and_reset(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, model, messages, temperature):
            captured["model"] = model
            captured["messages"] = messages
            captured["temperature"] = temperature

            class Choice:
                message = type("Message", (), {"content": "hello"})()

            return type("Response", (), {"choices": [Choice()]})()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    monkeypatch.setattr("contextrag.providers.openai_chat.OpenAI", lambda: FakeClient())
    manager = ChatManager()
    response = manager.complete(
        "gpt-3.5-turbo-1106",
        "user message",
        system_message="system",
        temperature=0.5,
    )
    assert response.choices[0].message.content == "hello"
    assert captured["messages"][0]["role"] == "system"
    manager.conversation_history = [{"role": "user", "content": "old"}]
    manager.reset()
    assert manager.conversation_history == []
