import pytest
import requests

from chromaroute import OpenRouterEmbeddingFunction


def test_openrouter_embedding_success(monkeypatch):
    def fake_post(url, headers, json, timeout):
        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "data": [
                        {"embedding": [0.1, 0.2], "index": 0},
                        {"embedding": [0.3, 0.4], "index": 1},
                    ]
                }

        return Response()

    monkeypatch.setattr("requests.post", fake_post)
    embedding_fn = OpenRouterEmbeddingFunction(
        model="thenlper/gte-base",
        api_key="test-key",
    )
    result = embedding_fn(["a", "b"])
    assert len(result) == 2
    assert pytest.approx(result[0].tolist()) == [0.1, 0.2]
    assert pytest.approx(result[1].tolist()) == [0.3, 0.4]


def test_openrouter_embedding_http_error(monkeypatch):
    def fake_post(url, headers, json, timeout):
        class Response:
            status_code = 404
            text = "not found"

            @staticmethod
            def json():
                return {"error": {"message": "not found"}}

        return Response()

    monkeypatch.setattr("requests.post", fake_post)
    embedding_fn = OpenRouterEmbeddingFunction(
        model="thenlper/gte-base",
        api_key="test-key",
    )
    with pytest.raises(ValueError, match="HTTP 404"):
        embedding_fn(["a"])


def test_openrouter_embedding_missing_data(monkeypatch):
    def fake_post(url, headers, json, timeout):
        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {"data": []}

        return Response()

    monkeypatch.setattr("requests.post", fake_post)
    embedding_fn = OpenRouterEmbeddingFunction(
        model="thenlper/gte-base",
        api_key="test-key",
    )
    with pytest.raises(ValueError, match="No embedding data received"):
        embedding_fn(["a"])


def test_openrouter_embedding_request_exception(monkeypatch):
    def fake_post(url, headers, json, timeout):
        raise requests.exceptions.RequestException("boom")

    monkeypatch.setattr("requests.post", fake_post)
    embedding_fn = OpenRouterEmbeddingFunction(
        model="thenlper/gte-base",
        api_key="test-key",
    )
    with pytest.raises(ValueError, match="OpenRouter embeddings request failed"):
        embedding_fn(["a"])


def test_openrouter_embedding_missing_embedding(monkeypatch):
    def fake_post(url, headers, json, timeout):
        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {"data": [{"index": 0}]}

        return Response()

    monkeypatch.setattr("requests.post", fake_post)
    embedding_fn = OpenRouterEmbeddingFunction(
        model="thenlper/gte-base",
        api_key="test-key",
    )
    with pytest.raises(ValueError, match="Missing embedding"):
        embedding_fn(["a"])


def test_openrouter_embedding_http_error_hint(monkeypatch):
    def fake_post(url, headers, json, timeout):
        class Response:
            status_code = 401
            text = "unauthorized"

            @staticmethod
            def json():
                return {"error": {"message": "unauthorized"}}

        return Response()

    monkeypatch.setattr("requests.post", fake_post)
    embedding_fn = OpenRouterEmbeddingFunction(
        model="thenlper/gte-base",
        api_key="test-key",
    )
    with pytest.raises(ValueError, match="HTTP 401"):
        embedding_fn(["a"])


def test_openrouter_embedding_build_from_config(monkeypatch):
    config = {
        "api_key_env_var": "OPENROUTER_API_KEY",
        "model": "thenlper/gte-base",
        "base_url": "https://openrouter.ai/api/v1",
        "referer": "https://example.com",
        "title": "test",
        "provider": {"order": ["x"]},
        "timeout_s": 1,
    }
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    embedding_fn = OpenRouterEmbeddingFunction.build_from_config(config)
    assert embedding_fn.model == "thenlper/gte-base"
    assert embedding_fn.base_url == "https://openrouter.ai/api/v1"


def test_openrouter_embedding_config_helpers():
    embedding_fn = OpenRouterEmbeddingFunction(
        model="thenlper/gte-base",
        api_key="test-key",
    )
    assert embedding_fn.name() == "openrouter"
    config = embedding_fn.get_config()
    assert config["model"] == "thenlper/gte-base"
    assert embedding_fn.default_space() == "cosine"
    assert "cosine" in embedding_fn.supported_spaces()


def test_openrouter_embedding_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterEmbeddingFunction(model="thenlper/gte-base", api_key=None)


def test_openrouter_embedding_empty_input():
    embedding_fn = OpenRouterEmbeddingFunction(
        model="thenlper/gte-base",
        api_key="test-key",
    )
    with pytest.raises(ValueError, match="non-empty"):
        embedding_fn([])
