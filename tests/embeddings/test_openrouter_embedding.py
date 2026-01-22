import pytest

from contextrag.embeddings.openrouter_embedding import OpenRouterEmbeddingFunction


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
