import json

from click.testing import CliRunner

from contextrag import cli
from contextrag.core import io
from contextrag.cli import main
from contextrag.config import AppConfig


def test_ingest_markdown_command(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "note.md").write_text("Hello world", encoding="utf-8")

    monkeypatch.setattr("contextrag.cli.modify_markdown", lambda text: text + "!")
    monkeypatch.setattr("contextrag.cli.count_tokens", lambda text: 2)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "ingest",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--format",
            "markdown",
        ],
    )
    assert result.exit_code == 0

    output_file = output_dir / "note.md"
    assert output_file.read_text(encoding="utf-8") == "Hello world!"
    manifest = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest[0]["tokens"] == 2


def test_ingest_markdown_respects_token_bounds(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "keep.md").write_text("keep", encoding="utf-8")
    (input_dir / "skip.md").write_text("skip", encoding="utf-8")

    def fake_count_tokens(text):
        return {"keep": 3, "skip": 6}[text]

    monkeypatch.setattr("contextrag.cli.modify_markdown", lambda text: text)
    monkeypatch.setattr("contextrag.cli.count_tokens", fake_count_tokens)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "ingest",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--min-tokens",
            "2",
            "--max-tokens",
            "5",
        ],
    )
    assert result.exit_code == 0
    manifest = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest) == 1
    assert "keep.md" in manifest[0]["source"]


def test_ingest_auto_html_uses_converter(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "doc.html").write_text("<p>hi</p>", encoding="utf-8")

    class FakeConverter:
        def __init__(self, folder):
            self.folder = folder

        def _read_html_file(self, file_path):
            return "<p>hi</p>"

        def _remove_html_footer(self, html_content):
            return html_content

        def _html_to_markdown(self, html_content):
            return "hi"

    monkeypatch.setattr("contextrag.cli.HTMLToMarkdownConverter", FakeConverter)
    monkeypatch.setattr("contextrag.cli.count_tokens", lambda text: 1)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "ingest",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--format",
            "auto",
        ],
    )
    assert result.exit_code == 0
    assert (output_dir / "doc.md").exists()


def test_route_command_buckets(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "short.md").write_text("short", encoding="utf-8")
    (input_dir / "medium.md").write_text("medium", encoding="utf-8")
    (input_dir / "long.md").write_text("long", encoding="utf-8")

    def fake_count_tokens(text):
        return {"short": 10, "medium": 500, "long": 2000}[text]

    monkeypatch.setattr("contextrag.cli.count_tokens", fake_count_tokens)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "route",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--short-max",
            "100",
            "--medium-max",
            "1000",
        ],
    )
    assert result.exit_code == 0
    assert (output_dir / "short" / "short.md").exists()
    assert (output_dir / "medium" / "medium.md").exists()
    assert (output_dir / "long" / "long.md").exists()


def test_embed_command_writes_cache(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    cache_path = tmp_path / "cache.json"
    input_dir.mkdir()
    (input_dir / "doc.md").write_text("hello", encoding="utf-8")

    config = AppConfig(
        openai_api_key="key",
        openai_embeddings_model="model",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="openai",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    )

    monkeypatch.setattr("contextrag.cli.load_config", lambda: config)
    monkeypatch.setattr("contextrag.cli.count_tokens", lambda text: 1)

    captured = {}

    def fake_build_embedding_function(config, embedding_model, embed_provider):
        captured["model"] = embedding_model
        captured["provider"] = embed_provider

        def embed(texts):
            assert texts == ["hello"]
            return [[0.1, 0.2, 0.3]]

        return embed

    monkeypatch.setattr(
        "contextrag.cli.build_embedding_function", fake_build_embedding_function
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "embed",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--cache",
            str(cache_path),
        ],
    )
    assert result.exit_code == 0
    assert cache_path.exists()
    assert captured["provider"] == "openai"
    assert captured["model"] is None

    rows = [
        json.loads(line)
        for line in (output_dir / "embeddings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["embedding"] == [0.1, 0.2, 0.3]


def test_embed_command_requires_api_key(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "doc.md").write_text("hello", encoding="utf-8")

    config = AppConfig(
        openai_api_key=None,
        openai_embeddings_model="model",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="openai",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    )
    monkeypatch.setattr("contextrag.cli.load_config", lambda: config)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["embed", "--input", str(input_dir), "--output", str(output_dir)],
    )
    assert result.exit_code != 0
    assert "OPENAI_API_KEY or OPENROUTER_API_KEY" in result.output


def test_embed_command_uses_openrouter_cache(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    cache_path = tmp_path / "cache.json"
    input_dir.mkdir()
    (input_dir / "doc.md").write_text("hello", encoding="utf-8")

    config = AppConfig(
        openai_api_key=None,
        openai_embeddings_model="model",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key="key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="auto",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    )
    monkeypatch.setattr("contextrag.cli.load_config", lambda: config)
    monkeypatch.setattr("contextrag.cli.count_tokens", lambda text: 1)

    checksum = io.checksum("hello")
    cache_path.write_text(json.dumps({checksum: [0.9]}), encoding="utf-8")

    captured = {}

    def fake_build_embedding_function(config, embedding_model, embed_provider):
        captured["provider"] = embed_provider

        def embed(texts):
            raise AssertionError("should use cache")

        return embed

    monkeypatch.setattr(
        "contextrag.cli.build_embedding_function", fake_build_embedding_function
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "embed",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--cache",
            str(cache_path),
        ],
    )
    assert result.exit_code == 0
    assert captured["provider"] == "openrouter"


def test_embed_command_allows_local_provider_without_keys(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "doc.md").write_text("hello", encoding="utf-8")

    config = AppConfig(
        openai_api_key=None,
        openai_embeddings_model="model",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="auto",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    )
    monkeypatch.setattr("contextrag.cli.load_config", lambda: config)
    monkeypatch.setattr("contextrag.cli.count_tokens", lambda text: 1)

    captured = {}

    def fake_build_embedding_function(config, embedding_model, embed_provider):
        captured["provider"] = embed_provider

        def embed(texts):
            return [[0.5]]

        return embed

    monkeypatch.setattr(
        "contextrag.cli.build_embedding_function", fake_build_embedding_function
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "embed",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--embed-provider",
            "local",
        ],
    )
    assert result.exit_code == 0
    assert captured["provider"] == "local"
    rows = [
        json.loads(line)
        for line in (output_dir / "embeddings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["embedding"] == [0.5]


def test_embed_command_uses_configured_local_provider(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "doc.md").write_text("hello", encoding="utf-8")

    config = AppConfig(
        openai_api_key=None,
        openai_embeddings_model="model",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="local",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    )
    monkeypatch.setattr("contextrag.cli.load_config", lambda: config)
    monkeypatch.setattr("contextrag.cli.count_tokens", lambda text: 1)

    captured = {}

    def fake_build_embedding_function(config, embedding_model, embed_provider):
        captured["provider"] = embed_provider

        def embed(texts):
            return [[0.7]]

        return embed

    monkeypatch.setattr(
        "contextrag.cli.build_embedding_function", fake_build_embedding_function
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "embed",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0
    assert captured["provider"] == "local"
    rows = [
        json.loads(line)
        for line in (output_dir / "embeddings.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["embedding"] == [0.7]


def test_index_command_chunks(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "doc.md").write_text("one two three four five", encoding="utf-8")

    class FakeVectorDB:
        last_instance = None

        def __init__(self, *args, **kwargs):
            FakeVectorDB.last_instance = self
            self.documents = []
            self.ids = []

        def add_documents(self, documents, ids):
            self.documents = documents
            self.ids = ids

    monkeypatch.setattr("contextrag.cli.VectorDB", FakeVectorDB)
    monkeypatch.setattr("contextrag.cli.resolve_embed_provider", lambda config, provider: "openai")
    monkeypatch.setattr("contextrag.cli.load_config", lambda: AppConfig(
        openai_api_key="key",
        openai_embeddings_model="text-embedding-3-small",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="openai",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    ))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "index",
            "--input",
            str(input_dir),
            "--chunk-words",
            "2",
            "--chunk-overlap",
            "0",
        ],
    )
    assert result.exit_code == 0
    instance = FakeVectorDB.last_instance
    assert instance is not None
    assert instance.documents == ["one two", "three four", "five"]
    assert instance.ids == ["doc::chunk0", "doc::chunk1", "doc::chunk2"]


def test_index_command_defaults_openrouter_chunk_words(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "doc.md").write_text("one two three", encoding="utf-8")

    class FakeVectorDB:
        def __init__(self, *args, **kwargs):
            self.documents = []
            self.ids = []

        def add_documents(self, documents, ids):
            self.documents = documents
            self.ids = ids

    captured = {}

    def fake_chunk_words(text, chunk_words, overlap):
        captured["chunk_words"] = chunk_words
        return [text]

    monkeypatch.setattr("contextrag.cli.VectorDB", FakeVectorDB)
    monkeypatch.setattr("contextrag.cli._chunk_words", fake_chunk_words)
    monkeypatch.setattr("contextrag.cli.resolve_embed_provider", lambda config, provider: "openrouter")
    monkeypatch.setattr("contextrag.cli.load_config", lambda: AppConfig(
        openai_api_key=None,
        openai_embeddings_model="text-embedding-3-small",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key="key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="openai",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    ))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "index",
            "--input",
            str(input_dir),
        ],
    )
    assert result.exit_code == 0
    assert captured["chunk_words"] == 400


def test_query_command_outputs_results(monkeypatch, tmp_path):
    class FakeVectorDB:
        def __init__(self, *args, **kwargs):
            pass

        def query(self, query_texts, n_results):
            assert query_texts == ["question"]
            assert n_results == 2
            return {"documents": [["doc1", "doc2"]], "distances": [[0.1, 0.2]]}

    monkeypatch.setattr("contextrag.cli.VectorDB", FakeVectorDB)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["query", "--collection", "col", "--persist", "path", "--query", "question", "--k", "2"],
    )
    assert result.exit_code == 0
    assert "1. doc1" in result.output
    assert "distance: 0.1" in result.output


def test_eval_command_uses_config(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "documents").mkdir()
    (dataset_dir / "queries.jsonl").write_text("{}", encoding="utf-8")

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "dataset: {path}".format(path=dataset_dir.as_posix()),
                "baseline: uniform",
                "k: 3",
                "output: {path}".format(path=(tmp_path / "out.json").as_posix()),
                "persist: /tmp/index",
                "embed_provider: local",
                "embedding_model: fake",
                "run_dir: {path}".format(path=(tmp_path / "run").as_posix()),
            ]
        ),
        encoding="utf-8",
    )

    called = {}

    def fake_run_eval(**kwargs):
        called.update(kwargs)
        return {"summary": {"precision_at_k": 1.0, "recall_at_k": 1.0, "k": kwargs["k"]}}

    monkeypatch.setattr("contextrag.cli.run_eval", fake_run_eval)
    monkeypatch.setattr("contextrag.cli.write_run_artifacts", lambda **kwargs: called.update({"run_artifacts": True}))

    runner = CliRunner()
    result = runner.invoke(main, ["eval", "--config", str(config_path)])
    assert result.exit_code == 0
    assert called["baseline"] == "uniform"
    assert called["k"] == 3
    assert called["embedding_model"] == "fake"
    assert called.get("run_artifacts") is True


def test_eval_command_requires_dataset(monkeypatch, tmp_path):
    output_path = tmp_path / "out.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["eval", "--output", str(output_path)],
    )
    assert result.exit_code != 0
    assert "--dataset is required" in result.output


def test_eval_command_requires_output(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "documents").mkdir()
    (dataset_dir / "queries.jsonl").write_text("{}", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["eval", "--dataset", str(dataset_dir)],
    )
    assert result.exit_code != 0
    assert "--output is required" in result.output


def test_doctor_reports_status(monkeypatch):
    config = AppConfig(
        openai_api_key=None,
        openai_embeddings_model="text-embedding-3-small",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key="key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="openai",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    )
    monkeypatch.setattr("contextrag.cli.load_config", lambda: config)
    monkeypatch.setattr("contextrag.cli.resolve_embed_provider", lambda cfg, provider: "openrouter")

    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "OPENAI_API_KEY: missing" in result.output
    assert "OPENROUTER_API_KEY: ok" in result.output
    assert "embeddings_provider: openrouter" in result.output


def test_doctor_reports_missing_tiktoken(monkeypatch):
    import builtins

    config = AppConfig(
        openai_api_key=None,
        openai_embeddings_model="text-embedding-3-small",
        openai_chat_model_short="gpt-3.5-turbo-1106",
        openai_chat_model_medium="gpt-3.5-turbo-16k",
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_chat_model="mistralai/devstral-2512:free",
        openrouter_embeddings_model="qwen/qwen3-embedding-8b",
        local_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        contextrag_chat_provider="openai",
        contextrag_embed_provider="openai",
        openrouter_referer=None,
        openrouter_title=None,
        openrouter_embed_provider_json=None,
        openrouter_embed_provider_order=None,
        openrouter_embed_allow_fallbacks=None,
    )

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("contextrag.cli.load_config", lambda: config)
    monkeypatch.setattr("contextrag.cli.resolve_embed_provider", lambda cfg, provider: "local")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "tiktoken: missing" in result.output
