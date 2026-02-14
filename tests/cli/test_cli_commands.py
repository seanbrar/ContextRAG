"""Tests for ContextRAG CLI commands."""


import json

from click.testing import CliRunner

from contextrag.cli import main
from contextrag.core.io import write_jsonl
from tests.conftest import make_test_config


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

    monkeypatch.setattr("contextrag.cli.VectorStore", FakeVectorDB)
    monkeypatch.setattr("contextrag.cli.load_config", lambda: make_test_config(
        openai_api_key=None,
        openrouter_api_key=None,
        embed_provider="local",
    ))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "db", "index",
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

    monkeypatch.setattr("contextrag.cli.VectorStore", FakeVectorDB)
    monkeypatch.setattr("contextrag.cli.chunk_text_by_words", fake_chunk_words)
    monkeypatch.setattr("contextrag.cli.load_config", lambda: make_test_config(
        openai_api_key=None,
        embed_provider="auto",
    ))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "db", "index",
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

    monkeypatch.setattr("contextrag.cli.VectorStore", FakeVectorDB)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["db", "query", "--collection", "col", "--persist", "path", "--query", "question", "--k", "2"],
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
    config = make_test_config(openai_api_key=None)
    monkeypatch.setattr("contextrag.cli.load_config", lambda: config)

    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "OPENROUTER_API_KEY: ok" in result.output
    assert "embed_provider: openrouter" in result.output


def test_doctor_reports_missing_keys(monkeypatch):
    config = make_test_config(openai_api_key=None, openrouter_api_key=None)
    monkeypatch.setattr("contextrag.cli.load_config", lambda: config)

    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "OPENROUTER_API_KEY: missing" in result.output
    assert "OPENAI_API_KEY: missing" in result.output
    assert "embed_provider: local" in result.output


def test_compare_command_writes_output(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()

    (run_a / "summary.json").write_text(
        json.dumps({"precision_at_k": 0.5, "recall_at_k": 1.0}),
        encoding="utf-8",
    )
    (run_b / "summary.json").write_text(
        json.dumps({"precision_at_k": 0.5, "recall_at_k": 1.0}),
        encoding="utf-8",
    )
    write_jsonl(
        run_a / "per_query.jsonl",
        [{"query": "q", "relevant_ids": ["d"], "retrieved_ids": ["x"], "precision_at_k": 0.0}],
    )
    write_jsonl(
        run_b / "per_query.jsonl",
        [{"query": "q", "relevant_ids": ["d"], "retrieved_ids": ["d"], "precision_at_k": 1.0}],
    )

    output = tmp_path / "compare.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["compare", "--run-a", str(run_a), "--run-b", str(run_b), "--output", str(output)],
    )
    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["counts"]["queries_compared"] == 1
    assert payload["counts"]["retrieved_ids_changed"] == 1


def test_validate_dataset_command(tmp_path):
    dataset = tmp_path / "dataset"
    docs = dataset / "documents"
    docs.mkdir(parents=True)
    (docs / "doc1.txt").write_text("content", encoding="utf-8")
    (dataset / "queries.jsonl").write_text(
        json.dumps({"query": "q1", "relevant_ids": ["doc1"]}) + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(main, ["validate-dataset", "--dataset", str(dataset)])
    assert result.exit_code == 0
    assert "dataset_ok" in result.output


def test_matrix_command(monkeypatch, tmp_path):
    captured = {}

    def fake_run_matrix(**kwargs):
        captured.update(kwargs)
        return {"rows": [1, 2], "comparisons": [1]}

    monkeypatch.setattr("contextrag.cli.run_matrix", fake_run_matrix)
    monkeypatch.setattr(
        "contextrag.cli.load_config",
        lambda: make_test_config(openai_api_key=None, openrouter_api_key=None, embed_provider="local"),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "matrix",
            "--dataset",
            str(tmp_path),
            "--baselines",
            "uniform,router",
            "--k-values",
            "3,5",
        ],
    )
    assert result.exit_code == 0
    assert captured["baselines"] == ["uniform", "router"]
    assert captured["k_values"] == [3, 5]
