"""ContextRAG CLI - RAG evaluation framework."""

from __future__ import annotations

import json
from pathlib import Path

import click
from chromaroute import VectorStore, build_embedding_function

from contextrag.config import load_config
from contextrag.core.io import iter_files
from contextrag.core.text import chunk_text_by_words
from contextrag.eval.compare import compare_runs
from contextrag.eval.query_schema import load_and_validate_queries
from contextrag.eval.runner import run_eval
from contextrag.experiments.eval_config import load_eval_config
from contextrag.experiments.matrix import run_matrix
from contextrag.experiments.run_logger import write_run_artifacts

TEXT_EXTENSIONS = (".md", ".txt")


def _parse_csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_evaluation(
    *,
    dataset_path: Path,
    baseline: str,
    top_k: int,
    output_path: Path,
    persist_path: str | None,
    embed_provider: str | None,
    embedding_model: str | None,
    run_dir: Path | None,
    retrieval_mode: str,
    uniform_chunk_tokens: int | None,
    chunk_overlap_tokens: int,
    retrieval_candidates: int,
    config_path: Path | None = None,
) -> None:
    """Run evaluation and write output artifacts."""
    results = run_eval(
        dataset_path=dataset_path,
        baseline=baseline,
        k=top_k,
        persist_path=persist_path,
        embed_provider=embed_provider,
        embedding_model=embedding_model,
        retrieval_mode=retrieval_mode,
        uniform_chunk_tokens=uniform_chunk_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        retrieval_candidates=retrieval_candidates,
    )

    if config_path:
        results["summary"]["config_path"] = str(config_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if run_dir:
        metadata = {
            "dataset": str(dataset_path),
            "baseline": baseline,
            "k": top_k,
            "retrieval_mode": retrieval_mode,
            "embed_provider": embed_provider,
            "embedding_model": embedding_model,
            "uniform_chunk_tokens": uniform_chunk_tokens,
            "chunk_overlap_tokens": chunk_overlap_tokens,
            "retrieval_candidates": retrieval_candidates,
            "output": str(output_path),
            "persist": persist_path,
        }
        if config_path:
            metadata["config_path"] = str(config_path)
        write_run_artifacts(
            run_dir=run_dir,
            results=results,
            metadata=metadata,
            dataset_path=dataset_path,
        )

    summary = results["summary"]
    click.echo(
        f"precision@k={summary['precision_at_k']:.3f} "
        f"recall@k={summary['recall_at_k']:.3f} "
        f"(k={summary['k']})"
    )


@click.group()
def main() -> None:
    """ContextRAG - RAG evaluation framework.

    Evaluate retrieval-augmented generation strategies with comprehensive
    metrics for accuracy, efficiency, and cost.

    Primary commands:
      eval    Run retrieval evaluation
      demo    Run offline demo with local embeddings
      doctor  Check configuration health

    Database commands:
      db index  Build a vector index
      db query  Query a vector index
    """


@main.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option("--dataset", "dataset_path", type=click.Path(path_type=Path))
@click.option("--baseline", type=click.Choice(["uniform", "adaptive", "router", "semantic"]))
@click.option("--k", "top_k", type=int)
@click.option(
    "--retrieval-mode",
    "retrieval_mode",
    type=click.Choice(["dense", "bm25", "hybrid", "dense-rerank"]),
)
@click.option("--output", "output_path", type=click.Path(path_type=Path))
@click.option("--run-dir", "run_dir", type=click.Path(path_type=Path))
@click.option("--persist", "persist_path")
@click.option("--embedding-model", "embedding_model")
@click.option(
    "--embed-provider",
    "embed_provider",
    type=click.Choice(["auto", "openrouter", "local"]),
)
@click.option("--uniform-chunk-tokens", "uniform_chunk_tokens", type=int)
@click.option("--chunk-overlap-tokens", "chunk_overlap_tokens", type=int)
@click.option("--retrieval-candidates", "retrieval_candidates", type=int)
@click.option("--dry-run", is_flag=True, help="Validate inputs/config and exit.")
def eval(
    config_path: Path | None,
    dataset_path: Path | None,
    baseline: str | None,
    top_k: int | None,
    retrieval_mode: str | None,
    output_path: Path | None,
    run_dir: Path | None,
    persist_path: str | None,
    embedding_model: str | None,
    embed_provider: str | None,
    uniform_chunk_tokens: int | None,
    chunk_overlap_tokens: int | None,
    retrieval_candidates: int | None,
    dry_run: bool,
) -> None:
    """Run retrieval evaluation.

    Evaluate RAG retrieval strategies with uniform or adaptive chunking.
    Use --config for reproducible experiments via YAML configuration.

    Examples:
      contextrag eval --config experiments/uniform.yaml
      contextrag eval --dataset data/corpus --output results.json
    """
    eval_config = load_eval_config(config_path) if config_path else None

    if eval_config:
        dataset_path = dataset_path or Path(eval_config.dataset)
        baseline = baseline or eval_config.baseline
        top_k = top_k or eval_config.k
        retrieval_mode = retrieval_mode or eval_config.retrieval_mode
        embed_provider = embed_provider or eval_config.embed_provider
        embedding_model = embedding_model or eval_config.embedding_model
        if uniform_chunk_tokens is None:
            uniform_chunk_tokens = eval_config.uniform_chunk_tokens
        if chunk_overlap_tokens is None:
            chunk_overlap_tokens = eval_config.chunk_overlap_tokens
        if retrieval_candidates is None:
            retrieval_candidates = eval_config.retrieval_candidates
        persist_path = persist_path or eval_config.persist
        output_path = output_path or Path(eval_config.output)
        run_dir = run_dir or (Path(eval_config.run_dir) if eval_config.run_dir else None)

    if not dataset_path:
        raise click.ClickException("--dataset is required (or provide --config).")
    if not output_path:
        raise click.ClickException("--output is required (or provide --config).")

    baseline = baseline or "uniform"
    top_k = top_k or 5
    retrieval_mode = retrieval_mode or "dense"
    chunk_overlap_tokens = chunk_overlap_tokens if chunk_overlap_tokens is not None else 0
    retrieval_candidates = retrieval_candidates or 50
    if chunk_overlap_tokens < 0:
        raise click.ClickException("--chunk-overlap-tokens must be >= 0.")
    if retrieval_candidates <= 0:
        raise click.ClickException("--retrieval-candidates must be > 0.")
    if uniform_chunk_tokens is not None and uniform_chunk_tokens <= 0:
        raise click.ClickException("--uniform-chunk-tokens must be > 0.")

    config = load_config()
    resolved_provider = "none"
    resolved_model = "none"
    if retrieval_mode != "bm25":
        resolved_provider = config.resolve_embed_provider(embed_provider)
        config.require_embed_provider(resolved_provider, embed_provider, click.ClickException)
        resolved_model = embedding_model or config.embed.resolve_model(resolved_provider)

    if dry_run:
        click.echo("eval_dry_run_ok")
        click.echo(f"dataset={dataset_path}")
        click.echo(f"baseline={baseline}")
        click.echo(f"k={top_k}")
        click.echo(f"retrieval_mode={retrieval_mode}")
        click.echo(f"embed_provider={resolved_provider}")
        click.echo(f"embedding_model={resolved_model}")
        click.echo(f"uniform_chunk_tokens={uniform_chunk_tokens}")
        click.echo(f"chunk_overlap_tokens={chunk_overlap_tokens}")
        click.echo(f"retrieval_candidates={retrieval_candidates}")
        click.echo(f"output={output_path}")
        click.echo(f"run_dir={run_dir}")
        click.echo(f"persist={persist_path}")
        if config_path:
            click.echo(f"config_path={config_path}")
        return

    _run_evaluation(
        dataset_path=dataset_path,
        baseline=baseline,
        top_k=top_k,
        output_path=output_path,
        persist_path=persist_path,
        embed_provider=embed_provider,
        embedding_model=embedding_model,
        run_dir=run_dir,
        retrieval_mode=retrieval_mode,
        uniform_chunk_tokens=uniform_chunk_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        retrieval_candidates=retrieval_candidates,
        config_path=config_path,
    )


@main.command()
@click.option(
    "--dataset",
    "dataset_path",
    default=Path("data/demo"),
    type=click.Path(path_type=Path),
)
@click.option("--baseline", type=click.Choice(["uniform", "adaptive", "router", "semantic"]), default="uniform")
@click.option("--k", "top_k", type=int, default=5)
@click.option(
    "--output",
    "output_path",
    default=Path("runs/demo_eval.json"),
    type=click.Path(path_type=Path),
)
@click.option(
    "--run-dir",
    "run_dir",
    default=Path("runs/demo_eval"),
    type=click.Path(path_type=Path),
)
def demo(
    dataset_path: Path,
    baseline: str,
    top_k: int,
    output_path: Path,
    run_dir: Path,
) -> None:
    """Run offline demo with local embeddings.

    No API keys required. Uses sentence-transformers for embeddings
    and runs against the bundled demo dataset.
    """
    _run_evaluation(
        dataset_path=dataset_path,
        baseline=baseline,
        top_k=top_k,
        output_path=output_path,
        persist_path=None,
        embed_provider="local",
        embedding_model=None,
        run_dir=run_dir,
        retrieval_mode="dense",
        uniform_chunk_tokens=None,
        chunk_overlap_tokens=0,
        retrieval_candidates=50,
    )


@main.command()
def doctor() -> None:
    """Check configuration and environment health."""
    from importlib.util import find_spec

    config = load_config()

    click.echo("=== API Keys ===")
    click.echo(
        f"OPENROUTER_API_KEY: {'ok' if config.embed.openrouter_api_key else 'missing'}"
    )
    click.echo(f"OPENAI_API_KEY: {'ok' if config.openai_api_key else 'missing'}")

    click.echo("\n=== Providers ===")
    click.echo(f"embed_provider: {config.resolve_embed_provider()}")
    click.echo(f"chat_provider: {config.chat_provider}")

    click.echo("\n=== Models ===")
    click.echo(f"openrouter_embeddings: {config.embed.openrouter_embeddings_model}")
    click.echo(f"local_embeddings: {config.embed.local_embeddings_model}")
    click.echo(f"openai_chat: {config.openai_chat_model}")
    click.echo(f"openrouter_chat: {config.openrouter_chat_model}")

    click.echo("\n=== Dependencies ===")
    deps = [
        ("chromadb", "chromadb"),
        ("chromaroute", "chromaroute"),
        ("tiktoken", "tiktoken"),
        ("sentence-transformers", "sentence_transformers"),
    ]
    for name, module in deps:
        click.echo(f"{name}: {'ok' if find_spec(module) else 'missing'}")

    if not config.embed.openrouter_api_key and not config.openai_api_key:
        click.echo("\nnote: Set OPENROUTER_API_KEY for hosted embeddings")


@main.command()
@click.option("--run-a", "run_a", required=True, type=click.Path(path_type=Path))
@click.option("--run-b", "run_b", required=True, type=click.Path(path_type=Path))
@click.option(
    "--output",
    "output_path",
    default=Path("runs/compare.json"),
    type=click.Path(path_type=Path),
)
def compare(run_a: Path, run_b: Path, output_path: Path) -> None:
    """Compare two evaluation runs and write query-level delta diagnostics."""
    comparison = compare_runs(run_a, run_b)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    counts = comparison["counts"]
    click.echo(
        f"compared={counts['queries_compared']} "
        f"retrieved_changed={counts['retrieved_ids_changed']} "
        f"metrics_changed={counts['metric_values_changed']}"
    )


@main.command("validate-dataset")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(path_type=Path),
)
def validate_dataset(dataset_path: Path) -> None:
    """Validate dataset structure and query schema."""
    documents_dir = dataset_path / "documents"
    queries_path = dataset_path / "queries.jsonl"
    if not documents_dir.exists():
        raise click.ClickException(f"Missing documents directory: {documents_dir}")
    if not queries_path.exists():
        raise click.ClickException(f"Missing queries file: {queries_path}")

    queries = load_and_validate_queries(queries_path)
    file_count = sum(1 for path in documents_dir.glob("*") if path.is_file())
    click.echo(
        f"dataset_ok documents={file_count} queries={len(queries)} path={dataset_path}"
    )


@main.command()
@click.option("--dataset", "dataset_path", required=True, type=click.Path(path_type=Path))
@click.option("--baselines", default="uniform,router")
@click.option("--k-values", default="3,5,10")
@click.option(
    "--retrieval-mode",
    "retrieval_mode",
    default="dense",
    type=click.Choice(["dense", "bm25", "hybrid", "dense-rerank"]),
)
@click.option("--uniform-chunk-tokens", "uniform_chunk_tokens", type=int)
@click.option("--chunk-overlap-tokens", "chunk_overlap_tokens", type=int, default=0)
@click.option("--retrieval-candidates", "retrieval_candidates", type=int, default=50)
@click.option(
    "--run-root",
    "run_root",
    default=Path("runs/matrix"),
    type=click.Path(path_type=Path),
)
@click.option(
    "--persist-root",
    "persist_root",
    default=Path("runs/chroma-matrix"),
    type=click.Path(path_type=Path),
)
@click.option("--embedding-model", "embedding_model")
@click.option(
    "--embed-provider",
    "embed_provider",
    type=click.Choice(["auto", "openrouter", "local"]),
)
def matrix(
    dataset_path: Path,
    baselines: str,
    k_values: str,
    retrieval_mode: str,
    uniform_chunk_tokens: int | None,
    chunk_overlap_tokens: int,
    retrieval_candidates: int,
    run_root: Path,
    persist_root: Path,
    embedding_model: str | None,
    embed_provider: str | None,
) -> None:
    """Run a baseline-by-k experiment matrix and write aggregate reports."""
    baseline_list = _parse_csv_items(baselines)
    if not baseline_list:
        raise click.ClickException("--baselines must include at least one value.")
    invalid_baselines = sorted(
        baseline
        for baseline in baseline_list
        if baseline not in {"uniform", "adaptive", "router", "semantic"}
    )
    if invalid_baselines:
        joined = ", ".join(invalid_baselines)
        raise click.ClickException(
            f"Unsupported baseline(s): {joined}. Use uniform, adaptive, router, semantic."
        )

    try:
        parsed_k_values = [int(item) for item in _parse_csv_items(k_values)]
    except ValueError as exc:
        raise click.ClickException("--k-values must be comma-separated integers.") from exc
    if not parsed_k_values or any(k <= 0 for k in parsed_k_values):
        raise click.ClickException("--k-values must include integers greater than 0.")
    if chunk_overlap_tokens < 0:
        raise click.ClickException("--chunk-overlap-tokens must be >= 0.")
    if retrieval_candidates <= 0:
        raise click.ClickException("--retrieval-candidates must be > 0.")
    if uniform_chunk_tokens is not None and uniform_chunk_tokens <= 0:
        raise click.ClickException("--uniform-chunk-tokens must be > 0.")

    if retrieval_mode != "bm25":
        config = load_config()
        resolved_provider = config.resolve_embed_provider(embed_provider)
        config.require_embed_provider(resolved_provider, embed_provider, click.ClickException)

    summary = run_matrix(
        dataset_path=dataset_path,
        baselines=baseline_list,
        k_values=parsed_k_values,
        run_root=run_root,
        persist_root=persist_root,
        retrieval_mode=retrieval_mode,
        uniform_chunk_tokens=uniform_chunk_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        retrieval_candidates=retrieval_candidates,
        embed_provider=embed_provider,
        embedding_model=embedding_model,
    )
    click.echo(
        f"matrix_done runs={len(summary['rows'])} "
        f"comparisons={len(summary['comparisons'])} "
        f"summary={run_root / 'matrix_summary.json'}"
    )


# Database subgroup for lower-level operations
@main.group()
def db() -> None:
    """Database operations for vector indices."""


@db.command("index")
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path))
@click.option("--collection", default="contextrag")
@click.option("--persist", "persist_path")
@click.option("--embedding-model", "embedding_model")
@click.option("--embed-provider", type=click.Choice(["auto", "openrouter", "local"]))
@click.option("--chunk-words", type=int)
@click.option("--chunk-overlap", type=int, default=50)
def db_index(
    input_path: Path,
    collection: str,
    persist_path: str | None,
    embedding_model: str | None,
    embed_provider: str | None,
    chunk_words: int | None,
    chunk_overlap: int,
) -> None:
    """Build a vector index from documents."""
    config = load_config()
    resolved_provider = config.resolve_embed_provider(embed_provider)
    config.require_embed_provider(resolved_provider, embed_provider, click.ClickException)

    embedding_fn = build_embedding_function(
        config=config.embed,
        embedding_model=embedding_model,
        embed_provider=embed_provider,
    )
    vector_store = VectorStore(
        collection_name=collection,
        persist_path=persist_path,
        embedding_function=embedding_fn,
    )

    # Default chunk size for OpenRouter to stay within embedding limits
    if resolved_provider == "openrouter" and chunk_words is None:
        chunk_words = 400

    documents: list[str] = []
    ids: list[str] = []
    for doc_file in iter_files(input_path, TEXT_EXTENSIONS):
        content = doc_file.read_text(encoding="utf-8")
        if chunk_words:
            chunks = chunk_text_by_words(content, chunk_words, chunk_overlap)
            for idx, chunk in enumerate(chunks):
                documents.append(chunk)
                ids.append(f"{doc_file.stem}::chunk{idx}")
        else:
            documents.append(content)
            ids.append(doc_file.stem)

    vector_store.add_documents(documents=documents, ids=ids)
    click.echo(f"Indexed {len(documents)} documents into {collection}")


@db.command("query")
@click.option("--collection", required=True)
@click.option("--persist", "persist_path", required=True)
@click.option("--query", "query_text", required=True)
@click.option("--k", "top_k", type=int, default=5)
def db_query(
    collection: str,
    persist_path: str,
    query_text: str,
    top_k: int,
) -> None:
    """Query a vector index."""
    config = load_config()
    embedding_fn = build_embedding_function(config=config.embed)
    vector_store = VectorStore(
        collection_name=collection,
        persist_path=persist_path,
        embedding_function=embedding_fn,
    )

    results = vector_store.query(query_texts=[query_text], n_results=top_k)
    for i, doc in enumerate(results["documents"][0], start=1):
        distance = results["distances"][0][i - 1]
        click.echo(f"{i}. {doc}")
        click.echo(f"   distance: {distance}")


if __name__ == "__main__":
    main()
