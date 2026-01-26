"""ContextRAG CLI - RAG evaluation framework."""

from __future__ import annotations

import json
from pathlib import Path

import click
from chromaroute import VectorStore, build_embedding_function

from contextrag.config import load_config
from contextrag.core.chunking import chunk_text_by_words
from contextrag.core.io import iter_files
from contextrag.eval.runner import run_eval
from contextrag.experiments.eval_config import load_eval_config
from contextrag.experiments.run_logger import write_run_artifacts

TEXT_EXTENSIONS = (".md", ".txt")


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
            "embed_provider": embed_provider,
            "embedding_model": embedding_model,
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
@click.option("--baseline", type=click.Choice(["uniform", "adaptive", "router"]))
@click.option("--k", "top_k", type=int)
@click.option("--output", "output_path", type=click.Path(path_type=Path))
@click.option("--run-dir", "run_dir", type=click.Path(path_type=Path))
@click.option("--persist", "persist_path")
@click.option("--embedding-model", "embedding_model")
@click.option(
    "--embed-provider",
    "embed_provider",
    type=click.Choice(["auto", "openrouter", "local"]),
)
def eval(
    config_path: Path | None,
    dataset_path: Path | None,
    baseline: str | None,
    top_k: int | None,
    output_path: Path | None,
    run_dir: Path | None,
    persist_path: str | None,
    embedding_model: str | None,
    embed_provider: str | None,
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
        embed_provider = embed_provider or eval_config.embed_provider
        embedding_model = embedding_model or eval_config.embedding_model
        persist_path = persist_path or eval_config.persist
        output_path = output_path or Path(eval_config.output)
        run_dir = run_dir or (Path(eval_config.run_dir) if eval_config.run_dir else None)

    if not dataset_path:
        raise click.ClickException("--dataset is required (or provide --config).")
    if not output_path:
        raise click.ClickException("--output is required (or provide --config).")

    baseline = baseline or "router"
    top_k = top_k or 5

    # Validate provider requirements
    config = load_config()
    resolved_provider = config.resolve_embed_provider(embed_provider)
    config.require_embed_provider(resolved_provider, embed_provider, click.ClickException)

    _run_evaluation(
        dataset_path=dataset_path,
        baseline=baseline,
        top_k=top_k,
        output_path=output_path,
        persist_path=persist_path,
        embed_provider=embed_provider,
        embedding_model=embedding_model,
        run_dir=run_dir,
        config_path=config_path,
    )


@main.command()
@click.option(
    "--dataset",
    "dataset_path",
    default=Path("data/demo"),
    type=click.Path(path_type=Path),
)
@click.option("--baseline", type=click.Choice(["uniform", "adaptive", "router"]), default="uniform")
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
    )


@main.command()
def doctor() -> None:
    """Check configuration and environment health."""
    from importlib.util import find_spec

    config = load_config()

    click.echo("=== API Keys ===")
    click.echo(f"OPENROUTER_API_KEY: {'ok' if config.openrouter_api_key else 'missing'}")
    click.echo(f"OPENAI_API_KEY: {'ok' if config.openai_api_key else 'missing'}")

    click.echo("\n=== Providers ===")
    click.echo(f"embed_provider: {config.resolve_embed_provider()}")
    click.echo(f"chat_provider: {config.chat_provider}")

    click.echo("\n=== Models ===")
    click.echo(f"openrouter_embeddings: {config.openrouter_embeddings_model}")
    click.echo(f"local_embeddings: {config.local_embeddings_model}")
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

    if not config.openrouter_api_key and not config.openai_api_key:
        click.echo("\nnote: Set OPENROUTER_API_KEY for hosted embeddings")


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

    embed_config = config.to_embed_config()
    embedding_fn = build_embedding_function(
        config=embed_config,
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
    embed_config = config.to_embed_config()
    embedding_fn = build_embedding_function(config=embed_config)
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
