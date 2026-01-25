"""ContextRAG CLI - RAG evaluation framework."""

from __future__ import annotations

import json
from pathlib import Path

import click

from contextrag.config import load_config, require_embedding_provider, resolve_embed_provider
from contextrag.core.chunking import chunk_text_by_words
from contextrag.core.io import iter_files
from contextrag.core.tokenizer import count_tokens
from contextrag.eval.runner import run_eval
from contextrag.experiments.eval_config import load_eval_config
from contextrag.experiments.run_logger import write_run_artifacts
from contextrag.index.vector_store import VectorDB


TEXT_EXTENSIONS = (".md", ".txt")


def _chunk_words(text: str, chunk_words: int, overlap: int) -> list[str]:
    """Split text into word-based chunks."""
    return chunk_text_by_words(text, chunk_words, overlap)


def _write_eval_outputs(
    *,
    dataset_path: Path,
    baseline: str,
    top_k: int,
    output_path: Path,
    persist_path: str | None,
    embed_provider: str | None,
    embedding_model: str | None,
    run_dir: Path | None,
    summary_updates: dict[str, str] | None = None,
    metadata_updates: dict[str, str] | None = None,
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
    if summary_updates:
        results["summary"].update(summary_updates)
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
        if metadata_updates:
            metadata.update(metadata_updates)
        write_run_artifacts(
            run_dir=run_dir,
            results=results,
            metadata=metadata,
            dataset_path=dataset_path,
        )
    summary = results["summary"]
    click.echo(
        "precision@k={precision:.3f} recall@k={recall:.3f} (k={k})".format(
            precision=summary["precision_at_k"],
            recall=summary["recall_at_k"],
            k=summary["k"],
        )
    )


@click.group()
def main() -> None:
    """ContextRAG - RAG evaluation framework.
    
    A CLI tool for evaluating retrieval-augmented generation strategies
    with comprehensive metrics (accuracy, efficiency, and cost).
    """


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path))
@click.option("--collection", default="contextrag")
@click.option("--persist", "persist_path", default=None)
@click.option("--embedding-model", "embedding_model", default=None)
@click.option(
    "--embed-provider",
    "embed_provider",
    type=click.Choice(["auto", "openrouter", "local"]),
    default=None,
)
@click.option("--chunk-words", type=int, default=None)
@click.option("--chunk-overlap", type=int, default=50)
def index(
    input_path: Path,
    collection: str,
    persist_path: str | None,
    embedding_model: str | None,
    embed_provider: str | None,
    chunk_words: int | None,
    chunk_overlap: int,
) -> None:
    """Build a vector index from Markdown documents."""
    config = load_config()
    resolved_provider = resolve_embed_provider(config, embed_provider)
    require_embedding_provider(
        config,
        resolved_provider=resolved_provider,
        explicit_provider=embed_provider,
        error_cls=click.ClickException,
    )
    vector_db = VectorDB(
        collection_name=collection,
        persist_path=persist_path,
        embedding_model=embedding_model,
        embed_provider=embed_provider,
    )
    if resolved_provider == "openrouter" and chunk_words is None:
        chunk_words = 400

    documents: list[str] = []
    ids: list[str] = []
    for md_file in iter_files(input_path, TEXT_EXTENSIONS):
        content = md_file.read_text(encoding="utf-8")
        if chunk_words:
            chunks = _chunk_words(content, chunk_words, chunk_overlap)
            for idx, chunk in enumerate(chunks):
                documents.append(chunk)
                ids.append(f"{md_file.stem}::chunk{idx}")
        else:
            documents.append(content)
            ids.append(md_file.stem)
    vector_db.add_documents(documents=documents, ids=ids)
    click.echo(f"Indexed {len(documents)} documents into {collection}")


@main.command()
@click.option("--collection", required=True)
@click.option("--persist", "persist_path", required=True)
@click.option("--query", "query_text", required=True)
@click.option("--k", "top_k", type=int, default=5)
def query(collection: str, persist_path: str, query_text: str, top_k: int) -> None:
    """Query the vector index."""
    vector_db = VectorDB(collection_name=collection, persist_path=persist_path)
    results = vector_db.query(query_texts=[query_text], n_results=top_k)
    for i, doc in enumerate(results["documents"][0], start=1):
        distance = results["distances"][0][i - 1]
        click.echo(f"{i}. {doc}")
        click.echo(f"   distance: {distance}")


@main.command()
@click.option("--dataset", "dataset_path", required=False, type=click.Path(path_type=Path))
@click.option(
    "--baseline",
    type=click.Choice(["uniform", "router"]),
    default=None,
)
@click.option("--k", "top_k", type=int, default=None)
@click.option("--output", "output_path", required=False, type=click.Path(path_type=Path))
@click.option("--persist", "persist_path", default=None)
@click.option("--embedding-model", "embedding_model", default=None)
@click.option(
    "--embed-provider",
    "embed_provider",
    type=click.Choice(["auto", "openrouter", "local"]),
    default=None,
)
@click.option("--run-dir", "run_dir", type=click.Path(path_type=Path))
@click.option("--config", "config_path", type=click.Path(path_type=Path))
def eval(
    dataset_path: Path,
    baseline: str,
    top_k: int,
    output_path: Path,
    persist_path: str | None,
    embedding_model: str | None,
    embed_provider: str | None,
    run_dir: Path | None,
    config_path: Path | None,
) -> None:
    """Run retrieval evaluation.
    
    This is the core command for evaluating RAG retrieval strategies.
    Supports both uniform chunking and adaptive (router) baselines.
    """
    eval_config = None
    if config_path:
        eval_config = load_eval_config(config_path)

    if eval_config:
        dataset_path = dataset_path or Path(eval_config.dataset)
        baseline = baseline or eval_config.baseline
        top_k = top_k or eval_config.k
        embed_provider = embed_provider or eval_config.embed_provider
        embedding_model = embedding_model or eval_config.embedding_model
        persist_path = persist_path or eval_config.persist
        output_path = output_path or Path(eval_config.output)
        run_dir = run_dir or (
            Path(eval_config.run_dir) if eval_config.run_dir else None
        )

    if not dataset_path:
        raise click.ClickException("--dataset is required (or provide --config).")
    if not baseline:
        baseline = "router"
    if not top_k:
        top_k = 5
    if not output_path:
        raise click.ClickException("--output is required (or provide --config).")

    config = load_config()
    resolved_provider = resolve_embed_provider(config, embed_provider)
    require_embedding_provider(
        config,
        resolved_provider=resolved_provider,
        explicit_provider=embed_provider,
        error_cls=click.ClickException,
    )
    summary_updates = {}
    metadata_updates = {}
    if config_path:
        summary_updates["config_path"] = str(config_path)
        metadata_updates["config_path"] = str(config_path)
    _write_eval_outputs(
        dataset_path=dataset_path,
        baseline=baseline,
        top_k=top_k,
        output_path=output_path,
        persist_path=persist_path,
        embed_provider=embed_provider,
        embedding_model=embedding_model,
        run_dir=run_dir,
        summary_updates=summary_updates or None,
        metadata_updates=metadata_updates or None,
    )


@main.command()
@click.option(
    "--dataset",
    "dataset_path",
    default=Path("data/demo"),
    type=click.Path(path_type=Path),
)
@click.option(
    "--baseline",
    type=click.Choice(["uniform", "router"]),
    default="uniform",
)
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
@click.option("--persist", "persist_path", default=None)
@click.option("--embedding-model", "embedding_model", default=None)
def demo(
    dataset_path: Path,
    baseline: str,
    top_k: int,
    output_path: Path,
    run_dir: Path,
    persist_path: str | None,
    embedding_model: str | None,
) -> None:
    """Run the offline demo evaluation with local embeddings.
    
    This command requires no API keys - it uses local sentence-transformers
    for embeddings and runs against the bundled demo dataset.
    """
    config = load_config()
    resolved_provider = resolve_embed_provider(config, explicit_provider="local")
    require_embedding_provider(
        config,
        resolved_provider=resolved_provider,
        explicit_provider="local",
        error_cls=click.ClickException,
    )
    _write_eval_outputs(
        dataset_path=dataset_path,
        baseline=baseline,
        top_k=top_k,
        output_path=output_path,
        persist_path=persist_path,
        embed_provider="local",
        embedding_model=embedding_model,
        run_dir=run_dir,
    )


@main.command()
def doctor() -> None:
    """Check configuration and environment health."""
    from importlib.util import find_spec

    config = load_config()
    
    # API Keys
    click.echo("=== API Keys ===")
    click.echo(f"OPENROUTER_API_KEY: {'ok' if config.openrouter_api_key else 'missing'}")
    click.echo(f"OPENAI_API_KEY: {'ok' if config.openai_api_key else 'missing'}")
    
    # Resolved providers
    click.echo("\n=== Providers ===")
    resolved_embed = resolve_embed_provider(config, None)
    click.echo(f"embed_provider: {resolved_embed}")
    click.echo(f"chat_provider: {config.chat_provider}")
    
    # Models
    click.echo("\n=== Models ===")
    click.echo(f"openrouter_embeddings_model: {config.openrouter_embeddings_model}")
    click.echo(f"local_embeddings_model: {config.local_embeddings_model}")
    click.echo(f"openai_chat_model: {config.openai_chat_model}")
    
    # Dependencies
    click.echo("\n=== Dependencies ===")
    deps = [
        ("chromadb", "chromadb"),
        ("chromaroute", "chromaroute"),
        ("tiktoken", "tiktoken"),
        ("sentence-transformers", "sentence_transformers"),
    ]
    for name, module in deps:
        status = "ok" if find_spec(module) else "missing"
        click.echo(f"{name}: {status}")

    if not config.openrouter_api_key and not config.openai_api_key:
        click.echo("\nnote: Set OPENROUTER_API_KEY for hosted embeddings")


if __name__ == "__main__":
    main()
