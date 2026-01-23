from __future__ import annotations

import json
from pathlib import Path

import click

from contextrag.config import (
    load_config,
    require_embedding_provider,
    resolve_embed_provider,
)
from contextrag.core.chunking import chunk_text_by_words
from contextrag.core.constants import MEDIUM_MAX_TOKENS, SHORT_MAX_TOKENS
from contextrag.core.io import checksum as checksum_text, iter_files, write_jsonl
from contextrag.core.routing import route_bucket
from contextrag.core.tokenizer import count_tokens
from contextrag.embeddings.provider import build_embedding_function
from contextrag.eval.runner import run_eval
from contextrag.experiments.eval_config import EvalConfig, load_eval_config
from contextrag.experiments.run_logger import write_run_artifacts
from contextrag.ingest.html_to_markdown import HTMLToMarkdownConverter
from contextrag.ingest.markdown_processing import modify_markdown
from contextrag.index.vector_store import VectorDB


TEXT_EXTENSIONS = (".md", ".txt")


def _chunk_words(text: str, chunk_words: int, overlap: int) -> list[str]:
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
    """ContextRAG command line interface."""


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--format",
    "format_",
    type=click.Choice(["html", "markdown", "auto"]),
    default="auto",
)
@click.option("--min-tokens", type=int, default=None)
@click.option("--max-tokens", type=int, default=None)
def ingest(
    input_path: Path,
    output_path: Path,
    format_: str,
    min_tokens: int | None,
    max_tokens: int | None,
) -> None:
    """Convert raw documents into cleaned Markdown."""
    output_path.mkdir(parents=True, exist_ok=True)

    html_files = iter_files(input_path, [".html"])
    md_files = iter_files(input_path, TEXT_EXTENSIONS)

    if format_ == "auto":
        format_ = "html" if html_files else "markdown"

    manifest_rows: list[dict] = []

    if format_ == "html":
        converter = HTMLToMarkdownConverter(str(input_path))
        for html_file in html_files:
            html_content = converter._read_html_file(html_file)
            if html_content is None:
                continue
            html_content = converter._remove_html_footer(html_content)
            markdown = converter._html_to_markdown(html_content)
            token_count = count_tokens(markdown)
            if min_tokens is not None and token_count < min_tokens:
                continue
            if max_tokens is not None and token_count > max_tokens:
                continue
            output_file = output_path / html_file.with_suffix(".md").name
            output_file.write_text(markdown, encoding="utf-8")
            manifest_rows.append(
                {
                    "source": str(html_file),
                    "output": str(output_file),
                    "tokens": token_count,
                }
            )
    else:
        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            content = modify_markdown(content)
            token_count = count_tokens(content)
            if min_tokens is not None and token_count < min_tokens:
                continue
            if max_tokens is not None and token_count > max_tokens:
                continue
            output_file = output_path / md_file.name
            output_file.write_text(content, encoding="utf-8")
            manifest_rows.append(
                {
                    "source": str(md_file),
                    "output": str(output_file),
                    "tokens": token_count,
                }
            )

    write_jsonl(output_path / "manifest.jsonl", manifest_rows)
    click.echo(f"Ingested {len(manifest_rows)} files into {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--short-max", type=int, default=SHORT_MAX_TOKENS)
@click.option("--medium-max", type=int, default=MEDIUM_MAX_TOKENS)
@click.option(
    "--chat-provider",
    type=click.Choice(["openai", "openrouter"]),
    default=None,
)
def route(
    input_path: Path,
    output_path: Path,
    short_max: int,
    medium_max: int,
    chat_provider: str | None,
) -> None:
    """Route documents into length-based buckets."""
    output_path.mkdir(parents=True, exist_ok=True)
    buckets = {
        "short": output_path / "short",
        "medium": output_path / "medium",
        "long": output_path / "long",
    }
    for path in buckets.values():
        path.mkdir(parents=True, exist_ok=True)

    routing_rows: list[dict] = []
    for md_file in iter_files(input_path, TEXT_EXTENSIONS):
        content = md_file.read_text(encoding="utf-8")
        token_count = count_tokens(content)
        bucket = route_bucket(
            token_count,
            short_max=short_max,
            medium_max=medium_max,
        )
        output_file = buckets[bucket] / md_file.name
        output_file.write_text(content, encoding="utf-8")
        routing_rows.append(
            {
                "source": str(md_file),
                "output": str(output_file),
                "tokens": token_count,
                "bucket": bucket,
                "chat_provider": chat_provider,
            }
        )

    write_jsonl(output_path / "routing.jsonl", routing_rows)
    click.echo(f"Routed {len(routing_rows)} files into {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--model", default=None)
@click.option("--cache", "cache_path", default=None, type=click.Path(path_type=Path))
@click.option(
    "--embed-provider",
    "embed_provider",
    type=click.Choice(["auto", "openai", "openrouter", "local"]),
    default=None,
)
def embed(
    input_path: Path,
    output_path: Path,
    model: str | None,
    cache_path: Path | None,
    embed_provider: str | None,
) -> None:
    """Generate embeddings for Markdown documents."""
    config = load_config()
    explicit_provider = embed_provider
    resolved_provider = resolve_embed_provider(config, explicit_provider=explicit_provider)
    require_embedding_provider(
        config,
        resolved_provider=resolved_provider,
        explicit_provider=explicit_provider,
        error_cls=click.ClickException,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_path / "embeddings.jsonl"

    cache: dict[str, list[float]] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    embedding_function = build_embedding_function(
        config=config,
        embedding_model=model,
        embed_provider=resolved_provider,
    )

    rows: list[dict] = []
    for md_file in iter_files(input_path, TEXT_EXTENSIONS):
        content = md_file.read_text(encoding="utf-8")
        content_checksum = checksum_text(content)
        embedding = cache.get(content_checksum)
        if embedding is None:
            embedding = embedding_function([content])[0]
            cache[content_checksum] = embedding
        rows.append(
            {
                "id": md_file.stem,
                "path": str(md_file),
                "tokens": count_tokens(content),
                "embedding": embedding,
            }
        )

    write_jsonl(embeddings_path, rows)
    if cache_path:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    click.echo(f"Embedded {len(rows)} files into {embeddings_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path))
@click.option("--collection", default="contextrag")
@click.option("--persist", "persist_path", default=None)
@click.option("--embedding-model", "embedding_model", default=None)
@click.option(
    "--embed-provider",
    "embed_provider",
    type=click.Choice(["auto", "openai", "openrouter", "local"]),
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
    type=click.Choice(["auto", "openai", "openrouter", "local"]),
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
    """Run retrieval evaluation."""
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
    """Run the offline demo evaluation with local embeddings."""
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
    checks = [
        ("OPENAI_API_KEY", bool(config.openai_api_key)),
        ("OPENROUTER_API_KEY", bool(config.openrouter_api_key)),
    ]
    for name, ok in checks:
        status = "ok" if ok else "missing"
        click.echo(f"{name}: {status}")
    resolved_provider = resolve_embed_provider(config, None)
    click.echo(f"embeddings_provider: {resolved_provider}")
    click.echo(f"local_embeddings_model: {config.local_embeddings_model}")
    if not config.openai_api_key and not config.openrouter_api_key:
        click.echo("note: set OPENAI_API_KEY or OPENROUTER_API_KEY for hosted embeddings")

    try:
        import tiktoken  # noqa: F401

        click.echo("tiktoken: ok")
    except ImportError:
        click.echo("tiktoken: missing")

    click.echo(
        "chromadb: {}".format("ok" if find_spec("chromadb") else "missing")
    )
    click.echo(
        "sentence-transformers: {}".format(
            "ok" if find_spec("sentence_transformers") else "missing"
        )
    )


if __name__ == "__main__":
    main()
