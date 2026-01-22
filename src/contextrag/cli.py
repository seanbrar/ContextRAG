from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import click
from openai import OpenAI

from contextrag.config import load_config
from contextrag.core.tokenizer import count_tokens
from contextrag.eval.runner import run_eval
from contextrag.ingest.html_to_markdown import HTMLToMarkdownConverter
from contextrag.ingest.markdown_processing import modify_markdown
from contextrag.index.vector_store import VectorDB


def _iter_files(root: Path, extensions: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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

    html_files = _iter_files(input_path, [".html"])
    md_files = _iter_files(input_path, [".md"])

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

    _write_jsonl(output_path / "manifest.jsonl", manifest_rows)
    click.echo(f"Ingested {len(manifest_rows)} files into {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--short-max", type=int, default=3500)
@click.option("--medium-max", type=int, default=15000)
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
    for md_file in _iter_files(input_path, [".md"]):
        content = md_file.read_text(encoding="utf-8")
        token_count = count_tokens(content)
        if token_count <= short_max:
            bucket = "short"
        elif token_count <= medium_max:
            bucket = "medium"
        else:
            bucket = "long"
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

    _write_jsonl(output_path / "routing.jsonl", routing_rows)
    click.echo(f"Routed {len(routing_rows)} files into {output_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--model", default=None)
@click.option("--cache", "cache_path", default=None, type=click.Path(path_type=Path))
def embed(
    input_path: Path,
    output_path: Path,
    model: str | None,
    cache_path: Path | None,
) -> None:
    """Generate embeddings for Markdown documents."""
    config = load_config()
    if not config.openai_api_key:
        raise click.ClickException("OPENAI_API_KEY is required for embeddings.")

    output_path.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_path / "embeddings.jsonl"

    cache: dict[str, list[float]] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    client = OpenAI(api_key=config.openai_api_key)
    used_model = model or config.openai_embeddings_model

    rows: list[dict] = []
    for md_file in _iter_files(input_path, [".md"]):
        content = md_file.read_text(encoding="utf-8")
        checksum = _checksum(content)
        embedding = cache.get(checksum)
        if embedding is None:
            response = client.embeddings.create(
                model=used_model,
                input=content,
                encoding_format="float",
            )
            embedding = response.data[0].embedding
            cache[checksum] = embedding
        rows.append(
            {
                "id": md_file.stem,
                "path": str(md_file),
                "tokens": count_tokens(content),
                "embedding": embedding,
            }
        )

    _write_jsonl(embeddings_path, rows)
    if cache_path:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    click.echo(f"Embedded {len(rows)} files into {embeddings_path}")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path))
@click.option("--collection", default="contextrag")
@click.option("--persist", "persist_path", default=None)
def index(input_path: Path, collection: str, persist_path: str | None) -> None:
    """Build a vector index from Markdown documents."""
    vector_db = VectorDB(collection_name=collection, persist_path=persist_path)
    documents: list[str] = []
    ids: list[str] = []
    for md_file in _iter_files(input_path, [".md"]):
        documents.append(md_file.read_text(encoding="utf-8"))
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
@click.option("--dataset", "dataset_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--baseline",
    type=click.Choice(["uniform", "router"]),
    default="router",
)
@click.option("--k", "top_k", type=int, default=5)
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--persist", "persist_path", default=None)
def eval(
    dataset_path: Path,
    baseline: str,
    top_k: int,
    output_path: Path,
    persist_path: str | None,
) -> None:
    """Run retrieval evaluation."""
    results = run_eval(
        dataset_path=dataset_path,
        baseline=baseline,
        k=top_k,
        persist_path=persist_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = results["summary"]
    click.echo(
        "precision@k={precision:.3f} recall@k={recall:.3f} (k={k})".format(
            precision=summary["precision_at_k"],
            recall=summary["recall_at_k"],
            k=summary["k"],
        )
    )


@main.command()
def doctor() -> None:
    """Check configuration and environment health."""
    config = load_config()
    checks = [
        ("OPENAI_API_KEY", bool(config.openai_api_key)),
        ("OPENROUTER_API_KEY", bool(config.openrouter_api_key)),
    ]
    for name, ok in checks:
        status = "ok" if ok else "missing"
        click.echo(f"{name}: {status}")

    try:
        import tiktoken  # noqa: F401

        click.echo("tiktoken: ok")
    except ImportError:
        click.echo("tiktoken: missing")


if __name__ == "__main__":
    main()
