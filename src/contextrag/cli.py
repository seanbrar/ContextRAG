from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import click
from openai import OpenAI

from contextrag.config import load_config, resolve_embed_provider
from contextrag.core.tokenizer import count_tokens
from contextrag.eval.runner import run_eval
from contextrag.experiments.eval_config import EvalConfig, load_eval_config
from contextrag.experiments.run_logger import write_run_artifacts
from contextrag.ingest.html_to_markdown import HTMLToMarkdownConverter
from contextrag.ingest.markdown_processing import modify_markdown
from contextrag.index.vector_store import VectorDB


def _iter_files(root: Path, extensions: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


TEXT_EXTENSIONS = (".md", ".txt")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _chunk_words(text: str, chunk_words: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_words <= 0:
        return [" ".join(words)]
    chunks = []
    step = max(chunk_words - overlap, 1)
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
    return chunks


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
    md_files = _iter_files(input_path, TEXT_EXTENSIONS)

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
    for md_file in _iter_files(input_path, TEXT_EXTENSIONS):
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
    if not config.openai_api_key and not config.openrouter_api_key:
        raise click.ClickException(
            "OPENAI_API_KEY or OPENROUTER_API_KEY is required for embeddings."
        )

    output_path.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_path / "embeddings.jsonl"

    cache: dict[str, list[float]] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    if config.openai_api_key:
        client = OpenAI(api_key=config.openai_api_key)
        used_model = model or config.openai_embeddings_model
    else:
        client = OpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
        )
        used_model = model or config.openrouter_embeddings_model

    rows: list[dict] = []
    for md_file in _iter_files(input_path, TEXT_EXTENSIONS):
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
    for md_file in _iter_files(input_path, TEXT_EXTENSIONS):
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
        write_run_artifacts(
            run_dir=run_dir,
            results=results,
            metadata={
                "dataset": str(dataset_path),
                "baseline": baseline,
                "k": top_k,
                "embed_provider": embed_provider,
                "embedding_model": embedding_model,
                "output": str(output_path),
                "persist": persist_path,
            },
        )
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
    click.echo(
        "embeddings_provider: {}".format(resolve_embed_provider(config, None))
    )

    try:
        import tiktoken  # noqa: F401

        click.echo("tiktoken: ok")
    except ImportError:
        click.echo("tiktoken: missing")


if __name__ == "__main__":
    main()
