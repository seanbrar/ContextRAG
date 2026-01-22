# ContextRAG

[![GitHub License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
[![Python 3.11-3.12](https://img.shields.io/badge/python-3.11--3.12-blue.svg)](https://www.python.org/downloads/)

A scalable vector database system for semantic search and document retrieval with context-aware processing.

## Table of Contents

- [Background](#background)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Install](#install)
- [Environment](#environment)
- [Usage](#usage)
- [Context Length Management](#context-length-management)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Docs](#docs)
- [Results Summary](#results-summary)
- [Future Enhancements](#future-enhancements)
- [Related Work](#related-work)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background

ContextRAG began in 2022–2023 as an exploration of **cost-aware model routing** for RAG systems. The original motivation was practical: GPT-3.5 (4K context) was significantly cheaper than GPT-3.5-16K, so routing documents to the appropriate model based on length could reduce costs without sacrificing capability.

As context windows expanded dramatically (128K–2M tokens by 2024–2025), the cost arbitrage diminished. The project evolved to explore whether **adaptive chunking strategies** could improve retrieval quality by:

1. Preserving semantic coherence in short documents (no chunking)
2. Using larger chunks for medium documents (fewer boundary artifacts)
3. Applying fine-grained chunking only to very long documents

**Key finding**: Rigorous evaluation shows that adaptive chunking does not improve retrieval accuracy over uniform chunking. Both strategies achieve identical precision and recall across heterogeneous document collections. See [Results Summary](#results-summary) for details.

The project's value lies in its **infrastructure**: a provider-agnostic embedding layer, reproducible evaluation framework with efficiency metrics, and well-documented methodology for testing RAG strategies.

## Key Features

- **Intelligent Document Processing**: Convert HTML to Markdown, clean document structure, and prepare text for embedding
- **Context-Length Awareness**: Automatically categorize documents by token length to optimize processing
- **Vector Embeddings**: Utilize OpenAI embeddings for semantic understanding of document content
- **Similarity Matching**: Find related documents using cosine similarity between document vectors
- **Markdown Processing**: Specialized handling for Markdown syntax and document structure
- **Customizable Classification**: Group documents by topics and categories

## System Architecture

The system is built around these core components:

```
+---------------------+     +----------------------+     +------------------+
| Document Collection |---->| Processing Pipeline  |---->| Vector Database  |
+---------------------+     +----------------------+     +------------------+
         |                          |                           |
         |                          v                           v
         |                  +----------------+         +----------------+
         +----------------->| Length-Based   |         | Semantic       |
                            | Classification |         | Search Engine  |
                            +----------------+         +----------------+
```

1. **Data Processing**
   - HTML to Markdown conversion
   - Document cleaning and normalization
   - Token-length detection

2. **Markdown Grouping**
   - File categorization
   - Topic assignment
   - Similarity detection

3. **Vector Database**
   - ChromaDB integration
   - Embedding generation
   - Similarity search

## Install

Requires Python 3.11 or 3.12.

```bash
# Clone repository
git clone https://github.com/seanbrar/ContextRAG.git
cd ContextRAG

# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies using Poetry
poetry install

# Optional: set up environment variables
# cp .env.example .env
```

## Environment

Required for embeddings and indexing (OpenAI or OpenRouter):

- `OPENAI_API_KEY` (preferred for OpenAI embeddings)
- `OPENROUTER_API_KEY` (fallback for embeddings and chat)

Optional (for OpenRouter chat routing or smoke tests):

- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `OPENROUTER_CHAT_MODEL` (default: `mistralai/devstral-2512:free`)
- `OPENROUTER_EMBEDDINGS_MODEL` (default: `qwen/qwen3-embedding-8b`)
- `CONTEXTRAG_EMBED_PROVIDER` (default: `auto`; options: `openai`, `openrouter`, `local`)
- `LOCAL_EMBEDDINGS_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `OPENROUTER_REFERER` (optional header for OpenRouter rankings)
- `OPENROUTER_TITLE` (optional header for OpenRouter rankings)
- `OPENROUTER_EMBED_PROVIDER_JSON` (optional JSON for OpenRouter provider routing)
- `OPENROUTER_EMBED_PROVIDER_ORDER` (optional comma list for routing order)
- `OPENROUTER_EMBED_ALLOW_FALLBACKS` (optional `true`/`false`)

Model defaults can be overridden:

- `OPENAI_EMBEDDINGS_MODEL` (default: `thenlper/gte-base`)
- `OPENAI_CHAT_MODEL_SHORT` (default: `gpt-3.5-turbo-1106`)
- `OPENAI_CHAT_MODEL_MEDIUM` (default: `gpt-3.5-turbo-16k`)

If you use OpenAI embeddings, set `OPENAI_EMBEDDINGS_MODEL=text-embedding-3-large`.
If you use OpenRouter embeddings, set `OPENROUTER_EMBEDDINGS_MODEL` to a model that supports embeddings
(for example `qwen/qwen3-embedding-8b`).
If you use local embeddings, the model will be downloaded automatically.

For OpenRouter embeddings with short context limits, use chunking:

```bash
poetry run contextrag index --input data/processed --collection contextrag --persist ./runs/chroma --chunk-words 400
```

## Usage

### CLI Quickstart

```bash
# 1) Ingest raw documents into cleaned Markdown
poetry run contextrag ingest --input data/raw --output data/processed --format auto

# 2) Route by length buckets
poetry run contextrag route --input data/processed --output data/routed

# 3) Build a vector index (Chroma)
poetry run contextrag index --input data/processed --collection contextrag --persist ./runs/chroma

# 4) Query the index
poetry run contextrag query --collection contextrag --persist ./runs/chroma --query "token limits"
```

### Doctor

```bash
poetry run contextrag doctor
```

### Python API (selected)

```python
from contextrag.ingest.html_to_markdown import HTMLToMarkdownConverter

converter = HTMLToMarkdownConverter("./my_documents")
converter.convert_all_files(use_target_folder=True)
```
## Context Length Management

ContextRAG addresses the challenge of varying document lengths through a three-tier approach:

```
┌────────────────────────────────────────────────────────────┐
│                   Context Length Classification            │
├────────────────┬────────────────────────┬──────────────────┤
│  Short         │  Medium                │  Long            │
│  (≤3500 tokens)│  (3500-15000 tokens)   │  (>15000 tokens) │
├────────────────┼────────────────────────┼──────────────────┤
│ - Direct       │ - Chunked processing   │ - Advanced       │
│   processing   │ - Section-based        │   chunking       │
│ - Full context │   embeddings           │ - Hierarchical   │
│   embedding    │ - Summary generation   │   embeddings     │
└────────────────┴────────────────────────┴──────────────────┘
```

This approach ensures:
- Efficient processing of documents regardless of size
- Optimal token usage for embedding models
- Accurate semantic search across varying document lengths

## Evaluation

The evaluation framework compares chunking strategies with comprehensive metrics:

| Metric | Description |
|--------|-------------|
| Precision@k | Fraction of top-k retrieved documents that are relevant |
| Recall@k | Fraction of relevant documents appearing in top-k |
| Efficiency | Chunk count, token usage, indexing time, query latency |
| Variance | Multiple runs to verify determinism |

The eval command expects a dataset directory with `documents/` and `queries.jsonl`:

```bash
poetry run contextrag eval --dataset data/demo --baseline router --k 5 --output runs/eval.json
```

To create a small RFC-based dataset:

```bash
python scripts/datasets/download_rfc_dataset.py --rfcs 822,9110,9595 --output data/demo/documents
```

You can also run evals from a YAML config:

```bash
poetry run contextrag eval --config experiments/eval_rfc.yaml
```

To capture run artifacts (summary/per-query/metadata), add a run directory:

```bash
poetry run contextrag eval --config experiments/eval_rfc.yaml --run-dir runs/eval_rfc
```

## Testing

Run the test suite to verify system functionality:

```bash
pytest tests/
```

## Docs

- `docs/architecture.md` — Pipeline stages and data flow
- `docs/design-decisions.md` — Engineering rationale and tradeoffs
- `docs/paper.md` — Academic writeup of the approach
- `docs/results.md` — Evaluation methodology and results

## Results Summary

### Mixed Corpus (Primary Evaluation)

12 documents (3 short, 1 medium, 8 long), 493k tokens, 60 queries, 3 runs:

| Baseline | Precision@5 | Recall@5 | Chunks | Variance |
| -------- | ----------- | -------- | ------ | -------- |
| Uniform  | 0.197       | 0.983    | 499    | 0 (deterministic) |
| Router   | 0.197       | 0.983    | 490    | 0 (deterministic) |

### Key Finding

**Adaptive chunking does not improve retrieval accuracy.** Both strategies achieve identical precision and recall. The router produces 1.8% fewer chunks, but this marginal efficiency gain does not translate to accuracy improvement.

### Interpretation

This negative result is reproducible and informative:

1. **Modern embeddings are robust** — text-embedding-3-small handles chunk boundaries well
2. **Simplicity wins** — uniform chunking is equally effective with less complexity
3. **Original motivation obsolete** — cost-based model routing (GPT-3.5 vs GPT-3.5-16K) is less relevant with modern pricing

### What This Project Demonstrates

- **Rigorous evaluation methodology** with efficiency metrics and variance analysis
- **Provider-agnostic infrastructure** (OpenAI, OpenRouter, local embeddings)
- **Honest reporting** of negative results with clear interpretation

See `docs/results.md` for complete methodology, efficiency breakdowns, and artifact paths.

## Future Directions

Given the negative result on adaptive chunking, potential directions include:

- **Cost-aware routing**: Route to different embedding providers based on cost/quality tradeoffs
- **Hybrid search**: Combine dense embeddings with sparse retrieval (BM25)
- **Alternative chunking strategies**: Semantic chunking, overlapping windows, hierarchical embeddings
- **Benchmark expansion**: Evaluate on standard IR benchmarks (NQ, TriviaQA) for broader validation

## Related Work

This project builds upon and extends research in the following areas:

- Vector search systems like Facebook AI Similarity Search (FAISS)
- Hierarchical document embedding approaches (Cohere et al., 2023)
- Adaptive chunking strategies for long documents (OpenAI, 2023)

## Maintainers

[Sean Brar](https://github.com/seanbrar) - Project creator and primary maintainer

## Contributing

Contributions to ContextRAG are welcome! Here's how you can help:

- Report bugs by opening an issue
- Suggest enhancements or new features
- Submit pull requests with improvements
- Help with documentation

Please ensure that your contributions adhere to our coding standards and include appropriate tests.

## License

[MIT License](LICENSE)
