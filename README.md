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

ContextRAG addresses a critical challenge in large language model applications: efficiently processing and retrieving information from documents of varying lengths and complexities. Traditional RAG (Retrieval-Augmented Generation) systems often struggle with:

1. Context length limitations of embedding models
2. Loss of semantic relationships in excessively chunked documents
3. Inefficient processing of extremely long documents

This project implements a novel approach that dynamically adapts to document characteristics, preserving semantic meaning while optimizing for computational efficiency.

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

Evaluation of ContextRAG is currently in progress. The evaluation harness supports:

| Metric | Description | Status |
|--------|-------------|--------|
| Precision@k | Relevance of top-k retrieved documents | In progress |
| Recall@k | Proportion of relevant documents retrieved | In progress |
| Processing Efficiency | Time and resource usage across document sizes | Initial testing |
| Accuracy vs. Context Length | Performance correlation with document length | Planned |

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

- `docs/architecture.md`
- `docs/paper.md`
- `docs/results.md`

## Results Summary

RFC demo (precision@5 / recall@5, `qwen/qwen3-embedding-8b`):

| Baseline | Precision@5 | Recall@5 | Notes |
| -------- | ----------- | -------- | ----- |
| uniform  | 0.133       | 0.667    | `runs/eval_uniform/summary.json` |
| router   | 0.133       | 0.667    | `runs/eval_router/summary.json` |

These results are on a small demo dataset and are intended for reproducibility.

## Future Enhancements

- Add support for additional document formats (PDF, DOCX)
- Implement more advanced embedding models
- Develop a query optimization layer
- Create a web interface for document exploration
- Add document versioning and change tracking

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
