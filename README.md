# ContextRAG

[![GitHub License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A scalable vector database system for semantic search and document retrieval with context-aware processing.

## Table of Contents

- [Background](#background)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Install](#install)
- [Usage](#usage)
- [Context Length Management](#context-length-management)
- [Evaluation](#evaluation)
- [Testing](#testing)
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

```bash
# Clone repository
git clone https://github.com/seanbrar/ContextRAG.git
cd ContextRAG

# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies using Poetry
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env to add your API keys
```

## Usage

### Processing Documents

```python
from src.data_processing.html_to_markdown import HTMLToMarkdownConverter

# Convert HTML files to Markdown
converter = HTMLToMarkdownConverter("./my_documents")
converter.convert_all_files(use_target_folder=True)
```

### Finding Similar Documents

```python
from src.markdown_grouping.file_grouping import read_markdown_files, compute_similarity, group_similar_files

# Read markdown files
files, checksums = read_markdown_files("./processed_documents")

# Compute similarity matrix
similarity_matrix = compute_similarity(files, checksums, {})

# Group similar files
groups = group_similar_files(similarity_matrix, threshold=0.7)

# Print results
for file_index, similar_files in groups.items():
    print(f"File: {list(files.keys())[file_index]} is similar to:")
    for similar_file_index in similar_files:
        print(f" - {list(files.keys())[similar_file_index]}")
```

### Querying the Vector Database

```python
from src.vector_db.main import VectorDB

# Initialize vector database
vector_db = VectorDB(collection_name="documentation")

# Add documents to the vector database
documents = ["Document 1 content", "Document 2 content", "Document 3 content"]
vector_db.add_documents(documents)

# Query the vector database
results = vector_db.query(query_texts=["How to configure settings?"], n_results=3)

# Process results
for i, doc in enumerate(results["documents"][0]):
    print(f"Result {i+1}: {doc}")
    print(f"Distance: {results['distances'][0][i]}")
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

Evaluation of ContextRAG is currently in progress. We are assessing performance across these dimensions:

| Metric | Description | Status |
|--------|-------------|--------|
| Precision@k | Relevance of top-k retrieved documents | In progress |
| Recall@k | Proportion of relevant documents retrieved | In progress |
| Processing Efficiency | Time and resource usage across document sizes | Initial testing |
| Accuracy vs. Context Length | Performance correlation with document length | Planned |

Preliminary observations suggest significant improvements in retrieval quality for longer documents compared to fixed-chunking approaches, but comprehensive benchmarks are still being developed.

If you're interested in contributing to the evaluation effort or have suggestions for benchmark datasets, please open an issue to discuss.

## Testing

Run the test suite to verify system functionality:

```bash
pytest tests/
```

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