# ContextRAG

A scalable vector database system for semantic search and document retrieval with context-aware processing.

## Overview

ContextRAG is a modular system designed to process, analyze, and retrieve information from large document collections using vector embeddings and semantic search. The system handles documents of varying lengths and complexities while maintaining efficient search capabilities.

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

## Key Features

- **Intelligent Document Processing**: Convert HTML to Markdown, clean document structure, and prepare text for embedding
- **Context-Length Awareness**: Automatically categorize documents by token length to optimize processing
- **Vector Embeddings**: Utilize OpenAI embeddings for semantic understanding of document content
- **Similarity Matching**: Find related documents using cosine similarity between document vectors
- **Markdown Processing**: Specialized handling for Markdown syntax and document structure
- **Customizable Classification**: Group documents by topics and categories

## System Architecture

The system is built around these core components:

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

## Installation

```bash
# Clone repository
git clone https://github.com/seanbrar/ContextRAG.git
cd ContextRAG

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

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
│                   Context Length Classification             │
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

## Requirements

- Python 3.8+
- ChromaDB
- OpenAI API access (for embeddings)
- BeautifulSoup4
- NLTK
- Scikit-learn

## License

[MIT License](LICENSE)