# ContextRAG

[![Tests](https://github.com/seanbrar/ContextRAG/actions/workflows/test.yml/badge.svg)](https://github.com/seanbrar/ContextRAG/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)](https://github.com/seanbrar/ContextRAG/actions/workflows/test.yml)
[![Python 3.11-3.12](https://img.shields.io/badge/python-3.11--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

RAG evaluation framework demonstrating that **length-based adaptive chunking does not outperform uniform chunking on the committed benchmarks**.

## Project Scope

Core scope (recommended for review/research claims):
- `uniform` vs `router` chunking
- dense retrieval (`retrieval_mode=dense`)
- datasets: `data/eval-expanded`, `data/eval-external`

Exploratory scope (kept for follow-up research, not canonical claims):
- semantic chunking, overlap sweeps, BM25/hybrid/rerank retrieval
- public transfer slice `data/eval-scifact-mini`

## The Research Question

> Does routing documents to different chunk sizes based on length improve RAG retrieval quality?

**Hypothesis**: Short documents (<=3,500 tokens) should remain whole, medium documents (3,501-15,000) should use larger chunks, and long documents (>15,000) should use smaller chunks. A "router" that adapts chunk size to document length should outperform uniform chunking.

## Canonical Finding

Across all committed evaluations, length-based routing **never beats** uniform chunking:

- Mixed corpus + hosted embeddings: **tie** (identical precision@5 and recall@5)
- Expanded local matrix (`k={3,5,10}`): router is **consistently worse**

Mixed-corpus hosted run slice:

| Strategy | Precision@5 | Recall@5 |
|----------|-------------|----------|
| Uniform chunking | 0.197 | 0.983 |
| Adaptive router | 0.197 | 0.983 |

Scope of this claim:

- Mixed corpus (`data/eval-mixed`): hosted `text-embedding-3-small`, `k=5`, 3 repeated runs
- RFC corpus (`data/demo`): OpenRouter `qwen/qwen3-embedding-8b`, `k=5`, uniform vs router
- External holdout (`data/eval-external`): local MiniLM matrix, `k={3,5,10}`
- Cost/quality side study: OpenAI `text-embedding-3-small` vs `text-embedding-3-large` (uniform baseline)

## Why This Matters

Within this evaluation scope, routing by document length did not outperform uniform chunking (and sometimes underperformed it). This finding simplifies RAG system design:

- **Use uniform chunking** - simpler baseline with equal or better observed quality
- **Skip adaptive complexity** - no accuracy benefit to justify the cost
- **Focus elsewhere** - retrieval improvements likely come from better embeddings or reranking, not chunk routing

For full methodology, see [docs/paper.md](docs/paper.md).

## Quickstart

```bash
# Install
git clone https://github.com/seanbrar/ContextRAG.git
cd ContextRAG
uv sync --all-extras

# Run offline demo (no API keys needed)
uv run contextrag demo
```

Output: `runs/demo_eval.json` with precision/recall metrics.

## Reproduce Core Study

```bash
# 1) Validate datasets
uv run contextrag validate-dataset --dataset data/eval-expanded
uv run contextrag validate-dataset --dataset data/eval-external

# 2) Run core local matrix (uniform/router × k={3,5,10}, dense retrieval)
uv run contextrag core matrix \
  --dataset data/eval-expanded \
  --k-values 3,5,10 \
  --embed-provider local \
  --run-root runs/matrix_eval_expanded_local \
  --persist-root runs/chroma-matrix-eval-expanded-local

# 3) Compare any two runs (example: uniform vs router at k=5)
uv run contextrag compare \
  --run-a runs/matrix_eval_expanded_local/uniform_k5 \
  --run-b runs/matrix_eval_expanded_local/router_k5 \
  --output runs/matrix_eval_expanded_local/comparisons/uniform_vs_router_k5_manual.json

# 4) Render a reviewer-friendly report
python3 scripts/render_matrix_report.py \
  --input runs/matrix_eval_expanded_local/matrix_summary.json \
  --output docs/matrix_eval_expanded_local.md
```

Primary artifacts:
- `runs/matrix_eval_expanded_local/matrix_summary.json`
- `runs/matrix_eval_expanded_local/matrix_summary.md`
- `runs/matrix_eval_expanded_local/comparisons/*.json`
- `docs/matrix_eval_expanded_local.md`

## Build Reviewer Bundle

```bash
make reviewer-bundle
```

This one command regenerates:
- expanded local matrix artifacts and comparisons
- external holdout matrix artifacts and comparisons
- `docs/paper_tables.md` (paper-ready aggregate + inference tables)
- `docs/reviewer_bundle.md` (review checklist/report index)

## Exploratory Extensions

Exploratory commands and datasets are documented in
[`docs/exploratory.md`](docs/exploratory.md).

## Dev Helpers

```bash
# Install deps
make install

# Lint, typecheck, tests
make all
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `contextrag core eval` | Claim-aligned single eval (`uniform/router + dense`, core datasets) |
| `contextrag core matrix` | Claim-aligned matrix (`uniform/router × k`, dense, core datasets) |
| `contextrag eval` | Core evaluation (`uniform/router + dense`) with optional exploratory modes |
| `contextrag demo` | Offline evaluation with local embeddings |
| `contextrag matrix` | Run matrix experiments (core and exploratory) |
| `contextrag compare` | Compare two run directories with per-query deltas + inference |
| `contextrag validate-dataset` | Validate dataset/query schema before eval |
| `contextrag artifact-eval` | Rebuild/verify committed artifact checksums |
| `contextrag doctor` | Check configuration health |
| `contextrag db index` | Build vector index from documents |
| `contextrag db query` | Query the vector index |

### Example: Core Evaluation

```bash
# Core eval path (recommended)
uv run contextrag core eval \
    --dataset data/eval-expanded \
    --baseline uniform \
    --k 5 \
    --embed-provider local \
    --output runs/core_eval_uniform_k5.json
```

### Example: Build and Query Index

```bash
# Build index
uv run contextrag db index \
    --input data/demo/documents \
    --collection my_docs \
    --persist ./runs/chroma

# Query
uv run contextrag db query \
    --collection my_docs \
    --persist ./runs/chroma \
    --query "HTTP caching headers"
```

## Configuration

Set environment variables or use `.env`:

```bash
# Embeddings (via chromaroute)
OPENROUTER_API_KEY=sk-or-...        # For hosted embeddings
EMBED_PROVIDER=auto                  # auto | openrouter | local
OPENROUTER_EMBEDDINGS_MODEL=openai/text-embedding-3-small
LOCAL_EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Chat (for future semantic chunking research)
OPENAI_API_KEY=sk-...               # For OpenAI chat
OPENAI_CHAT_MODEL=gpt-4o-mini
CONTEXTRAG_CHAT_PROVIDER=auto       # auto | openai | openrouter
```

Note: direct `embed_provider=openai` is intentionally unsupported for embeddings.
Use OpenAI embedding models through OpenRouter model IDs (for example,
`openai/text-embedding-3-small`).

## Historical Context

This project evolved over 2022–2025:

**2022–2023**: Cost-based model routing. GPT-3.5 (4K context) was significantly cheaper than GPT-3.5-16K, motivating intelligent routing.

**2024**: Context windows expanded to 128K–2M tokens. Focus shifted to chunking strategies.

**2025**: Rigorous evaluation infrastructure revealed the null result. The embedding abstraction was extracted into [chromaroute](https://github.com/seanbrar/chromaroute).

See [docs/evolution.md](docs/evolution.md) for the full journey.

## Architecture

```mermaid
flowchart LR
    A[Documents] --> B[Index]
    B --> C[chromaroute]
    C --> D{Provider}
    D -->|OpenRouter| E[text-embedding-3]
    D -->|Local| F[MiniLM]
    E & F --> G[(ChromaDB)]
    G --> H[Query]
    H --> I[Evaluate]
```

ContextRAG is a CLI tool built on [chromaroute](https://github.com/seanbrar/chromaroute), a provider-agnostic embedding library for ChromaDB.

## Testing

```bash
make test-cov
```

Target: high test coverage with CI gate (`--cov-fail-under=95`).

## Docs

- [docs/paper.md](docs/paper.md) - Full research methodology and results
- [docs/preregistration.md](docs/preregistration.md) - Locked hypotheses, endpoints, and decision rules
- [docs/annotation_protocol.md](docs/annotation_protocol.md) - Dual-annotation and agreement workflow
- [docs/matrix_eval_expanded_local.md](docs/matrix_eval_expanded_local.md) - Latest local matrix dashboard
- [docs/matrix_eval_external_local.md](docs/matrix_eval_external_local.md) - External holdout matrix dashboard
- [docs/baseline_study.md](docs/baseline_study.md) - Expanded baseline fairness study
- [docs/paper_tables.md](docs/paper_tables.md) - Generated paper-ready tables
- [docs/reviewer_bundle.md](docs/reviewer_bundle.md) - Reviewer-oriented artifact index
- [docs/evolution.md](docs/evolution.md) - Project history 2022–2025
- [docs/design-decisions.md](docs/design-decisions.md) - Architecture rationale

## Related Work

- [chromaroute](https://github.com/seanbrar/chromaroute) - Provider-agnostic embeddings for ChromaDB (extracted from this project)
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [OpenRouter](https://openrouter.ai/) - Multi-provider API gateway

## License

[MIT](LICENSE)
