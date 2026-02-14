# Preregistration (v1)

Date locked: February 14, 2026

This document defines the pre-registered analysis plan for ContextRAG's core
question: whether length-based chunk routing improves retrieval quality versus a
uniform baseline.

## Research Question

Does routing documents into short/medium/long chunking policies improve
retrieval performance compared with uniform chunking?

## Primary Hypothesis

`H1`: The router baseline improves rank-aware retrieval quality over uniform
chunking on held-out evaluations.

Operationalized as:
- Positive delta in `nDCG@k` and `MRR@k` for router - uniform
- Tested at `k in {3, 5, 10}` on `data/eval-expanded` and `data/eval-external`

## Null Hypothesis

`H0`: There is no meaningful improvement from router relative to uniform.

## Primary Endpoint

- `nDCG@k` (primary endpoint, rank-aware, graded relevance compatible)

## Secondary Endpoints

- `MRR@k`
- `Hit@1`
- `Precision@k`
- `Recall@k`
- `unique_doc_ratio@k` (diagnostic for duplicate-doc retrieval behavior)

## Experimental Conditions

- Baselines: `uniform`, `router`
- Provider: `local` (MiniLM) for deterministic artifact generation
- Datasets: `data/eval-expanded`, `data/eval-external`
- k-values: `3, 5, 10`

## Decision Rules

For each dataset/k pair:
1. Compute paired per-query deltas (`router - uniform`)
2. Report mean delta, bootstrap 95% CI, and paired randomization p-value
3. Apply Holm-Bonferroni correction across tested metrics per dataset/k slice
4. Evaluate equivalence for the primary endpoint (`nDCG@k`) using TOST with a
   symmetric margin `epsilon = 0.02`

Interpretation:
- Improvement claim: corrected p-value < 0.05 and positive mean delta for the
  primary endpoint without violating equivalence bounds.
- No-win claim: primary endpoint fails improvement criteria and is either
  equivalent (TOST pass) or negative in direction.

## Exclusions and Data Integrity Rules

- Drop blank lines in `queries.jsonl`
- Reject malformed records and duplicate relevant IDs
- Do not alter query text or relevance labels during scoring
- Track dataset SHA-256 and run metadata in `manifest.json`

## Reporting Requirements

Each comparison report must include:
- Raw per-metric p-values
- Holm-adjusted p-values
- Effect sizes (Cohen's d and Cliff's delta)
- Equivalence/non-inferiority fields for the primary metric

## Deviations Policy

Any deviation from this preregistration must be listed in the output artifact
and in `docs/results.md` under a "Deviations" subsection.
