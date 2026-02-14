# Annotation Protocol

This protocol defines dual-annotation and adjudication requirements for
ContextRAG relevance labels.

## Required Workflow

1. Two independent annotation rounds (`annotator_a`, `annotator_b`)
2. Agreement report generation (Jaccard, macro P/R/F1, Cohen's kappa)
3. Adjudicated final labels committed to `queries.jsonl`
4. Provenance metadata committed as `provenance.json`

For legacy datasets where independent dual-human rounds were not originally
captured, a retrospective dual-pass workflow is permitted if:
- the method is explicitly documented in `provenance.json`
- agreement metrics are regenerated and committed
- adjudication status is explicitly marked as retrospective

## Annotation Round Format

One JSON object per line:

```json
{"query": "...", "relevant_ids": ["doc1", "doc2"]}
```

## Agreement Computation

Use:

```bash
python3 scripts/compute_annotation_agreement.py \
  --round-a data/<dataset>/annotations/annotator_a.jsonl \
  --round-b data/<dataset>/annotations/annotator_b.jsonl \
  --documents-dir data/<dataset>/documents \
  --output data/<dataset>/annotations/agreement.json
```

## Provenance Requirements

Each `provenance.json` should include:
- source corpus and version
- annotation protocol version
- annotator round file paths
- agreement report path
- adjudication date and owner
- data generation scripts used
