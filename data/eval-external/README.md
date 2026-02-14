External holdout benchmark for ContextRAG.

Purpose:
- Provide an out-of-corpus split to check whether findings from `data/eval-mixed`
  and `data/eval-expanded` hold on a distinct RFC set.
- Increase protocol difficulty with contrastive prompts and hard-negative phrasing.

Contents:
- `documents/`: 10 RFC texts not used in the mixed/expanded corpora
- `queries.jsonl`: 36 total queries
- `provenance.json`: dataset/source and annotation metadata
- 20 single-label queries (`relevant_ids`)
- 16 multi-label/graded queries (`relevant`)

Schema:
- Legacy line shape: `{"query": "...", "relevant_ids": ["doc1", ...]}`
- v2 line shape: `{"query": "...", "relevant": [{"id": "doc1", "score": 2.0}, ...]}`

Sources (retrieved from rfc-editor.org):
- RFC 2119, RFC 3986, RFC 6265, RFC 6750, RFC 7230, RFC 7231,
  RFC 7540, RFC 8259, RFC 8441, RFC 9114

Validation:

```bash
uv run contextrag validate-dataset --dataset data/eval-external
```
