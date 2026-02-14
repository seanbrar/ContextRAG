# Baseline Study

Expanded baseline comparison on `data/eval-expanded`.

| Scenario | Baseline | Retrieval | Chunk | Overlap | Precision@k | Recall@k | Hit@1 | MRR@k | nDCG@k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform-512-dense | uniform | dense | 512 | 0 | 0.258 | 0.818 | 0.860 | 0.897 | 0.820 |
| uniform-1000-dense | uniform | dense | 1000 | 0 | 0.258 | 0.835 | 0.800 | 0.877 | 0.808 |
| uniform-2000-dense | uniform | dense | 2000 | 0 | 0.264 | 0.815 | 0.740 | 0.838 | 0.783 |
| uniform-overlap-dense | uniform | dense | 1000 | 200 | 0.256 | 0.812 | 0.850 | 0.897 | 0.809 |
| semantic-dense | semantic | dense | 1000 | 0 | 0.258 | 0.815 | 0.760 | 0.847 | 0.787 |
| uniform-bm25 | uniform | bm25 | 1000 | 0 | 0.244 | 0.802 | 0.840 | 0.887 | 0.812 |
| uniform-hybrid | uniform | hybrid | 1000 | 0 | 0.268 | 0.849 | 0.860 | 0.907 | 0.847 |
| uniform-dense-rerank | uniform | dense-rerank | 1000 | 0 | 0.260 | 0.835 | 0.780 | 0.855 | 0.807 |
