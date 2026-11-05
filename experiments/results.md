# RAG Experiment Results (Before vs. After)

| Pipeline Stage | Retrieval Hit Rate | Answer Correctness | Refusal Accuracy |
|---|---|---|---|
| 1. Baseline (Dense Only, top-3) | 46.2% | 23.1% | 100.0% |
| 2. Hybrid Search (BM25 + Dense RRF, top-4) | 100.0% | 84.6% | 100.0% |
| 3. Hybrid + Cross-Encoder Re-ranker (top-15 -> top-4) | 100.0% | 84.6% | 100.0% |
