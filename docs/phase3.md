# Phase 3 — FAISS-Based Scalable Retrieval

Phase 3 adds a scalable retrieval benchmark on top of the Phase 2 dual-encoder.

## What this phase does

1. Loads the best dual-encoder checkpoint from `data/artifacts/dual_encoder_best/`.
2. Generates **catalog embeddings with batch inference**.
3. Saves embeddings as a **float32 numpy array**.
4. Builds a FAISS exact baseline (`IndexFlatIP`).
5. Benchmarks approximate index configurations (`IndexIVFFlat` by default, optional HNSW).
6. Reports:
   - p50 / p95 / mean latency
   - ANN recall and recall drop vs exact
   - whether phase targets are met (`p95 < 50ms`, `recall_drop < 2%`).

## Run

```bash
python scripts/06_faiss_retrieval_benchmark.py --config configs/default.yaml
```

## Outputs

- `data/artifacts/phase3_catalog_embeddings.npy`
- `data/logs/phase3_faiss_profile_report.json`

## Tuning guidance

### IVF (`ann_index_type: ivf_flat`)

- Increase `ivf_nlist_options` to improve selectivity (often improves speed at larger scales).
- Increase `ivf_nprobe` to improve recall (typically increases latency).

### HNSW (`ann_index_type: hnsw`)

- Increase `hnsw_m` for richer graph connectivity.
- Increase `hnsw_ef_search` for better recall at higher latency.

## Batched query retrieval

Batched query retrieval is implemented with `FaissRetriever.query_batch`, and all profiling is done in batch mode via `benchmark_retriever`.
