Marketplace Search & Personalization Engine (Arts Dataset Proxy)
===============================================================

Goal
----
Build a production-minded marketplace search and personalization prototype using:
- arts.txt.gz reviews as implicit feedback (rating + timestamp)
- synthetic queries derived from product metadata
- retrieval (TF-IDF baseline -> dual-encoder -> FAISS ANN)
- personalization (user embedding blend)
- two-stage reranking (LightGBM or shallow MLP)
- explore-exploit (epsilon-greedy + diversity penalty)
- offline evaluation + replay simulation
- logging schema and deterministic experiment bucketing

Dataset Assumptions
-------------------
Input files (not committed to git):
- data/raw/arts.txt.gz
  Streaming JSON lines (one review per line).
  Required fields expected: reviewerID, asin, overall (rating), unixReviewTime (timestamp)
- data/raw/meta_arts.json.gz (or similar)
  Product metadata. Required fields expected: asin, title, description (optional), category (optional), brand (optional)

If your metadata file schema differs, update:
- src/marketplace_search/data/ingest_meta.py

Environment
-----------
Python: 3.10+
Recommended: uv or poetry

Setup (uv example)
------------------
1) Create env and install:
   uv venv
   uv pip install -e ".[dev]"

2) Place raw data:
   mkdir -p data/raw
   cp /path/to/arts.txt.gz data/raw/
   cp /path/to/meta_arts.json.gz data/raw/

3) Run full pipeline (baseline -> full system):
   make all

Make Targets
------------
make format        Run ruff/black
make test          Run unit tests
make build-data    Build catalog, interactions, queries, time splits
make tfidf         Train/eval TF-IDF baseline
make dual          Train dual encoder, eval
make faiss         Embed catalog, build FAISS, benchmark
make rerank        Train reranker, eval
make replay        Run offline replay eval (explore-exploit)
make report        Generate consolidated report into data/logs/
make all           Runs build-data + tfidf + dual + faiss + rerank + replay + report

Core Design Choices
-------------------
Implicit Feedback Labels:
- positive: rating >= 4
- negative: rating <= 2
- neutral: ignore by default

Splits:
- time-based global split on timestamp
  Train: oldest 70%
  Valid: next 15%
  Test: most recent 15%
Leakage checks enforced in src/marketplace_search/data/splits.py

Synthetic Queries:
- Default: product title as pseudo-query
- Optional: n-gram extraction from titles for more realistic query distribution
Config in: configs/default.yaml

Retrieval:
1) TF-IDF baseline (sklearn) for strong, cheap baseline
2) Dual-encoder retrieval (MiniLM or DistilBERT) trained on (query, purchased_product)
   Hard negatives:
   - same category but not interacted
   - in-batch negatives
3) FAISS ANN:
   - IndexFlatIP for exact baseline
   - IVF or HNSW for latency/recall tradeoff
Benchmarks logged with p50/p95 latency and recall deltas

Personalization:
- user embedding = time-decayed average of purchased product embeddings
- blended query = alpha*query + (1-alpha)*user
- cold start uses category priors

Reranking:
- retrieve top 200 from FAISS
- build features:
  similarity, category match, product popularity, user-category affinity, simulated global CTR
- train LightGBM to predict positive interaction
- evaluate NDCG@10, MRR, Recall@10

Explore-Exploit:
- epsilon-greedy swaps in underexposed items
- diversity penalty discourages category concentration
- replay evaluator uses historical positives as proxy clicks

Logging / Experiment Readiness:
- impression schema:
  {
    request_id,
    user_id,
    query_text,
    candidate_ids,
    predicted_scores,
    timestamp,
    experiment_bucket
  }
- deterministic split:
  bucket = hash(user_id) % 2
Implementation: src/marketplace_search/common/hashing.py

Repo Layout
-----------
- src/marketplace_search/data        ingestion, cleaning, splits, synthetic queries
- src/marketplace_search/eval        metrics + evaluation harness + replay
- src/marketplace_search/retrieval   TF-IDF, dual-encoder, FAISS
- src/marketplace_search/personalization user profiling and blending
- src/marketplace_search/ranking     reranker training/inference
- src/marketplace_search/explore     exploration and diversity
- src/marketplace_search/serving     end-to-end pipeline + logging + fallback + cache
- scripts/                           runnable pipeline steps

How To Run (Step-by-step)
-------------------------
1) Build dataset artifacts:
   python scripts/01_build_dataset.py --config configs/default.yaml

2) TF-IDF baseline:
   python scripts/02_train_tfidf.py --config configs/experiments/tfidf.yaml
   python scripts/03_eval_tfidf.py  --config configs/experiments/tfidf.yaml

3) Dual encoder:
   python scripts/04_train_dual_encoder.py --config configs/experiments/dual_encoder.yaml

4) FAISS:
   python scripts/05_embed_catalog.py --config configs/experiments/faiss.yaml
   python scripts/06_build_faiss.py   --config configs/experiments/faiss.yaml
   python scripts/07_bench_faiss.py   --config configs/experiments/faiss.yaml

5) Personalization + reranking:
   python scripts/08_personalize_eval.py --config configs/experiments/personalize.yaml
   python scripts/09_train_reranker.py   --config configs/experiments/reranker.yaml
   python scripts/10_eval_full_system.py --config configs/experiments/reranker.yaml

6) Explore-exploit replay:
   python scripts/11_run_replay_eval.py --config configs/experiments/explore_exploit.yaml

Outputs
-------
- data/processed/    splits, query sets, mappings
- data/artifacts/    embeddings, faiss index, trained models
- data/logs/         metrics json, latency bench, per-query results, impression logs

Non-goals
---------
- microservices
- real online traffic
- distributed index serving

This repo is built to demonstrate:
- strong baselines
- reproducible offline evaluation
- scalable retrieval
- personalization
- two-stage ranking maturity
- experiment readiness mindset
