# Phase 5 — Two-Stage Reranking Model

Phase 5 adds a learned reranker on top of embedding retrieval.

## What this phase does

1. Retrieves top-200 candidates per query from a FAISS index built over dual-encoder catalog embeddings.
2. Builds a candidate-level feature matrix with:
   - `similarity_score` (FAISS inner product)
   - `category_match_binary` (candidate category equals query product category)
   - `historical_global_ctr` (simulated CTR from historical ratings/labels)
   - `user_category_affinity` (historical click propensity for user/category)
   - `product_popularity` (log-normalized interaction count)
3. Trains a shallow MLP (`MLPClassifier`) to predict `p(click | query, user, product)`.
4. Re-ranks candidates by predicted probability.
5. Evaluates reranking against pure embedding ranking and reports:
   - NDCG@10
   - MRR
   - % lift on test set
6. Logs permutation-based feature importance.

## Run

```bash
python scripts/08_two_stage_reranking.py --config configs/default.yaml
```

## Outputs

- `data/logs/phase5_two_stage_reranker_report.json`

The report includes:

- validation/test baseline vs reranked metrics
- ranking lift beyond embeddings
- feature importance analysis

## Config knobs (`configs/default.yaml`)

```yaml
phase5:
  batch_size: 128
  candidate_k: 200
  eval_k: 10
  hidden_dim: 32
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  max_iter: 30
  ctr_smoothing: 10.0
  user_affinity_smoothing: 5.0
  catalog_embeddings_filename: phase3_catalog_embeddings.npy
```
