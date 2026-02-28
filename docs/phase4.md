# Phase 4 — Personalization Layer

Phase 4 adds user-aware retrieval on top of the Phase 2/3 embedding stack.

## What this phase does

1. Builds user embeddings from past positive interactions by aggregating purchased product embeddings.
2. Applies exponential time decay to each historical interaction:
   - `weight = exp(-lambda * age)`
3. L2-normalizes user vectors.
4. Builds final personalized query vectors:
   - `v_final = alpha * v_query + (1 - alpha) * v_user`
5. Tunes `alpha` on validation NDCG@10 using a configurable alpha grid.
6. Handles cold-start users with category priors:
   - Category centroids are computed from catalog embeddings.
   - For unseen users, the closest category prior to query embedding is used.
7. Evaluates personalized vs non-personalized retrieval and reports segmentation:
   - overall NDCG@10
   - repeat-user sessions
   - cold-start sessions

## Run

```bash
python scripts/07_personalization_eval.py --config configs/default.yaml
```

## Outputs

- `data/logs/phase4_personalization_report.json`

The report includes:

- tuned alpha and validation score
- baseline vs personalized NDCG@10
- `% lift` for overall, repeat-user, and cold-start segments

## Config knobs (`configs/default.yaml`)

```yaml
phase4:
  batch_size: 128
  ndcg_k: 10
  decay_lambda: 0.01
  alpha_grid: [0.2, 0.4, 0.6, 0.8, 1.0]
  catalog_embeddings_filename: phase3_catalog_embeddings.npy
```

### Notes

- If `catalog_embeddings_filename` exists in artifacts, it is reused.
- Otherwise catalog embeddings are regenerated from the best Phase 2 dual-encoder checkpoint.
