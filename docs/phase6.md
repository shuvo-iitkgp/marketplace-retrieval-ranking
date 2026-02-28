# Phase 6 — Constrained Explore–Exploit Reranking (Exposure + Diversity)

Phase 6 adds safe exploration to the Phase 5 reranker while preserving relevance.

## What this phase does

1. Maintains an `exposure_store[(query_cluster_id, product_id)]` with decay (`0.99`) and updates only for actually shown items (`top eval_k`).
2. Computes novelty bonus:
   - `novelty_bonus = 1 / sqrt(1 + exposure)`
3. Applies constrained epsilon-greedy reranking:
   - `base_score = reranker_prob` (or normalized similarity)
   - `final_score = base_score + alpha * novelty_bonus`
   - exploration only inside `safe_pool = top_pool_k` by `base_score`
4. Applies intent-aware diversity cap for broad queries:
   - broad if `query_cluster_entropy >= entropy_threshold` OR query has fewer than 3 tokens
   - greedy cap with backfill (`max_per_category=3` for `eval_k=10`)
5. Runs offline parameter sweeps over `(epsilon, alpha, top_pool_k, max_per_category)` and logs tradeoffs.

## Offline metrics reported

- NDCG@10
- MRR
- Coverage@10
- LongTail@10
- CategoryEntropy@10
- Tradeoff curve: `Coverage@10 vs ΔNDCG@10`

## Run

```bash
python scripts/09_constrained_explore_exploit.py --config configs/default.yaml
```

## Outputs

- `data/logs/phase6_constrained_explore_exploit_report.json`

The report contains full sweep metrics plus baseline deltas (`ΔNDCG`, `ΔMRR`) to ensure relevance loss is explicitly bounded.
