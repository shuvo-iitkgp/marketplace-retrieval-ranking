# Action: Problem Framing and Metric Definition

## Define business framing:

- Objective: Improve product discovery and engagement in a marketplace setting.
- Translate Arts dataset into marketplace proxy: users = retailers, products = SKUs, purchases = implicit clicks.

## Define evaluation metrics:

- Offline: Recall@10, NDCG@10, MRR
- Personalization: HitRate@K for repeat users
- System: p95 latency, index memory footprint

## Create time-based split:

- Train: oldest 70%
- Validation: next 15%
- Test: most recent 15%
- Ensure no future leakage.

## Define implicit positive:

- rating ≥ 4 → positive
- rating ≤ 2 → negative

## Write metric module:

- vectorized NDCG@K
- batched Recall@K
- ensure reproducibility with seed control

## Deliverable:

- Formal evaluation harness before model training.
- Baseline metric values logged and versioned.
