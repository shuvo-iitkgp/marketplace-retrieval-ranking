# Phase 7 — Logging and A/B Test Readiness

Phase 7 prepares the system for production experimentation with deterministic traffic split, impression logging, and offline A/B simulation.

## What this phase does

1. Uses a flat impression schema for each served request:
   - `request_id`
   - `user_id`
   - `query_text`
   - `candidate_ids`
   - `predicted_scores`
   - `timestamp`
   - `experiment_bucket`
2. Uses deterministic split with two buckets (`assign_bucket(user_id, 2)`), equivalent to `hash(user_id) % 2` with a stable hash implementation.
3. Runs an offline A/B simulation:
   - **Control**: TF-IDF-like baseline ranker
   - **Treatment**: two-stage-like ranker with stronger top-rank placement
4. Enforces guardrails:
   - null result rate alert
   - p95 latency threshold alert
5. Produces an experiment hypothesis statement suitable for a production A/B launch checklist.

## Run

```bash
python scripts/10_offline_ab_simulation.py --config configs/default.yaml
```

## Outputs

- `data/logs/phase7_offline_ab_report.json`
- `data/logs/phase7_control_impressions.jsonl`
- `data/logs/phase7_treatment_impressions.jsonl`

## Notebook

Open:

- `notebooks/phase7_ab_evaluation.ipynb`

The notebook summarizes:

- control vs treatment metrics from simulation output
- guardrail status
- hypothesis statement for production experiment readiness
