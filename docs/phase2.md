# Phase 2 — Dual-Encoder Training Notes

This phase introduces a trainable retrieval model with:

- query/product transformer encoders (shared or separate towers)
- normalized embeddings + dot-product similarity
- cross-entropy over the query-to-candidate similarity matrix
- negatives from both in-batch positives and category-aware hard negatives

## Training/Eval Script

```bash
python scripts/05_train_eval_dual_encoder.py --config configs/default.yaml
```

## Deliverables Produced by the Script

- Best checkpoint selected by **validation NDCG@10** in `data/artifacts/dual_encoder_best/`
- Comparison report in `data/logs/phase2_dual_encoder_report.json` containing:
  - TF-IDF baseline metrics
  - Best dual-encoder metrics
  - Recall@10 lift vs TF-IDF
  - Ablation runs (`full`, `no_hard_negatives`, `separate_towers`)

## Ablation Interpretation Guidance

Use the report to answer:

1. Does hard-negative mining improve Recall@10 and/or NDCG@10?
2. Do separate towers outperform shared weights for the dataset?
3. Is the best model consistently better than TF-IDF on Recall@10?

Record conclusions directly from the JSON report after each run.
