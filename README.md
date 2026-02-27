# Marketplace Search & Personalization Engine — Phase 0

## Problem Framing

This project builds a **production-minded marketplace search and personalization prototype**
using Amazon Arts & Crafts reviews as an implicit feedback proxy for a real marketplace.

### Domain Mapping

| Real Marketplace | Arts Dataset Proxy |
|---|---|
| Retailer / Shopper | `review/userId` |
| SKU / Product | `product/productId` (ASIN) |
| Purchase / Click | Review with `score ≥ 4` |
| Product Browse | Review with `score 3` (neutral) |
| Product Return / Complaint | Review with `score ≤ 2` |
| Product Title | `product/title` → synthetic query |
| Brand | `brands_txt.gz` lookup |

### Objective

> Improve product discovery and engagement: given a user's history and a search query,
> surface the most relevant SKUs in the top-K results.

---

## Phase 0 Deliverables

1. **Data ingestion** — parse the raw text format into structured Parquet
2. **Time-based splits** — train/val/test with strict temporal ordering (no future leakage)
3. **Implicit label rules** — positive / negative / neutral interaction labeling
4. **Evaluation harness** — Recall@K, NDCG@K, MRR, HitRate@K, latency/memory utilities
5. **Baseline logging** — versioned metric snapshots in `data/logs/`

---

## Evaluation Metrics

### Offline Retrieval
| Metric | Definition | Target |
|---|---|---|
| Recall@10 | Fraction of positives in top-10 | ≥ 0.30 (baseline) |
| NDCG@10 | Discounted gain at rank 10 | ≥ 0.20 (baseline) |
| MRR | Mean reciprocal rank of first hit | ≥ 0.15 (baseline) |

### Personalization
| Metric | Definition |
|---|---|
| HitRate@K | % of repeat users with ≥1 future positive in top-K |

### System
| Metric | Definition |
|---|---|
| p95 Latency | 95th percentile query latency (ms) |
| Index Memory | FAISS index footprint (MB) |

---

## Data Splits

```
Timeline ──────────────────────────────────────────────────▶
         [────── TRAIN 70% ──────][─ VAL 15% ─][─ TEST 15% ─]
```

Split is performed on **global interaction timestamp** (not per-user), ensuring:
- No user's future interactions appear in training
- Temporal ordering is strictly preserved

---

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Build dataset artifacts
python scripts/01_build_dataset.py --config configs/default.yaml

# Run evaluation harness smoke test
python scripts/02_eval_smoke_test.py --config configs/default.yaml
```

---

## Project Layout

```
src/marketplace_search/
├── data/
│   ├── ingest.py          # Parse raw Arts text format
│   ├── splits.py          # Time-based train/val/test split
│   └── labels.py          # Implicit positive/negative rules
├── eval/
│   ├── metrics.py         # Recall@K, NDCG@K, MRR, HitRate@K
│   ├── harness.py         # Evaluation orchestration
│   └── system_metrics.py  # Latency + memory utilities
└── common/
    ├── hashing.py         # Deterministic experiment bucketing
    ├── logging_schema.py  # Impression log schema
    └── config.py          # Config loading
configs/
├── default.yaml
scripts/
├── 01_build_dataset.py
└── 02_eval_smoke_test.py
data/
├── raw/                   # Arts_txt.gz, brands_txt.gz
├── processed/             # Parquet outputs
└── logs/                  # Versioned metric snapshots
```
