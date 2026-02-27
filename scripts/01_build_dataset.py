"""
scripts/01_build_dataset.py
────────────────────────────
Phase 0 — Build all dataset artifacts:

1. Ingest raw Arts_txt.gz reviews + brands_txt.gz
2. Assign implicit labels (positive / negative)
3. Filter by minimum activity
4. Build product catalog
5. Apply time-based train/val/test split
6. Validate no future leakage
7. Save processed splits + catalog to data/processed/ as Parquet
8. Log dataset statistics to data/logs/

Run:
    python scripts/01_build_dataset.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# ── ensure src/ is importable ────────────────────────────────────────────── #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_search.common.config import load_config
from marketplace_search.data.ingest import build_catalog, load_brands, load_reviews
from marketplace_search.data.labels import assign_labels, filter_active_entities
from marketplace_search.data.splits import (
    check_no_future_leakage,
    time_split,
    get_repeat_users,
)

# ── logging setup ────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_dataset")


# ─────────────────────────────────────────────────────────────────────────── #
# Main                                                                         #
# ─────────────────────────────────────────────────────────────────────────── #


def main(config_path: str) -> None:
    cfg = load_config(config_path, project_root=PROJECT_ROOT)
    start_time = time.time()

    # ── Paths ──────────────────────────────────────────────────────────── #
    raw_reviews_path = PROJECT_ROOT / cfg.paths.raw_reviews
    raw_brands_path = PROJECT_ROOT / cfg.paths.raw_brands
    processed_dir = PROJECT_ROOT / cfg.paths.processed_dir
    logs_dir = PROJECT_ROOT / cfg.paths.logs_dir

    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Ingest ──────────────────────────────────────────────────────── #
    logger.info("Step 1/7: Ingesting reviews from %s", raw_reviews_path)
    reviews_raw = load_reviews(raw_reviews_path)

    logger.info("Step 1b: Ingesting brands from %s", raw_brands_path)
    brands = load_brands(raw_brands_path)

    # ── 2. Label assignment ────────────────────────────────────────────── #
    logger.info("Step 2/7: Assigning implicit labels")
    data_cfg = cfg.data
    interactions = assign_labels(
        reviews_raw,
        positive_threshold=data_cfg.positive_threshold,
        negative_threshold=data_cfg.negative_threshold,
        keep_neutral=False,
    )

    # ── 3. Activity filter ─────────────────────────────────────────────── #
    logger.info("Step 3/7: Filtering by minimum activity")
    interactions = filter_active_entities(
        interactions,
        min_user_interactions=data_cfg.min_user_interactions,
        min_product_reviews=data_cfg.min_product_reviews,
    )

    # ── 4. Product catalog ─────────────────────────────────────────────── #
    logger.info("Step 4/7: Building product catalog")
    catalog = build_catalog(interactions, brands)
    catalog_path = processed_dir / "catalog.csv.gz"
    catalog.to_csv(catalog_path, index=False)
    logger.info("Catalog saved → %s", catalog_path)

    # ── 5. Time-based split ────────────────────────────────────────────── #
    logger.info("Step 5/7: Applying time-based split")
    splits = time_split(
        interactions,
        train_frac=data_cfg.train_frac,
        val_frac=data_cfg.val_frac,
        test_frac=data_cfg.test_frac,
    )

    # ── 6. Leakage check ───────────────────────────────────────────────── #
    logger.info("Step 6/7: Checking for future leakage")
    check_no_future_leakage(splits)

    # ── 7. Save splits ─────────────────────────────────────────────────── #
    logger.info("Step 7/7: Saving processed splits")
    splits.train.to_csv(processed_dir / "train.csv.gz", index=False)
    splits.val.to_csv(processed_dir / "val.csv.gz", index=False)
    splits.test.to_csv(processed_dir / "test.csv.gz", index=False)
    logger.info("Splits saved → %s", processed_dir)

    # ── Dataset statistics ─────────────────────────────────────────────── #
    eval_cfg = cfg.evaluation
    repeat_users_val = get_repeat_users(
        splits.train, splits.val,
        min_train_interactions=eval_cfg.repeat_user_min_interactions,
    )
    repeat_users_test = get_repeat_users(
        splits.train, splits.test,
        min_train_interactions=eval_cfg.repeat_user_min_interactions,
    )

    stats = {
        "timestamp_utc": int(time.time()),
        "raw_reviews": len(reviews_raw),
        "after_labeling": len(interactions),
        "n_users": interactions["user_id"].nunique(),
        "n_products": interactions["product_id"].nunique(),
        "n_brands": brands["product_id"].nunique(),
        "label_distribution": interactions["label"].value_counts().to_dict(),
        "splits": {
            "train": {
                "n_interactions": len(splits.train),
                "n_users": splits.train["user_id"].nunique(),
                "n_products": splits.train["product_id"].nunique(),
            },
            "val": {
                "n_interactions": len(splits.val),
                "n_users": splits.val["user_id"].nunique(),
                "n_products": splits.val["product_id"].nunique(),
                "repeat_users": len(repeat_users_val),
            },
            "test": {
                "n_interactions": len(splits.test),
                "n_users": splits.test["user_id"].nunique(),
                "n_products": splits.test["product_id"].nunique(),
                "repeat_users": len(repeat_users_test),
            },
        },
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    stats_path = logs_dir / "dataset_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Dataset statistics saved → %s", stats_path)

    # ── Summary ────────────────────────────────────────────────────────── #
    logger.info("\n" + "=" * 60)
    logger.info("  Dataset Build Complete")
    logger.info("=" * 60)
    logger.info("  Raw reviews:        %d", stats["raw_reviews"])
    logger.info("  After label filter: %d", stats["after_labeling"])
    logger.info("  Unique users:       %d", stats["n_users"])
    logger.info("  Unique products:    %d", stats["n_products"])
    logger.info("  Train:  %d interactions", stats["splits"]["train"]["n_interactions"])
    logger.info("  Val:    %d interactions", stats["splits"]["val"]["n_interactions"])
    logger.info("  Test:   %d interactions", stats["splits"]["test"]["n_interactions"])
    logger.info("  Elapsed: %.1fs", stats["elapsed_seconds"])
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Phase 0 dataset artifacts.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (relative to project root).",
    )
    args = parser.parse_args()
    main(args.config)
