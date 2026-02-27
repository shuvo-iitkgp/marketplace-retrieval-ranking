"""
scripts/02_eval_smoke_test.py
──────────────────────────────
Phase 0 — Evaluation harness smoke test with a popularity baseline.

This script validates the evaluation harness end-to-end BEFORE any
model training.  It uses a trivial "most-popular-items" retrieval
function as a sanity-check baseline.

Why this matters
----------------
* Confirms the harness, metrics, and data splits all work correctly.
* Establishes a known-bad lower bound to beat with real models.
* Surfaces data issues (empty eval sets, leakage, degenerate metrics).

Baseline: "Most Popular Items"
    For every user, retrieve the globally most interacted-with products
    (by training set positive count), sorted descending.
    This is equivalent to a non-personalised popularity recommender.

Run:
    python scripts/02_eval_smoke_test.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── ensure src/ is importable ────────────────────────────────────────────── #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_search.common.config import load_config
from marketplace_search.eval.harness import EvalHarness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_smoke_test")


# ─────────────────────────────────────────────────────────────────────────── #
# Popularity baseline                                                          #
# ─────────────────────────────────────────────────────────────────────────── #


class PopularityBaseline:
    """
    Non-personalised retrieval: return globally popular products.

    For every user query we return the same ranked list — the products
    most frequently interacted with positively in training.
    """

    def __init__(self, train_df: pd.DataFrame, k: int = 200) -> None:
        # Rank products by number of positive training interactions
        popular = (
            train_df[train_df["label"] == 1]
            .groupby("product_id")
            .size()
            .sort_values(ascending=False)
        )
        self.popular_ids: list[str] = popular.index.tolist()[:k]
        logger.info(
            "PopularityBaseline built with top-%d products from training", len(self.popular_ids)
        )

    def retrieve(self, user_id: str) -> list[str]:
        """Ignores user_id — returns the same popular list for everyone."""
        return self.popular_ids


# ─────────────────────────────────────────────────────────────────────────── #
# Main                                                                         #
# ─────────────────────────────────────────────────────────────────────────── #


def main(config_path: str) -> None:
    cfg = load_config(config_path, project_root=PROJECT_ROOT)
    processed_dir = PROJECT_ROOT / cfg.paths.processed_dir

    # ── Load splits ────────────────────────────────────────────────────── #
    logger.info("Loading processed splits from %s", processed_dir)
    train_df = pd.read_csv(processed_dir / "train.csv.gz")
    val_df = pd.read_csv(processed_dir / "val.csv.gz")
    test_df = pd.read_csv(processed_dir / "test.csv.gz")

    logger.info(
        "Loaded splits — train: %d, val: %d, test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # ── Build baseline ─────────────────────────────────────────────────── #
    baseline = PopularityBaseline(train_df, k=200)

    # ── Evaluate on validation split ───────────────────────────────────── #
    logger.info("\n--- Evaluating on VALIDATION split ---")
    val_harness = EvalHarness(cfg, eval_df=val_df, train_df=train_df, split_name="val")
    val_metrics = val_harness.run(
        retrieve_fn=baseline.retrieve,
        k=200,
        measure_latency_flag=True,
        latency_sample=100,
    )
    val_harness.print_summary(val_metrics)
    val_harness.save_results(
        val_metrics,
        model_name="popularity_baseline",
        extra_meta={"description": "Non-personalised popularity baseline for smoke test"},
    )

    # ── Evaluate on test split ─────────────────────────────────────────── #
    logger.info("\n--- Evaluating on TEST split ---")
    test_harness = EvalHarness(cfg, eval_df=test_df, train_df=train_df, split_name="test")
    test_metrics = test_harness.run(
        retrieve_fn=baseline.retrieve,
        k=200,
        measure_latency_flag=True,
        latency_sample=100,
    )
    test_harness.print_summary(test_metrics)
    test_harness.save_results(
        test_metrics,
        model_name="popularity_baseline",
        extra_meta={"description": "Non-personalised popularity baseline for smoke test"},
    )

    logger.info("Smoke test complete. Baseline metrics logged to %s", cfg.paths.logs_dir)
    logger.info(
        "Key baseline → recall@10: %.4f | ndcg@10: %.4f | mrr: %.4f",
        test_metrics.get("recall@10", 0),
        test_metrics.get("ndcg@10", 0),
        test_metrics.get("mrr", 0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval harness smoke test (Phase 0).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)
