"""
scripts/04_train_eval_tfidf.py
───────────────────────────────
Phase 1 — Train TF-IDF retrieval model and run full evaluation.

Two evaluation modes are run and compared:

Mode A — Synthetic Query Eval (primary for retrieval quality)
    Query: product title / n-gram → retrieve → check if GT product is in top-K
    This directly measures how well TF-IDF can find a product given a text query.

Mode B — User History Eval (continuity with Phase 0 harness)
    Query: user's training purchase history → retrieve → check against test positives
    This measures personalisation quality as a sanity check.

Run:
    python scripts/04_train_eval_tfidf.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_search.common.config import load_config
from marketplace_search.eval.harness import EvalHarness
from marketplace_search.eval.query_harness import QueryEvalHarness, compare_metrics
from marketplace_search.retrieval.tfidf import TFIDFRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_eval_tfidf")


def main(config_path: str) -> None:
    cfg = load_config(config_path, project_root=PROJECT_ROOT)
    t0 = time.time()

    processed_dir = PROJECT_ROOT / cfg.paths.processed_dir
    artifacts_dir = PROJECT_ROOT / cfg.paths.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────── #
    logger.info("Loading processed data...")
    train_df = pd.read_csv(processed_dir / "train.csv.gz")
    val_df = pd.read_csv(processed_dir / "val.csv.gz")
    test_df = pd.read_csv(processed_dir / "test.csv.gz")
    catalog = pd.read_csv(processed_dir / "rich_catalog.csv.gz")
    queries_title = pd.read_csv(processed_dir / "queries_title.csv.gz")
    queries_ngram = pd.read_csv(processed_dir / "queries_ngram.csv.gz")

    logger.info(
        "Data loaded — train: %d, val: %d, test: %d, catalog: %d products",
        len(train_df), len(val_df), len(test_df), len(catalog),
    )
    logger.info(
        "Queries — title: %d, ngram: %d",
        len(queries_title), len(queries_ngram),
    )

    # ── 2. Train TF-IDF ────────────────────────────────────────────── #
    tfidf_cfg = cfg.tfidf
    logger.info("Fitting TF-IDF retriever...")

    retriever = TFIDFRetriever(
        max_features=tfidf_cfg.max_features,
        ngram_range=tuple(tfidf_cfg.ngram_range),
        min_df=tfidf_cfg.min_df,
        sublinear_tf=tfidf_cfg.sublinear_tf,
    )
    retriever.fit(catalog, text_col="cleaned_text")

    logger.info("TF-IDF: %s", retriever)

    # Save model
    model_path = artifacts_dir / "tfidf_retriever.pkl"
    retriever.save(model_path)

    # System metrics
    system_info = {
        "vocab_size": retriever.vocab_size,
        "n_products": retriever.n_products,
        "index_memory_mb": retriever.index_memory_mb(),
    }
    logger.info(
        "TF-IDF index — vocab: %d, products: %d, memory: %.2f MB",
        system_info["vocab_size"],
        system_info["n_products"],
        system_info["index_memory_mb"],
    )

    # ── 3a. Mode A: Synthetic query eval (title strategy) ─────────── #
    logger.info("\n=== MODE A: Synthetic Query Eval (title strategy) ===")

    # Filter queries to only products in the catalog
    catalog_pids = set(catalog["product_id"].tolist())
    q_title_filtered = queries_title[
        queries_title["product_id"].isin(catalog_pids)
    ].reset_index(drop=True)

    title_harness = QueryEvalHarness(cfg, q_title_filtered, split_name="title_queries")
    title_metrics = title_harness.run(
        retrieve_fn=retriever.query_retrieve,
        k=tfidf_cfg.retrieval_k,
        measure_latency_flag=True,
        latency_sample=200,
        save_per_query=True,
    )
    title_harness.print_summary(title_metrics, title="TF-IDF — Title Query Eval")
    title_harness.save_results(
        title_metrics,
        model_name="tfidf_title_queries",
        extra_meta={**system_info, "strategy": "title"},
    )

    # ── 3b. Mode A: Synthetic query eval (ngram strategy) ─────────── #
    logger.info("\n=== MODE A: Synthetic Query Eval (ngram strategy) ===")

    q_ngram_filtered = queries_ngram[
        queries_ngram["product_id"].isin(catalog_pids)
    ].reset_index(drop=True)

    ngram_harness = QueryEvalHarness(cfg, q_ngram_filtered, split_name="ngram_queries")
    ngram_metrics = ngram_harness.run(
        retrieve_fn=retriever.query_retrieve,
        k=tfidf_cfg.retrieval_k,
        measure_latency_flag=True,
        latency_sample=200,
        save_per_query=True,
    )
    ngram_harness.print_summary(ngram_metrics, title="TF-IDF — Ngram Query Eval")
    ngram_harness.save_results(
        ngram_metrics,
        model_name="tfidf_ngram_queries",
        extra_meta={**system_info, "strategy": "ngram"},
    )

    # ── 4. Mode B: User history eval (Phase 0 harness) ────────────── #
    logger.info("\n=== MODE B: User History Eval (test split) ===")

    user_retrieve_fn = retriever.build_user_query_fn(
        train_df=train_df,
        catalog=catalog,
        k=tfidf_cfg.retrieval_k,
    )

    test_harness = EvalHarness(cfg, eval_df=test_df, train_df=train_df, split_name="test")
    user_metrics = test_harness.run(
        retrieve_fn=user_retrieve_fn,
        k=tfidf_cfg.retrieval_k,
        measure_latency_flag=True,
        latency_sample=200,
    )
    test_harness.print_summary(user_metrics)
    test_harness.save_results(
        user_metrics,
        model_name="tfidf_user_history",
        extra_meta={**system_info, "mode": "user_history"},
    )

    # ── 5. Comparison table ────────────────────────────────────────── #
    all_results = {
        "tfidf_title_queries": title_metrics,
        "tfidf_ngram_queries": ngram_metrics,
        "tfidf_user_history": user_metrics,
    }
    comparison = compare_metrics(all_results)
    logger.info("\nComparison table:\n%s", comparison.to_string())

    # ── 6. Final report ────────────────────────────────────────────── #
    report = {
        "timestamp_utc": int(time.time()),
        "model": "tfidf_baseline",
        "system": system_info,
        "title_query_eval": {k: v for k, v in title_metrics.items() if isinstance(v, float)},
        "ngram_query_eval": {k: v for k, v in ngram_metrics.items() if isinstance(v, float)},
        "user_history_eval": {k: v for k, v in user_metrics.items() if isinstance(v, float)},
        "elapsed_total_seconds": round(time.time() - t0, 2),
    }
    report_path = PROJECT_ROOT / cfg.paths.logs_dir / "tfidf_phase1_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("  Phase 1 TF-IDF Complete")
    logger.info("=" * 60)
    logger.info("  [Title Queries]  NDCG@10: %.4f  Recall@10: %.4f  MRR: %.4f",
                title_metrics.get("ndcg@10", 0),
                title_metrics.get("recall@10", 0),
                title_metrics.get("mrr", 0))
    logger.info("  [Ngram Queries]  NDCG@10: %.4f  Recall@10: %.4f  MRR: %.4f",
                ngram_metrics.get("ndcg@10", 0),
                ngram_metrics.get("recall@10", 0),
                ngram_metrics.get("mrr", 0))
    logger.info("  [User History]   NDCG@10: %.4f  Recall@10: %.4f  MRR: %.4f",
                user_metrics.get("ndcg@10", 0),
                user_metrics.get("recall@10", 0),
                user_metrics.get("mrr", 0))
    logger.info("  Index memory:    %.2f MB", system_info["index_memory_mb"])
    logger.info("  Elapsed:         %.1fs", report["elapsed_total_seconds"])
    logger.info("=" * 60)
    logger.info("Report saved → %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
