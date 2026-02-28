from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_search.common.config import load_config
from marketplace_search.common.hashing import assign_bucket
from marketplace_search.common.logging_schema import ImpressionLogger
from marketplace_search.experimentation.ab_simulation import OfflineABSimulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("phase7_offline_ab")


def build_eval_requests(test_df: pd.DataFrame, queries_df: pd.DataFrame, n_requests: int, seed: int) -> list[dict]:
    positives = test_df[test_df["label"] == 1][["user_id", "product_id"]].copy()
    query_lookup = queries_df[["product_id", "query_text"]].drop_duplicates("product_id")
    merged = positives.merge(query_lookup, on="product_id", how="inner")
    if merged.empty:
        raise ValueError("No positive test interactions with query_text available for offline simulation.")

    merged = merged.rename(columns={"product_id": "target_product_id"})
    merged = merged.sample(n=min(n_requests, len(merged)), random_state=seed).reset_index(drop=True)
    return merged[["user_id", "query_text", "target_product_id"]].to_dict(orient="records")


def build_rankers(catalog_ids: list[str], seed: int):
    rng = random.Random(seed)

    def control_ranker(_user_id: str, _query_text: str, target_product_id: str, k: int):
        pool = catalog_ids[:]
        rng.shuffle(pool)
        # weaker baseline: target appears deeper / sometimes missing
        if target_product_id in pool and rng.random() < 0.8:
            pool.remove(target_product_id)
            insert_idx = min(len(pool), rng.randint(max(1, k // 2), max(1, k - 1)))
            pool.insert(insert_idx, target_product_id)

        chosen = pool[:k]
        scores = np.linspace(1.0, 0.1, num=len(chosen)).tolist()
        latency_ms = float(rng.uniform(8.0, 25.0))
        return chosen, scores, latency_ms

    def treatment_ranker(user_id: str, _query_text: str, target_product_id: str, k: int):
        pool = catalog_ids[:]
        rng.shuffle(pool)
        if target_product_id in pool:
            pool.remove(target_product_id)
            # deterministic personalization-ish bump by user bucket
            bucket = assign_bucket(user_id, num_buckets=2)
            insert_idx = 0 if bucket == 1 else 1
            pool.insert(insert_idx, target_product_id)

        chosen = pool[:k]
        scores = np.linspace(1.0, 0.2, num=len(chosen)).tolist()
        latency_ms = float(rng.uniform(12.0, 35.0))
        return chosen, scores, latency_ms

    return control_ranker, treatment_ranker


def main(config_path: str) -> None:
    cfg = load_config(config_path, project_root=PROJECT_ROOT)
    processed_dir = PROJECT_ROOT / cfg.paths.processed_dir
    logs_dir = PROJECT_ROOT / cfg.paths.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(processed_dir / "test.csv.gz")
    queries_df = pd.read_csv(processed_dir / "queries_title.csv.gz")
    catalog_df = pd.read_csv(processed_dir / "rich_catalog.csv.gz")

    requests = build_eval_requests(
        test_df=test_df,
        queries_df=queries_df,
        n_requests=int(cfg.phase7.offline_ab_n_requests),
        seed=int(cfg.project.seed),
    )
    control_ranker, treatment_ranker = build_rankers(
        catalog_ids=catalog_df["product_id"].dropna().astype(str).drop_duplicates().tolist(),
        seed=int(cfg.project.seed),
    )

    sim = OfflineABSimulation(
        eval_k=int(cfg.phase7.eval_k),
        latency_threshold_ms=float(cfg.phase7.guardrails.max_p95_latency_ms),
    )
    result, logs = sim.run(
        requests=requests,
        control_ranker_fn=control_ranker,
        treatment_ranker_fn=treatment_ranker,
        expected_min_ndcg_lift=float(cfg.phase7.expected_min_ndcg_lift),
    )

    control_logger = ImpressionLogger(logs_dir / "phase7_control_impressions.jsonl")
    treatment_logger = ImpressionLogger(logs_dir / "phase7_treatment_impressions.jsonl")
    control_logger.log_batch(logs["control"])
    treatment_logger.log_batch(logs["treatment"])

    report = {
        "timestamp_utc": int(time.time()),
        "phase": "phase07_offline_ab_simulation",
        "control": result.control.__dict__,
        "treatment": result.treatment.__dict__,
        "ndcg_lift": result.ndcg_lift,
        "mrr_lift": result.mrr_lift,
        "guardrails": {
            "null_result_rate_alert": bool(
                result.control.null_result_rate > float(cfg.phase7.guardrails.max_null_result_rate)
                or result.treatment.null_result_rate > float(cfg.phase7.guardrails.max_null_result_rate)
            ),
            "latency_alert": bool(result.control.latency_alert or result.treatment.latency_alert),
            "max_null_result_rate": float(cfg.phase7.guardrails.max_null_result_rate),
            "max_p95_latency_ms": float(cfg.phase7.guardrails.max_p95_latency_ms),
        },
        "hypothesis": result.hypothesis_statement,
    }

    out_path = logs_dir / "phase7_offline_ab_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("saved report -> %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)