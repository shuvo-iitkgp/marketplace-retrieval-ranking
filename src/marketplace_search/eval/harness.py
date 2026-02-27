"""
src/marketplace_search/eval/harness.py
───────────────────────────────────────
Evaluation harness for the marketplace search system.

This module orchestrates the offline evaluation workflow:

1.  Build per-query (user) relevance sets from the eval split
2.  Run a retrieval function (or pre-computed ranked lists) against each query
3.  Compute all configured metrics
4.  Log results to a versioned JSON file in ``data/logs/``

Designed to be called identically for:
  - TF-IDF baseline
  - Dual-encoder retrieval
  - Full re-ranked system

Usage
-----
    harness = EvalHarness(config, split="test")
    results = harness.run(retrieve_fn=my_model.retrieve)
    harness.save_results(results, model_name="tfidf_baseline")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from marketplace_search.eval.metrics import compute_all_metrics
from marketplace_search.eval.system_metrics import measure_latency

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────── #
# Data preparation helpers                                                     #
# ─────────────────────────────────────────────────────────────────────────── #


def build_query_relevance(
    eval_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "product_id",
    label_col: str = "label",
    positive_label: int = 1,
) -> Tuple[List[str], List[List[str]]]:
    """
    Convert an eval DataFrame into (query_ids, relevant_items_per_query).

    One "query" per user — we predict which products the user will engage
    with positively.

    Parameters
    ----------
    eval_df:
        Evaluation interactions DataFrame.
    user_col, item_col, label_col:
        Column names.
    positive_label:
        Label value considered relevant.

    Returns
    -------
    (query_user_ids, relevant_ids_per_user)
        Both lists are aligned (same index = same user).
    """
    positives = eval_df[eval_df[label_col] == positive_label]
    grouped = (
        positives.groupby(user_col)[item_col]
        .apply(list)
        .reset_index()
    )
    query_ids = grouped[user_col].tolist()
    relevant_ids = grouped[item_col].tolist()
    return query_ids, relevant_ids


# ─────────────────────────────────────────────────────────────────────────── #
# Harness                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #


class EvalHarness:
    """
    Orchestrates offline evaluation for one model / system variant.

    Parameters
    ----------
    config:
        Config object (marketplace_search.common.config.Config).
    eval_df:
        Evaluation split DataFrame (val or test).
    train_df:
        Training split DataFrame (used for user history lookups).
    split_name:
        "val" or "test" — used in log file naming.
    """

    def __init__(
        self,
        config,
        eval_df: pd.DataFrame,
        train_df: pd.DataFrame,
        split_name: str = "test",
    ) -> None:
        self.config = config
        self.eval_df = eval_df
        self.train_df = train_df
        self.split_name = split_name

        # Resolve log dir
        self.log_dir = Path(config.paths.logs_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compute query relevance sets
        self.query_ids, self.relevant_ids = build_query_relevance(eval_df)
        logger.info(
            "EvalHarness ready: %d queries in %s split", len(self.query_ids), split_name
        )

    # ------------------------------------------------------------------ #
    # Core evaluation                                                      #
    # ------------------------------------------------------------------ #

    def run(
        self,
        retrieve_fn: Callable[[str], List[str]],
        k: int = 200,
        measure_latency_flag: bool = True,
        latency_sample: int = 200,
        rng_seed: int = 42,
    ) -> dict:
        """
        Run evaluation of a retrieval function.

        Parameters
        ----------
        retrieve_fn:
            Callable that takes a user_id and returns an ordered list of
            product_ids (most relevant first, up to ``k`` items).
        k:
            Maximum number of items to retrieve per query.
        measure_latency_flag:
            Whether to benchmark latency on a sample of queries.
        latency_sample:
            Number of queries to use for latency benchmarking.
        rng_seed:
            Random seed for reproducibility.

        Returns
        -------
        dict
            All metric values + latency + metadata.
        """
        np.random.seed(rng_seed)

        logger.info("Running evaluation on %d queries...", len(self.query_ids))

        # Collect retrieved lists for each query user
        retrieved_ids: List[List[str]] = []
        for user_id in self.query_ids:
            try:
                results = retrieve_fn(user_id)
                retrieved_ids.append(list(results)[:k])
            except Exception as exc:
                logger.warning("retrieve_fn failed for user %s: %s", user_id, exc)
                retrieved_ids.append([])

        # Compute ranking metrics
        eval_cfg = self.config.evaluation
        metrics = compute_all_metrics(
            relevant_ids=self.relevant_ids,
            retrieved_ids=retrieved_ids,
            recall_ks=eval_cfg.recall_k,
            ndcg_ks=eval_cfg.ndcg_k,
            hit_rate_ks=eval_cfg.hit_rate_k,
            mrr_k=eval_cfg.mrr_k,
            rng_seed=rng_seed,
        )

        # Latency benchmark
        if measure_latency_flag and len(self.query_ids) > 0:
            sample_size = min(latency_sample, len(self.query_ids))
            rng = np.random.default_rng(rng_seed)
            sample_idx = rng.choice(len(self.query_ids), size=sample_size, replace=False)
            sample_queries = [self.query_ids[i] for i in sample_idx]

            latency = measure_latency(
                fn=retrieve_fn,
                queries=sample_queries,
                warmup=min(5, len(sample_queries)),
                percentiles=eval_cfg.latency_percentiles,
            )
            metrics.update({f"latency_{k}": v for k, v in latency.items()})

            sla_ok = latency.get("p95_ms", 0) <= eval_cfg.max_p95_latency_ms
            metrics["latency_sla_pass"] = sla_ok

        metrics["split"] = self.split_name
        metrics["n_queries"] = len(self.query_ids)
        return metrics

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    def save_results(
        self,
        metrics: dict,
        model_name: str,
        extra_meta: Optional[dict] = None,
    ) -> Path:
        """
        Save metric results to a versioned JSON file.

        File name format:
            ``{log_dir}/{model_name}_{split_name}_{timestamp}.json``

        Parameters
        ----------
        metrics:
            Dict of metric values returned by ``run()``.
        model_name:
            Short name for the model/system (e.g. "tfidf_baseline").
        extra_meta:
            Any additional key-value pairs to include in the log.

        Returns
        -------
        Path
            Path to the written log file.
        """
        ts = int(time.time())
        filename = f"{model_name}_{self.split_name}_{ts}.json"
        output_path = self.log_dir / filename

        payload = {
            "model": model_name,
            "split": self.split_name,
            "timestamp_utc": ts,
            "metrics": metrics,
        }
        if extra_meta:
            payload["meta"] = extra_meta

        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)

        logger.info("Saved evaluation results to %s", output_path)
        self._log_summary(metrics)
        return output_path

    # ------------------------------------------------------------------ #
    # Pretty-print                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _log_summary(metrics: dict) -> None:
        logger.info("─" * 50)
        logger.info("Evaluation Summary")
        logger.info("─" * 50)
        for key in sorted(metrics.keys()):
            val = metrics[key]
            if isinstance(val, float):
                logger.info("  %-25s %.4f", key, val)
            else:
                logger.info("  %-25s %s", key, val)
        logger.info("─" * 50)

    def print_summary(self, metrics: dict) -> None:
        """Print metrics to stdout in a readable table."""
        print("\n" + "=" * 52)
        print(f"  Evaluation Results — {self.split_name.upper()} split")
        print("=" * 52)
        for key in sorted(metrics.keys()):
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {key:<30} {val:.4f}")
            else:
                print(f"  {key:<30} {val}")
        print("=" * 52 + "\n")
