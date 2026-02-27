"""
src/marketplace_search/eval/query_harness.py
─────────────────────────────────────────────
Query-centric evaluation harness for Phase 1.

This complements the Phase 0 ``EvalHarness`` (which is user-centric) with a
**query-centric** evaluation mode where:

  - Each query is a (query_text, ground_truth_product_id) pair.
  - A "hit" at rank K means the ground-truth product appears in top-K.
  - This directly measures retrieval quality for synthetic queries.

Two evaluation modes
--------------------
1. ``run_query_eval``  — synthetic query dataset (query_text → product_id)
2. ``run_user_eval``   — user history as query (user_id → future products)
   (delegates to Phase 0 EvalHarness for continuity)

This module also handles the per-query result logging that lets us
inspect failure cases and build error analysis tables.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from marketplace_search.eval.metrics import (
    compute_all_metrics,
    recall_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    hit_rate_at_k,
)
from marketplace_search.eval.system_metrics import measure_latency

logger = logging.getLogger(__name__)


class QueryEvalHarness:
    """
    Evaluates a retrieval function against a synthetic query dataset.

    Each row in ``queries_df`` is one (query_text, product_id) pair.
    The retrieval function receives the query_text and must return an
    ordered list of product_ids.

    Parameters
    ----------
    config:
        Config object.
    queries_df:
        DataFrame with columns: query_id, product_id, query_text.
    split_name:
        Label used in log file names ("train_queries", "test_queries", …).
    """

    def __init__(
        self,
        config,
        queries_df: pd.DataFrame,
        split_name: str = "query_eval",
    ) -> None:
        self.config = config
        self.queries_df = queries_df.reset_index(drop=True)
        self.split_name = split_name
        self.log_dir = Path(config.paths.logs_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "QueryEvalHarness ready: %d queries (%d unique products)",
            len(queries_df),
            queries_df["product_id"].nunique(),
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
        save_per_query: bool = False,
    ) -> dict:
        """
        Evaluate retrieve_fn over all synthetic queries.

        Parameters
        ----------
        retrieve_fn:
            Callable that takes a query_text string and returns an ordered
            list of product_ids (most relevant first).
        k:
            Maximum retrieval depth.
        save_per_query:
            If True, save per-query result details for error analysis.

        Returns
        -------
        dict of metric_name → float, plus system metrics.
        """
        np.random.seed(rng_seed)

        query_texts = self.queries_df["query_text"].tolist()
        gt_product_ids = self.queries_df["product_id"].tolist()

        logger.info("Running query eval on %d queries...", len(query_texts))

        # Each query has exactly one ground-truth product
        relevant_ids = [[pid] for pid in gt_product_ids]
        retrieved_ids: List[List[str]] = []
        per_query_results = []

        for i, (qtext, gt_pid) in enumerate(zip(query_texts, gt_product_ids)):
            try:
                results = retrieve_fn(qtext)[:k]
            except Exception as exc:
                logger.warning("retrieve_fn failed for query '%s': %s", qtext[:40], exc)
                results = []

            retrieved_ids.append(results)

            if save_per_query:
                rank = next(
                    (r + 1 for r, pid in enumerate(results) if pid == gt_pid),
                    None,
                )
                per_query_results.append(
                    {
                        "query_id": self.queries_df.iloc[i]["query_id"],
                        "query_text": qtext,
                        "gt_product_id": gt_pid,
                        "rank": rank,
                        "hit@10": rank is not None and rank <= 10,
                        "hit@50": rank is not None and rank <= 50,
                    }
                )

        # ── Compute all metrics ─────────────────────────────────────── #
        eval_cfg = self.config.evaluation
        metrics = compute_all_metrics(
            relevant_ids=relevant_ids,
            retrieved_ids=retrieved_ids,
            recall_ks=eval_cfg.recall_k,
            ndcg_ks=eval_cfg.ndcg_k,
            hit_rate_ks=eval_cfg.hit_rate_k,
            mrr_k=eval_cfg.mrr_k,
            rng_seed=rng_seed,
        )

        # ── Latency benchmark ───────────────────────────────────────── #
        if measure_latency_flag and query_texts:
            sample_n = min(latency_sample, len(query_texts))
            rng = np.random.default_rng(rng_seed)
            idx = rng.choice(len(query_texts), size=sample_n, replace=False)
            sample_q = [query_texts[i] for i in idx]

            lat = measure_latency(
                fn=retrieve_fn,
                queries=sample_q,
                warmup=min(5, len(sample_q)),
                percentiles=eval_cfg.latency_percentiles,
            )
            metrics.update({f"latency_{kk}": vv for kk, vv in lat.items()})
            metrics["latency_sla_pass"] = (
                lat.get("p95_ms", 0) <= eval_cfg.max_p95_latency_ms
            )

        metrics["split"] = self.split_name
        metrics["n_queries"] = len(query_texts)
        metrics["n_unique_products"] = self.queries_df["product_id"].nunique()

        # ── Save per-query details if requested ─────────────────────── #
        if save_per_query and per_query_results:
            pq_path = self.log_dir / f"per_query_{self.split_name}.json"
            with pq_path.open("w") as f:
                json.dump(per_query_results, f, indent=2)
            logger.info("Per-query results saved → %s", pq_path)

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
        ts = int(time.time())
        filename = f"{model_name}_{self.split_name}_{ts}.json"
        out = self.log_dir / filename
        payload = {
            "model": model_name,
            "split": self.split_name,
            "timestamp_utc": ts,
            "metrics": metrics,
        }
        if extra_meta:
            payload["meta"] = extra_meta
        with out.open("w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved eval results → %s", out)
        return out

    def print_summary(self, metrics: dict, title: str = "") -> None:
        label = title or self.split_name.upper()
        print(f"\n{'=' * 56}")
        print(f"  {label}")
        print("=" * 56)
        # Print the most important metrics first
        priority = ["ndcg@10", "ndcg@5", "recall@10", "recall@5", "mrr",
                    "hit_rate@10", "latency_p95_ms"]
        printed = set()
        for key in priority:
            if key in metrics:
                val = metrics[key]
                fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
                print(f"  {key:<32} {fmt}")
                printed.add(key)
        print("  " + "─" * 52)
        for key in sorted(metrics.keys()):
            if key not in printed:
                val = metrics[key]
                fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
                print(f"  {key:<32} {fmt}")
        print("=" * 56 + "\n")


# ─────────────────────────────────────────────────────────────────────────── #
# Comparison utility                                                           #
# ─────────────────────────────────────────────────────────────────────────── #


def compare_metrics(
    results: dict[str, dict],
    keys: List[str] = ("ndcg@10", "recall@10", "mrr", "hit_rate@10"),
) -> pd.DataFrame:
    """
    Build a comparison table across multiple model runs.

    Parameters
    ----------
    results:
        Dict mapping model_name → metrics_dict.
    keys:
        Which metrics to include in the table.

    Returns
    -------
    pd.DataFrame with models as rows and metrics as columns.
    """
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        for k in keys:
            row[k] = metrics.get(k, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
