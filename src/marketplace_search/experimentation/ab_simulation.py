from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from marketplace_search.common.hashing import assign_bucket
from marketplace_search.common.logging_schema import ImpressionLog
from marketplace_search.eval.metrics import mean_reciprocal_rank, ndcg_at_k

RankerFn = Callable[[str, str, str, int], tuple[list[str], list[float], float]]


@dataclass
class ExperimentArmSummary:
    arm_name: str
    n_requests: int
    ndcg_at_k: float
    mrr: float
    null_result_rate: float
    p95_latency_ms: float
    mean_latency_ms: float
    latency_alert: bool


@dataclass
class OfflineABResult:
    control: ExperimentArmSummary
    treatment: ExperimentArmSummary
    ndcg_lift: float
    mrr_lift: float
    hypothesis_statement: str


class OfflineABSimulation:
    """Offline A/B simulator for control-vs-treatment ranking systems."""

    def __init__(self, eval_k: int = 10, latency_threshold_ms: float = 100.0) -> None:
        self.eval_k = eval_k
        self.latency_threshold_ms = latency_threshold_ms

    def _summarize_arm(
        self,
        arm_name: str,
        requests: list[dict],
        ranker_fn: RankerFn,
    ) -> tuple[ExperimentArmSummary, list[ImpressionLog]]:
        relevant, retrieved = [], []
        latencies_ms: list[float] = []
        null_count = 0
        impressions: list[ImpressionLog] = []

        for req in requests:
            user_id = str(req["user_id"])
            query_text = str(req["query_text"])
            target_product_id = str(req["target_product_id"])
            bucket = assign_bucket(user_id, num_buckets=2)

            t0 = time.perf_counter()
            cand_ids, scores, latency_ms = ranker_fn(user_id, query_text, target_product_id, self.eval_k)
            measured_ms = (time.perf_counter() - t0) * 1000.0

            effective_latency_ms = float(latency_ms if latency_ms is not None else measured_ms)
            latencies_ms.append(effective_latency_ms)

            if not cand_ids:
                null_count += 1

            relevant.append([target_product_id])
            retrieved.append(cand_ids)

            impressions.append(
                ImpressionLog(
                    user_id=user_id,
                    query_text=query_text,
                    candidate_ids=cand_ids,
                    predicted_scores=[float(s) for s in scores],
                    experiment_bucket=bucket,
                )
            )

        ndcg = float(ndcg_at_k(relevant, retrieved, k=self.eval_k))
        mrr = float(mean_reciprocal_rank(relevant, retrieved, k=self.eval_k))
        null_rate = float(null_count / max(len(requests), 1))

        lat_arr = np.array(latencies_ms, dtype=np.float32) if latencies_ms else np.array([0.0], dtype=np.float32)
        p95 = float(np.percentile(lat_arr, 95))
        mean_ms = float(lat_arr.mean())

        summary = ExperimentArmSummary(
            arm_name=arm_name,
            n_requests=len(requests),
            ndcg_at_k=ndcg,
            mrr=mrr,
            null_result_rate=null_rate,
            p95_latency_ms=p95,
            mean_latency_ms=mean_ms,
            latency_alert=bool(p95 > self.latency_threshold_ms),
        )
        return summary, impressions

    def run(
        self,
        requests: list[dict],
        control_ranker_fn: RankerFn,
        treatment_ranker_fn: RankerFn,
        expected_min_ndcg_lift: float,
    ) -> tuple[OfflineABResult, dict[str, list[ImpressionLog]]]:
        control_summary, control_logs = self._summarize_arm("control_tfidf", requests, control_ranker_fn)
        treatment_summary, treatment_logs = self._summarize_arm(
            "treatment_two_stage", requests, treatment_ranker_fn
        )

        ndcg_lift = treatment_summary.ndcg_at_k - control_summary.ndcg_at_k
        mrr_lift = treatment_summary.mrr - control_summary.mrr

        result = OfflineABResult(
            control=control_summary,
            treatment=treatment_summary,
            ndcg_lift=float(ndcg_lift),
            mrr_lift=float(mrr_lift),
            hypothesis_statement=build_hypothesis_statement(
                expected_min_ndcg_lift=expected_min_ndcg_lift,
                observed_ndcg_lift=float(ndcg_lift),
            ),
        )
        return result, {"control": control_logs, "treatment": treatment_logs}


def build_hypothesis_statement(expected_min_ndcg_lift: float, observed_ndcg_lift: float) -> str:
    relation = "meets" if observed_ndcg_lift >= expected_min_ndcg_lift else "does not meet"
    return (
        "Hypothesis: the two-stage treatment improves ranking quality over TF-IDF "
        f"by at least +{expected_min_ndcg_lift:.4f} NDCG@K while keeping guardrails healthy. "
        f"Offline estimate: ΔNDCG={observed_ndcg_lift:.4f}, which {relation} this bar."
    )