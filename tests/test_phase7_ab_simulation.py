import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_search.common.hashing import assign_bucket
from marketplace_search.common.logging_schema import ImpressionLog
from marketplace_search.experimentation.ab_simulation import OfflineABSimulation, build_hypothesis_statement


def test_impression_schema_validation():
    ok = ImpressionLog(
        user_id="u1",
        query_text="ceramic mug",
        candidate_ids=["p1", "p2"],
        predicted_scores=[0.9, 0.2],
        experiment_bucket=1,
    )
    assert ok.request_id

    try:
        ImpressionLog(
            user_id="u1",
            query_text="ceramic mug",
            candidate_ids=["p1"],
            predicted_scores=[0.9, 0.2],
            experiment_bucket=1,
        )
        assert False, "Expected ValueError for shape mismatch"
    except ValueError:
        pass


def test_deterministic_bucket_split_binary():
    buckets = [assign_bucket(f"user_{i}", num_buckets=2) for i in range(100)]
    assert set(buckets).issubset({0, 1})
    assert assign_bucket("stable-user", num_buckets=2) == assign_bucket("stable-user", num_buckets=2)


def test_offline_ab_simulation_and_guardrails():
    requests = [
        {"user_id": "u1", "query_text": "q1", "target_product_id": "p1"},
        {"user_id": "u2", "query_text": "q2", "target_product_id": "p2"},
        {"user_id": "u3", "query_text": "q3", "target_product_id": "p3"},
    ]

    def control_ranker(_user_id, _query_text, _target_product_id, _k):
        return ["x", "y", "z"], [0.8, 0.4, 0.1], 20.0

    def treatment_ranker(_user_id, _query_text, target_product_id, _k):
        return [target_product_id, "x", "y"], [0.9, 0.4, 0.1], 30.0

    sim = OfflineABSimulation(eval_k=3, latency_threshold_ms=25.0)
    result, logs = sim.run(
        requests=requests,
        control_ranker_fn=control_ranker,
        treatment_ranker_fn=treatment_ranker,
        expected_min_ndcg_lift=0.1,
    )

    assert result.ndcg_lift > 0
    assert result.mrr_lift > 0
    assert result.treatment.latency_alert is True
    assert result.control.null_result_rate == 0.0
    assert len(logs["control"]) == len(requests)


def test_hypothesis_statement_text():
    text = build_hypothesis_statement(expected_min_ndcg_lift=0.01, observed_ndcg_lift=0.02)
    assert "Hypothesis" in text
    assert "meets" in text