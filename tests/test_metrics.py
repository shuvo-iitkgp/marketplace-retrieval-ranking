"""
tests/test_metrics.py
──────────────────────
Unit tests for the evaluation metrics module.

All tests use manually constructed inputs with known expected outputs
to verify correctness of the vectorized metric implementations.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Ensure src/ is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_search.eval.metrics import (
    recall_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    hit_rate_at_k,
    compute_all_metrics,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Recall@K tests                                                               #
# ─────────────────────────────────────────────────────────────────────────── #


class TestRecallAtK:
    def test_perfect_retrieval(self):
        relevant = [["a", "b", "c"]]
        retrieved = [["a", "b", "c", "d", "e"]]
        assert recall_at_k(relevant, retrieved, k=3) == pytest.approx(1.0)

    def test_no_relevant_retrieved(self):
        relevant = [["a", "b"]]
        retrieved = [["x", "y", "z"]]
        assert recall_at_k(relevant, retrieved, k=3) == pytest.approx(0.0)

    def test_partial_retrieval(self):
        relevant = [["a", "b", "c", "d"]]
        retrieved = [["a", "x", "b", "y"]]
        # 2 of 4 relevant in top-4
        assert recall_at_k(relevant, retrieved, k=4) == pytest.approx(0.5)

    def test_k_cutoff(self):
        # relevant item 'a' is at rank 3, so recall@2 = 0, recall@3 = 0.5
        relevant = [["a", "b"]]
        retrieved = [["x", "y", "a", "b"]]
        assert recall_at_k(relevant, retrieved, k=2) == pytest.approx(0.0)
        assert recall_at_k(relevant, retrieved, k=3) == pytest.approx(0.5)

    def test_empty_relevant(self):
        relevant = [[]]
        retrieved = [["a", "b", "c"]]
        assert recall_at_k(relevant, retrieved, k=3) == pytest.approx(0.0)

    def test_multiple_queries_mean(self):
        relevant = [["a"], ["b"]]
        retrieved = [["a", "x"], ["x", "b"]]
        # Query 1: recall@2 = 1.0, Query 2: recall@2 = 1.0
        assert recall_at_k(relevant, retrieved, k=2) == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            recall_at_k([["a"]], [["a"], ["b"]], k=3)


# ─────────────────────────────────────────────────────────────────────────── #
# NDCG@K tests                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #


class TestNdcgAtK:
    def test_perfect_ndcg(self):
        relevant = [["a", "b"]]
        retrieved = [["a", "b", "c", "d"]]
        # Ideal: [1, 1, 0, 0] → NDCG = 1.0
        assert ndcg_at_k(relevant, retrieved, k=4) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        relevant = [["a", "b"]]
        retrieved = [["x", "y", "z"]]
        assert ndcg_at_k(relevant, retrieved, k=3) == pytest.approx(0.0)

    def test_ordering_matters(self):
        relevant = [["a", "b"]]
        # 'b' first, 'a' second — suboptimal ordering
        retrieved_suboptimal = [["b", "a"]]
        retrieved_optimal = [["a", "b"]]
        # Both have 2/2 relevant, but suboptimal scores lower
        # Actually with binary relevance and perfect retrieval they're equal
        # (both get NDCG=1.0 if all relevant items are retrieved)
        score_sub = ndcg_at_k(relevant, retrieved_suboptimal, k=2)
        score_opt = ndcg_at_k(relevant, retrieved_optimal, k=2)
        assert score_sub == pytest.approx(score_opt)  # both perfect here

    def test_first_item_beats_second(self):
        # Single relevant item — placing it at rank 1 vs rank 2
        relevant = [["a"]]
        at_rank_1 = [["a", "b", "c"]]
        at_rank_2 = [["b", "a", "c"]]
        score_1 = ndcg_at_k(relevant, at_rank_1, k=3)
        score_2 = ndcg_at_k(relevant, at_rank_2, k=3)
        assert score_1 > score_2

    def test_empty_relevant(self):
        relevant = [[]]
        retrieved = [["a", "b"]]
        assert ndcg_at_k(relevant, retrieved, k=2) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────── #
# MRR tests                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #


class TestMRR:
    def test_first_rank(self):
        relevant = [["a"]]
        retrieved = [["a", "b", "c"]]
        assert mean_reciprocal_rank(relevant, retrieved) == pytest.approx(1.0)

    def test_second_rank(self):
        relevant = [["b"]]
        retrieved = [["a", "b", "c"]]
        assert mean_reciprocal_rank(relevant, retrieved) == pytest.approx(0.5)

    def test_third_rank(self):
        relevant = [["c"]]
        retrieved = [["a", "b", "c"]]
        assert mean_reciprocal_rank(relevant, retrieved) == pytest.approx(1 / 3)

    def test_not_in_list(self):
        relevant = [["z"]]
        retrieved = [["a", "b", "c"]]
        assert mean_reciprocal_rank(relevant, retrieved, k=3) == pytest.approx(0.0)

    def test_average_over_queries(self):
        # Query 1: relevant at rank 1 (RR=1.0)
        # Query 2: relevant at rank 2 (RR=0.5)
        # MRR = (1.0 + 0.5) / 2 = 0.75
        relevant = [["a"], ["b"]]
        retrieved = [["a", "x"], ["x", "b"]]
        assert mean_reciprocal_rank(relevant, retrieved) == pytest.approx(0.75)

    def test_k_cutoff(self):
        # Relevant item is at rank 5, but k=3 → MRR = 0
        relevant = [["e"]]
        retrieved = [["a", "b", "c", "d", "e"]]
        assert mean_reciprocal_rank(relevant, retrieved, k=3) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────── #
# HitRate@K tests                                                              #
# ─────────────────────────────────────────────────────────────────────────── #


class TestHitRateAtK:
    def test_hit(self):
        relevant = [["a", "b"]]
        retrieved = [["x", "a", "y"]]
        assert hit_rate_at_k(relevant, retrieved, k=3) == pytest.approx(1.0)

    def test_miss(self):
        relevant = [["a"]]
        retrieved = [["x", "y", "z"]]
        assert hit_rate_at_k(relevant, retrieved, k=3) == pytest.approx(0.0)

    def test_k_cutoff_miss(self):
        # 'a' is at rank 3, so hit@2 = 0
        relevant = [["a"]]
        retrieved = [["x", "y", "a"]]
        assert hit_rate_at_k(relevant, retrieved, k=2) == pytest.approx(0.0)
        assert hit_rate_at_k(relevant, retrieved, k=3) == pytest.approx(1.0)

    def test_average(self):
        # 1 hit, 1 miss → 0.5
        relevant = [["a"], ["z"]]
        retrieved = [["a", "b"], ["x", "y"]]
        assert hit_rate_at_k(relevant, retrieved, k=2) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────── #
# compute_all_metrics                                                          #
# ─────────────────────────────────────────────────────────────────────────── #


class TestComputeAllMetrics:
    def test_returns_expected_keys(self):
        relevant = [["a", "b", "c"]]
        retrieved = [["a", "b", "c", "d"]]
        result = compute_all_metrics(
            relevant, retrieved,
            recall_ks=[5, 10],
            ndcg_ks=[5, 10],
            hit_rate_ks=[5, 10],
            mrr_k=10,
        )
        assert "recall@5" in result
        assert "recall@10" in result
        assert "ndcg@5" in result
        assert "ndcg@10" in result
        assert "mrr" in result
        assert "hit_rate@5" in result
        assert "hit_rate@10" in result

    def test_all_values_in_range(self):
        relevant = [["a"], ["b", "c"]]
        retrieved = [["a", "x", "y"], ["x", "b", "c"]]
        result = compute_all_metrics(relevant, retrieved)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"


# ─────────────────────────────────────────────────────────────────────────── #
# Hashing tests                                                                #
# ─────────────────────────────────────────────────────────────────────────── #


class TestHashing:
    def test_deterministic(self):
        from marketplace_search.common.hashing import assign_bucket
        b1 = assign_bucket("user_42", num_buckets=2)
        b2 = assign_bucket("user_42", num_buckets=2)
        assert b1 == b2

    def test_valid_range(self):
        from marketplace_search.common.hashing import assign_bucket
        for i in range(100):
            b = assign_bucket(f"user_{i}", num_buckets=4)
            assert 0 <= b < 4

    def test_roughly_uniform(self):
        from marketplace_search.common.hashing import assign_bucket
        buckets = [assign_bucket(f"user_{i}", num_buckets=2) for i in range(1000)]
        n_zero = buckets.count(0)
        # With 1000 users and 2 buckets, expect ~500 each; allow 10% tolerance
        assert 400 <= n_zero <= 600
