"""
tests/test_splits_labels.py
────────────────────────────
Unit tests for data splitting and label assignment.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_search.data.labels import assign_labels, filter_active_entities, POSITIVE, NEGATIVE, NEUTRAL
from marketplace_search.data.splits import time_split, check_no_future_leakage


# ─────────────────────────────────────────────────────────────────────────── #
# Label tests                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #


def make_reviews(scores, timestamps=None):
    n = len(scores)
    return pd.DataFrame({
        "user_id": [f"u{i}" for i in range(n)],
        "product_id": [f"p{i}" for i in range(n)],
        "score": scores,
        "timestamp": timestamps or list(range(n)),
        "title": ["t"] * n,
    })


class TestAssignLabels:
    def test_positive_label(self):
        df = make_reviews([5.0, 4.0])
        result = assign_labels(df)
        assert (result["label"] == POSITIVE).all()

    def test_negative_label(self):
        df = make_reviews([1.0, 2.0])
        result = assign_labels(df)
        assert (result["label"] == NEGATIVE).all()

    def test_neutral_dropped_by_default(self):
        df = make_reviews([1.0, 3.0, 5.0])
        result = assign_labels(df)
        assert len(result) == 2
        assert 3 not in result["score"].values

    def test_neutral_kept_when_requested(self):
        df = make_reviews([1.0, 3.0, 5.0])
        result = assign_labels(df, keep_neutral=True)
        assert len(result) == 3
        assert NEUTRAL in result["label"].values

    def test_custom_thresholds(self):
        df = make_reviews([3.0, 4.0])
        result = assign_labels(df, positive_threshold=3, negative_threshold=1)
        assert (result["label"] == POSITIVE).all()


class TestFilterActiveEntities:
    def test_removes_low_activity_users(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "product_id": ["p1", "p2", "p3"],
            "score": [5, 5, 5],
            "timestamp": [1, 2, 3],
            "label": [1, 1, 1],
        })
        result = filter_active_entities(df, min_user_interactions=2)
        assert "u2" not in result["user_id"].values
        assert "u1" in result["user_id"].values


# ─────────────────────────────────────────────────────────────────────────── #
# Split tests                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #


def make_interactions(n=100):
    return pd.DataFrame({
        "user_id": [f"u{i % 10}" for i in range(n)],
        "product_id": [f"p{i % 20}" for i in range(n)],
        "score": [5.0] * n,
        "timestamp": list(range(1000, 1000 + n)),
        "label": [1] * n,
    })


class TestTimeSplit:
    def test_sizes(self):
        df = make_interactions(100)
        splits = time_split(df, 0.70, 0.15, 0.15)
        assert len(splits.train) == 70
        assert len(splits.val) == 15
        assert len(splits.test) == 15

    def test_temporal_ordering(self):
        df = make_interactions(100)
        splits = time_split(df)
        assert splits.train["timestamp"].max() <= splits.val["timestamp"].min()
        assert splits.val["timestamp"].max() <= splits.test["timestamp"].min()

    def test_fractions_must_sum_to_one(self):
        df = make_interactions(100)
        with pytest.raises(ValueError):
            time_split(df, 0.5, 0.3, 0.3)  # sums to 1.1

    def test_no_leakage_passes(self):
        """Leakage check should pass on a well-formed time split."""
        df = make_interactions(200)
        splits = time_split(df)
        # Should not raise
        check_no_future_leakage(splits)
