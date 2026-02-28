import pytest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_search.retrieval.personalization import (
    blend_query_user_embedding,
    build_category_priors,
    build_user_embeddings,
    compute_time_decay_weights,
    rank_products_for_embeddings,
    tune_alpha,
)


def test_compute_time_decay_weights_decreases_with_age():
    timestamps = np.array([100.0, 98.0, 90.0])
    weights = compute_time_decay_weights(timestamps, reference_time=100.0, decay_lambda=0.1)
    assert weights[0] > weights[1] > weights[2]


def test_build_user_embeddings_applies_decay_and_normalization():
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "product_id": ["p_recent", "p_old", "p_other"],
            "timestamp": [100.0, 50.0, 100.0],
            "label": [1, 1, 1],
        }
    )
    product_embeddings = {
        "p_recent": np.array([1.0, 0.0], dtype=np.float32),
        "p_old": np.array([0.0, 1.0], dtype=np.float32),
        "p_other": np.array([0.0, 1.0], dtype=np.float32),
    }

    user_embeddings = build_user_embeddings(
        interactions_df=interactions,
        product_embeddings=product_embeddings,
        decay_lambda=0.1,
    )

    u1 = user_embeddings["u1"]
    assert np.linalg.norm(u1) == pytest.approx(1.0, rel=1e-5)
    assert u1[0] > u1[1]


def test_rank_and_tune_alpha_uses_personalization_and_cold_start_prior():
    catalog_ids = ["p1", "p2", "p3"]
    catalog_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    product_embeddings = dict(zip(catalog_ids, catalog_embeddings, strict=True))

    catalog_df = pd.DataFrame(
        {
            "product_id": catalog_ids,
            "category_id": ["cat_a", "cat_b", "cat_a"],
        }
    )
    priors = build_category_priors(catalog_df, product_embeddings)

    query_embeddings = np.array(
        [
            [0.2, 0.8],
            [0.95, 0.05],
        ],
        dtype=np.float32,
    )
    user_ids = ["u_repeat", "u_new"]
    user_embeddings = {"u_repeat": np.array([1.0, 0.0], dtype=np.float32)}
    ground_truth = ["p1", "p1"]

    baseline = rank_products_for_embeddings(
        query_embeddings=query_embeddings,
        user_ids=user_ids,
        catalog_ids=catalog_ids,
        catalog_embeddings=catalog_embeddings,
        user_embeddings=user_embeddings,
        category_priors=priors,
        alpha=1.0,
        k=3,
    )
    personalized = rank_products_for_embeddings(
        query_embeddings=query_embeddings,
        user_ids=user_ids,
        catalog_ids=catalog_ids,
        catalog_embeddings=catalog_embeddings,
        user_embeddings=user_embeddings,
        category_priors=priors,
        alpha=0.2,
        k=3,
    )

    assert baseline[0][0] == "p2"
    assert personalized[0][0] == "p1"
    assert personalized[1][0] in {"p1", "p3"}

    tuned = tune_alpha(
        candidate_alphas=[1.0, 0.8, 0.2],
        query_embeddings=query_embeddings,
        user_ids=user_ids,
        ground_truth_product_ids=ground_truth,
        catalog_ids=catalog_ids,
        catalog_embeddings=catalog_embeddings,
        user_embeddings=user_embeddings,
        category_priors=priors,
        k=3,
    )
    assert tuned.alpha == 0.2


def test_blend_query_user_embedding_is_normalized():
    q = np.array([1.0, 0.0], dtype=np.float32)
    u = np.array([0.0, 1.0], dtype=np.float32)
    mixed = blend_query_user_embedding(q, u, alpha=0.5)
    assert np.linalg.norm(mixed) == pytest.approx(1.0, rel=1e-5)