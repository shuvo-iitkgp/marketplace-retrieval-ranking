import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_search.eval.metrics import mean_reciprocal_rank, ndcg_at_k
from marketplace_search.retrieval.reranker import (
    add_reranker_features,
    build_candidate_frame,
    build_product_feature_table,
    build_user_category_affinity,
    evaluate_rankings,
    resolve_category_column,
)


def test_build_product_feature_table_and_affinity_have_expected_columns():
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "product_id": ["p1", "p2", "p1", "p3"],
            "label": [1, 0, 1, 1],
            "score": [5, 2, 4, 5],
        }
    )
    catalog = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3"],
            "category": ["cat_a", "cat_a", "cat_b"],
        }
    )

    product_table = build_product_feature_table(interactions, catalog)
    assert {"global_ctr", "product_popularity", "category"}.issubset(set(product_table.columns))
    assert (product_table["global_ctr"].between(0, 1)).all()

    affinity = build_user_category_affinity(interactions, catalog)
    assert affinity[("u1", "cat_a")] > 0.0


def test_reranker_metrics_improve_over_similarity_baseline():
    eval_df = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "product_id": ["p1", "p3"],
            "category": ["cat_a", "cat_b"],
        }
    )
    candidate_ids = np.array([["p2", "p1"], ["p2", "p3"]], dtype=object)
    candidate_scores = np.array([[0.9, 0.8], [0.6, 0.5]], dtype=np.float32)
    candidate_df = build_candidate_frame(eval_df, candidate_ids, candidate_scores)

    product_table = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3"],
            "category": ["cat_a", "cat_a", "cat_b"],
            "global_ctr": [0.9, 0.1, 0.8],
            "product_popularity": [0.5, 1.0, 0.4],
        }
    )
    affinity = {("u1", "cat_a"): 0.8, ("u2", "cat_b"): 0.9}

    enriched, _ = add_reranker_features(candidate_df, product_table, affinity)
    enriched["rerank_probability"] = (
        0.7 * enriched["historical_global_ctr"] + 0.3 * enriched["category_match_binary"]
    )

    metrics = evaluate_rankings(
        enriched,
        baseline_score_col="similarity_score",
        rerank_score_col="rerank_probability",
        ndcg_fn=ndcg_at_k,
        mrr_fn=mean_reciprocal_rank,
        k=2,
    )

    assert metrics["baseline_ndcg@10"] < metrics["reranked_ndcg@10"]
    assert metrics["baseline_mrr"] < metrics["reranked_mrr"]


def test_category_column_resolution_supports_category_id():
    catalog = pd.DataFrame({"product_id": ["p1"], "category_id": ["cat_a"]})
    assert resolve_category_column(catalog) == "category_id"