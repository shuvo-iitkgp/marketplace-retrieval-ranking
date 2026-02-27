"""
src/marketplace_search/eval/metrics.py
───────────────────────────────────────
Vectorized ranking metrics for offline evaluation.

All functions accept numpy arrays or Python lists and are designed to be
fast enough to run over the full test set in a single call.

Metrics implemented
-------------------
* Recall@K          — fraction of relevant items retrieved in top-K
* NDCG@K            — normalized discounted cumulative gain at rank K
* MRR               — mean reciprocal rank of the first relevant item
* HitRate@K         — binary: ≥1 relevant item in top-K (averaged across queries)

Reproducibility
---------------
All functions accept an optional ``rng_seed`` parameter.  When random
tie-breaking is needed (currently unused but reserved), the seed ensures
deterministic results.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────── #
# Recall@K                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #


def recall_at_k(
    relevant_ids: List[Sequence],
    retrieved_ids: List[Sequence],
    k: int,
) -> float:
    """
    Mean Recall@K over a list of queries.

    For each query:
        recall@k = |{relevant} ∩ {retrieved[:k]}| / |{relevant}|

    If a query has zero relevant items it contributes 0 to the mean
    (rather than being skipped) to avoid inflating the metric.

    Parameters
    ----------
    relevant_ids:
        List of sequences — one per query — of ground-truth item IDs.
    retrieved_ids:
        List of sequences — one per query — of retrieved item IDs (ordered,
        most relevant first).
    k:
        Cut-off rank.

    Returns
    -------
    float
        Mean Recall@K in [0, 1].
    """
    if len(relevant_ids) != len(retrieved_ids):
        raise ValueError(
            f"relevant_ids and retrieved_ids must have the same length. "
            f"Got {len(relevant_ids)} and {len(retrieved_ids)}."
        )

    scores = []
    for rel, ret in zip(relevant_ids, retrieved_ids):
        rel_set = set(rel)
        if not rel_set:
            scores.append(0.0)
            continue
        top_k = set(list(ret)[:k])
        scores.append(len(rel_set & top_k) / len(rel_set))

    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────── #
# NDCG@K                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #


def _dcg(gains: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain for a single ranked list."""
    gains_k = gains[:k]
    if len(gains_k) == 0:
        return 0.0
    positions = np.arange(1, len(gains_k) + 1)
    return float(np.sum(gains_k / np.log2(positions + 1)))


def ndcg_at_k(
    relevant_ids: List[Sequence],
    retrieved_ids: List[Sequence],
    k: int,
    relevance_scores: Optional[List[dict]] = None,
) -> float:
    """
    Mean NDCG@K over a list of queries.

    By default uses binary relevance (1 if relevant, 0 otherwise).
    Pass ``relevance_scores`` — a list of {item_id: score} dicts — to use
    graded relevance.

    Parameters
    ----------
    relevant_ids:
        Ground-truth relevant item IDs per query.
    retrieved_ids:
        Retrieved item IDs per query (most relevant first).
    k:
        Cut-off rank.
    relevance_scores:
        Optional list of per-query {item_id: float} relevance dicts.
        If None, binary relevance is assumed.

    Returns
    -------
    float
        Mean NDCG@K in [0, 1].
    """
    if len(relevant_ids) != len(retrieved_ids):
        raise ValueError("relevant_ids and retrieved_ids must have the same length.")

    scores = []
    for i, (rel, ret) in enumerate(zip(relevant_ids, retrieved_ids)):
        rel_set = set(rel)
        if not rel_set:
            scores.append(0.0)
            continue

        # Build gain vector for the retrieved list
        if relevance_scores is not None:
            rel_map = relevance_scores[i]
            gains = np.array([rel_map.get(item, 0.0) for item in list(ret)[:k]])
            ideal_gains = np.array(
                sorted(rel_map.values(), reverse=True)[:k], dtype=float
            )
        else:
            # Binary relevance
            gains = np.array([1.0 if item in rel_set else 0.0 for item in list(ret)[:k]])
            ideal_n = min(len(rel_set), k)
            ideal_gains = np.ones(ideal_n)

        dcg = _dcg(gains, k)
        idcg = _dcg(ideal_gains, k)
        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────── #
# MRR                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #


def mean_reciprocal_rank(
    relevant_ids: List[Sequence],
    retrieved_ids: List[Sequence],
    k: int = 100,
) -> float:
    """
    Mean Reciprocal Rank (capped at rank ``k``).

    MRR = mean over queries of 1 / rank_of_first_relevant_item.
    If no relevant item appears in the top-k, the query contributes 0.

    Parameters
    ----------
    relevant_ids, retrieved_ids:
        Same convention as other metric functions.
    k:
        Rank cut-off.

    Returns
    -------
    float
        MRR in [0, 1].
    """
    scores = []
    for rel, ret in zip(relevant_ids, retrieved_ids):
        rel_set = set(rel)
        if not rel_set:
            scores.append(0.0)
            continue
        rr = 0.0
        for rank, item in enumerate(list(ret)[:k], start=1):
            if item in rel_set:
                rr = 1.0 / rank
                break
        scores.append(rr)

    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────── #
# HitRate@K                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #


def hit_rate_at_k(
    relevant_ids: List[Sequence],
    retrieved_ids: List[Sequence],
    k: int,
) -> float:
    """
    Mean HitRate@K — fraction of queries with at least one relevant item in top-K.

    hit@k = 1 if |{relevant} ∩ {retrieved[:k]}| >= 1 else 0

    Parameters
    ----------
    relevant_ids, retrieved_ids:
        Same convention as other metric functions.
    k:
        Cut-off rank.

    Returns
    -------
    float
        Mean HitRate@K in [0, 1].
    """
    scores = []
    for rel, ret in zip(relevant_ids, retrieved_ids):
        rel_set = set(rel)
        if not rel_set:
            scores.append(0.0)
            continue
        top_k = set(list(ret)[:k])
        scores.append(1.0 if rel_set & top_k else 0.0)

    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────── #
# Convenience: compute all metrics at once                                     #
# ─────────────────────────────────────────────────────────────────────────── #


def compute_all_metrics(
    relevant_ids: List[Sequence],
    retrieved_ids: List[Sequence],
    recall_ks: Sequence[int] = (1, 5, 10, 20),
    ndcg_ks: Sequence[int] = (5, 10, 20),
    hit_rate_ks: Sequence[int] = (5, 10, 20),
    mrr_k: int = 100,
    rng_seed: int = 42,  # reserved for future tie-breaking
) -> dict:
    """
    Compute and return a flat dict of all metrics.

    Returns
    -------
    dict
        Keys follow the pattern ``"recall@10"``, ``"ndcg@10"``, ``"mrr"``,
        ``"hit_rate@10"``, etc.
    """
    _ = rng_seed  # currently unused; reserved for reproducible tie-breaking

    results = {}

    for k in recall_ks:
        results[f"recall@{k}"] = recall_at_k(relevant_ids, retrieved_ids, k)

    for k in ndcg_ks:
        results[f"ndcg@{k}"] = ndcg_at_k(relevant_ids, retrieved_ids, k)

    results["mrr"] = mean_reciprocal_rank(relevant_ids, retrieved_ids, k=mrr_k)

    for k in hit_rate_ks:
        results[f"hit_rate@{k}"] = hit_rate_at_k(relevant_ids, retrieved_ids, k)

    return results
