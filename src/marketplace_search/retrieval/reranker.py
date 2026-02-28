from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd

@dataclass
class RerankerArtifacts:
    feature_columns: list[str]
    product_feature_table: pd.DataFrame
    user_category_affinity: dict[tuple[str, str], float]

ExposureStore = dict[tuple[str, str], float]


@dataclass
class ConstrainedExploreExploitConfig:
    epsilon: float = 0.05
    alpha: float = 0.05
    top_pool_k: int = 50
    eval_k: int = 10
    max_per_category: int = 3
    broad_entropy_threshold: float = 1.0
    exposure_decay: float = 0.99

def resolve_category_column(catalog_df: pd.DataFrame) -> str:
    """Resolve category column name across dataset versions."""
    if "category" in catalog_df.columns:
        return "category"
    if "category_id" in catalog_df.columns:
        return "category_id"
    raise KeyError("catalog_df must contain either 'category' or 'category_id'")


def build_product_feature_table(
    interactions_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    ctr_smoothing: float = 10.0,
) -> pd.DataFrame:
    """Build product-level reranking features used by every query."""
    if "product_id" not in interactions_df.columns:
        raise ValueError("interactions_df must contain product_id")

    ratings = interactions_df.get("score")
    if ratings is not None:
        ctr_signal = ((ratings.astype(float) - 1.0) / 4.0).clip(0.0, 1.0)
    else:
        ctr_signal = interactions_df["label"].astype(float).clip(0.0, 1.0)

    product_stats = (
        interactions_df.assign(_ctr_signal=ctr_signal)
        .groupby("product_id", as_index=False)
        .agg(global_ctr=("_ctr_signal", "mean"), interactions=("_ctr_signal", "count"))
    )

    prior = float(product_stats["global_ctr"].mean()) if not product_stats.empty else 0.0
    product_stats["global_ctr"] = (
        product_stats["global_ctr"] * product_stats["interactions"] + ctr_smoothing * prior
    ) / (product_stats["interactions"] + ctr_smoothing)

    max_count = float(product_stats["interactions"].max()) if not product_stats.empty else 1.0
    if max_count <= 0:
        max_count = 1.0
    product_stats["product_popularity"] = np.log1p(product_stats["interactions"]) / np.log1p(max_count)

    category_col = resolve_category_column(catalog_df)
    categories = catalog_df[["product_id", category_col]].drop_duplicates("product_id").rename(
        columns={category_col: "category"}
    )
    return categories.merge(
        product_stats[["product_id", "global_ctr", "product_popularity"]],
        on="product_id",
        how="left",
    ).fillna({"global_ctr": prior, "product_popularity": 0.0, "category": "unknown"})


def build_user_category_affinity(
    interactions_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    smoothing: float = 5.0,
) -> dict[tuple[str, str], float]:
    """Estimate p(click|user, category) from historical labeled interactions."""
    category_col = resolve_category_column(catalog_df)
    frame = interactions_df[["user_id", "product_id", "label"]].merge(
        catalog_df[["product_id", category_col]].drop_duplicates("product_id").rename(columns={category_col: "category"}),
        on="product_id",
        how="left",
    )
    frame["category"] = frame["category"].fillna("unknown")
    grouped = frame.groupby(["user_id", "category"], as_index=False).agg(
        positives=("label", "sum"),
        impressions=("label", "count"),
    )

    global_ctr = float(frame["label"].mean()) if not frame.empty else 0.0
    grouped["affinity"] = (grouped["positives"] + smoothing * global_ctr) / (grouped["impressions"] + smoothing)
    return {
        (str(row.user_id), str(row.category)): float(row.affinity)
        for row in grouped.itertuples(index=False)
    }


def fetch_faiss_candidates_with_scores(retriever, query_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return product ids and similarity scores from a fitted FaissRetriever."""
    if retriever.index is None:
        raise RuntimeError("Call fit before querying")
    q = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    scores, indices = retriever.index.search(q, int(k))

    candidate_ids = np.full(indices.shape, "", dtype=object)
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            idx = int(indices[i, j])
            if 0 <= idx < len(retriever.product_ids):
                candidate_ids[i, j] = retriever.product_ids[idx]
    return candidate_ids, scores.astype(np.float32)


def build_candidate_frame(
    eval_df: pd.DataFrame,
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
) -> pd.DataFrame:
    """Build one row per (query, candidate) pair."""
    rows: list[dict] = []
    for q_idx, row in enumerate(eval_df.itertuples(index=False)):
        for rank_idx, cand in enumerate(candidate_ids[q_idx]):
            if not cand:
                continue
            rows.append(
                {
                    "query_idx": q_idx,
                    "user_id": str(row.user_id),
                    "target_product_id": str(row.product_id),
                    "query_category": str(row.category),
                    "candidate_product_id": str(cand),
                    "similarity_score": float(candidate_scores[q_idx, rank_idx]),
                    "label": int(str(cand) == str(row.product_id)),
                }
            )
    return pd.DataFrame(rows)


def add_reranker_features(
    candidate_df: pd.DataFrame,
    product_feature_table: pd.DataFrame,
    user_category_affinity: dict[tuple[str, str], float],
) -> tuple[pd.DataFrame, list[str]]:
    enriched = candidate_df.merge(
        product_feature_table.rename(columns={"product_id": "candidate_product_id", "category": "candidate_category"}),
        on="candidate_product_id",
        how="left",
    )
    enriched["candidate_category"] = enriched["candidate_category"].fillna("unknown")

    enriched["category_match_binary"] = (
        enriched["candidate_category"].astype(str) == enriched["query_category"].astype(str)
    ).astype(np.float32)
    enriched["historical_global_ctr"] = enriched["global_ctr"].astype(np.float32)
    enriched["product_popularity"] = enriched["product_popularity"].astype(np.float32)
    enriched["user_category_affinity"] = [
        float(user_category_affinity.get((u, c), 0.0))
        for u, c in zip(enriched["user_id"].astype(str), enriched["candidate_category"].astype(str))
    ]

    feature_columns = [
        "similarity_score",
        "category_match_binary",
        "historical_global_ctr",
        "user_category_affinity",
        "product_popularity",
    ]
    enriched[feature_columns] = enriched[feature_columns].fillna(0.0)
    return enriched, feature_columns


def ranking_lists_from_frame(candidate_df: pd.DataFrame, score_column: str) -> list[list[str]]:
    ordered = candidate_df.sort_values(["query_idx", score_column], ascending=[True, False])
    return ordered.groupby("query_idx")["candidate_product_id"].apply(list).tolist()


def evaluate_rankings(
    candidate_df: pd.DataFrame,
    baseline_score_col: str,
    rerank_score_col: str,
    ndcg_fn,
    mrr_fn,
    k: int,
) -> dict[str, float]:
    relevant = [[pid] for pid in candidate_df.groupby("query_idx")["target_product_id"].first().tolist()]
    baseline = ranking_lists_from_frame(candidate_df, baseline_score_col)
    reranked = ranking_lists_from_frame(candidate_df, rerank_score_col)
    return {
        "baseline_ndcg@10": float(ndcg_fn(relevant, baseline, k=k)),
        "reranked_ndcg@10": float(ndcg_fn(relevant, reranked, k=k)),
        "baseline_mrr": float(mrr_fn(relevant, baseline, k=k)),
        "reranked_mrr": float(mrr_fn(relevant, reranked, k=k)),
    }


def compute_permutation_feature_importance(model, x: np.ndarray, y: np.ndarray, feature_names: list[str], seed: int) -> list[dict]:
    try:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(model, x, y, n_repeats=5, random_state=seed, scoring="roc_auc")
        order = np.argsort(-result.importances_mean)
        return [
            {
                "feature": feature_names[i],
                "importance_mean": float(result.importances_mean[i]),
                "importance_std": float(result.importances_std[i]),
            }
            for i in order
        ]
    except ModuleNotFoundError:
        # Fallback: absolute point-biserial proxy to keep reporting available in lightweight envs.
        y_arr = y.astype(np.float32)
        importances: list[dict] = []
        for i, name in enumerate(feature_names):
            feat = x[:, i].astype(np.float32)
            if np.std(feat) < 1e-12:
                score = 0.0
            else:
                score = float(abs(np.corrcoef(feat, y_arr)[0, 1]))
                if np.isnan(score):
                    score = 0.0
            importances.append({"feature": name, "importance_mean": score, "importance_std": 0.0})
        # return sorted(importances, key=lambda d: d["importance_mean"], reverse=True)
        return sorted(importances, key=lambda d: d["importance_mean"], reverse=True)

def build_exposure_store() -> ExposureStore:
    return defaultdict(float)


def novelty_bonus(query_cluster_id: str, product_id: str, exposure_store: ExposureStore) -> float:
    exposure = float(exposure_store.get((str(query_cluster_id), str(product_id)), 0.0))
    return float(1.0 / np.sqrt(1.0 + exposure))


def apply_decay_to_cluster(exposure_store: ExposureStore, query_cluster_id: str, decay: float) -> None:
    cluster = str(query_cluster_id)
    for key in list(exposure_store.keys()):
        if key[0] == cluster:
            exposure_store[key] = float(decay) * float(exposure_store[key])


def update_exposure_store(
    exposure_store: ExposureStore,
    query_cluster_id: str,
    shown_product_ids: list[str],
    decay: float,
    eval_k: int,
) -> None:
    cluster = str(query_cluster_id)
    apply_decay_to_cluster(exposure_store, cluster, decay=decay)
    for pid in shown_product_ids[: int(eval_k)]:
        key = (cluster, str(pid))
        exposure_store[key] = float(exposure_store.get(key, 0.0)) + 1.0


def _is_broad_query(query_cluster_entropy: float, query_text: str, entropy_threshold: float) -> bool:
    short_query = len(str(query_text).split()) < 3
    return bool(float(query_cluster_entropy) >= float(entropy_threshold) or short_query)


def _apply_diversity_cap(
    ranked_frame: pd.DataFrame,
    eval_k: int,
    max_per_category: int,
) -> pd.DataFrame:
    category_counts: dict[str, int] = defaultdict(int)
    selected_idx: list[int] = []
    skipped_idx: list[int] = []
    for row in ranked_frame.itertuples():
        category = str(row.candidate_category)
        if category_counts[category] < int(max_per_category):
            selected_idx.append(int(row.Index))
            category_counts[category] += 1
            if len(selected_idx) >= int(eval_k):
                break
        else:
            skipped_idx.append(int(row.Index))

    if len(selected_idx) < int(eval_k):
        for idx in skipped_idx:
            if idx not in selected_idx:
                selected_idx.append(idx)
            if len(selected_idx) >= int(eval_k):
                break

    selected = ranked_frame.loc[selected_idx].copy()
    selected["final_rank"] = np.arange(1, len(selected) + 1)
    return selected


def rerank_with_constrained_explore_exploit(
    candidate_df: pd.DataFrame,
    exposure_store: ExposureStore,
    cfg: ConstrainedExploreExploitConfig,
    seed: int = 42,
) -> tuple[pd.DataFrame, ExposureStore]:
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []

    for _, query_df in candidate_df.groupby("query_idx", sort=True):
        query_df = query_df.copy()
        cluster_id = str(query_df["query_cluster_id"].iloc[0])
        query_text = str(query_df.get("query_text", pd.Series([""])).iloc[0])
        entropy = float(query_df.get("query_cluster_entropy", pd.Series([0.0])).iloc[0])

        query_df["base_score"] = query_df["base_score"].astype(float)
        query_df["novelty_bonus"] = [
            novelty_bonus(cluster_id, pid, exposure_store)
            for pid in query_df["candidate_product_id"].astype(str)
        ]
        query_df["final_score"] = query_df["base_score"] + float(cfg.alpha) * query_df["novelty_bonus"]

        baseline = query_df.sort_values("base_score", ascending=False, kind="mergesort")
        safe_k = min(int(cfg.top_pool_k), len(baseline))
        safe_pool = baseline.iloc[:safe_k].copy()
        outside_pool = baseline.iloc[safe_k:].copy()

        explore = bool(rng.random() < float(cfg.epsilon))
        if explore:
            safe_pool = safe_pool.sort_values("final_score", ascending=False, kind="mergesort")

        ranked = pd.concat([safe_pool, outside_pool], ignore_index=False)
        ranked = ranked.sort_values(["base_score"], ascending=False, kind="mergesort") if not explore else ranked

        broad = _is_broad_query(entropy, query_text, cfg.broad_entropy_threshold)
        if broad:
            top = _apply_diversity_cap(ranked, eval_k=cfg.eval_k, max_per_category=cfg.max_per_category)
        else:
            top = ranked.head(int(cfg.eval_k)).copy()
            top["final_rank"] = np.arange(1, len(top) + 1)

        update_exposure_store(
            exposure_store,
            query_cluster_id=cluster_id,
            shown_product_ids=top["candidate_product_id"].astype(str).tolist(),
            decay=float(cfg.exposure_decay),
            eval_k=int(cfg.eval_k),
        )
        top["did_explore"] = int(explore)
        rows.append(top)

    return pd.concat(rows, ignore_index=True), exposure_store



