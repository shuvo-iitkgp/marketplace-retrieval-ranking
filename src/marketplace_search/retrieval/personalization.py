from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from marketplace_search.eval.metrics import ndcg_at_k
from marketplace_search.retrieval.faiss_retrieval import l2_normalize, to_float32_contiguous


@dataclass
class CategoryPriors:
    category_embeddings: dict[str, np.ndarray]
    global_embedding: np.ndarray

    def pick_for_query(self, query_embedding: np.ndarray) -> np.ndarray:
        if not self.category_embeddings:
            return self.global_embedding
        q = to_float32_contiguous(np.expand_dims(query_embedding, axis=0))[0]
        best_score = -np.inf
        best_embedding = self.global_embedding
        for embedding in self.category_embeddings.values():
            score = float(np.dot(q, embedding))
            if score > best_score:
                best_score = score
                best_embedding = embedding
        return best_embedding


@dataclass
class PersonalizationEvalResult:
    alpha: float
    ndcg_at_10: float


def compute_time_decay_weights(
    timestamps: np.ndarray,
    reference_time: float,
    decay_lambda: float,
) -> np.ndarray:
    ages = np.maximum(reference_time - timestamps.astype(np.float64), 0.0)
    return np.exp(-decay_lambda * ages)


def build_user_embeddings(
    interactions_df: pd.DataFrame,
    product_embeddings: dict[str, np.ndarray],
    user_col: str = "user_id",
    product_col: str = "product_id",
    timestamp_col: str = "timestamp",
    label_col: str = "label",
    positive_label: int = 1,
    decay_lambda: float = 0.01,
    reference_time: float | None = None,
) -> dict[str, np.ndarray]:
    if interactions_df.empty:
        return {}

    df = interactions_df.copy()
    if label_col in df.columns:
        df = df[df[label_col] == positive_label]
    if df.empty:
        return {}

    ref_time = float(reference_time if reference_time is not None else df[timestamp_col].max())

    user_vectors: dict[str, np.ndarray] = {}
    for user_id, group in df.groupby(user_col):
        weighted_vectors = []
        weights = compute_time_decay_weights(
            group[timestamp_col].to_numpy(dtype=np.float64),
            reference_time=ref_time,
            decay_lambda=decay_lambda,
        )
        for (_, row), weight in zip(group.iterrows(), weights):
            embedding = product_embeddings.get(row[product_col])
            if embedding is None:
                continue
            weighted_vectors.append(weight * embedding)

        if not weighted_vectors:
            continue

        aggregated = np.sum(np.vstack(weighted_vectors), axis=0, dtype=np.float32)
        normalized = l2_normalize(np.expand_dims(aggregated, axis=0))[0]
        user_vectors[str(user_id)] = normalized

    return user_vectors


def build_category_priors(
    catalog_df: pd.DataFrame,
    product_embeddings: dict[str, np.ndarray],
    product_col: str = "product_id",
    category_col: str = "category_id",
) -> CategoryPriors:
    category_vectors: dict[str, list[np.ndarray]] = {}
    all_vectors: list[np.ndarray] = []

    for row in catalog_df[[product_col, category_col]].drop_duplicates(product_col).itertuples(index=False):
        product_id, category_id = row
        emb = product_embeddings.get(product_id)
        if emb is None:
            continue
        key = str(category_id)
        category_vectors.setdefault(key, []).append(emb)
        all_vectors.append(emb)

    if not all_vectors:
        raise ValueError("No catalog embeddings available to build priors")

    category_embeddings = {
        category: l2_normalize(np.mean(np.vstack(vectors), axis=0, dtype=np.float32)[None, :])[0]
        for category, vectors in category_vectors.items()
        if vectors
    }
    global_embedding = l2_normalize(np.mean(np.vstack(all_vectors), axis=0, dtype=np.float32)[None, :])[0]
    return CategoryPriors(category_embeddings=category_embeddings, global_embedding=global_embedding)


def blend_query_user_embedding(query_embedding: np.ndarray, user_embedding: np.ndarray, alpha: float) -> np.ndarray:
    blended = alpha * query_embedding + (1.0 - alpha) * user_embedding
    return l2_normalize(np.expand_dims(blended.astype(np.float32), axis=0))[0]


def rank_products_for_embeddings(
    query_embeddings: np.ndarray,
    user_ids: list[str],
    catalog_ids: list[str],
    catalog_embeddings: np.ndarray,
    user_embeddings: dict[str, np.ndarray],
    category_priors: CategoryPriors,
    alpha: float,
    k: int,
) -> list[list[str]]:
    if query_embeddings.shape[0] != len(user_ids):
        raise ValueError("query_embeddings and user_ids must have the same length")

    final_query_embeddings = []
    for query_embedding, user_id in zip(query_embeddings, user_ids):
        user_vector = user_embeddings.get(user_id)
        if user_vector is None:
            user_vector = category_priors.pick_for_query(query_embedding)
        final_query_embeddings.append(blend_query_user_embedding(query_embedding, user_vector, alpha=alpha))

    final_matrix = to_float32_contiguous(np.vstack(final_query_embeddings))
    scores = final_matrix @ to_float32_contiguous(catalog_embeddings).T
    top_indices = np.argsort(-scores, axis=1)[:, :k]
    return [[catalog_ids[idx] for idx in row] for row in top_indices]


def tune_alpha(
    candidate_alphas: list[float],
    query_embeddings: np.ndarray,
    user_ids: list[str],
    ground_truth_product_ids: list[str],
    catalog_ids: list[str],
    catalog_embeddings: np.ndarray,
    user_embeddings: dict[str, np.ndarray],
    category_priors: CategoryPriors,
    k: int = 10,
) -> PersonalizationEvalResult:
    relevant_ids = [[pid] for pid in ground_truth_product_ids]

    best = PersonalizationEvalResult(alpha=1.0, ndcg_at_10=-1.0)
    for alpha in candidate_alphas:
        retrieved = rank_products_for_embeddings(
            query_embeddings=query_embeddings,
            user_ids=user_ids,
            catalog_ids=catalog_ids,
            catalog_embeddings=catalog_embeddings,
            user_embeddings=user_embeddings,
            category_priors=category_priors,
            alpha=alpha,
            k=k,
        )
        ndcg = ndcg_at_k(relevant_ids=relevant_ids, retrieved_ids=retrieved, k=k)
        if ndcg > best.ndcg_at_10:
            best = PersonalizationEvalResult(alpha=float(alpha), ndcg_at_10=float(ndcg))

    return best