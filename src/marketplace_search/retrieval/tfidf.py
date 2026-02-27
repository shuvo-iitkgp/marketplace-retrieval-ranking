"""
src/marketplace_search/retrieval/tfidf.py
──────────────────────────────────────────
TF-IDF baseline retrieval engine.

Design
------
1. Index the product catalog using sklearn's TfidfVectorizer.
2. At query time, transform the query text to a TF-IDF vector and rank
   products by cosine similarity.
3. Return the top-K product IDs.

Two retrieval modes
-------------------
query_retrieve(query_text)  — takes a raw text query (for synthetic query eval)
user_retrieve(user_id)      — takes a user_id, builds a query from their
                               training history titles (for harness compatibility)

Persistence
-----------
The fitted vectorizer + document matrix can be saved/loaded with pickle so
we don't re-fit on every evaluation run.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from marketplace_search.data.catalog import clean_text

logger = logging.getLogger(__name__)


class TFIDFRetriever:
    """
    TF-IDF retrieval engine over a product catalog.

    Parameters
    ----------
    max_features:
        Vocabulary size cap for TfidfVectorizer.
    ngram_range:
        N-gram range passed to TfidfVectorizer.
    min_df:
        Minimum document frequency for a term to be included.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple = (1, 2),
        min_df: int = 1,
        sublinear_tf: bool = True,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,   # log(1+tf) dampening
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self._doc_matrix = None          # (n_products, vocab) sparse matrix
        self._product_ids: List[str] = []
        self._is_fitted = False

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, catalog: pd.DataFrame, text_col: str = "cleaned_text") -> "TFIDFRetriever":
        """
        Fit the vectorizer and build the document matrix.

        Parameters
        ----------
        catalog:
            Product catalog DataFrame.  Must contain ``product_id`` and
            ``text_col`` columns.
        text_col:
            Column containing the pre-cleaned text to index.

        Returns
        -------
        self
        """
        logger.info("Fitting TF-IDF on %d products...", len(catalog))
        t0 = time.perf_counter()

        texts = catalog[text_col].fillna("").tolist()
        self._product_ids = catalog["product_id"].tolist()
        self._doc_matrix = self.vectorizer.fit_transform(texts)
        self._is_fitted = True

        elapsed = time.perf_counter() - t0
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(
            "TF-IDF fitted: vocab=%d, doc_matrix=%s, elapsed=%.2fs",
            vocab_size,
            self._doc_matrix.shape,
            elapsed,
        )
        return self

    # ------------------------------------------------------------------ #
    # Query retrieval (text-based)                                         #
    # ------------------------------------------------------------------ #

    def query_retrieve(self, query_text: str, k: int = 200) -> List[str]:
        """
        Retrieve top-K products for a raw text query.

        Parameters
        ----------
        query_text:
            Raw search query (will be cleaned internally).
        k:
            Number of results to return.

        Returns
        -------
        List of product_ids ordered by descending cosine similarity.
        """
        self._check_fitted()
        cleaned = clean_text(query_text)
        if not cleaned:
            return []

        q_vec = self.vectorizer.transform([cleaned])
        scores = linear_kernel(q_vec, self._doc_matrix).flatten()

        top_k_idx = scores.argsort()[::-1][:k]
        return [self._product_ids[i] for i in top_k_idx if scores[i] > 0]

    # ------------------------------------------------------------------ #
    # User retrieval (history-based, for harness compatibility)            #
    # ------------------------------------------------------------------ #

    def build_user_query_fn(
        self,
        train_df: pd.DataFrame,
        catalog: pd.DataFrame,
        k: int = 200,
    ):
        """
        Build a ``retrieve_fn(user_id) -> List[str]`` closure compatible
        with the Phase 0 EvalHarness interface.

        The user's "query" is constructed by concatenating the titles of
        their positively-interacted products in training — i.e., their
        purchase history acts as a bag-of-words query profile.

        Parameters
        ----------
        train_df:
            Training interactions DataFrame (with label column).
        catalog:
            Product catalog with ``product_id``, ``cleaned_text`` columns.
        k:
            Retrieval depth.

        Returns
        -------
        Callable[[str], List[str]]
        """
        # Build user → title-tokens lookup from training positives
        title_lookup = catalog.set_index("product_id")["cleaned_text"].to_dict()

        pos_train = train_df[train_df["label"] == 1]
        user_history: dict[str, str] = {}
        for user_id, group in pos_train.groupby("user_id"):
            titles = [
                title_lookup.get(pid, "")
                for pid in group["product_id"].tolist()
            ]
            user_history[user_id] = " ".join(filter(None, titles))

        def retrieve_fn(user_id: str) -> List[str]:
            query_text = user_history.get(user_id, "")
            if not query_text:
                return []
            return self.query_retrieve(query_text, k=k)

        return retrieve_fn

    # ------------------------------------------------------------------ #
    # Scores (for reranker feature generation in later phases)            #
    # ------------------------------------------------------------------ #

    def score_candidates(
        self, query_text: str, candidate_ids: List[str]
    ) -> dict[str, float]:
        """
        Return cosine similarity scores for a list of candidate product IDs.

        Used by the reranker to get TF-IDF scores as a feature.
        """
        self._check_fitted()
        cleaned = clean_text(query_text)
        if not cleaned:
            return {pid: 0.0 for pid in candidate_ids}

        pid_to_idx = {pid: i for i, pid in enumerate(self._product_ids)}
        q_vec = self.vectorizer.transform([cleaned])

        scores = {}
        for pid in candidate_ids:
            idx = pid_to_idx.get(pid)
            if idx is None:
                scores[pid] = 0.0
            else:
                doc_vec = self._doc_matrix[idx]
                scores[pid] = float(linear_kernel(q_vec, doc_vec).flatten()[0])
        return scores

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Pickle the fitted retriever to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = path.stat().st_size / (1024 ** 2)
        logger.info("Saved TFIDFRetriever to %s (%.2f MB)", path, size_mb)

    @classmethod
    def load(cls, path: str | Path) -> "TFIDFRetriever":
        """Load a previously saved TFIDFRetriever."""
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        logger.info("Loaded TFIDFRetriever from %s", path)
        return obj

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("TFIDFRetriever must be fitted before retrieval. Call .fit() first.")

    @property
    def vocab_size(self) -> int:
        self._check_fitted()
        return len(self.vectorizer.vocabulary_)

    @property
    def n_products(self) -> int:
        return len(self._product_ids)

    def index_memory_mb(self) -> float:
        """Approximate memory footprint of the document matrix in MB."""
        if self._doc_matrix is None:
            return 0.0
        # Sparse matrix: data + indices + indptr arrays
        m = self._doc_matrix
        nbytes = m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
        return nbytes / (1024 ** 2)

    def __repr__(self) -> str:
        if self._is_fitted:
            return (
                f"TFIDFRetriever(vocab={self.vocab_size}, "
                f"products={self.n_products}, "
                f"index_mb={self.index_memory_mb():.2f})"
            )
        return "TFIDFRetriever(unfitted)"
