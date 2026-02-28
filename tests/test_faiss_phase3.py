import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

faiss = pytest.importorskip("faiss")

from marketplace_search.retrieval.faiss_retrieval import (
    FaissIndexConfig,
    FaissRetriever,
    benchmark_retriever,
    compute_expected_topk_ids,
    l2_normalize,
)


def _toy_embeddings() -> tuple[list[str], np.ndarray]:
    ids = ["p1", "p2", "p3", "p4"]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return ids, l2_normalize(embeddings)


def test_flatip_query_batch_returns_expected_neighbors():
    ids, emb = _toy_embeddings()
    retriever = FaissRetriever(product_ids=ids, config=FaissIndexConfig(index_type="flatip"))
    retriever.fit(emb)

    q = l2_normalize(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    results = retriever.query_batch(q, k=2)

    assert results[0][0] == "p1"
    assert results[1][0] == "p3"


def test_ivf_recall_drop_small_on_toy_data():
    ids, emb = _toy_embeddings()
    queries = emb.copy()

    exact = FaissRetriever(product_ids=ids, config=FaissIndexConfig(index_type="flatip"))
    exact.fit(emb)

    ivf = FaissRetriever(
        product_ids=ids,
        config=FaissIndexConfig(index_type="ivf_flat", nlist=2, nprobe=2),
    )
    ivf.fit(emb)

    expected = compute_expected_topk_ids(exact, queries, k=2, batch_size=2)
    exact_metrics = benchmark_retriever(exact, queries, expected, k=2, batch_size=2)
    ivf_metrics = benchmark_retriever(ivf, queries, expected, k=2, batch_size=2)

    assert exact_metrics.recall_at_k == pytest.approx(1.0)
    assert ivf_metrics.recall_at_k >= 0.98


def test_query_before_fit_raises():
    retriever = FaissRetriever(product_ids=["p1"], config=FaissIndexConfig(index_type="flatip"))
    with pytest.raises(RuntimeError):
        retriever.query_batch(np.array([[1.0, 0.0]], dtype=np.float32), k=1)