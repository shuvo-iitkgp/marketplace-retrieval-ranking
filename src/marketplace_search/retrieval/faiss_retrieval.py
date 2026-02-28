from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _require_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "FAISS retrieval requires optional dependency `faiss-cpu`. "
            "Install with `pip install faiss-cpu`."
        ) from exc
    return faiss


def to_float32_contiguous(embeddings: np.ndarray) -> np.ndarray:
    """Ensure embeddings are C-contiguous float32 for FAISS."""
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows so inner-product search behaves like cosine similarity."""
    arr = to_float32_contiguous(embeddings)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


@dataclass
class FaissIndexConfig:
    index_type: str = "flatip"  # flatip | ivf_flat | hnsw
    nlist: int = 256
    nprobe: int = 16
    hnsw_m: int = 32
    ef_search: int = 64
    metric: str = "ip"  # ip | l2


class FaissRetriever:
    def __init__(self, product_ids: list[str], config: FaissIndexConfig) -> None:
        self.product_ids = list(product_ids)
        self.config = config
        self.index = None

    def fit(self, embeddings: np.ndarray) -> None:
        faiss = _require_faiss()
        vectors = to_float32_contiguous(embeddings)
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2D array")

        dim = vectors.shape[1]
        metric = faiss.METRIC_INNER_PRODUCT if self.config.metric == "ip" else faiss.METRIC_L2
        index_type = self.config.index_type.lower()

        if index_type == "flatip":
            index = faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
            index.add(vectors)
        elif index_type == "ivf_flat":
            quantizer = faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
            nlist = max(1, min(int(self.config.nlist), vectors.shape[0]))
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric)
            if not index.is_trained:
                index.train(vectors)
            index.add(vectors)
            index.nprobe = int(self.config.nprobe)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, int(self.config.hnsw_m), metric)
            index.add(vectors)
            index.hnsw.efSearch = int(self.config.ef_search)
        else:
            raise ValueError(f"Unsupported index_type: {self.config.index_type}")

        self.index = index
        logger.info("Built FAISS index type=%s ntotal=%d", self.config.index_type, self.index.ntotal)

    def query(self, query_embedding: np.ndarray, k: int = 100) -> list[str]:
        ids_batch = self.query_batch(np.expand_dims(query_embedding, axis=0), k=k)
        return ids_batch[0]

    def query_batch(self, query_embeddings: np.ndarray, k: int = 100) -> list[list[str]]:
        if self.index is None:
            raise RuntimeError("Call fit before querying")
        q = to_float32_contiguous(query_embeddings)
        if q.ndim != 2:
            raise ValueError("query_embeddings must be a 2D array")
        _, idx = self.index.search(q, k)
        results: list[list[str]] = []
        for row in idx:
            results.append([self.product_ids[i] for i in row if 0 <= i < len(self.product_ids)])
        return results


@dataclass
class BenchmarkResult:
    recall_at_k: float
    p50_ms: float
    p95_ms: float
    mean_ms: float


def benchmark_retriever(
    retriever: FaissRetriever,
    query_embeddings: np.ndarray,
    expected_topk: list[set[str]],
    k: int,
    batch_size: int,
) -> BenchmarkResult:
    latencies_ms: list[float] = []
    retrieved: list[list[str]] = []

    q = to_float32_contiguous(query_embeddings)
    for start in range(0, len(q), batch_size):
        batch = q[start:start + batch_size]
        t0 = time.perf_counter()
        batch_results = retriever.query_batch(batch, k=k)
        latencies_ms.append((time.perf_counter() - t0) * 1_000.0 / max(len(batch), 1))
        retrieved.extend(batch_results)

    recalls = []
    for found, expected in zip(retrieved, expected_topk):
        if not expected:
            continue
        overlap = len(set(found) & expected)
        recalls.append(overlap / len(expected))

    arr = np.asarray(latencies_ms, dtype=np.float64)
    return BenchmarkResult(
        recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
        p50_ms=float(np.percentile(arr, 50)) if arr.size else 0.0,
        p95_ms=float(np.percentile(arr, 95)) if arr.size else 0.0,
        mean_ms=float(np.mean(arr)) if arr.size else 0.0,
    )


def compute_expected_topk_ids(
    exact_retriever: FaissRetriever,
    query_embeddings: np.ndarray,
    k: int,
    batch_size: int,
) -> list[set[str]]:
    expected: list[set[str]] = []
    q = to_float32_contiguous(query_embeddings)
    for start in range(0, len(q), batch_size):
        batch = q[start:start + batch_size]
        expected.extend([set(ids) for ids in exact_retriever.query_batch(batch, k=k)])
    return expected