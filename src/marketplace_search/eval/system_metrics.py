"""
src/marketplace_search/eval/system_metrics.py
──────────────────────────────────────────────
Utilities for measuring system-level metrics:
  - Query latency (p50, p95, p99)
  - Index / model memory footprint

These are used in the eval harness to ensure the system meets
SLA requirements (e.g., p95 < 100ms, index < X MB).
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Callable, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────── #
# Latency measurement                                                          #
# ─────────────────────────────────────────────────────────────────────────── #


@contextmanager
def timer_ms():
    """
    Context manager that measures wall-clock time in milliseconds.

    Usage::

        with timer_ms() as t:
            do_something()
        print(t.elapsed_ms)
    """

    class _Timer:
        elapsed_ms: float = 0.0

    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed_ms = (time.perf_counter() - start) * 1_000.0


def measure_latency(
    fn: Callable,
    queries: list,
    warmup: int = 5,
    percentiles: List[int] = (50, 95, 99),
) -> dict:
    """
    Benchmark a retrieval function over a list of queries and return
    latency percentiles in milliseconds.

    Parameters
    ----------
    fn:
        Callable that takes a single query and returns results.
        Signature: fn(query) -> Any
    queries:
        List of inputs to pass to ``fn``.
    warmup:
        Number of warm-up calls before recording timings.
    percentiles:
        Percentiles to compute (default: 50, 95, 99).

    Returns
    -------
    dict
        Keys: ``"p{N}_ms"`` for each requested percentile, plus
        ``"mean_ms"`` and ``"n_queries"``.
    """
    # Warm up (fill caches, JIT, etc.)
    for q in queries[:warmup]:
        fn(q)

    latencies_ms: List[float] = []
    for q in queries:
        start = time.perf_counter()
        fn(q)
        latencies_ms.append((time.perf_counter() - start) * 1_000.0)

    arr = np.array(latencies_ms)
    results = {f"p{p}_ms": float(np.percentile(arr, p)) for p in percentiles}
    results["mean_ms"] = float(arr.mean())
    results["n_queries"] = len(latencies_ms)
    return results


# ─────────────────────────────────────────────────────────────────────────── #
# Memory footprint                                                             #
# ─────────────────────────────────────────────────────────────────────────── #


def file_size_mb(path: str) -> float:
    """Return the size of a file in megabytes."""
    return os.path.getsize(path) / (1024 ** 2)


def object_size_mb(obj) -> float:
    """
    Estimate the in-memory size of an object in megabytes using sys.getsizeof.

    Note: This is a *shallow* estimate. For numpy arrays and PyTorch tensors
    the reported value reflects actual data memory.  For complex Python objects
    it may undercount nested references.
    """
    import sys
    return sys.getsizeof(obj) / (1024 ** 2)


def numpy_array_size_mb(arr: np.ndarray) -> float:
    """Return the exact memory footprint of a numpy array in MB."""
    return arr.nbytes / (1024 ** 2)


# ─────────────────────────────────────────────────────────────────────────── #
# SLA checker                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #


def check_latency_sla(
    latency_results: dict,
    max_p95_ms: float = 100.0,
) -> bool:
    """
    Return True if p95 latency is within the SLA budget.

    Parameters
    ----------
    latency_results:
        Dict returned by ``measure_latency``.
    max_p95_ms:
        SLA target for p95 latency in milliseconds.
    """
    p95 = latency_results.get("p95_ms")
    if p95 is None:
        raise KeyError("latency_results must contain 'p95_ms'")
    passed = p95 <= max_p95_ms
    return passed
