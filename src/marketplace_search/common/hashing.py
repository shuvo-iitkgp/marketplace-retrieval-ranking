"""
src/marketplace_search/common/hashing.py
─────────────────────────────────────────
Deterministic experiment bucketing utilities.

Design goals
------------
* Reproducible: same user_id always maps to the same bucket, regardless of
  when or where the code runs.
* Collision-resistant: uses SHA-256 (not Python's built-in hash(), which is
  randomised by PYTHONHASHSEED).
* Lightweight: no external dependencies beyond stdlib.

Usage
-----
    from marketplace_search.common.hashing import assign_bucket, bucket_split

    bucket = assign_bucket("user_123", num_buckets=2)  # → 0 or 1
    is_treatment = bucket_split("user_123", treatment_fraction=0.5)
"""

from __future__ import annotations

import hashlib
from typing import Iterable


def _sha256_int(value: str) -> int:
    """Return a stable non-negative integer derived from ``value``."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest, 16)


def assign_bucket(entity_id: str, num_buckets: int = 2) -> int:
    """
    Map an entity (user, session, …) to a bucket in [0, num_buckets).

    Parameters
    ----------
    entity_id:
        String identifier — typically user_id or session_id.
    num_buckets:
        Total number of buckets.  Must be ≥ 1.

    Returns
    -------
    int
        Bucket index in [0, num_buckets).
    """
    if num_buckets < 1:
        raise ValueError(f"num_buckets must be >= 1, got {num_buckets}")
    return _sha256_int(entity_id) % num_buckets


def bucket_split(entity_id: str, treatment_fraction: float = 0.5) -> bool:
    """
    Return True if entity falls in the *treatment* group.

    Parameters
    ----------
    entity_id:
        Entity identifier.
    treatment_fraction:
        Fraction of population assigned to treatment (0 < f < 1).
    """
    if not (0 < treatment_fraction < 1):
        raise ValueError("treatment_fraction must be in (0, 1)")
    threshold = int(treatment_fraction * (2**256))
    return _sha256_int(entity_id) < threshold


def assign_buckets_bulk(
    entity_ids: Iterable[str], num_buckets: int = 2
) -> dict[str, int]:
    """
    Batch version of ``assign_bucket``.

    Returns a dict mapping each entity_id to its bucket index.
    """
    return {eid: assign_bucket(eid, num_buckets) for eid in entity_ids}
