"""
src/marketplace_search/common/logging_schema.py
────────────────────────────────────────────────
Impression log schema and helper functions.

Every served result page is recorded as an ImpressionLog entry.
This schema is intentionally flat (JSONL-friendly) so it can be
ingested by any downstream system (BigQuery, Redshift, Kafka, etc.).

Fields
------
request_id       : Unique identifier for this search request (UUID).
user_id          : Retailer / shopper identifier.
query_text       : Raw query string.
candidate_ids    : Ordered list of returned product ASINs (top-K).
predicted_scores : Float scores aligned with candidate_ids.
timestamp        : Unix epoch seconds (UTC) when the request was served.
experiment_bucket: Integer bucket from deterministic hash assignment.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ImpressionLog:
    """Single impression record."""

    user_id: str
    query_text: str
    candidate_ids: List[str]
    predicted_scores: List[float]
    experiment_bucket: int
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        if len(self.candidate_ids) != len(self.predicted_scores):
            raise ValueError(
                "candidate_ids and predicted_scores must have the same length. "
                f"Got {len(self.candidate_ids)} ids and "
                f"{len(self.predicted_scores)} scores."
            )

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "ImpressionLog":
        return cls(**data)

    @classmethod
    def from_json(cls, line: str) -> "ImpressionLog":
        return cls.from_dict(json.loads(line))


# ─────────────────────────────────────────────────────────────────────────── #
# File-level writer                                                            #
# ─────────────────────────────────────────────────────────────────────────── #


class ImpressionLogger:
    """
    Appends ImpressionLog records to a JSONL file.

    Parameters
    ----------
    log_path:
        File path.  Parent directories are created automatically.
    """

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, impression: ImpressionLog) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(impression.to_json() + "\n")

    def log_batch(self, impressions: list[ImpressionLog]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            for imp in impressions:
                f.write(imp.to_json() + "\n")

    def read_all(self) -> list[ImpressionLog]:
        if not self.log_path.exists():
            return []
        with self.log_path.open(encoding="utf-8") as f:
            return [ImpressionLog.from_json(line) for line in f if line.strip()]
