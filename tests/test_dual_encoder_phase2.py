import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from marketplace_search.retrieval.dual_encoder import (
    DualEncoderTrainDataset,
    similarity_cross_entropy_loss,
)


def test_similarity_loss_scalar():
    q = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    p = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    n = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    loss = similarity_cross_entropy_loss(q, p, n, temperature=0.1)
    assert loss.ndim == 0
    assert float(loss.item()) > 0


def test_hard_negative_not_positive_product_when_possible():
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "product_id": ["p1", "p2", "p3"],
            "label": [1, 1, 1],
        }
    )
    queries = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3"],
            "query_text": ["paint set", "brush set", "canvas"],
        }
    )
    catalog = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3", "p4"],
            "cleaned_text": ["paint set", "brush set", "canvas", "ink"],
            "category_id": ["art", "art", "paper", "art"],
        }
    )

    ds = DualEncoderTrainDataset(interactions, queries, catalog, seed=7)
    sample = ds[0]
    assert sample["query_text"]
    assert sample["pos_text"] != sample["neg_text"]