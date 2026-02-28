from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DualEncoderArtifacts:
    model_dir: Path
    model_state: Path
    catalog_embeddings: Path
    catalog_ids: Path
    metadata: Path


def _require_torch_transformers() -> tuple:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "Dual-encoder training requires optional deps: torch and transformers. "
            "Install with `pip install torch transformers`."
        ) from exc
    return torch, AutoModel, AutoTokenizer


class DualEncoderModel:  # thin wrapper to avoid import-time torch dependency
    def __init__(self, model_name: str = "distilbert-base-uncased", shared_weights: bool = True):
        torch, AutoModel, _ = _require_torch_transformers()
        self.torch = torch
        self.model_name = model_name
        self.shared_weights = shared_weights
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.product_encoder = self.query_encoder if shared_weights else AutoModel.from_pretrained(model_name)

    def to(self, device: str):
        self.query_encoder.to(device)
        if not self.shared_weights:
            self.product_encoder.to(device)
        return self

    def parameters(self):
        if self.shared_weights:
            return self.query_encoder.parameters()
        return list(self.query_encoder.parameters()) + list(self.product_encoder.parameters())

    def train(self):
        self.query_encoder.train()
        if not self.shared_weights:
            self.product_encoder.train()

    def eval(self):
        self.query_encoder.eval()
        if not self.shared_weights:
            self.product_encoder.eval()

    def state_dict(self):
        return {
            "query_encoder": self.query_encoder.state_dict(),
            "product_encoder": None if self.shared_weights else self.product_encoder.state_dict(),
            "model_name": self.model_name,
            "shared_weights": self.shared_weights,
        }

    def load_state_dict(self, state: dict):
        self.query_encoder.load_state_dict(state["query_encoder"])
        if not self.shared_weights and state.get("product_encoder") is not None:
            self.product_encoder.load_state_dict(state["product_encoder"])

    def encode_queries(self, batch_inputs: dict):
        return self._encode(batch_inputs, tower="query")

    def encode_products(self, batch_inputs: dict):
        return self._encode(batch_inputs, tower="product")

    def _encode(self, batch_inputs: dict, tower: str):
        torch = self.torch
        encoder = self.query_encoder if tower == "query" else self.product_encoder
        outputs = encoder(**batch_inputs)
        pooled = outputs.last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


class DualEncoderTrainDataset:
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        queries_df: pd.DataFrame,
        catalog_df: pd.DataFrame,
        seed: int = 42,
    ) -> None:
        positives = interactions_df[interactions_df["label"] == 1].copy()
        positives = positives.merge(
            queries_df[["product_id", "query_text"]],
            on="product_id",
            how="inner",
        )
        self.samples = positives[["user_id", "product_id", "query_text"]].drop_duplicates().reset_index(drop=True)
        catalog_base = catalog_df[["product_id", "cleaned_text", "category_id"]].drop_duplicates("product_id")
        self.catalog_lookup = catalog_base.set_index("product_id")["cleaned_text"].to_dict()
        self.product_to_category = catalog_base.set_index("product_id")["category_id"].to_dict()
        self.category_to_products = (
            catalog_base.groupby("category_id")["product_id"].apply(list).to_dict()
        )
        self.user_pos = positives.groupby("user_id")["product_id"].apply(set).to_dict()
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]
        user_id, pos_pid, query_text = row["user_id"], row["product_id"], row["query_text"]
        category = self.product_to_category.get(pos_pid, "unknown")
        candidates = [
            pid for pid in self.category_to_products.get(category, [])
            if pid != pos_pid and pid not in self.user_pos.get(user_id, set())
        ]
        if not candidates:
            candidates = [pid for pid in self.catalog_lookup if pid != pos_pid]
        neg_pid = self.rng.choice(candidates)
        return {
            "query_text": query_text,
            "pos_text": self.catalog_lookup[pos_pid],
            "neg_text": self.catalog_lookup[neg_pid],
        }


def make_collate_fn(tokenizer, max_length: int = 64):
    def collate(batch: list[dict]) -> dict:
        query_texts = [b["query_text"] for b in batch]
        pos_texts = [b["pos_text"] for b in batch]
        neg_texts = [b["neg_text"] for b in batch]
        return {
            "query": tokenizer(query_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"),
            "pos": tokenizer(pos_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"),
            "neg": tokenizer(neg_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"),
        }

    return collate


def similarity_cross_entropy_loss(query_emb, pos_emb, neg_emb, temperature: float = 1.0):
    import torch
    candidates = torch.cat([pos_emb, neg_emb], dim=0)
    logits = query_emb @ candidates.T / temperature
    labels = torch.arange(query_emb.size(0), device=query_emb.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def encode_corpus(
    model: DualEncoderModel,
    tokenizer,
    texts: list[str],
    device: str,
    batch_size: int = 64,
    max_length: int = 64,
) -> np.ndarray:
    torch, _, _ = _require_torch_transformers()
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            emb = model.encode_products(encoded).cpu().numpy()
            all_embeddings.append(emb)
    return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 768), dtype=np.float32)


def build_retrieve_fn(
    model: DualEncoderModel,
    tokenizer,
    catalog_df: pd.DataFrame,
    device: str,
    batch_size: int = 64,
    max_length: int = 64,
) -> Callable[[str], list[str]]:
    catalog_ids = catalog_df["product_id"].tolist()
    catalog_texts = catalog_df["cleaned_text"].fillna("").tolist()
    catalog_embeddings = encode_corpus(model, tokenizer, catalog_texts, device=device, batch_size=batch_size, max_length=max_length)

    def retrieve(query_text: str, k: int = 200) -> list[str]:
        torch, _, _ = _require_torch_transformers()
        model.eval()
        with torch.no_grad():
            encoded = tokenizer([query_text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {kk: vv.to(device) for kk, vv in encoded.items()}
            q_emb = model.encode_queries(encoded).cpu().numpy()[0]
        scores = catalog_embeddings @ q_emb
        top_idx = np.argsort(-scores)[:k]
        return [catalog_ids[i] for i in top_idx]

    return retrieve


def save_artifacts(
    output_dir: Path,
    model: DualEncoderModel,
    catalog_ids: list[str],
    catalog_embeddings: np.ndarray,
    metadata: dict,
) -> DualEncoderArtifacts:
    torch, _, _ = _require_torch_transformers()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_state = output_dir / "dual_encoder_state.pt"
    catalog_emb_path = output_dir / "catalog_embeddings.npy"
    catalog_ids_path = output_dir / "catalog_ids.json"
    meta_path = output_dir / "metadata.json"

    torch.save(model.state_dict(), model_state)
    np.save(catalog_emb_path, catalog_embeddings)
    with catalog_ids_path.open("w") as f:
        json.dump(catalog_ids, f)
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return DualEncoderArtifacts(output_dir, model_state, catalog_emb_path, catalog_ids_path, meta_path)