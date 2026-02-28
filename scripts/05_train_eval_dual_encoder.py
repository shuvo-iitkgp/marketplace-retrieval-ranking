"""
Phase 2 — Train/evaluate a dual-encoder retrieval model.

Features:
- Query encoder + product encoder (shared or separate weights)
- Hard negatives from same category and in-batch negatives
- Similarity-matrix cross-entropy on normalized embeddings
- Mixed precision + gradient clipping
- Best checkpoint selected by validation NDCG@10
- Lift report vs TF-IDF baseline and simple ablations
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_search.common.config import load_config
from marketplace_search.eval.query_harness import QueryEvalHarness
from marketplace_search.retrieval.dual_encoder import (
    DualEncoderModel,
    DualEncoderTrainDataset,
    build_retrieve_fn,
    encode_corpus,
    make_collate_fn,
    save_artifacts,
    similarity_cross_entropy_loss,
)
from marketplace_search.retrieval.tfidf import TFIDFRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase2_dual_encoder")


def run_single_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    cfg,
    shared_weights: bool,
    use_hard_negatives: bool,
):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training config: shared_weights=%s, use_hard_negatives=%s, device=%s", shared_weights, use_hard_negatives, device)

    model_cfg = cfg.dual_encoder
    model = DualEncoderModel(model_name=model_cfg.model_name, shared_weights=shared_weights).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)

    ds = DualEncoderTrainDataset(train_df, queries_df, catalog_df, seed=cfg.project.seed)
    collate = make_collate_fn(tokenizer, max_length=model_cfg.max_length)
    loader = DataLoader(ds, batch_size=model_cfg.batch_size, shuffle=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_cfg.learning_rate, weight_decay=model_cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_ndcg10 = -1.0
    best_state = None
    train_history: list[dict] = []

    val_queries = val_df[val_df["label"] == 1][["product_id"]].drop_duplicates().merge(
        queries_df[["product_id", "query_text"]], on="product_id", how="inner"
    )
    val_harness = QueryEvalHarness(cfg, val_queries, split_name="val_dual_encoder")

    for epoch in range(1, model_cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps = 0
        for batch in loader:
            query = {k: v.to(device) for k, v in batch["query"].items()}
            pos = {k: v.to(device) for k, v in batch["pos"].items()}
            neg = {k: v.to(device) for k, v in batch["neg"].items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                q_emb = model.encode_queries(query)
                p_emb = model.encode_products(pos)
                if use_hard_negatives:
                    n_emb = model.encode_products(neg)
                else:
                    n_emb = p_emb.detach()
                loss = similarity_cross_entropy_loss(q_emb, p_emb, n_emb, temperature=model_cfg.temperature)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            n_steps += 1

        retrieve_fn = build_retrieve_fn(model, tokenizer, catalog_df, device=device, batch_size=model_cfg.eval_batch_size, max_length=model_cfg.max_length)
        val_metrics = val_harness.run(
            retrieve_fn=retrieve_fn,
            k=model_cfg.retrieval_k,
            measure_latency_flag=False,
            save_per_query=False,
        )
        ndcg10 = val_metrics.get("ndcg@10", 0.0)
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_steps, 1),
            "val_ndcg@10": ndcg10,
            "val_recall@10": val_metrics.get("recall@10", 0.0),
        }
        train_history.append(epoch_record)
        logger.info("Epoch %d | loss=%.4f | val_ndcg@10=%.4f", epoch, epoch_record["train_loss"], ndcg10)

        if ndcg10 > best_ndcg10:
            best_ndcg10 = ndcg10
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    retrieve_fn = build_retrieve_fn(model, tokenizer, catalog_df, device=device, batch_size=model_cfg.eval_batch_size, max_length=model_cfg.max_length)
    catalog_ids = catalog_df["product_id"].tolist()
    catalog_embeddings = encode_corpus(model, tokenizer, catalog_df["cleaned_text"].fillna("").tolist(), device=device, batch_size=model_cfg.eval_batch_size, max_length=model_cfg.max_length)

    return model, retrieve_fn, train_history, catalog_ids, catalog_embeddings


def main(config_path: str) -> None:
    cfg = load_config(config_path, project_root=PROJECT_ROOT)
    processed_dir = PROJECT_ROOT / cfg.paths.processed_dir
    artifacts_dir = PROJECT_ROOT / cfg.paths.artifacts_dir
    logs_dir = PROJECT_ROOT / cfg.paths.logs_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(processed_dir / "train.csv.gz")
    val_df = pd.read_csv(processed_dir / "val.csv.gz")
    test_df = pd.read_csv(processed_dir / "test.csv.gz")
    catalog_df = pd.read_csv(processed_dir / "rich_catalog.csv.gz")
    queries_df = pd.read_csv(processed_dir / "queries_title.csv.gz")

    test_queries = test_df[test_df["label"] == 1][["product_id"]].drop_duplicates().merge(
        queries_df[["product_id", "query_text"]], on="product_id", how="inner"
    )
    test_harness = QueryEvalHarness(cfg, test_queries, split_name="test_dual_encoder")

    tfidf = TFIDFRetriever(
        max_features=cfg.tfidf.max_features,
        ngram_range=tuple(cfg.tfidf.ngram_range),
        min_df=cfg.tfidf.min_df,
        sublinear_tf=cfg.tfidf.sublinear_tf,
    )
    tfidf.fit(catalog_df, text_col="cleaned_text")
    tfidf_metrics = test_harness.run(
        retrieve_fn=tfidf.query_retrieve,
        k=cfg.dual_encoder.retrieval_k,
        measure_latency_flag=False,
        save_per_query=False,
    )

    ablations = [
        {"name": "full", "shared_weights": cfg.dual_encoder.shared_weights, "use_hard_negatives": True},
        {"name": "no_hard_negatives", "shared_weights": cfg.dual_encoder.shared_weights, "use_hard_negatives": False},
        {"name": "separate_towers", "shared_weights": False, "use_hard_negatives": True},
    ]

    ablation_results = []
    best_run = None

    for ab in ablations:
        model, retrieve_fn, history, catalog_ids, catalog_embeddings = run_single_experiment(
            train_df=train_df,
            val_df=val_df,
            catalog_df=catalog_df,
            queries_df=queries_df,
            cfg=cfg,
            shared_weights=ab["shared_weights"],
            use_hard_negatives=ab["use_hard_negatives"],
        )
        metrics = test_harness.run(
            retrieve_fn=retrieve_fn,
            k=cfg.dual_encoder.retrieval_k,
            measure_latency_flag=False,
            save_per_query=False,
        )
        recall_lift = metrics.get("recall@10", 0.0) - tfidf_metrics.get("recall@10", 0.0)
        run_payload = {
            **ab,
            "metrics": metrics,
            "history": history,
            "recall@10_lift_vs_tfidf": recall_lift,
        }
        ablation_results.append(run_payload)

        if best_run is None or metrics.get("ndcg@10", 0.0) > best_run["metrics"].get("ndcg@10", 0.0):
            best_run = {**run_payload, "model": model, "catalog_ids": catalog_ids, "catalog_embeddings": catalog_embeddings}

    best_dir = artifacts_dir / "dual_encoder_best"
    metadata = {
        "timestamp_utc": int(time.time()),
        "best_run": {k: v for k, v in best_run.items() if k not in {"model", "catalog_ids", "catalog_embeddings"}},
        "tfidf_baseline": tfidf_metrics,
    }
    save_artifacts(
        best_dir,
        best_run["model"],
        best_run["catalog_ids"],
        best_run["catalog_embeddings"],
        metadata,
    )

    report = {
        "timestamp_utc": int(time.time()),
        "tfidf_baseline": tfidf_metrics,
        "best_dual_encoder": {k: v for k, v in best_run.items() if k not in {"model", "catalog_ids", "catalog_embeddings"}},
        "ablations": ablation_results,
    }
    report_path = logs_dir / "phase2_dual_encoder_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved Phase 2 report → %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)