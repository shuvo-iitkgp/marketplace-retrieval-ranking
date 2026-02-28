from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from marketplace_search.common.config import load_config
from marketplace_search.eval.metrics import ndcg_at_k
from marketplace_search.retrieval.dual_encoder import (
    DualEncoderModel,
    _require_torch_transformers,
    encode_corpus,
)
from marketplace_search.retrieval.faiss_retrieval import l2_normalize
from marketplace_search.retrieval.personalization import (
    build_category_priors,
    build_user_embeddings,
    rank_products_for_embeddings,
    tune_alpha,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("phase4_personalization")


def load_phase2_model(artifacts_dir: Path, cfg):
    torch, _, AutoTokenizer = _require_torch_transformers()

    model_cfg = cfg.dual_encoder
    model = DualEncoderModel(model_name=model_cfg.model_name, shared_weights=model_cfg.shared_weights)
    state = torch.load(artifacts_dir / "dual_encoder_state.pt", map_location="cpu")
    model.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    return model, tokenizer


def encode_queries(model, tokenizer, texts: list[str], cfg, device: str) -> np.ndarray:
    return l2_normalize(
        encode_corpus(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            batch_size=cfg.phase4.batch_size,
            max_length=cfg.dual_encoder.max_length,
        )
    )


def make_eval_df(interactions_df: pd.DataFrame, queries_df: pd.DataFrame) -> pd.DataFrame:
    positives = interactions_df[interactions_df["label"] == 1][["user_id", "product_id", "timestamp"]].copy()
    query_lookup = queries_df[["product_id", "query_text"]].drop_duplicates("product_id")
    eval_df = positives.merge(query_lookup, on="product_id", how="inner")
    return eval_df


def evaluate_split(
    eval_df: pd.DataFrame,
    query_embeddings: np.ndarray,
    catalog_ids: list[str],
    catalog_embeddings: np.ndarray,
    user_embeddings: dict[str, np.ndarray],
    category_priors,
    alpha: float,
    k: int,
) -> float:
    retrieved = rank_products_for_embeddings(
        query_embeddings=query_embeddings,
        user_ids=eval_df["user_id"].astype(str).tolist(),
        catalog_ids=catalog_ids,
        catalog_embeddings=catalog_embeddings,
        user_embeddings=user_embeddings,
        category_priors=category_priors,
        alpha=alpha,
        k=k,
    )
    relevant = [[pid] for pid in eval_df["product_id"].tolist()]
    return ndcg_at_k(relevant_ids=relevant, retrieved_ids=retrieved, k=k)


def main(config_path: str) -> None:
    cfg = load_config(config_path, project_root=PROJECT_ROOT)

    processed_dir = PROJECT_ROOT / cfg.paths.processed_dir
    artifacts_dir = PROJECT_ROOT / cfg.paths.artifacts_dir
    logs_dir = PROJECT_ROOT / cfg.paths.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    phase2_best = artifacts_dir / "dual_encoder_best"
    if not phase2_best.exists():
        raise FileNotFoundError("Phase 2 artifacts not found. Run scripts/05_train_eval_dual_encoder.py first.")

    catalog_df = pd.read_csv(processed_dir / "rich_catalog.csv.gz")
    train_df = pd.read_csv(processed_dir / "train.csv.gz")
    val_df = pd.read_csv(processed_dir / "val.csv.gz")
    test_df = pd.read_csv(processed_dir / "test.csv.gz")
    queries_df = pd.read_csv(processed_dir / "queries_title.csv.gz")

    catalog_ids = catalog_df["product_id"].tolist()
    catalog_texts = catalog_df["cleaned_text"].fillna("").tolist()

    embeddings_path = artifacts_dir / cfg.phase4.catalog_embeddings_filename
    torch, _, _ = _require_torch_transformers()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if embeddings_path.exists():
        catalog_embeddings = np.load(embeddings_path).astype(np.float32)
        logger.info("Loaded precomputed catalog embeddings from %s", embeddings_path)
    else:
        model, tokenizer = load_phase2_model(phase2_best, cfg)
        model.to(device)
        catalog_embeddings = l2_normalize(
            encode_corpus(
                model=model,
                tokenizer=tokenizer,
                texts=catalog_texts,
                device=device,
                batch_size=cfg.phase4.batch_size,
                max_length=cfg.dual_encoder.max_length,
            )
        ).astype(np.float32)
        np.save(embeddings_path, catalog_embeddings)
        logger.info("Saved catalog embeddings -> %s", embeddings_path)

    model, tokenizer = load_phase2_model(phase2_best, cfg)
    model.to(device)

    product_embedding_map = dict(zip(catalog_ids, catalog_embeddings))#, strict=True))
    category_priors = build_category_priors(catalog_df=catalog_df, product_embeddings=product_embedding_map)

    val_eval_df = make_eval_df(val_df, queries_df)
    test_eval_df = make_eval_df(test_df, queries_df)

    val_query_embeddings = encode_queries(model, tokenizer, val_eval_df["query_text"].fillna("").tolist(), cfg, device)
    test_query_embeddings = encode_queries(model, tokenizer, test_eval_df["query_text"].fillna("").tolist(), cfg, device)

    val_user_embeddings = build_user_embeddings(
        interactions_df=train_df,
        product_embeddings=product_embedding_map,
        decay_lambda=cfg.phase4.decay_lambda,
    )
    test_user_embeddings = build_user_embeddings(
        interactions_df=pd.concat([train_df, val_df], ignore_index=True),
        product_embeddings=product_embedding_map,
        decay_lambda=cfg.phase4.decay_lambda,
    )

    alpha_grid = [float(a) for a in cfg.phase4.alpha_grid]
    alpha_result = tune_alpha(
        candidate_alphas=alpha_grid,
        query_embeddings=val_query_embeddings,
        user_ids=val_eval_df["user_id"].astype(str).tolist(),
        ground_truth_product_ids=val_eval_df["product_id"].tolist(),
        catalog_ids=catalog_ids,
        catalog_embeddings=catalog_embeddings,
        user_embeddings=val_user_embeddings,
        category_priors=category_priors,
        k=cfg.phase4.ndcg_k,
    )

    baseline_ndcg_test = evaluate_split(
        eval_df=test_eval_df,
        query_embeddings=test_query_embeddings,
        catalog_ids=catalog_ids,
        catalog_embeddings=catalog_embeddings,
        user_embeddings=test_user_embeddings,
        category_priors=category_priors,
        alpha=1.0,
        k=cfg.phase4.ndcg_k,
    )
    personalized_ndcg_test = evaluate_split(
        eval_df=test_eval_df,
        query_embeddings=test_query_embeddings,
        catalog_ids=catalog_ids,
        catalog_embeddings=catalog_embeddings,
        user_embeddings=test_user_embeddings,
        category_priors=category_priors,
        alpha=alpha_result.alpha,
        k=cfg.phase4.ndcg_k,
    )

    train_positive_counts = (
        train_df[train_df["label"] == 1].groupby("user_id").size().to_dict()
    )
    repeat_threshold = int(cfg.evaluation.repeat_user_min_interactions)
    test_is_repeat = test_eval_df["user_id"].map(lambda u: train_positive_counts.get(u, 0) >= repeat_threshold)

    repeat_df = test_eval_df[test_is_repeat].reset_index(drop=True)
    repeat_query_embeddings = test_query_embeddings[test_is_repeat.to_numpy()]
    cold_df = test_eval_df[~test_is_repeat].reset_index(drop=True)
    cold_query_embeddings = test_query_embeddings[(~test_is_repeat).to_numpy()]

    repeat_baseline = evaluate_split(
        eval_df=repeat_df,
        query_embeddings=repeat_query_embeddings,
        catalog_ids=catalog_ids,
        catalog_embeddings=catalog_embeddings,
        user_embeddings=test_user_embeddings,
        category_priors=category_priors,
        alpha=1.0,
        k=cfg.phase4.ndcg_k,
    ) if not repeat_df.empty else 0.0
    repeat_personalized = evaluate_split(
        eval_df=repeat_df,
        query_embeddings=repeat_query_embeddings,
        catalog_ids=catalog_ids,
        catalog_embeddings=catalog_embeddings,
        user_embeddings=test_user_embeddings,
        category_priors=category_priors,
        alpha=alpha_result.alpha,
        k=cfg.phase4.ndcg_k,
    ) if not repeat_df.empty else 0.0

    cold_baseline = evaluate_split(
        eval_df=cold_df,
        query_embeddings=cold_query_embeddings,
        catalog_ids=catalog_ids,
        catalog_embeddings=catalog_embeddings,
        user_embeddings=test_user_embeddings,
        category_priors=category_priors,
        alpha=1.0,
        k=cfg.phase4.ndcg_k,
    ) if not cold_df.empty else 0.0
    cold_personalized = evaluate_split(
        eval_df=cold_df,
        query_embeddings=cold_query_embeddings,
        catalog_ids=catalog_ids,
        catalog_embeddings=catalog_embeddings,
        user_embeddings=test_user_embeddings,
        category_priors=category_priors,
        alpha=alpha_result.alpha,
        k=cfg.phase4.ndcg_k,
    ) if not cold_df.empty else 0.0

    lift_pct = ((personalized_ndcg_test - baseline_ndcg_test) / max(baseline_ndcg_test, 1e-12)) * 100.0

    report = {
        "timestamp_utc": int(time.time()),
        "phase": "phase04_personalization",
        "objective": "personalized_query_embedding",
        "params": {
            "decay_lambda": float(cfg.phase4.decay_lambda),
            "alpha_grid": alpha_grid,
            "best_alpha": float(alpha_result.alpha),
            "ndcg_k": int(cfg.phase4.ndcg_k),
        },
        "validation": {
            "best_alpha": float(alpha_result.alpha),
            "ndcg@10": float(alpha_result.ndcg_at_10),
            "num_examples": int(len(val_eval_df)),
        },
        "test": {
            "baseline_ndcg@10": float(baseline_ndcg_test),
            "personalized_ndcg@10": float(personalized_ndcg_test),
            "lift_percent": float(lift_pct),
            "num_examples": int(len(test_eval_df)),
        },
        "segmentation": {
            "repeat_users": {
                "num_examples": int(len(repeat_df)),
                "baseline_ndcg@10": float(repeat_baseline),
                "personalized_ndcg@10": float(repeat_personalized),
                "lift_percent": float(((repeat_personalized - repeat_baseline) / max(repeat_baseline, 1e-12)) * 100.0),
            },
            "cold_start_users": {
                "num_examples": int(len(cold_df)),
                "baseline_ndcg@10": float(cold_baseline),
                "personalized_ndcg@10": float(cold_personalized),
                "lift_percent": float(((cold_personalized - cold_baseline) / max(cold_baseline, 1e-12)) * 100.0),
            },
        },
    }

    report_path = logs_dir / "phase4_personalization_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved phase 4 report -> %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)