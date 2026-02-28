from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_search.common.config import load_config
from marketplace_search.eval.metrics import mean_reciprocal_rank, ndcg_at_k
from marketplace_search.retrieval.dual_encoder import (
    DualEncoderModel,
    _require_torch_transformers,
    encode_corpus,
)
from marketplace_search.retrieval.faiss_retrieval import FaissIndexConfig, FaissRetriever, l2_normalize
from marketplace_search.retrieval.reranker import (
    add_reranker_features,
    build_candidate_frame,
    build_product_feature_table,
    build_user_category_affinity,
    compute_permutation_feature_importance,
    evaluate_rankings,
    fetch_faiss_candidates_with_scores,
    resolve_category_column,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("phase5_reranker")


def load_phase2_model(artifacts_dir: Path, cfg):
    torch, _, AutoTokenizer = _require_torch_transformers()
    model_cfg = cfg.dual_encoder
    model = DualEncoderModel(model_name=model_cfg.model_name, shared_weights=model_cfg.shared_weights)
    state = torch.load(artifacts_dir / "dual_encoder_state.pt", map_location="cpu")
    model.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    return model, tokenizer


def build_eval_queries(interactions_df: pd.DataFrame, queries_df: pd.DataFrame, catalog_df: pd.DataFrame) -> pd.DataFrame:
    positives = interactions_df[interactions_df["label"] == 1][["user_id", "product_id"]].copy()
    query_lookup = queries_df[["product_id", "query_text"]].drop_duplicates("product_id")
    category_col = resolve_category_column(catalog_df)
    category_lookup = catalog_df[["product_id", category_col]].drop_duplicates("product_id").rename(
        columns={category_col: "category"}
    )
    return (
        positives.merge(query_lookup, on="product_id", how="inner")
        .merge(category_lookup, on="product_id", how="left")
        .fillna({"category": "unknown"})
        .reset_index(drop=True)
    )


def encode_queries(model, tokenizer, texts: list[str], cfg, device: str) -> np.ndarray:
    return l2_normalize(
        encode_corpus(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            batch_size=cfg.phase5.batch_size,
            max_length=cfg.dual_encoder.max_length,
        )
    )


def main(config_path: str) -> None:
    cfg = load_config(config_path, project_root=PROJECT_ROOT)
    processed_dir = PROJECT_ROOT / cfg.paths.processed_dir
    artifacts_dir = PROJECT_ROOT / cfg.paths.artifacts_dir
    logs_dir = PROJECT_ROOT / cfg.paths.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    phase2_best = artifacts_dir / "dual_encoder_best"
    if not phase2_best.exists():
        raise FileNotFoundError("Phase 2 artifacts not found. Run scripts/05_train_eval_dual_encoder.py first.")

    train_df = pd.read_csv(processed_dir / "train.csv.gz")
    val_df = pd.read_csv(processed_dir / "val.csv.gz")
    test_df = pd.read_csv(processed_dir / "test.csv.gz")
    catalog_df = pd.read_csv(processed_dir / "rich_catalog.csv.gz")
    queries_df = pd.read_csv(processed_dir / "queries_title.csv.gz")

    catalog_ids = catalog_df["product_id"].astype(str).tolist()
    catalog_texts = catalog_df["cleaned_text"].fillna("").tolist()

    embeddings_path = artifacts_dir / cfg.phase5.catalog_embeddings_filename
    torch, _, _ = _require_torch_transformers()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if embeddings_path.exists():
        catalog_embeddings = np.load(embeddings_path).astype(np.float32)
        logger.info("Loaded catalog embeddings from %s", embeddings_path)
    else:
        model, tokenizer = load_phase2_model(phase2_best, cfg)
        model.to(device)
        catalog_embeddings = l2_normalize(
            encode_corpus(
                model=model,
                tokenizer=tokenizer,
                texts=catalog_texts,
                device=device,
                batch_size=cfg.phase5.batch_size,
                max_length=cfg.dual_encoder.max_length,
            )
        ).astype(np.float32)
        np.save(embeddings_path, catalog_embeddings)
        logger.info("Saved catalog embeddings -> %s", embeddings_path)

    model, tokenizer = load_phase2_model(phase2_best, cfg)
    model.to(device)

    retriever = FaissRetriever(
        product_ids=catalog_ids,
        config=FaissIndexConfig(index_type="flatip", metric="ip"),
    )
    retriever.fit(catalog_embeddings)

    product_feature_table = build_product_feature_table(
        interactions_df=train_df,
        catalog_df=catalog_df,
        ctr_smoothing=float(cfg.phase5.ctr_smoothing),
    )
    user_cat_affinity = build_user_category_affinity(
        interactions_df=train_df,
        catalog_df=catalog_df,
        smoothing=float(cfg.phase5.user_affinity_smoothing),
    )

    split_queries = {
        "train": build_eval_queries(train_df, queries_df, catalog_df),
        "val": build_eval_queries(val_df, queries_df, catalog_df),
        "test": build_eval_queries(test_df, queries_df, catalog_df),
    }

    split_candidates: dict[str, pd.DataFrame] = {}
    for split_name, eval_df in split_queries.items():
        query_embeddings = encode_queries(
            model,
            tokenizer,
            eval_df["query_text"].fillna("").tolist(),
            cfg,
            device,
        )
        cand_ids, cand_scores = fetch_faiss_candidates_with_scores(
            retriever,
            query_embeddings,
            k=int(cfg.phase5.candidate_k),
        )
        candidate_df = build_candidate_frame(eval_df, cand_ids, cand_scores)
        candidate_df, feature_cols = add_reranker_features(candidate_df, product_feature_table, user_cat_affinity)
        split_candidates[split_name] = candidate_df

    x_train = split_candidates["train"][feature_cols].to_numpy(dtype=np.float32)
    y_train = split_candidates["train"]["label"].to_numpy(dtype=np.int32)
    x_val = split_candidates["val"][feature_cols].to_numpy(dtype=np.float32)
    y_val = split_candidates["val"]["label"].to_numpy(dtype=np.int32)

    model_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(int(cfg.phase5.hidden_dim),),
                    activation="relu",
                    alpha=float(cfg.phase5.weight_decay),
                    learning_rate_init=float(cfg.phase5.learning_rate),
                    max_iter=int(cfg.phase5.max_iter),
                    early_stopping=True,
                    random_state=int(cfg.project.seed),
                ),
            ),
        ]
    )
    model_pipe.fit(x_train, y_train)

    for split_name, frame in split_candidates.items():
        probs = model_pipe.predict_proba(frame[feature_cols].to_numpy(dtype=np.float32))[:, 1]
        split_candidates[split_name]["rerank_probability"] = probs

    val_metrics = evaluate_rankings(
        split_candidates["val"],
        baseline_score_col="similarity_score",
        rerank_score_col="rerank_probability",
        ndcg_fn=ndcg_at_k,
        mrr_fn=mean_reciprocal_rank,
        k=int(cfg.phase5.eval_k),
    )
    test_metrics = evaluate_rankings(
        split_candidates["test"],
        baseline_score_col="similarity_score",
        rerank_score_col="rerank_probability",
        ndcg_fn=ndcg_at_k,
        mrr_fn=mean_reciprocal_rank,
        k=int(cfg.phase5.eval_k),
    )

    importances = compute_permutation_feature_importance(
        model=model_pipe,
        x=x_val,
        y=y_val,
        feature_names=feature_cols,
        seed=int(cfg.project.seed),
    )

    ndcg_lift_pct = (
        (test_metrics["reranked_ndcg@10"] - test_metrics["baseline_ndcg@10"])
        / max(test_metrics["baseline_ndcg@10"], 1e-12)
    ) * 100.0
    mrr_lift_pct = (
        (test_metrics["reranked_mrr"] - test_metrics["baseline_mrr"])
        / max(test_metrics["baseline_mrr"], 1e-12)
    ) * 100.0

    report = {
        "timestamp_utc": int(time.time()),
        "phase": "phase05_two_stage_reranking",
        "candidate_retrieval": "faiss_top200",
        "model": "mlp_classifier",
        "feature_columns": feature_cols,
        "params": {
            "candidate_k": int(cfg.phase5.candidate_k),
            "eval_k": int(cfg.phase5.eval_k),
            "hidden_dim": int(cfg.phase5.hidden_dim),
            "max_iter": int(cfg.phase5.max_iter),
            "learning_rate": float(cfg.phase5.learning_rate),
        },
        "validation": val_metrics,
        "test": {
            **test_metrics,
            "ndcg_lift_percent": float(ndcg_lift_pct),
            "mrr_lift_percent": float(mrr_lift_pct),
        },
        "feature_importance": importances,
    }

    report_path = logs_dir / "phase5_two_stage_reranker_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved phase 5 report -> %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)