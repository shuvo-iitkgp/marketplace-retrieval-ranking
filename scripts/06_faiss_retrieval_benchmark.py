from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from marketplace_search.common.config import load_config
from marketplace_search.retrieval.dual_encoder import (
    DualEncoderModel,
    _require_torch_transformers,
    encode_corpus,
)
from marketplace_search.retrieval.faiss_retrieval import (
    FaissIndexConfig,
    FaissRetriever,
    benchmark_retriever,
    compute_expected_topk_ids,
    l2_normalize,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("phase3_faiss")


def load_phase2_model(artifacts_dir: Path, cfg):
    torch, _, AutoTokenizer = _require_torch_transformers()

    model_cfg = cfg.dual_encoder
    model = DualEncoderModel(model_name=model_cfg.model_name, shared_weights=model_cfg.shared_weights)
    state = torch.load(artifacts_dir / "dual_encoder_state.pt", map_location="cpu")
    model.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    return model, tokenizer


def make_query_embeddings(model, tokenizer, query_texts: list[str], cfg, device: str) -> np.ndarray:
    arr = encode_corpus(
        model=model,
        tokenizer=tokenizer,
        texts=query_texts,
        device=device,
        batch_size=cfg.phase3.batch_size,
        max_length=cfg.dual_encoder.max_length,
    )
    return l2_normalize(arr)


def main(config_path: str) -> None:
    cfg = load_config(config_path, project_root=PROJECT_ROOT)

    processed_dir = PROJECT_ROOT / cfg.paths.processed_dir
    artifacts_dir = PROJECT_ROOT / cfg.paths.artifacts_dir
    logs_dir = PROJECT_ROOT / cfg.paths.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    phase2_best = artifacts_dir / "dual_encoder_best"
    if not phase2_best.exists():
        raise FileNotFoundError(
            "Phase 2 artifacts not found. Run scripts/05_train_eval_dual_encoder.py first."
        )

    catalog_df = pd.read_csv(processed_dir / "rich_catalog.csv.gz")
    test_df = pd.read_csv(processed_dir / "test.csv.gz")
    queries_df = pd.read_csv(processed_dir / "queries_title.csv.gz")
    test_queries = test_df[test_df["label"] == 1][["product_id"]].drop_duplicates().merge(
        queries_df[["product_id", "query_text"]], on="product_id", how="inner"
    )

    product_ids = catalog_df["product_id"].tolist()
    catalog_texts = catalog_df["cleaned_text"].fillna("").tolist()
    query_texts = test_queries["query_text"].fillna("").tolist()

    torch, _, _ = _require_torch_transformers()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_phase2_model(phase2_best, cfg)
    model.to(device)

    logger.info("Encoding full catalog (%d products) via batch inference...", len(catalog_texts))
    t0 = time.perf_counter()
    catalog_embeddings = encode_corpus(
        model=model,
        tokenizer=tokenizer,
        texts=catalog_texts,
        device=device,
        batch_size=cfg.phase3.batch_size,
        max_length=cfg.dual_encoder.max_length,
    )
    catalog_embeddings = l2_normalize(catalog_embeddings).astype(np.float32)
    catalog_path = artifacts_dir / "phase3_catalog_embeddings.npy"
    np.save(catalog_path, catalog_embeddings)
    embedding_elapsed = time.perf_counter() - t0

    query_embeddings = make_query_embeddings(model, tokenizer, query_texts, cfg, device)

    exact_cfg = FaissIndexConfig(index_type="flatip", metric="ip")
    exact = FaissRetriever(product_ids=product_ids, config=exact_cfg)
    exact.fit(catalog_embeddings)

    expected_topk = compute_expected_topk_ids(
        exact_retriever=exact,
        query_embeddings=query_embeddings,
        k=cfg.phase3.retrieval_k,
        batch_size=cfg.phase3.query_batch_size,
    )
    exact_metrics = benchmark_retriever(
        retriever=exact,
        query_embeddings=query_embeddings,
        expected_topk=expected_topk,
        k=cfg.phase3.retrieval_k,
        batch_size=cfg.phase3.query_batch_size,
    )

    ann_results = []
    ann_index_type = str(cfg.phase3.ann_index_type).lower()
    if ann_index_type == "ivf_flat":
        trial_nlists = list(cfg.phase3.ivf_nlist_options)
    else:
        trial_nlists = [None]

    for nlist in trial_nlists:
        ann_cfg = FaissIndexConfig(
            index_type=ann_index_type,
            metric="ip",
            nlist=nlist or cfg.phase3.ivf_nlist_options[0],
            nprobe=cfg.phase3.ivf_nprobe,
            hnsw_m=cfg.phase3.hnsw_m,
            ef_search=cfg.phase3.hnsw_ef_search,
        )
        ann = FaissRetriever(product_ids=product_ids, config=ann_cfg)
        ann.fit(catalog_embeddings)
        metrics = benchmark_retriever(
            retriever=ann,
            query_embeddings=query_embeddings,
            expected_topk=expected_topk,
            k=cfg.phase3.retrieval_k,
            batch_size=cfg.phase3.query_batch_size,
        )
        recall_drop = max(0.0, exact_metrics.recall_at_k - metrics.recall_at_k)
        ann_results.append(
            {
                "index_type": ann_index_type,
                "nlist": int(nlist) if nlist is not None else None,
                "nprobe": int(cfg.phase3.ivf_nprobe),
                "hnsw_m": int(cfg.phase3.hnsw_m),
                "hnsw_ef_search": int(cfg.phase3.hnsw_ef_search),
                "recall_at_k": metrics.recall_at_k,
                "recall_drop_vs_exact": recall_drop,
                "p50_ms": metrics.p50_ms,
                "p95_ms": metrics.p95_ms,
                "mean_ms": metrics.mean_ms,
                "meets_latency_target": metrics.p95_ms < cfg.phase3.target_p95_latency_ms,
                "meets_recall_target": recall_drop < cfg.phase3.max_recall_drop,
            }
        )

    best_ann = min(
        ann_results,
        key=lambda x: (
            not (x["meets_latency_target"] and x["meets_recall_target"]),
            x["p95_ms"],
            x["recall_drop_vs_exact"],
        ),
    )

    profile_report = {
        "timestamp_utc": int(time.time()),
        "phase": "phase03_faiss_retrieval",
        "catalog_size": len(product_ids),
        "num_eval_queries": len(query_texts),
        "embedding_generation_seconds": embedding_elapsed,
        "saved_catalog_embeddings": str(catalog_path.relative_to(PROJECT_ROOT)),
        "exact_index": {
            "index_type": "flatip",
            "recall_at_k": exact_metrics.recall_at_k,
            "p50_ms": exact_metrics.p50_ms,
            "p95_ms": exact_metrics.p95_ms,
            "mean_ms": exact_metrics.mean_ms,
        },
        "ann_trials": ann_results,
        "best_ann": best_ann,
        "targets": {
            "p95_latency_ms_lt": cfg.phase3.target_p95_latency_ms,
            "recall_drop_lt": cfg.phase3.max_recall_drop,
        },
    }

    report_path = logs_dir / "phase3_faiss_profile_report.json"
    with report_path.open("w") as f:
        json.dump(profile_report, f, indent=2)

    logger.info("Saved profiling report -> %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)