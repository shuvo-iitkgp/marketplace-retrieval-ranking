"""
scripts/03_build_catalog_queries.py
────────────────────────────────────
Phase 1 — Build rich product catalog and synthetic query dataset.

Steps:
1. Load raw reviews (reuse Phase 0 ingest)
2. Build rich catalog with cleaned_text (title + brand + summary)
3. Generate synthetic queries (title strategy + ngram strategy)
4. Save catalog and query sets to data/processed/

Run:
    python scripts/03_build_catalog_queries.py --config configs/default.yaml
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
from marketplace_search.data.ingest import load_brands, load_reviews
from marketplace_search.data.catalog import build_rich_catalog, generate_queries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_catalog_queries")


def main(config_path: str) -> None:
    cfg = load_config(config_path, project_root=PROJECT_ROOT)
    t0 = time.time()

    processed_dir = PROJECT_ROOT / cfg.paths.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw data ───────────────────────────────────────────── #
    logger.info("Loading raw reviews...")
    reviews_raw = load_reviews(PROJECT_ROOT / cfg.paths.raw_reviews)

    # Add summary column from review text (use first 10 words of review/text as proxy)
    # The raw file has review/text — we re-parse to get it
    logger.info("Extracting review summaries for richer text...")
    reviews_raw = _add_summary_from_raw(PROJECT_ROOT / cfg.paths.raw_reviews, reviews_raw)

    logger.info("Loading brands...")
    brands = load_brands(PROJECT_ROOT / cfg.paths.raw_brands)

    # ── 2. Rich catalog ────────────────────────────────────────────── #
    logger.info("Building rich catalog...")
    catalog_cfg = cfg.catalog
    catalog = build_rich_catalog(
        reviews_raw,
        brands,
        max_summary_words=catalog_cfg.max_summary_words,
    )

    catalog_path = processed_dir / "rich_catalog.csv.gz"
    catalog.to_csv(catalog_path, index=False)
    logger.info("Rich catalog saved → %s (%d products)", catalog_path, len(catalog))

    # ── 3. Synthetic queries — title strategy ─────────────────────── #
    q_cfg = cfg.queries
    logger.info("Generating title-strategy queries...")
    queries_title = generate_queries(
        catalog,
        strategy="title",
        seed=q_cfg.seed,
    )
    title_path = processed_dir / "queries_title.csv.gz"
    queries_title.to_csv(title_path, index=False)
    logger.info("Title queries saved → %s (%d rows)", title_path, len(queries_title))

    # ── 4. Synthetic queries — ngram strategy ─────────────────────── #
    logger.info("Generating ngram-strategy queries...")
    queries_ngram = generate_queries(
        catalog,
        strategy="ngram",
        ngram_sizes=q_cfg.ngram_sizes,
        max_queries_per_product=q_cfg.max_queries_per_product,
        seed=q_cfg.seed,
    )
    ngram_path = processed_dir / "queries_ngram.csv.gz"
    queries_ngram.to_csv(ngram_path, index=False)
    logger.info("Ngram queries saved → %s (%d rows)", ngram_path, len(queries_ngram))

    # ── 5. Stats ───────────────────────────────────────────────────── #
    stats = {
        "timestamp_utc": int(time.time()),
        "catalog_products": int(len(catalog)),
        "catalog_categories": int(catalog["category_id"].nunique()),
        "catalog_brands": int((catalog["brand"] != "unknown").sum()),
        "queries_title": int(len(queries_title)),
        "queries_ngram": int(len(queries_ngram)),
        "avg_text_length": float(catalog["cleaned_text"].str.len().mean()),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    stats_path = PROJECT_ROOT / cfg.paths.logs_dir / "catalog_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────── #
    logger.info("\n" + "=" * 58)
    logger.info("  Catalog & Query Build Complete")
    logger.info("=" * 58)
    logger.info("  Catalog products:   %d", stats["catalog_products"])
    logger.info("  Categories:         %d", stats["catalog_categories"])
    logger.info("  Products w/ brand:  %d", stats["catalog_brands"])
    logger.info("  Title queries:      %d", stats["queries_title"])
    logger.info("  Ngram queries:      %d", stats["queries_ngram"])
    logger.info("  Avg text length:    %.0f chars", stats["avg_text_length"])
    logger.info("  Elapsed:            %.1fs", stats["elapsed_seconds"])
    logger.info("=" * 58)


def _add_summary_from_raw(filepath, reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-parse the raw file to extract review/summary field.
    Merges back onto reviews_df by (user_id, product_id).
    """
    import gzip
    from pathlib import Path as P

    filepath = P(filepath)
    open_fn = gzip.open if filepath.suffix == ".gz" else open

    summaries = []
    current: dict = {}

    with open_fn(filepath, "rt", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                if current:
                    summaries.append({
                        "user_id": current.get("review/userId", ""),
                        "product_id": current.get("product/productId", ""),
                        "summary": current.get("review/summary", ""),
                    })
                    current = {}
                continue
            if ": " in line:
                key, _, value = line.partition(": ")
                current[key.strip()] = value.strip()
        if current:
            summaries.append({
                "user_id": current.get("review/userId", ""),
                "product_id": current.get("product/productId", ""),
                "summary": current.get("review/summary", ""),
            })

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df[summary_df["user_id"] != ""]

    merged = reviews_df.merge(
        summary_df[["user_id", "product_id", "summary"]],
        on=["user_id", "product_id"],
        how="left",
    )
    merged["summary"] = merged["summary"].fillna("")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
