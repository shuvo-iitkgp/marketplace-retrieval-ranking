"""
tests/test_phase1.py
─────────────────────
Unit tests for Phase 1:
  - Catalog building and text cleaning
  - Synthetic query generation
  - TF-IDF retriever (fit, retrieve, persist)
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ─────────────────────────────────────────────────────────────────────────── #
# Catalog tests                                                                #
# ─────────────────────────────────────────────────────────────────────────── #


def make_reviews_df():
    return pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4"],
        "product_id": ["p1", "p1", "p2", "p3"],
        "score": [5.0, 4.0, 3.0, 5.0],
        "timestamp": [1000, 1001, 1002, 1003],
        "title": ["Watercolor Paint Set", "Watercolor Paint Set", "Sketch Pencil Kit", "Canvas Board"],
        "summary": ["Great paints", "Nice colors", "Good pencils", ""],
    })


def make_brands_df():
    return pd.DataFrame({
        "product_id": ["p1", "p3"],
        "brand": ["Winsor", "Strathmore"],
    })


class TestBuildRichCatalog:
    def test_returns_expected_columns(self):
        from marketplace_search.data.catalog import build_rich_catalog
        reviews = make_reviews_df()
        brands = make_brands_df()
        catalog = build_rich_catalog(reviews, brands)
        for col in ["product_id", "title", "brand", "cleaned_text", "category_id", "review_count"]:
            assert col in catalog.columns, f"Missing column: {col}"

    def test_brand_join(self):
        from marketplace_search.data.catalog import build_rich_catalog
        reviews = make_reviews_df()
        brands = make_brands_df()
        catalog = build_rich_catalog(reviews, brands)
        p1 = catalog[catalog["product_id"] == "p1"].iloc[0]
        assert p1["brand"] == "Winsor"
        p2 = catalog[catalog["product_id"] == "p2"].iloc[0]
        assert p2["brand"] == "unknown"

    def test_cleaned_text_nonempty(self):
        from marketplace_search.data.catalog import build_rich_catalog
        reviews = make_reviews_df()
        brands = make_brands_df()
        catalog = build_rich_catalog(reviews, brands)
        assert (catalog["cleaned_text"].str.len() > 0).all()

    def test_cleaned_text_lowercase(self):
        from marketplace_search.data.catalog import build_rich_catalog
        reviews = make_reviews_df()
        brands = make_brands_df()
        catalog = build_rich_catalog(reviews, brands)
        for text in catalog["cleaned_text"]:
            assert text == text.lower(), f"Text not lowercase: {text}"

    def test_category_assigned(self):
        from marketplace_search.data.catalog import build_rich_catalog
        reviews = make_reviews_df()
        brands = make_brands_df()
        catalog = build_rich_catalog(reviews, brands)
        # "Watercolor Paint Set" should map to "Paint & Color"
        paint_row = catalog[catalog["product_id"] == "p1"].iloc[0]
        assert paint_row["category_id"] == "Paint & Color"

    def test_deduplication(self):
        from marketplace_search.data.catalog import build_rich_catalog
        reviews = make_reviews_df()
        brands = make_brands_df()
        catalog = build_rich_catalog(reviews, brands)
        # p1 has 2 reviews but should appear once in catalog
        assert len(catalog[catalog["product_id"] == "p1"]) == 1


class TestCleanText:
    def test_lowercase(self):
        from marketplace_search.data.catalog import clean_text
        assert clean_text("Hello WORLD") == "hello world"

    def test_strips_punctuation(self):
        from marketplace_search.data.catalog import clean_text
        result = clean_text("paint, brush!")
        assert "," not in result and "!" not in result

    def test_handles_empty(self):
        from marketplace_search.data.catalog import clean_text
        assert clean_text("") == ""

    def test_collapses_whitespace(self):
        from marketplace_search.data.catalog import clean_text
        assert clean_text("a   b   c") == "a b c"


# ─────────────────────────────────────────────────────────────────────────── #
# Query generation tests                                                       #
# ─────────────────────────────────────────────────────────────────────────── #


def make_catalog_df():
    from marketplace_search.data.catalog import build_rich_catalog
    return build_rich_catalog(make_reviews_df(), make_brands_df())


class TestGenerateQueries:
    def test_title_strategy_one_per_product(self):
        from marketplace_search.data.catalog import generate_queries
        catalog = make_catalog_df()
        queries = generate_queries(catalog, strategy="title")
        # Each product should have exactly one title query
        assert len(queries) == len(catalog)

    def test_title_query_matches_product(self):
        from marketplace_search.data.catalog import generate_queries
        catalog = make_catalog_df()
        queries = generate_queries(catalog, strategy="title")
        # All returned queries must reference a valid product
        catalog_pids = set(catalog["product_id"])
        assert set(queries["product_id"]).issubset(catalog_pids)

    def test_ngram_strategy_produces_multiple(self):
        from marketplace_search.data.catalog import generate_queries
        catalog = make_catalog_df()
        queries = generate_queries(catalog, strategy="ngram", max_queries_per_product=2)
        # Should generally produce more queries than products
        assert len(queries) >= len(catalog)

    def test_ngram_queries_are_shorter_than_title(self):
        from marketplace_search.data.catalog import generate_queries
        catalog = make_catalog_df()
        title_q = generate_queries(catalog, strategy="title")
        ngram_q = generate_queries(catalog, strategy="ngram")
        avg_title_len = title_q["query_text"].str.len().mean()
        avg_ngram_len = ngram_q["query_text"].str.len().mean()
        # n-grams should be shorter on average
        assert avg_ngram_len <= avg_title_len

    def test_invalid_strategy_raises(self):
        from marketplace_search.data.catalog import generate_queries
        catalog = make_catalog_df()
        try:
            generate_queries(catalog, strategy="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_unique_query_ids(self):
        from marketplace_search.data.catalog import generate_queries
        catalog = make_catalog_df()
        queries = generate_queries(catalog, strategy="ngram", max_queries_per_product=3)
        assert queries["query_id"].nunique() == len(queries)


# ─────────────────────────────────────────────────────────────────────────── #
# TF-IDF retriever tests                                                       #
# ─────────────────────────────────────────────────────────────────────────── #


class TestTFIDFRetriever:
    def _make_catalog(self):
        return pd.DataFrame({
            "product_id": ["p1", "p2", "p3", "p4"],
            "cleaned_text": [
                "watercolor paint brushes set",
                "oil painting canvas board",
                "sketch pencil drawing kit",
                "fabric scissors cutting tool",
            ],
        })

    def test_fit_sets_fitted(self):
        from marketplace_search.retrieval.tfidf import TFIDFRetriever
        r = TFIDFRetriever()
        catalog = self._make_catalog()
        r.fit(catalog)
        assert r._is_fitted

    def test_retrieve_returns_list(self):
        from marketplace_search.retrieval.tfidf import TFIDFRetriever
        r = TFIDFRetriever()
        r.fit(self._make_catalog())
        results = r.query_retrieve("watercolor paint", k=10)
        assert isinstance(results, list)

    def test_retrieve_top1_matches_query(self):
        from marketplace_search.retrieval.tfidf import TFIDFRetriever
        r = TFIDFRetriever()
        r.fit(self._make_catalog())
        # "watercolor paint" should rank p1 first
        results = r.query_retrieve("watercolor paint", k=4)
        assert results[0] == "p1"

    def test_retrieve_k_respected(self):
        from marketplace_search.retrieval.tfidf import TFIDFRetriever
        r = TFIDFRetriever()
        r.fit(self._make_catalog())
        for k in [1, 2, 4]:
            results = r.query_retrieve("paint", k=k)
            assert len(results) <= k

    def test_empty_query_returns_empty(self):
        from marketplace_search.retrieval.tfidf import TFIDFRetriever
        r = TFIDFRetriever()
        r.fit(self._make_catalog())
        assert r.query_retrieve("", k=10) == []

    def test_unfitted_raises(self):
        from marketplace_search.retrieval.tfidf import TFIDFRetriever
        r = TFIDFRetriever()
        try:
            r.query_retrieve("paint")
            assert False, "Should raise RuntimeError"
        except RuntimeError:
            pass

    def test_score_candidates(self):
        from marketplace_search.retrieval.tfidf import TFIDFRetriever
        r = TFIDFRetriever()
        r.fit(self._make_catalog())
        scores = r.score_candidates("watercolor paint", ["p1", "p2"])
        assert "p1" in scores and "p2" in scores
        # p1 should score higher than p2 for "watercolor paint"
        assert scores["p1"] > scores["p2"]

    def test_save_load_roundtrip(self):
        from marketplace_search.retrieval.tfidf import TFIDFRetriever
        r = TFIDFRetriever()
        r.fit(self._make_catalog())
        original_results = r.query_retrieve("watercolor paint", k=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tfidf.pkl"
            r.save(path)
            r2 = TFIDFRetriever.load(path)
            loaded_results = r2.query_retrieve("watercolor paint", k=4)

        assert original_results == loaded_results

    def test_index_memory_positive(self):
        from marketplace_search.retrieval.tfidf import TFIDFRetriever
        r = TFIDFRetriever()
        r.fit(self._make_catalog())
        assert r.index_memory_mb() > 0

    def test_user_retrieve_fn(self):
        from marketplace_search.retrieval.tfidf import TFIDFRetriever
        r = TFIDFRetriever()
        catalog = self._make_catalog()
        r.fit(catalog)

        train_df = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "product_id": ["p1", "p2"],
            "label": [1, 1],
        })
        retrieve_fn = r.build_user_query_fn(train_df, catalog, k=10)
        results = retrieve_fn("u1")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_user_retrieve_unknown_user_returns_empty(self):
        from marketplace_search.retrieval.tfidf import TFIDFRetriever
        r = TFIDFRetriever()
        r.fit(self._make_catalog())
        train_df = pd.DataFrame({"user_id": [], "product_id": [], "label": []})
        retrieve_fn = r.build_user_query_fn(train_df, self._make_catalog(), k=10)
        assert retrieve_fn("unknown_user") == []


if __name__ == "__main__":
    # Run all test classes manually (no pytest required)
    import traceback

    test_classes = [
        TestBuildRichCatalog,
        TestCleanText,
        TestGenerateQueries,
        TestTFIDFRetriever,
    ]

    total = passed = failed = 0
    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for method in methods:
            total += 1
            try:
                getattr(instance, method)()
                passed += 1
                print(f"  PASS  {cls.__name__}.{method}")
            except Exception as e:
                failed += 1
                print(f"  FAIL  {cls.__name__}.{method}")
                traceback.print_exc()

    print(f"\n{'=' * 50}")
    print(f"  {passed}/{total} tests passed, {failed} failed")
    print("=" * 50)
    if failed:
        sys.exit(1)
