# Makefile — Marketplace Search & Personalization Engine
# Phase 0 targets

.PHONY: all install format test build-data eval-smoke clean

# ── Install ──────────────────────────────────────────────────────────────── #
install:
	pip install -e ".[dev]"

# ── Code quality ─────────────────────────────────────────────────────────── #
format:
	ruff check src/ scripts/ tests/ --fix
	ruff format src/ scripts/ tests/

lint:
	ruff check src/ scripts/ tests/

# ── Tests ────────────────────────────────────────────────────────────────── #
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=src/marketplace_search --cov-report=term-missing

# ── Phase 0 pipeline ─────────────────────────────────────────────────────── #

## Build all dataset artifacts (ingest → label → split → save)
build-data:
	python scripts/01_build_dataset.py --config configs/default.yaml

## Run evaluation harness smoke test (popularity baseline)
eval-smoke:
	python scripts/02_eval_smoke_test.py --config configs/default.yaml

## Run full Phase 0 pipeline
phase0: build-data eval-smoke
	@echo "✓ Phase 0 complete. Check data/logs/ for results."

# ── Cleanup ──────────────────────────────────────────────────────────────── #
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache

clean-data:
	rm -rf data/processed/ data/artifacts/ data/logs/

# ── Help ─────────────────────────────────────────────────────────────────── #
help:
	@echo ""
	@echo "  Phase 0 Targets:"
	@echo "  ────────────────────────────────────────"
	@echo "  install      Install package + dev deps"
	@echo "  format       Auto-format code (ruff)"
	@echo "  lint         Lint check (ruff)"
	@echo "  test         Run unit tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  build-data   Ingest, label, split, save"
	@echo "  eval-smoke   Popularity baseline eval"
	@echo "  phase0       Run full Phase 0 pipeline"
	@echo "  clean        Remove Python cache files"
	@echo ""
