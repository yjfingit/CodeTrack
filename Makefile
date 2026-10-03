.PHONY: help install install-dev lint format test test-cov smoke clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## install runtime deps + package (editable)
	pip install -r requirements.txt && pip install -e .

install-dev: ## install dev deps + pre-commit hooks
	pip install -r requirements-dev.txt && pip install -e . && pre-commit install

lint: ## run ruff lint
	ruff check codetrack tests

format: ## run ruff format
	ruff format codetrack tests

test: ## run test suite
	pytest -q

test-cov: ## run tests with coverage
	pytest -q --cov=codetrack --cov-report=term-missing

smoke: ## run the minimal end-to-end smoke test
	bash tools/smoke_test.sh

clean: ## remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
