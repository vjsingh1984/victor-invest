.PHONY: help install dev-install test test-cov lint format type-check clean run analyze status cache-clean

.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
RESET := \033[0m

help: ## Show this help message
	@echo "$(CYAN)InvestiGator - Investment Research Platform$(RESET)"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

install: ## Install package in editable mode
	pip install -e .

dev-install: ## Install package with development dependencies
	pip install -e ".[dev,viz,jupyter]"

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src/investigator --cov-report=html --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest tests/ -v -m unit

test-integration: ## Run integration tests only
	pytest tests/ -v -m integration

test-fast: ## Run tests excluding slow tests
	pytest tests/ -v -m "not slow"

lint: ## Run linters (flake8)
	flake8 src/investigator/ --max-line-length=120 --exclude=__pycache__

format: ## Format code with black and isort
	black src/investigator/ tests/
	isort src/investigator/ tests/

format-check: ## Check code formatting without making changes
	black --check src/investigator/ tests/
	isort --check src/investigator/ tests/

type-check: ## Run type checking with mypy
	mypy src/investigator/

clean: ## Clean build artifacts and cache files
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .coverage htmlcov/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean ## Clean everything including data caches
	rm -rf data/sec_cache/* data/llm_cache/* data/technical_cache/* data/market_context_cache/*
	rm -rf artifacts/results/* artifacts/metrics/* artifacts/logs/*
	@echo "$(GREEN)All caches and artifacts cleared$(RESET)"

analyze: ## Run analysis on a symbol (usage: make analyze SYMBOL=AAPL)
	python3 cli_orchestrator.py analyze $(SYMBOL) -m standard

analyze-force: ## Run analysis with force refresh (usage: make analyze-force SYMBOL=AAPL)
	python3 cli_orchestrator.py analyze $(SYMBOL) -m standard --force-refresh

batch: ## Run batch analysis (usage: make batch SYMBOLS="AAPL MSFT GOOGL")
	python3 cli_orchestrator.py batch $(SYMBOLS) --mode standard

status: ## Check system status
	python3 cli_orchestrator.py status

cache-inspect: ## Inspect cache for symbol (usage: make cache-inspect SYMBOL=AAPL)
	python3 cli_orchestrator.py inspect-cache --symbol $(SYMBOL) --verbose

cache-clean: ## Clean cache for symbol (usage: make cache-clean SYMBOL=AAPL)
	python3 cli_orchestrator.py clean-cache --symbol $(SYMBOL)

run-dev: ## Run development server (if API exists)
	uvicorn investigator.api.main:app --reload --port 8000

docker-build: ## Build Docker image
	docker build -t investigator:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 investigator:latest

pre-commit: format lint type-check test ## Run all pre-commit checks

ci: format-check lint type-check test-cov ## Run CI pipeline checks

build: clean ## Build distribution packages
	python -m build

publish-test: build ## Publish to Test PyPI
	python -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	python -m twine upload dist/*

verify-structure: ## Verify package structure is correct
	@echo "$(CYAN)Verifying package structure...$(RESET)"
	@test -d src/investigator || (echo "❌ src/investigator/ not found" && exit 1)
	@test -f src/investigator/__init__.py || (echo "❌ src/investigator/__init__.py not found" && exit 1)
	@test -d src/investigator/domain || (echo "❌ src/investigator/domain/ not found" && exit 1)
	@test -d src/investigator/infrastructure || (echo "❌ src/investigator/infrastructure/ not found" && exit 1)
	@test -d src/investigator/application || (echo "❌ src/investigator/application/ not found" && exit 1)
	@test -d src/investigator/interfaces || (echo "❌ src/investigator/interfaces/ not found" && exit 1)
	@echo "$(GREEN)✓ Package structure verified$(RESET)"

tag-release: ## Tag a new release (usage: make tag-release VERSION=v0.1.0)
	git tag -a $(VERSION) -m "Release $(VERSION)"
	git push origin $(VERSION)

show-todos: ## Show implementation tracker status
	@if [ -f REFACTORING_IMPLEMENTATION_TRACKER.md ]; then \
		echo "$(CYAN)Implementation Progress:$(RESET)"; \
		grep -E "^\- \[(x| )\]" REFACTORING_IMPLEMENTATION_TRACKER.md | head -20; \
	fi
