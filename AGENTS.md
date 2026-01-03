# Repository Guidelines

## Project Structure & Modules
- Source lives in `src/investigator/` with clean-architecture layers (domain/application/infrastructure/interfaces) and CLI entrypoints in `src/investigator/cli.py` + `cli_orchestrator.py`.
- API server code: `src/investigator/api/`; shared utilities in `src/investigator/shared/`. Tests mirror `src/` under `tests/`. Docs and architecture notes are in `docs/`. Config, data caches, and artifacts live in `config/`, `data/`, and `artifacts/`.

## Build, Test, and Development Commands
- Python 3.11+; create a venv, then `make install` (runtime deps) or `make dev-install` (adds lint/test/viz/jupyter).
- Core checks: `make format` (black + isort, autofix), `make lint` (flake8), `make type-check` (mypy), `make test` (pytest). One-shot CI surface: `make ci`. Coverage: `make test-cov`.
- Run analysis: `make analyze SYMBOL=AAPL` or `python3 cli_orchestrator.py analyze <SYMBOL> -m standard`; batch via `make batch SYMBOLS="AAPL MSFT"`. Health and cache: `python3 cli_orchestrator.py status`, `make cache-inspect SYMBOL=AAPL`, `make cache-clean SYMBOL=AAPL`.
- API dev server: `make run-dev` (uvicorn on :8000). Clean artifacts/caches with `make clean` or `make clean-all` before packaging.

## Coding Style & Naming Conventions
- Formatting: black + isort, 120-char lines. Lint with flake8; optional type coverage with mypy. Keep functions small and align modules with existing layer boundaries.
- Naming: files/modules snake_case; classes CamelCase; functions/variables snake_case; constants UPPER_SNAKE. Make targets stay kebab-case.
- Prefer explicit type hints on public surfaces and dataclasses/config objects. Avoid committing generated data under `data/` and `artifacts/`.

## Testing Guidelines
- Pytest discovery: `tests/**/test_*.py` or `*_test.py`; classes start with `Test`, functions with `test_`.
- Markers: `unit`, `integration`, `slow`, `cache`, `performance`, `comprehensive`. Default pipelines expect fast/unit runs, so mark heavy suites (`-m "not slow"`) and isolate network/LLM calls behind fixtures or fakes.
- Generate coverage locally with `make test-cov` and review `htmlcov/`. Keep fixtures under `tests/fixtures` (add as needed) and reuse sample SEC/LLM payloads for determinism.

## Commit & Pull Request Guidelines
- Follow the existing style: `type(scope): concise subject` (e.g., `fix(sec): correct canonical mapper path resolution`); imperative voice; ~72-char subjects.
- For PRs: include summary, rationale, and risk notes; link issues/tasks; list commands executed (format, lint, test) with key results; attach payload snippets or screenshots for API/analysis changes when useful.
- Keep secrets out of VCS (e.g., `config.json`, DB creds, SEC tokens); prefer `.env` or local overrides documented in `config/`.
