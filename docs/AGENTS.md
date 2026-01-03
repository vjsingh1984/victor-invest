# Repository Guidelines

## Project Structure & Module Organization
InvestiGator now exposes an installable package under `src/investigator/` (clean architecture split across `domain/`, `application/`, `infrastructure/`, `interfaces/`, `shared/`). Legacy orchestrator helpers in `core/` and async agents in `agents/` remain while migration completes; prompts live in `prompts/`, with data access and external connectors under `dao/` and `api/`. The CLI entry points are `cli_orchestrator.py` and `src/investigator/cli.py`, wrapped for convenience by `investigator_v2.sh`. Reference data and cached artifacts stay in `data/`, runtime logs land in `logs/`, and longer-lived outputs should be moved to `archive/` (git-ignored) after review. Tests mirror runtime modules inside `tests/`, while deployment assets live in `deployment/` and `docker-compose.yml`.

## Build, Test, and Development Commands
Use `./setup_dev.sh` for the full bootstrap (venv, requirements, Ollama models). Manual alternative: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt` or `make dev-install`. Verify the pipeline with `./investigator_v2.sh --symbol AAPL --mode quick`; append `--run-tests` to exercise diagnostics. For focused debugging call `python cli_orchestrator.py --symbol MSFT --mode comprehensive` or `python -m investigator.cli analyze --symbol MSFT`. When working with containers, bring up backing services via `docker-compose up postgres redis` and the API via `docker-compose up investigator-api`. Use `make clean` / `make clean-all` to purge bytecode, coverage, and cached artifacts.

## Coding Style & Naming Conventions
Format Python with `black` (120 columns) and organize imports via `isort`; run `pre-commit run --all-files` before commits. Enforce lint and type gates using `flake8`, `pylint`, and `mypy`, keeping async signatures annotated. Modules, files, and functions use snake_case; classes keep CamelCase; config artifacts prefer lower-kebab (e.g., `config.yaml`). Stick to four-space indentation, explicit docstrings for public orchestrators, and descriptive prompt IDs inside `prompts/`.

## Testing Guidelines
Pytest discovers specs in `tests/` named `test_*.py` or `*_test.py`. Quick pass: `python -m pytest -m "not slow and not integration"`; full suite: `python -m pytest --durations=10`. Tag expensive scenarios with `@pytest.mark.slow` or `@pytest.mark.integration` to keep CI lean. Share reusable fixtures through `tests/conftest.py` and keep golden outputs under `results/` (then archive to `archive/results/` once validated) for deterministic comparisons.

## Commit & Pull Request Guidelines
Commits follow the Conventional Commit style (`type(scope): summary`) already present in history; keep each change focused and include related docs or tests. Pull requests should outline context, key changes, and validation (`pytest` output or relevant `results/*.log`). Link issues, capture follow-up tasks explicitly, and request reviewers who own the impacted subsystems (`agents/`, `core/`, `api/`). Include screenshots when altering generated reports or dashboards.
