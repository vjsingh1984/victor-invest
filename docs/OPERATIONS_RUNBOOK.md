# Operations Runbook

## Environment & Services
- **Python/venv:** Python 3.11+, activate `venv` before running commands.
- **LLM pool:** Ollama running locally; pool settings loaded via `config.json` (servers, capacity caps). Restart Ollama if LLM calls hang.
- **Database:** PostgreSQL for market data; ensure `config.json` holds correct DSN.

## Routine Commands
- Health/status: `python3 cli_orchestrator.py status`
- Cache maintenance: `make cache-inspect SYMBOL=AAPL`, `make cache-clean SYMBOL=AAPL`, wipe all via `make clean-all`
- Core checks: `make format lint type-check test` or one-shot `make ci`
- Coverage review: `make test-cov` → open `htmlcov/index.html`
- API dev server: `make run-dev` (uvicorn on :8000)

## Failure Recovery
- **Orchestrator startup fails:** Restart after confirming Ollama servers are available and DB is reachable; review logs for resource pool errors.
- **Agent hangs:** Cancel running job and restart orchestrator; verify Ollama and network to SEC are responsive; clear caches for the symbol with `make cache-clean SYMBOL=<TICKER>`.
- **Database connectivity errors:** Validate DSN in `config.json`, check network/firewall, and run a simple SQL ping via `psql`.
- **Corrupted caches:** Remove affected symbol caches (`make cache-clean SYMBOL=<TICKER>`) or fully reset with `make clean-all` (clears SEC/LLM/technical caches and artifacts).

## Observability & Logging
- Metrics are emitted via `MetricsCollector`; orchestrator logs report worker health and queue stats every minute. Monitor for repeated worker error logs.
- Long-running tasks mark failures in `completed_tasks` with `error`; fetch status via API/CLI status endpoints if exposed.

## Change Management
- Before releases: run `make ci`, verify coverage, and ensure `README.adoc` + `docs/AGENTS.md` are current.
- Secrets: keep SEC user agent, DB creds, and tokens in `config.json` or `.env` (git-ignored). Never commit secrets.
