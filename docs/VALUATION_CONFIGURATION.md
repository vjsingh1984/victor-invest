# Valuation Configuration Guide

This document describes the configuration files and parameters that influence the multi-model valuation framework.

## Files

- `config.json`
  - `valuation.sector_multiples_path`: Path to the JSON file containing sector median multiples.
  - `valuation.sector_multiples_freshness_days`: Warn when reference data is older than this many days.
  - `valuation.sector_multiples_delta_threshold`: Warn when a new multiple shifts more than this ratio compared to the previous snapshot.
  - `valuation.liquidity_floor_usd`: Minimum daily dollar liquidity required before the P/S model is considered applicable.
  - `valuation.model_fallback`: Archetype-aware fallback weights used when model confidences are unavailable.

- `config/sector_multiples.json`
  - `_metadata`: Optional object containing source metadata and historical snapshots.
  - Each sector entry should include `pe`, `ev_ebitda`, `ps`, `pb`, `sample_size`, and `last_updated` (ISO8601).

- `config/company_archetype_rules.yaml`
  - Thresholds that map financial signals to primary/secondary archetypes, used by the model selector.

- `config/model_selection.yaml`
  - `defaults.include` / `defaults.exclude`: Baseline list of models considered for blending.
  - `archetypes.<name>.include` / `exclude`: Adjusts model participation for specific archetypes (e.g., `financial`, `high_growth`).
  - `min_models`: Optional advisory minimum count of applicable models.

## Sector Multiples Loader

The loader (`SectorMultiplesLoader`) reads the JSON reference and performs:

- **Freshness checks**: warns when `last_updated` is older than `valuation.sector_multiples_freshness_days`.
- **Sample size checks**: warns if `sample_size` < 5.
- **Delta monitoring**: if `_metadata.previous` is present, warns when new values diverge from previous by more than `valuation.sector_multiples_delta_threshold`.

To refresh the data:

1. Download the latest sector multiples from the chosen data vendor.
2. Update `config/sector_multiples.json` with the new values, including `sample_size` and `last_updated`.
3. Optionally populate `_metadata.previous` before overwriting so the loader can alert on larger-than-expected shifts.

## Model Fallback Weights

When the orchestrator cannot rely on per-model confidence scores (e.g., insufficient data), it falls back to the weights defined under `valuation.model_fallback`. Example:

```json
"model_fallback": {
  "default": {
    "weights": {"dcf": 0.6, "ev_ebitda": 0.2, "ps": 0.1, "pe": 0.1}
  },
  "financials": {
    "weights": {"pb": 0.5, "pe": 0.3, "ev_ebitda": 0.2}
  }
}
```

- Keys are archetype names in lowercase (e.g., `financials`).
- The `default` node is used when no archetype-specific weights are found.
- Only models present in the fallback map and available in the current run receive non-zero weight.

## Liquidity Guard Rails

The P/S model enforces a minimum `daily_liquidity_usd`. Adjust `valuation.liquidity_floor_usd` to raise or lower that threshold.

## High-Level Flow Summary

1. **Company profile** is constructed using `company_archetype_rules.yaml`.
2. **Multiples loader** fetches sector medians from `config/sector_multiples.json` (with freshness checks).
3. **Multiple models** (P/E, EV/EBITDA, P/S, P/B) compute valuation results, considering liquidity and archetype.
4. **Model selection rules** (`model_selection.yaml`) filter which models participate based on archetype.
5. **Multi-model orchestrator** blends the remaining models, applying fallback weights only when confidence-based weighting is not possible.

Keep these files up to date to ensure the valuation engine produces reliable outputs.

## CLI/API Quick Reference

Use these commands after adjusting sector multiples or model selection rules:

- **CLI wrapper** (auto-handles virtualenv/bootstrap):

  ```bash
  ./investigator_v2.sh --symbol AAPL --mode quick --force-refresh
  ```

- **Direct module invocation** (bypasses wrapper, useful inside dev shells):

  ```bash
  PYTHONPATH=src python -m investigator.cli analyze --symbol MSFT --mode comprehensive --format json
  ```

- **API request** (matches CLI behavior for remote clients):

  ```bash
  curl -X POST http://localhost:8080/api/analyze \
    -H "Content-Type: application/json" \
    -d '{"symbol": "TSLA", "mode": "quick", "force_refresh": true}'
  ```

All flows reload configuration on each run, so changes to `config/sector_multiples.json` or `config/model_selection.yaml` take effect immediately.
