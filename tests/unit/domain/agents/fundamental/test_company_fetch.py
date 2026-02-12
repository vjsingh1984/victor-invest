"""Unit tests for company-level processed-table fetch helper."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

from investigator.domain.agents.fundamental.company_fetch import (
    fetch_latest_company_data_from_processed_table,
)


def _build_db_manager_with_row(row_mapping):
    row = SimpleNamespace(_mapping=row_mapping)
    query_result = MagicMock()
    query_result.fetchone.return_value = row

    conn = MagicMock()
    conn.execute.side_effect = [query_result, MagicMock()]

    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    db_manager = SimpleNamespace(engine=engine)
    return db_manager, conn


def test_fetch_latest_company_data_maps_and_derives_fields():
    db_manager, _conn = _build_db_manager_with_row(
        {
            "symbol": "AAPL",
            "fiscal_year": 2024,
            "fiscal_period": "FY",
            "filed_date": date(2025, 1, 31),
            "total_revenue": 1_000.0,
            "net_income": 200.0,
            "gross_profit": 400.0,
            "operating_income": 250.0,
            "cost_of_revenue": 600.0,
            "total_assets": 5_000.0,
            "stockholders_equity": 2_000.0,
            "current_assets": 800.0,
            "current_liabilities": 300.0,
            "total_liabilities": 3_000.0,
            "total_debt": 0.0,
            "long_term_debt": 700.0,
            "short_term_debt": 100.0,
            "inventory": 50.0,
            "cash_and_equivalents": 600.0,
            "operating_cash_flow": 300.0,
            "capital_expenditures": -50.0,
            "free_cash_flow": 250.0,
            "shares_outstanding": 0.0,
            "weighted_average_diluted_shares_outstanding": 120.0,
            "current_ratio": 2.2,
            "quick_ratio": 1.9,
            "debt_to_equity": 0.4,
            "roe": 0.1,
            "roa": 0.08,
            "gross_margin": 0.4,
            "operating_margin": 0.25,
            "net_margin": 0.2,
            "data_quality_score": 92.0,
        }
    )
    result = fetch_latest_company_data_from_processed_table(
        symbol="AAPL",
        db_manager=db_manager,
        logger=MagicMock(),
        processed_additional_financial_keys=["cash"],
        processed_ratio_keys=["interest_coverage"],
    )

    assert result is not None
    metrics = result["financial_metrics"]
    ratios = result["financial_ratios"]
    assert metrics["total_debt"] == 800.0
    assert metrics["cash"] == 600.0
    assert metrics["shares_outstanding"] == 120.0
    assert ratios["interest_coverage"] == 0.0
    assert result["data_quality_score"] == 92.0
    assert result["source"] == "clean_architecture"


def test_fetch_latest_company_data_deletes_corrupt_negative_revenue():
    db_manager, conn = _build_db_manager_with_row(
        {
            "symbol": "ORCL",
            "fiscal_year": 2024,
            "fiscal_period": "Q2",
            "filed_date": date(2024, 12, 31),
            "total_revenue": -10.0,
            "operating_income": 1.0,
        }
    )
    logger = MagicMock()
    result = fetch_latest_company_data_from_processed_table(
        symbol="ORCL",
        db_manager=db_manager,
        logger=logger,
        processed_additional_financial_keys=[],
        processed_ratio_keys=[],
    )

    assert result is None
    conn.commit.assert_called_once()
    logger.error.assert_called()
    logger.warning.assert_called()


def test_fetch_latest_company_data_returns_none_when_missing():
    query_result = MagicMock()
    query_result.fetchone.return_value = None

    conn = MagicMock()
    conn.execute.return_value = query_result
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    db_manager = SimpleNamespace(engine=engine)

    result = fetch_latest_company_data_from_processed_table(
        symbol="MSFT",
        db_manager=db_manager,
        logger=MagicMock(),
        processed_additional_financial_keys=[],
        processed_ratio_keys=[],
    )

    assert result is None
