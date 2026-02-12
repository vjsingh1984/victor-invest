"""Unit tests for quarterly fetch helper functions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from investigator.domain.agents.fundamental.models import QuarterlyData
from investigator.domain.agents.fundamental.quarterly_fetch import (
    build_financials_from_bulk_tables,
    build_financials_from_processed_data,
    fetch_processed_quarter_payload,
    normalize_cached_quarter,
    query_recent_processed_periods,
)


def test_normalize_cached_quarter_deserializes_dict():
    logger = MagicMock()
    cached = {
        "fiscal_year": 2024,
        "fiscal_period": "Q4",
        "financial_data": {"revenues": 100.0},
        "ratios": {},
        "data_quality": {"score": 80},
        "filing_date": "2025-01-01",
    }

    normalized = normalize_cached_quarter(
        cached_quarter=cached,
        quarterly_data_cls=QuarterlyData,
        symbol="AAPL",
        fiscal_year=2024,
        fiscal_period="Q4",
        logger=logger,
    )

    assert isinstance(normalized, QuarterlyData)
    assert normalized.fiscal_year == 2024
    assert normalized.fiscal_period == "Q4"


def test_query_recent_processed_periods_maps_rows():
    row = SimpleNamespace(
        symbol="AAPL",
        fiscal_year=2024,
        fiscal_period="Q4",
        adsh="0001",
        filed="2025-01-01",
        period_end="2024-12-31",
        form="10-Q",
        total_revenue=100.0,
        net_income=10.0,
        gross_profit=40.0,
        operating_income=20.0,
        interest_expense=2.0,
        income_tax_expense=3.0,
        cost_of_revenue=60.0,
        total_assets=1000.0,
        total_liabilities=400.0,
        stockholders_equity=600.0,
        current_assets=500.0,
        current_liabilities=250.0,
        accounts_receivable=80.0,
        inventory=30.0,
        cash_and_equivalents=120.0,
        long_term_debt=200.0,
        short_term_debt=50.0,
        total_debt=250.0,
        operating_cash_flow=25.0,
        capital_expenditures=-8.0,
        free_cash_flow=17.0,
        dividends_paid=2.0,
        cash_flow_statement_qtrs=2,
        income_statement_qtrs=2,
        property_plant_equipment_net=300.0,
        shares_outstanding=100.0,
    )
    session = MagicMock()
    execute_result = MagicMock()
    execute_result.fetchall.return_value = [row]
    session.execute.return_value = execute_result

    db_manager = MagicMock()
    db_manager.get_session.return_value.__enter__.return_value = session

    fiscal_period_service = MagicMock()
    fiscal_period_service.is_ytd.side_effect = lambda q: q >= 2

    periods = query_recent_processed_periods(
        symbol="AAPL",
        num_quarters=12,
        db_manager=db_manager,
        fiscal_period_service=fiscal_period_service,
        logger=MagicMock(),
    )

    assert len(periods) == 1
    assert periods[0]["fiscal_period"] == "Q4"
    assert periods[0]["is_ytd_income"] is True
    assert periods[0]["is_ytd_cashflow"] is True


def test_build_financials_from_processed_data_rejects_zero_revenue():
    result = build_financials_from_processed_data(
        processed_data={"income_statement": {"total_revenue": 0}},
        shares_outstanding=123.0,
    )
    assert result is None


def test_build_financials_from_bulk_tables_derives_fcf():
    canonical_mapper = MagicMock()
    canonical_mapper.get_tags.side_effect = lambda key, _sector: [key]

    strategy = MagicMock()
    strategy.get_num_data_for_adsh.return_value = {
        "operating_cash_flow": 100.0,
        "capital_expenditures": -30.0,
        "free_cash_flow": 0.0,
    }

    financial_data = build_financials_from_bulk_tables(
        symbol="MSFT",
        fiscal_year=2024,
        fiscal_period="Q3",
        adsh="0002",
        sector="Technology",
        canonical_keys_needed=["operating_cash_flow", "capital_expenditures", "free_cash_flow", "total_revenue"],
        canonical_mapper=canonical_mapper,
        strategy=strategy,
        logger=MagicMock(),
    )

    assert financial_data["operating_cash_flow"] == 100.0
    assert financial_data["capital_expenditures"] == -30.0
    assert financial_data["free_cash_flow"] == 70.0


def test_fetch_processed_quarter_payload_maps_data():
    row_mapping = {
        "income_statement_qtrs": 2,
        "cash_flow_statement_qtrs": 2,
        "total_revenue": 500.0,
        "net_income": 80.0,
        "gross_profit": 200.0,
        "operating_income": 120.0,
        "cost_of_revenue": 300.0,
        "research_and_development_expense": 20.0,
        "selling_general_administrative_expense": 40.0,
        "operating_expenses": 80.0,
        "interest_expense": 5.0,
        "income_tax_expense": 10.0,
        "earnings_per_share": 2.0,
        "earnings_per_share_diluted": 1.9,
        "preferred_stock_dividends": 0.0,
        "common_stock_dividends": 1.0,
        "weighted_average_diluted_shares_outstanding": 100.0,
        "operating_cash_flow": 140.0,
        "capital_expenditures": -30.0,
        "free_cash_flow": None,
        "dividends_paid": 5.0,
        "investing_cash_flow": -50.0,
        "financing_cash_flow": -20.0,
        "depreciation_amortization": 15.0,
        "stock_based_compensation": 7.0,
        "total_assets": 1000.0,
        "total_liabilities": 400.0,
        "stockholders_equity": 600.0,
        "current_assets": 450.0,
        "current_liabilities": 220.0,
        "retained_earnings": 300.0,
        "accounts_payable": 90.0,
        "accrued_liabilities": 60.0,
        "long_term_debt": 180.0,
        "short_term_debt": 40.0,
        "total_debt": 220.0,
        "net_debt": 100.0,
        "cash_and_equivalents": 120.0,
        "accounts_receivable": 110.0,
        "inventory": 55.0,
        "property_plant_equipment": 350.0,
        "accumulated_depreciation": 100.0,
        "property_plant_equipment_net": 250.0,
        "goodwill": 20.0,
        "intangible_assets": 10.0,
        "deferred_revenue": 30.0,
        "treasury_stock": 5.0,
        "other_comprehensive_income": 4.0,
        "book_value": 600.0,
        "book_value_per_share": 6.0,
        "working_capital": 230.0,
        "market_cap": 2000.0,
        "enterprise_value": 2100.0,
        "shares_outstanding": 100.0,
        "current_ratio": 2.0,
        "quick_ratio": 1.5,
        "debt_to_equity": 0.3,
        "interest_coverage": 12.0,
        "roa": 0.08,
        "roe": 0.12,
        "gross_margin": 0.4,
        "operating_margin": 0.24,
        "net_margin": 0.16,
        "asset_turnover": 0.5,
        "dividend_payout_ratio": 0.2,
        "dividend_yield": 0.01,
        "effective_tax_rate": 0.18,
        "data_quality_score": 95.0,
    }
    row = SimpleNamespace(_mapping=row_mapping)

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = row
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    fiscal_period_service = MagicMock()
    fiscal_period_service.is_ytd.side_effect = lambda q: q >= 2

    data = fetch_processed_quarter_payload(
        symbol="NVDA",
        fiscal_year=2024,
        fiscal_period="Q2",
        adsh="0003",
        engine=engine,
        fiscal_period_service=fiscal_period_service,
        logger=MagicMock(),
    )

    assert data is not None
    assert data["income_statement"]["is_ytd"] is True
    assert data["cash_flow"]["is_ytd"] is True
    assert data["cash_flow"]["free_cash_flow"] == 110.0
