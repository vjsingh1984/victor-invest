from types import SimpleNamespace

from investigator.domain.agents.fundamental.summaries import (
    extract_latest_financials,
    get_historical_trend,
    summarize_company_data,
)


def test_get_historical_trend_returns_placeholder_structure():
    trend = get_historical_trend({"revenues": 1000})
    assert trend["revenue_trend"] == [100, 110, 121, 133]
    assert trend["earnings_trend"] == [10, 12, 15, 18]
    assert trend["years"] == [2021, 2022, 2023, 2024]


def test_summarize_company_data_uses_top_level_market_cap_then_market_data():
    company_data = {
        "symbol": "AAPL",
        "market_cap": 3_000_000_000,
        "current_price": 190.0,
        "market_data": {"market_cap": 2_900_000_000, "price": 191.0},
        "financials": {
            "revenues": 100_000,
            "net_income": 25_000,
            "total_assets": 350_000,
            "stockholders_equity": 75_000,
        },
    }
    summary = summarize_company_data(company_data)
    assert summary["symbol"] == "AAPL"
    assert summary["market_cap"] == 3_000_000_000
    assert summary["price"] == 191.0
    assert summary["revenue"] == 100_000
    assert summary["net_income"] == 25_000
    assert summary["total_assets"] == 350_000
    assert summary["total_equity"] == 75_000


def test_extract_latest_financials_from_dict_payload_calculates_ebitda():
    quarterly = [
        {
            "revenue": 5000,
            "net_income": 800,
            "operating_income": 900,
            "depreciation_amortization": 100,
            "total_assets": 15000,
            "total_liabilities": 7000,
            "stockholders_equity": 8000,
            "cash": 1200,
            "total_debt": 2000,
            "operating_cash_flow": 950,
            "capital_expenditures": 250,
            "free_cash_flow": 700,
        }
    ]
    latest = extract_latest_financials(quarterly)
    assert latest["revenues"] == 5000
    assert latest["net_income"] == 800
    assert latest["operating_income"] == 900
    assert latest["depreciation_amortization"] == 100
    assert latest["ebitda"] == 1000
    assert latest["free_cash_flow"] == 700


def test_extract_latest_financials_from_quarterly_data_object():
    financial_data = SimpleNamespace(
        income_statement={"revenue": 62000, "net_income": 22000, "operating_income": 26000, "gross_profit": 43000},
        balance_sheet={
            "total_assets": 500000,
            "total_liabilities": 200000,
            "stockholders_equity": 300000,
            "cash": 45000,
            "total_debt": 55000,
            "current_assets": 180000,
            "current_liabilities": 95000,
            "inventory": 3500,
            "shares_outstanding": 7400,
        },
        cash_flow_statement={
            "depreciation_amortization": 3200,
            "operating_cash_flow": 28000,
            "capital_expenditures": 9000,
            "free_cash_flow": 19000,
            "dividends": 5000,
        },
        quarterly_data={},
    )
    q = SimpleNamespace(financial_data=financial_data)
    latest = extract_latest_financials([q])
    assert latest["revenues"] == 62000
    assert latest["net_income"] == 22000
    assert latest["operating_income"] == 26000
    assert latest["depreciation_amortization"] == 3200
    assert latest["ebitda"] == 29200
    assert latest["stockholders_equity"] == 300000
    assert latest["shares_outstanding"] == 7400


def test_extract_latest_financials_handles_empty_input():
    assert extract_latest_financials([]) == {}
