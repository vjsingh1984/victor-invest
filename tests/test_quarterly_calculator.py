import math

from utils.quarterly_calculator import analyze_quarterly_patterns, compute_missing_quarter


def test_compute_missing_quarter_derives_free_cash_flow():
    fy_data = {
        "symbol": "TEST",
        "fiscal_year": 2025,
        "fiscal_period": "FY",
        "cash_flow": {
            "operating_cash_flow": 118_254_000_000.0,
            "capital_expenditures": 9_447_000_000.0,
        },
    }

    def quarter(quarter_period: str, ocf: float, capex: float):
        return {
            "fiscal_year": 2025,
            "fiscal_period": quarter_period,
            "cash_flow": {
                "operating_cash_flow": ocf,
                "capital_expenditures": capex,
            },
        }

    q1 = quarter("Q1", 39_895_000_000.0, 2_392_000_000.0)
    q2 = quarter("Q2", 31_548_000_000.0, 2_147_000_000.0)
    q3 = quarter("Q3", 20_000_000_000.0, 2_000_000_000.0)

    q4 = compute_missing_quarter(fy_data, q1, q2, q3)

    assert q4 is not None
    cash_flow = q4["cash_flow"]

    expected_ocf = fy_data["cash_flow"]["operating_cash_flow"] - (
        q1["cash_flow"]["operating_cash_flow"]
        + q2["cash_flow"]["operating_cash_flow"]
        + q3["cash_flow"]["operating_cash_flow"]
    )
    expected_capex = fy_data["cash_flow"]["capital_expenditures"] - (
        q1["cash_flow"]["capital_expenditures"]
        + q2["cash_flow"]["capital_expenditures"]
        + q3["cash_flow"]["capital_expenditures"]
    )
    expected_fcf = expected_ocf - abs(expected_capex)

    assert cash_flow["operating_cash_flow"] == expected_ocf
    assert cash_flow["capital_expenditures"] == expected_capex
    assert math.isclose(cash_flow["free_cash_flow"], expected_fcf)
    assert not cash_flow.get("is_ytd")


def test_compute_missing_quarter_normalizes_income_statement():
    fy_data = {
        "symbol": "TEST",
        "fiscal_year": 2025,
        "fiscal_period": "FY",
        "income_statement": {
            "total_revenue": 400_000_000_000.0,
            "net_income": 120_000_000_000.0,
            "operating_income": 150_000_000_000.0,
        },
    }

    def income_quarter(period: str, revenue: float, net_income: float, operating_income: float):
        return {
            "fiscal_year": 2025,
            "fiscal_period": period,
            "income_statement": {
                "total_revenue": revenue,
                "net_income": net_income,
                "operating_income": operating_income,
            },
        }

    q1 = income_quarter("Q1", 120_000_000_000.0, 40_000_000_000.0, 45_000_000_000.0)
    q2 = income_quarter("Q2", 100_000_000_000.0, 30_000_000_000.0, 35_000_000_000.0)
    q3 = income_quarter("Q3", 90_000_000_000.0, 25_000_000_000.0, 30_000_000_000.0)

    q4 = compute_missing_quarter(fy_data, q1, q2, q3)

    assert q4 is not None
    stmt = q4["income_statement"]

    expected_revenue = fy_data["income_statement"]["total_revenue"] - (
        q1["income_statement"]["total_revenue"]
        + q2["income_statement"]["total_revenue"]
        + q3["income_statement"]["total_revenue"]
    )
    expected_net_income = fy_data["income_statement"]["net_income"] - (
        q1["income_statement"]["net_income"]
        + q2["income_statement"]["net_income"]
        + q3["income_statement"]["net_income"]
    )
    expected_operating_income = fy_data["income_statement"]["operating_income"] - (
        q1["income_statement"]["operating_income"]
        + q2["income_statement"]["operating_income"]
        + q3["income_statement"]["operating_income"]
    )

    assert math.isclose(stmt["total_revenue"], expected_revenue)
    assert math.isclose(stmt["net_income"], expected_net_income)
    assert math.isclose(stmt["operating_income"], expected_operating_income)


def test_analyze_quarterly_patterns_reports_seasonality_and_growth():
    def quarter(year, period, fcf, revenue):
        return {
            "fiscal_year": year,
            "fiscal_period": period,
            "financial_data": {
                "cash_flow_statement": {"free_cash_flow": fcf},
                "income_statement": {"total_revenue": revenue},
            },
        }

    quarters = [
        quarter(2025, "Q4", 40_000_000_000.0, 110_000_000_000.0),
        quarter(2025, "Q3", 12_000_000_000.0, 90_000_000_000.0),
        quarter(2025, "Q2", 11_000_000_000.0, 88_000_000_000.0),
        quarter(2025, "Q1", 10_000_000_000.0, 86_000_000_000.0),
        quarter(2024, "Q4", 35_000_000_000.0, 100_000_000_000.0),
        quarter(2024, "Q3", 15_000_000_000.0, 85_000_000_000.0),
        quarter(2024, "Q2", 14_000_000_000.0, 83_000_000_000.0),
        quarter(2024, "Q1", 13_000_000_000.0, 81_000_000_000.0),
    ]

    patterns = analyze_quarterly_patterns(quarters, "free_cash_flow")

    assert patterns["metric"] == "free_cash_flow"
    assert "seasonality" in patterns
    assert patterns["trend"] in {"accelerating", "stable", "decelerating"}
    assert patterns["yoy_growth"]


def test_analyze_quarterly_patterns_for_revenue_trends():
    def quarter(year, period, revenue):
        return {
            "fiscal_year": year,
            "fiscal_period": period,
            "financial_data": {
                "income_statement": {"total_revenue": revenue},
            },
        }

    quarters = [
        quarter(2025, "Q4", 110_000_000_000.0),
        quarter(2025, "Q3", 95_000_000_000.0),
        quarter(2025, "Q2", 92_000_000_000.0),
        quarter(2025, "Q1", 90_000_000_000.0),
        quarter(2024, "Q4", 100_000_000_000.0),
        quarter(2024, "Q3", 92_000_000_000.0),
        quarter(2024, "Q2", 88_000_000_000.0),
        quarter(2024, "Q1", 86_000_000_000.0),
    ]

    patterns = analyze_quarterly_patterns(quarters, "total_revenue")

    assert patterns["metric"] == "total_revenue"
    assert patterns["yoy_growth"]
    assert patterns["trend"] in {"accelerating", "stable", "decelerating"}


def test_compute_missing_quarter_copies_balance_sheet_snapshot():
    fy_data = {
        "symbol": "TEST",
        "fiscal_year": 2025,
        "fiscal_period": "FY",
        "cash_flow": {
            "operating_cash_flow": 50_000_000_000.0,
            "capital_expenditures": 5_000_000_000.0,
        },
        "balance_sheet": {
            "total_assets": 350_000_000_000.0,
            "total_liabilities": 200_000_000_000.0,
            "stockholders_equity": 150_000_000_000.0,
        },
    }

    q1 = {
        "fiscal_year": 2025,
        "fiscal_period": "Q1",
        "cash_flow": {
            "operating_cash_flow": 15_000_000_000.0,
            "capital_expenditures": 1_500_000_000.0,
        },
    }
    q2 = {
        "fiscal_year": 2025,
        "fiscal_period": "Q2",
        "cash_flow": {
            "operating_cash_flow": 12_000_000_000.0,
            "capital_expenditures": 1_200_000_000.0,
        },
    }
    q3 = {
        "fiscal_year": 2025,
        "fiscal_period": "Q3",
        "cash_flow": {
            "operating_cash_flow": 10_000_000_000.0,
            "capital_expenditures": 1_100_000_000.0,
        },
    }

    q4 = compute_missing_quarter(fy_data, q1, q2, q3)
    assert q4 is not None
    balance_sheet = q4["balance_sheet"]
    assert balance_sheet == fy_data["balance_sheet"]
