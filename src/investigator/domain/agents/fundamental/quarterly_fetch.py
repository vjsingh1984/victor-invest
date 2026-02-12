"""Helpers for quarterly data ingestion from processed and fallback SEC sources."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import text


def to_float(value: Any) -> float:
    """Convert DB/scalar value to float with safe fallback to `0.0`."""
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def query_recent_processed_periods(
    *,
    symbol: str,
    num_quarters: int,
    db_manager: Any,
    fiscal_period_service: Any,
    logger: Any,
) -> List[Dict[str, Any]]:
    """Load recent FY/Q periods for a symbol from `sec_companyfacts_processed`."""
    query = text(
        """
        SELECT
            symbol, fiscal_year, fiscal_period, adsh,
            filed_date as filed,
            period_end_date as period_end,
            form_type as form,
            total_revenue,
            net_income,
            gross_profit,
            operating_income,
            interest_expense,
            income_tax_expense,
            cost_of_revenue,
            total_assets,
            total_liabilities,
            stockholders_equity,
            current_assets,
            current_liabilities,
            accounts_receivable,
            inventory,
            cash_and_equivalents,
            long_term_debt,
            short_term_debt,
            total_debt,
            operating_cash_flow,
            capital_expenditures,
            free_cash_flow,
            dividends_paid,
            cash_flow_statement_qtrs,
            income_statement_qtrs,
            property_plant_equipment_net,
            weighted_average_diluted_shares_outstanding as shares_outstanding
        FROM sec_companyfacts_processed
        WHERE symbol = :symbol
        ORDER BY
            fiscal_year DESC,
            CASE fiscal_period
                WHEN 'FY' THEN 4
                WHEN 'Q3' THEN 3
                WHEN 'Q2' THEN 2
                WHEN 'Q1' THEN 1
                ELSE 0
            END DESC
        LIMIT :sql_limit
    """
    )

    sql_limit = num_quarters + 3
    with db_manager.get_session() as session:
        result = session.execute(query, {"symbol": symbol, "sql_limit": sql_limit})
        rows = result.fetchall()

    if not rows:
        logger.warning("No quarterly data in processed table for %s", symbol)
        return []

    quarters_data: List[Dict[str, Any]] = []
    for row in rows:
        cf_qtrs = int(row.cash_flow_statement_qtrs) if row.cash_flow_statement_qtrs else 1
        inc_qtrs = int(row.income_statement_qtrs) if row.income_statement_qtrs else 1

        quarters_data.append(
            {
                "symbol": row.symbol,
                "fiscal_year": row.fiscal_year,
                "fiscal_period": row.fiscal_period,
                "adsh": row.adsh,
                "filed": str(row.filed) if row.filed else None,
                "period_end": str(row.period_end) if row.period_end else None,
                "form": row.form,
                "total_revenue": to_float(row.total_revenue),
                "net_income": to_float(row.net_income),
                "gross_profit": to_float(row.gross_profit),
                "operating_income": to_float(row.operating_income),
                "interest_expense": to_float(row.interest_expense),
                "income_tax_expense": to_float(row.income_tax_expense),
                "cost_of_revenue": to_float(row.cost_of_revenue),
                "total_assets": to_float(row.total_assets),
                "total_liabilities": to_float(row.total_liabilities),
                "stockholders_equity": to_float(row.stockholders_equity),
                "current_assets": to_float(row.current_assets),
                "current_liabilities": to_float(row.current_liabilities),
                "accounts_receivable": to_float(row.accounts_receivable),
                "inventory": to_float(row.inventory),
                "cash_and_equivalents": to_float(row.cash_and_equivalents),
                "long_term_debt": to_float(row.long_term_debt),
                "short_term_debt": to_float(row.short_term_debt),
                "total_debt": to_float(row.total_debt),
                "operating_cash_flow": to_float(row.operating_cash_flow),
                "capital_expenditures": to_float(row.capital_expenditures),
                "free_cash_flow": to_float(row.free_cash_flow),
                "dividends_paid": to_float(row.dividends_paid),
                "property_plant_equipment_net": to_float(row.property_plant_equipment_net),
                "shares_outstanding": to_float(row.shares_outstanding),
                "cash_flow_statement_qtrs": cf_qtrs,
                "income_statement_qtrs": inc_qtrs,
                "is_ytd_cashflow": fiscal_period_service.is_ytd(cf_qtrs),
                "is_ytd_income": fiscal_period_service.is_ytd(inc_qtrs),
            }
        )

    fy_count = sum(1 for q in quarters_data if q.get("fiscal_period") == "FY")
    q_count = sum(1 for q in quarters_data if q.get("fiscal_period", "").startswith("Q"))
    logger.info(
        "✅ Retrieved %s periods from processed table for %s "
        "(%s Q periods, %s FY periods - FY needed for Q4 computation)",
        len(quarters_data),
        symbol,
        q_count,
        fy_count,
    )
    if len(quarters_data) < num_quarters:
        logger.warning(
            "Only %s quarters available for %s (target: %s). "
            "Company may be newly public or have incomplete filing history.",
            len(quarters_data),
            symbol,
            num_quarters,
        )

    return quarters_data


def normalize_cached_quarter(
    *,
    cached_quarter: Any,
    quarterly_data_cls: Any,
    symbol: str,
    fiscal_year: int,
    fiscal_period: str,
    logger: Any,
) -> Optional[Any]:
    """Validate/deserialize quarter cache entry and return normalized quarter object."""
    if not cached_quarter:
        return None

    if isinstance(cached_quarter, dict):
        try:
            cached_quarter = quarterly_data_cls.from_dict(cached_quarter)
            logger.debug("Cache HIT (dict→QuarterlyData) for %s %s-%s", symbol, fiscal_year, fiscal_period)
        except Exception as exc:
            logger.warning("Failed to deserialize cached quarter for %s: %s, re-fetching", symbol, exc)
            return None
    elif not isinstance(cached_quarter, quarterly_data_cls):
        logger.warning(
            "Invalid cached quarter type for %s %s-%s: %s, re-fetching",
            symbol,
            fiscal_year,
            fiscal_period,
            type(cached_quarter),
        )
        return None
    else:
        logger.debug("Cache HIT for %s %s-%s", symbol, fiscal_year, fiscal_period)

    return cached_quarter


def build_financials_from_processed_data(
    *,
    processed_data: Dict[str, Any],
    shares_outstanding: float,
) -> Optional[Dict[str, Any]]:
    """Flatten statement-level processed payload into QuarterlyData financial_data layout."""
    income_statement = processed_data.get("income_statement", {})
    revenue = income_statement.get("total_revenue", 0)
    if not revenue or revenue <= 0:
        return None

    cash_flow = processed_data.get("cash_flow", {})
    balance_sheet = processed_data.get("balance_sheet", {})
    financial_data = {
        "revenues": income_statement.get("total_revenue", 0),
        "net_income": income_statement.get("net_income", 0),
        "gross_profit": income_statement.get("gross_profit", 0),
        "operating_income": income_statement.get("operating_income", 0),
        "interest_expense": income_statement.get("interest_expense", 0),
        "income_tax_expense": income_statement.get("income_tax_expense", 0),
        "total_assets": balance_sheet.get("total_assets", 0),
        "total_liabilities": balance_sheet.get("total_liabilities", 0),
        "stockholders_equity": balance_sheet.get("stockholders_equity", 0),
        "current_assets": balance_sheet.get("current_assets", 0),
        "current_liabilities": balance_sheet.get("current_liabilities", 0),
        "total_debt": balance_sheet.get("total_debt", 0),
        "long_term_debt": balance_sheet.get("long_term_debt", 0),
        "short_term_debt": balance_sheet.get("short_term_debt", 0),
        "cash_and_equivalents": balance_sheet.get("cash_and_equivalents", 0),
        "net_debt": balance_sheet.get("net_debt", 0),
        "operating_cash_flow": cash_flow.get("operating_cash_flow", 0),
        "capital_expenditures": cash_flow.get("capital_expenditures", 0),
        "dividends_paid": cash_flow.get("dividends_paid", 0),
        "weighted_average_diluted_shares_outstanding": shares_outstanding,
    }
    return {
        "financial_data": financial_data,
        "ratios": processed_data.get("ratios", {}),
        "quality": processed_data.get("data_quality_score", 0),
        "revenue": revenue,
        "is_ytd_cashflow": cash_flow.get("is_ytd", False),
        "is_ytd_income": income_statement.get("is_ytd", False),
    }


def build_financials_from_bulk_tables(
    *,
    symbol: str,
    fiscal_year: int,
    fiscal_period: str,
    adsh: str,
    sector: str,
    canonical_keys_needed: Sequence[str],
    canonical_mapper: Any,
    strategy: Any,
    logger: Any,
) -> Dict[str, float]:
    """Extract canonical values from bulk SEC tables for fallback quarter construction."""
    all_tags = set()
    for canonical_key in canonical_keys_needed:
        all_tags.update(canonical_mapper.get_tags(canonical_key, sector))
    tag_values = strategy.get_num_data_for_adsh(adsh, tags=list(all_tags))

    def extract_canonical_value(canonical_key: str) -> float:
        for tag in canonical_mapper.get_tags(canonical_key, sector):
            if tag in tag_values and tag_values[tag] is not None:
                return to_float(tag_values[tag])
        return 0.0

    ocf_value = extract_canonical_value("operating_cash_flow")
    capex_value = extract_canonical_value("capital_expenditures")
    fcf_value = extract_canonical_value("free_cash_flow")

    if ocf_value is not None or capex_value is not None:
        ocf_float = float(ocf_value or 0.0)
        capex_float = float(capex_value or 0.0)
        derived_fcf = ocf_float - abs(capex_float)
        if fcf_value is None or abs(float(fcf_value)) < 1e-6:
            fcf_value = derived_fcf
            if abs(derived_fcf) > 1e-6:
                logger.debug(
                    "🔄 [FALLBACK] Derived FCF for %s %s-%s via OCF %.2f - |CapEx| %.2f = %.2f",
                    symbol,
                    fiscal_year,
                    fiscal_period,
                    ocf_float,
                    capex_float,
                    derived_fcf,
                )

    return {
        "revenues": extract_canonical_value("total_revenue"),
        "net_income": extract_canonical_value("net_income"),
        "total_assets": extract_canonical_value("total_assets"),
        "total_liabilities": extract_canonical_value("total_liabilities"),
        "stockholders_equity": extract_canonical_value("stockholders_equity"),
        "current_assets": extract_canonical_value("current_assets"),
        "current_liabilities": extract_canonical_value("current_liabilities"),
        "long_term_debt": extract_canonical_value("long_term_debt"),
        "short_term_debt": extract_canonical_value("short_term_debt"),
        "total_debt": extract_canonical_value("total_debt"),
        "operating_cash_flow": ocf_value,
        "capital_expenditures": capex_value,
        "free_cash_flow": fcf_value if fcf_value is not None else 0,
        "dividends_paid": extract_canonical_value("dividends_paid"),
        "weighted_average_diluted_shares_outstanding": extract_canonical_value(
            "weighted_average_diluted_shares_outstanding"
        ),
    }


def fetch_processed_quarter_payload(
    *,
    symbol: str,
    fiscal_year: int,
    fiscal_period: str,
    adsh: str,
    engine: Any,
    fiscal_period_service: Any,
    logger: Any,
) -> Optional[Dict[str, Any]]:
    """Fetch one processed-quarter payload and map into statement-oriented structure."""
    logger.info("🔍 [PROCESSED_TABLE] Querying for %s %s-%s ADSH=%s...", symbol, fiscal_year, fiscal_period, adsh[:20])
    query = text(
        """
        SELECT *
        FROM sec_companyfacts_processed
        WHERE symbol = :symbol
          AND fiscal_year = :fiscal_year
          AND fiscal_period = :fiscal_period
          AND adsh = :adsh
        LIMIT 1
    """
    )
    with engine.connect() as conn:
        result = conn.execute(
            query,
            {
                "symbol": symbol.upper(),
                "fiscal_year": fiscal_year,
                "fiscal_period": fiscal_period,
                "adsh": adsh,
            },
        ).fetchone()

    if not result:
        logger.warning(
            "❌ [PROCESSED_TABLE] No data found for %s %s-%s ADSH=%s",
            symbol,
            fiscal_year,
            fiscal_period,
            adsh[:20],
        )
        return None

    row = dict(result._mapping)
    income_qtrs_val = row.get("income_statement_qtrs")
    cashflow_qtrs_val = row.get("cash_flow_statement_qtrs")
    is_ytd_income = fiscal_period_service.is_ytd(income_qtrs_val) if income_qtrs_val else False
    is_ytd_cashflow = fiscal_period_service.is_ytd(cashflow_qtrs_val) if cashflow_qtrs_val else False

    ocf_val = to_float(row.get("operating_cash_flow"))
    capex_val = to_float(row.get("capital_expenditures"))
    raw_fcf = row.get("free_cash_flow")
    free_cash_flow_val = to_float(raw_fcf)
    if (raw_fcf is None or abs(free_cash_flow_val) < 1e-6) and (ocf_val is not None and capex_val is not None):
        derived_fcf = float(ocf_val) - abs(float(capex_val))
        free_cash_flow_val = derived_fcf
        if abs(derived_fcf) > 1e-6:
            logger.debug(
                "🔄 [PROCESSED_TABLE] Derived FCF for %s %s-%s via OCF %.2f - |CapEx| %.2f = %.2f",
                symbol,
                fiscal_year,
                fiscal_period,
                ocf_val,
                capex_val,
                derived_fcf,
            )

    data = {
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "adsh": adsh,
        "income_statement": {
            "total_revenue": to_float(row.get("total_revenue")),
            "net_income": to_float(row.get("net_income")),
            "gross_profit": to_float(row.get("gross_profit")),
            "operating_income": to_float(row.get("operating_income")),
            "cost_of_revenue": to_float(row.get("cost_of_revenue")),
            "research_and_development_expense": to_float(row.get("research_and_development_expense")),
            "selling_general_administrative_expense": to_float(row.get("selling_general_administrative_expense")),
            "operating_expenses": to_float(row.get("operating_expenses")),
            "interest_expense": to_float(row.get("interest_expense")),
            "income_tax_expense": to_float(row.get("income_tax_expense")),
            "earnings_per_share": to_float(row.get("earnings_per_share")),
            "earnings_per_share_diluted": to_float(row.get("earnings_per_share_diluted")),
            "preferred_stock_dividends": to_float(row.get("preferred_stock_dividends")),
            "common_stock_dividends": to_float(row.get("common_stock_dividends")),
            "weighted_average_diluted_shares_outstanding": to_float(
                row.get("weighted_average_diluted_shares_outstanding")
            ),
            "is_ytd": is_ytd_income,
        },
        "cash_flow": {
            "operating_cash_flow": ocf_val,
            "capital_expenditures": capex_val,
            "free_cash_flow": free_cash_flow_val,
            "dividends_paid": to_float(row.get("dividends_paid")),
            "investing_cash_flow": to_float(row.get("investing_cash_flow")),
            "financing_cash_flow": to_float(row.get("financing_cash_flow")),
            "depreciation_amortization": to_float(row.get("depreciation_amortization")),
            "stock_based_compensation": to_float(row.get("stock_based_compensation")),
            "preferred_stock_dividends": to_float(row.get("preferred_stock_dividends")),
            "common_stock_dividends": to_float(row.get("common_stock_dividends")),
            "is_ytd": is_ytd_cashflow,
        },
        "balance_sheet": {
            "total_assets": to_float(row.get("total_assets")),
            "total_liabilities": to_float(row.get("total_liabilities")),
            "stockholders_equity": to_float(row.get("stockholders_equity")),
            "current_assets": to_float(row.get("current_assets")),
            "current_liabilities": to_float(row.get("current_liabilities")),
            "retained_earnings": to_float(row.get("retained_earnings")),
            "accounts_payable": to_float(row.get("accounts_payable")),
            "accrued_liabilities": to_float(row.get("accrued_liabilities")),
            "long_term_debt": to_float(row.get("long_term_debt")),
            "short_term_debt": to_float(row.get("short_term_debt")),
            "total_debt": to_float(row.get("total_debt")),
            "net_debt": to_float(row.get("net_debt")),
            "cash_and_equivalents": to_float(row.get("cash_and_equivalents")),
            "accounts_receivable": to_float(row.get("accounts_receivable")),
            "inventory": to_float(row.get("inventory")),
            "property_plant_equipment": to_float(row.get("property_plant_equipment")),
            "accumulated_depreciation": to_float(row.get("accumulated_depreciation")),
            "property_plant_equipment_net": to_float(row.get("property_plant_equipment_net")),
            "goodwill": to_float(row.get("goodwill")),
            "intangible_assets": to_float(row.get("intangible_assets")),
            "deferred_revenue": to_float(row.get("deferred_revenue")),
            "treasury_stock": to_float(row.get("treasury_stock")),
            "other_comprehensive_income": to_float(row.get("other_comprehensive_income")),
            "book_value": to_float(row.get("book_value")),
            "book_value_per_share": to_float(row.get("book_value_per_share")),
            "working_capital": to_float(row.get("working_capital")),
        },
        "market_metrics": {
            "market_cap": to_float(row.get("market_cap")),
            "enterprise_value": to_float(row.get("enterprise_value")),
            "shares_outstanding": to_float(row.get("shares_outstanding")),
        },
        "ratios": {
            "liquidity": {
                "current_ratio": to_float(row.get("current_ratio")),
                "quick_ratio": to_float(row.get("quick_ratio")),
            },
            "leverage": {
                "debt_to_equity": to_float(row.get("debt_to_equity")),
                "interest_coverage": to_float(row.get("interest_coverage")),
            },
            "profitability": {
                "roa": to_float(row.get("roa")),
                "roe": to_float(row.get("roe")),
                "gross_margin": to_float(row.get("gross_margin")),
                "operating_margin": to_float(row.get("operating_margin")),
                "net_margin": to_float(row.get("net_margin")),
            },
            "efficiency": {"asset_turnover": to_float(row.get("asset_turnover"))},
            "distribution": {
                "dividend_payout_ratio": to_float(row.get("dividend_payout_ratio")),
                "dividend_yield": to_float(row.get("dividend_yield")),
            },
            "tax": {"effective_tax_rate": to_float(row.get("effective_tax_rate"))},
        },
        "data_quality_score": to_float(row.get("data_quality_score")),
    }

    interest_expense = data["income_statement"].get("interest_expense", 0)
    income_tax_expense = data["income_statement"].get("income_tax_expense", 0)
    logger.info(
        "🔍 [FUNDAMENTAL_FETCH] %s %s-%s income_statement: interest_expense=$%.2fM, income_tax=$%.2fB",
        symbol,
        fiscal_year,
        fiscal_period,
        interest_expense / 1e6,
        income_tax_expense / 1e9,
    )

    revenue = data["income_statement"]["total_revenue"]
    ocf = data["cash_flow"]["operating_cash_flow"]
    capex = data["cash_flow"]["capital_expenditures"]
    fcf = data["cash_flow"]["free_cash_flow"]
    inc_ytd_str = "YTD" if is_ytd_income else "PIT"
    cf_ytd_str = "YTD" if is_ytd_cashflow else "PIT"
    logger.info(
        "✅ [PROCESSED_TABLE] %s %s-%s: Income(%s, qtrs=%s), CashFlow(%s, qtrs=%s)",
        symbol,
        fiscal_year,
        fiscal_period,
        inc_ytd_str,
        income_qtrs_val or "NULL",
        cf_ytd_str,
        cashflow_qtrs_val or "NULL",
    )
    logger.info(
        "   📊 Raw DB Values: Revenue=$%.2fB, OCF=$%.2fB, CapEx=$%.2fB, FCF=$%.2fB",
        revenue / 1e9,
        ocf / 1e9,
        capex / 1e9,
        fcf / 1e9,
    )
    return data
