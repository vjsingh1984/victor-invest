"""Helpers for company-level processed-table fetch and compatibility mapping."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from sqlalchemy import text


def fetch_latest_company_data_from_processed_table(
    *,
    symbol: str,
    db_manager: Any,
    logger: Any,
    processed_additional_financial_keys: Sequence[str],
    processed_ratio_keys: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """
    Fetch latest company-level snapshot from `sec_companyfacts_processed`.

    Returns a compatibility payload matching the legacy extractor contract:
    `financial_metrics`, `financial_ratios`, `data_quality_score`, and `source`.
    """
    with db_manager.engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT *
                FROM sec_companyfacts_processed
                WHERE symbol = :symbol
                ORDER BY
                    fiscal_year DESC,
                    CASE fiscal_period
                        WHEN 'FY' THEN 4
                        WHEN 'Q4' THEN 3
                        WHEN 'Q3' THEN 2
                        WHEN 'Q2' THEN 1
                        WHEN 'Q1' THEN 0
                    END DESC
                LIMIT 1
            """
            ),
            {"symbol": symbol},
        ).fetchone()

        if not result:
            logger.warning("[CLEAN ARCH] No processed data found for %s in sec_companyfacts_processed", symbol)
            return None

        row = dict(result._mapping)

        def safe_float(key: str) -> float:
            value = row.get(key)
            if value is None:
                return 0.0
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        revenue = safe_float("total_revenue")
        fiscal_year = row.get("fiscal_year")
        fiscal_period = row.get("fiscal_period")

        if revenue < 0:
            logger.error(
                "❌ CORRUPT DATA DETECTED: %s %s-%s has NEGATIVE revenue: $%s. "
                "This indicates failed YTD conversion. DELETING corrupt record and forcing re-fetch.",
                symbol,
                fiscal_year,
                fiscal_period,
                format(revenue, ",.0f"),
            )
            conn.execute(
                text(
                    """
                    DELETE FROM sec_companyfacts_processed
                    WHERE symbol = :symbol
                      AND fiscal_year = :fiscal_year
                      AND fiscal_period = :fiscal_period
                """
                ),
                {"symbol": symbol, "fiscal_year": fiscal_year, "fiscal_period": fiscal_period},
            )
            conn.commit()
            logger.warning(
                "⚠️  Deleted corrupt record for %s %s-%s. "
                "SEC Agent should re-fetch and reprocess this period in next run.",
                symbol,
                fiscal_year,
                fiscal_period,
            )
            return None

        if revenue == 0 and fiscal_period != "Q1":
            logger.warning(
                "⚠️  %s %s-%s has ZERO revenue. May indicate incomplete data or failed YTD conversion.",
                symbol,
                fiscal_year,
                fiscal_period,
            )

        logger.info(
            "[CLEAN ARCH] Fetched company data for %s from processed table: "
            "%s-%s (filed: %s) | Revenue: $%s",
            symbol,
            fiscal_year,
            fiscal_period,
            row.get("filed_date"),
            format(revenue, ",.0f"),
        )

        filed_date = row.get("filed_date").isoformat() if row.get("filed_date") else None
        financial_metrics: Dict[str, Any] = {
            "revenues": safe_float("total_revenue"),
            "net_income": safe_float("net_income"),
            "gross_profit": safe_float("gross_profit"),
            "operating_income": safe_float("operating_income"),
            "cost_of_revenue": safe_float("cost_of_revenue"),
            "assets": safe_float("total_assets"),
            "equity": safe_float("stockholders_equity"),
            "assets_current": safe_float("current_assets"),
            "liabilities_current": safe_float("current_liabilities"),
            "liabilities": safe_float("total_liabilities"),
            "total_debt": safe_float("total_debt"),
            "long_term_debt": safe_float("long_term_debt"),
            "debt_current": safe_float("short_term_debt"),
            "inventory": safe_float("inventory"),
            "cash_and_equivalents": safe_float("cash_and_equivalents"),
            "operating_cash_flow": safe_float("operating_cash_flow"),
            "capital_expenditures": safe_float("capital_expenditures"),
            "free_cash_flow": safe_float("free_cash_flow"),
            "shares_outstanding": safe_float("shares_outstanding"),
            "fiscal_year": row.get("fiscal_year"),
            "fiscal_period": row.get("fiscal_period"),
            "symbol": row.get("symbol"),
            "data_date": filed_date,
            "weighted_average_diluted_shares_outstanding": safe_float(
                "weighted_average_diluted_shares_outstanding"
            ),
        }

        for key in processed_additional_financial_keys:
            financial_metrics[key] = safe_float(key)

        long_term_debt = financial_metrics.get("long_term_debt") or 0.0
        short_term_debt = financial_metrics.get("debt_current") or 0.0
        if not financial_metrics.get("total_debt") and (long_term_debt or short_term_debt):
            financial_metrics["total_debt"] = long_term_debt + short_term_debt

        if not financial_metrics.get("cash"):
            cash_guess = financial_metrics.get("cash_and_equivalents") or safe_float("cash")
            financial_metrics["cash"] = cash_guess

        if not financial_metrics.get("shares_outstanding"):
            shares_guess = financial_metrics.get("weighted_average_diluted_shares_outstanding") or safe_float(
                "shares_outstanding"
            )
            financial_metrics["shares_outstanding"] = shares_guess

        financial_ratios: Dict[str, Any] = {
            "current_ratio": safe_float("current_ratio"),
            "quick_ratio": safe_float("quick_ratio"),
            "debt_to_equity": safe_float("debt_to_equity"),
            "roe": safe_float("roe"),
            "roa": safe_float("roa"),
            "gross_margin": safe_float("gross_margin"),
            "operating_margin": safe_float("operating_margin"),
            "net_margin": safe_float("net_margin"),
            "symbol": row.get("symbol"),
            "data_date": filed_date,
            "raw_metrics": financial_metrics,
        }

        for key in processed_ratio_keys:
            financial_ratios[key] = safe_float(key)

        return {
            "financial_metrics": financial_metrics,
            "financial_ratios": financial_ratios,
            "data_quality_score": safe_float("data_quality_score"),
            "source": "clean_architecture",
        }
