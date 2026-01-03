"""
SEC Data Sources

Provides SEC filing data including insider transactions (Form 4),
institutional holdings (Form 13F), and quarterly financials.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
import logging

from ..base import (
    DataSource, DataResult, SourceMetadata,
    DataCategory, DataFrequency, DataQuality
)
from ..registry import register_source

logger = logging.getLogger(__name__)


@register_source("insider_transactions", DataCategory.SENTIMENT)
class InsiderTransactionSource(DataSource):
    """
    SEC Form 4 Insider Transaction Data

    Provides:
    - Recent insider buys/sells
    - Aggregate insider sentiment
    - Cluster buying/selling detection
    """

    def __init__(self):
        super().__init__("insider_transactions", DataCategory.SENTIMENT, DataFrequency.DAILY)

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name="insider_transactions",
            category=DataCategory.SENTIMENT,
            frequency=DataFrequency.DAILY,
            description="SEC Form 4 insider trading data",
            provider="SEC Edgar",
            is_free=True,
            requires_api_key=False,
            lookback_days=365,
            symbols_supported=True,
        )

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """Fetch insider transaction data"""
        try:
            from investigator.infrastructure.database.db import get_db_manager
            from sqlalchemy import text

            engine = get_db_manager().engine
            target_date = as_of_date or date.today()
            start_date = target_date - timedelta(days=90)

            with engine.connect() as conn:
                # Get recent transactions from form4_filings
                result = conn.execute(
                    text("""
                        SELECT
                            owner_name,
                            filing_date,
                            transaction_code,
                            shares,
                            price_per_share,
                            total_value,
                            is_director,
                            is_officer,
                            owner_title
                        FROM form4_filings
                        WHERE symbol = :symbol
                        AND filing_date >= :start_date
                        AND filing_date <= :target_date
                        ORDER BY filing_date DESC
                        LIMIT 50
                    """),
                    {"symbol": symbol, "start_date": start_date, "target_date": target_date}
                )

                transactions = []
                total_bought = 0
                total_sold = 0

                for row in result:
                    tx_code = row[2] or ""
                    tx = {
                        "owner": row[0],
                        "date": row[1].isoformat() if row[1] else None,
                        "type": "BUY" if tx_code in ("P", "A") else "SELL" if tx_code == "S" else "OTHER",
                        "shares": float(row[3]) if row[3] else None,
                        "price": float(row[4]) if row[4] else None,
                        "total_value": float(row[5]) if row[5] else None,
                        "is_director": row[6],
                        "is_officer": row[7],
                        "title": row[8],
                    }
                    transactions.append(tx)

                    if tx["shares"]:
                        if tx["type"] == "BUY":
                            total_bought += tx["shares"]
                        elif tx["type"] == "SELL":
                            total_sold += tx["shares"]

                # Get sentiment score
                result = conn.execute(
                    text("""
                        SELECT
                            sentiment_score,
                            buy_count,
                            sell_count,
                            buy_value,
                            sell_value,
                            calculation_date
                        FROM insider_sentiment
                        WHERE symbol = :symbol
                        ORDER BY calculation_date DESC
                        LIMIT 1
                    """),
                    {"symbol": symbol}
                )
                sentiment_row = result.fetchone()

            if not transactions and not sentiment_row:
                return DataResult(
                    success=False,
                    error=f"No insider data for {symbol}",
                    source=self.name,
                )

            # Calculate metrics
            buy_count = sum(1 for t in transactions if t["type"] == "BUY")
            sell_count = sum(1 for t in transactions if t["type"] == "SELL")

            sentiment = "neutral"
            if buy_count > sell_count * 2:
                sentiment = "strongly_bullish"
            elif buy_count > sell_count:
                sentiment = "bullish"
            elif sell_count > buy_count * 2:
                sentiment = "strongly_bearish"
            elif sell_count > buy_count:
                sentiment = "bearish"

            data = {
                "symbol": symbol,
                "transactions": transactions[:20],  # Limit to 20 most recent
                "summary": {
                    "total_transactions": len(transactions),
                    "buys": buy_count,
                    "sells": sell_count,
                    "total_bought": total_bought,
                    "total_sold": total_sold,
                    "net_shares": total_bought - total_sold,
                    "sentiment": sentiment,
                },
                "period": {
                    "start": start_date.isoformat(),
                    "end": target_date.isoformat(),
                },
            }

            if sentiment_row:
                data["sentiment_score"] = {
                    "score": float(sentiment_row[0]) if sentiment_row[0] else None,
                    "db_buy_count": sentiment_row[1],
                    "db_sell_count": sentiment_row[2],
                    "buy_value": float(sentiment_row[3]) if sentiment_row[3] else None,
                    "sell_value": float(sentiment_row[4]) if sentiment_row[4] else None,
                    "as_of": sentiment_row[5].isoformat() if sentiment_row[5] else None,
                }

            return DataResult(
                success=True,
                data=data,
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except Exception as e:
            logger.error(f"Insider data fetch error: {e}")
            return DataResult(success=False, error=str(e), source=self.name)


@register_source("institutional_holdings", DataCategory.SENTIMENT)
class InstitutionalHoldingsSource(DataSource):
    """
    SEC Form 13F Institutional Holdings Data

    Provides:
    - Top institutional holders
    - Ownership changes
    - Institutional sentiment
    """

    def __init__(self):
        super().__init__("institutional_holdings", DataCategory.SENTIMENT, DataFrequency.QUARTERLY)

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name="institutional_holdings",
            category=DataCategory.SENTIMENT,
            frequency=DataFrequency.QUARTERLY,
            description="SEC Form 13F institutional holdings",
            provider="SEC Edgar",
            is_free=True,
            requires_api_key=False,
            lookback_days=365 * 2,
            symbols_supported=True,
        )

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """Fetch institutional holdings data"""
        try:
            from investigator.infrastructure.database.db import get_db_manager
            from sqlalchemy import text

            engine = get_db_manager().engine

            with engine.connect() as conn:
                # Get latest holdings - join with institutions for manager name
                result = conn.execute(
                    text("""
                        SELECT
                            i.name as manager_name,
                            h.shares,
                            h.value_thousands,
                            f.report_quarter,
                            f.total_value as filing_total_value
                        FROM form13f_holdings h
                        JOIN form13f_filings f ON h.filing_id = f.id
                        JOIN institutions i ON f.institution_id = i.id
                        WHERE h.symbol = :symbol
                        ORDER BY f.report_quarter DESC, h.value_thousands DESC
                        LIMIT 100
                    """),
                    {"symbol": symbol}
                )

                holdings = []
                latest_date = None
                total_shares = 0
                total_value = 0
                holder_count = 0

                for row in result:
                    if latest_date is None:
                        latest_date = row[3]

                    # Only include holdings from latest quarter
                    if row[3] == latest_date:
                        value_dollars = float(row[2]) * 1000 if row[2] else None
                        holdings.append({
                            "manager": row[0],
                            "shares": int(row[1]) if row[1] else None,
                            "value": value_dollars,
                        })
                        total_shares += int(row[1]) if row[1] else 0
                        total_value += value_dollars or 0
                        holder_count += 1

                ownership = None  # institutional_ownership table may not exist

            if not holdings:
                return DataResult(
                    success=False,
                    error=f"No institutional data for {symbol}",
                    source=self.name,
                )

            data = {
                "symbol": symbol,
                "top_holders": holdings[:20],
                "summary": {
                    "total_holders": holder_count,
                    "total_shares": total_shares,
                    "total_value": total_value,
                },
                "report_date": latest_date.isoformat() if latest_date else None,
            }

            return DataResult(
                success=True,
                data=data,
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except Exception as e:
            logger.error(f"Institutional holdings fetch error: {e}")
            return DataResult(success=False, error=str(e), source=self.name)


@register_source("sec_quarterly", DataCategory.FUNDAMENTAL)
class SECQuarterlySource(DataSource):
    """
    SEC Quarterly Financial Data

    Provides:
    - Income statement metrics
    - Balance sheet metrics
    - Cash flow metrics
    - Key ratios
    """

    def __init__(self):
        super().__init__("sec_quarterly", DataCategory.FUNDAMENTAL, DataFrequency.QUARTERLY)

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            name="sec_quarterly",
            category=DataCategory.FUNDAMENTAL,
            frequency=DataFrequency.QUARTERLY,
            description="SEC quarterly financial statements",
            provider="SEC Edgar",
            is_free=True,
            requires_api_key=False,
            lookback_days=365 * 5,
            symbols_supported=True,
        )

    def _fetch_impl(self, symbol: str, as_of_date: Optional[date] = None) -> DataResult:
        """Fetch quarterly financial data"""
        try:
            from investigator.infrastructure.database.db import get_db_manager
            from sqlalchemy import text

            engine = get_db_manager().engine
            target_date = as_of_date or date.today()

            with engine.connect() as conn:
                # quarterly_metrics stores data in JSONB metrics_data column
                result = conn.execute(
                    text("""
                        SELECT
                            fiscal_year,
                            fiscal_period,
                            metrics_data,
                            calculated_at
                        FROM quarterly_metrics
                        WHERE symbol = :symbol
                        ORDER BY fiscal_year DESC, fiscal_period DESC
                        LIMIT 8
                    """),
                    {"symbol": symbol}
                )

                quarters = []
                for row in result:
                    metrics = row[2] or {}  # JSONB data
                    quarters.append({
                        "fiscal_year": row[0],
                        "fiscal_period": row[1],
                        "revenue": metrics.get("revenue") or metrics.get("revenues"),
                        "net_income": metrics.get("net_income") or metrics.get("netIncome"),
                        "operating_income": metrics.get("operating_income") or metrics.get("operatingIncome"),
                        "gross_profit": metrics.get("gross_profit") or metrics.get("grossProfit"),
                        "eps_basic": metrics.get("eps_basic") or metrics.get("basicEPS"),
                        "eps_diluted": metrics.get("eps_diluted") or metrics.get("dilutedEPS"),
                        "total_assets": metrics.get("total_assets") or metrics.get("totalAssets"),
                        "total_liabilities": metrics.get("total_liabilities") or metrics.get("totalLiabilities"),
                        "stockholders_equity": metrics.get("stockholders_equity") or metrics.get("stockholdersEquity"),
                        "operating_cash_flow": metrics.get("operating_cash_flow") or metrics.get("operatingCashFlow"),
                        "free_cash_flow": metrics.get("free_cash_flow") or metrics.get("freeCashFlow"),
                        "calculated_at": row[3].isoformat() if row[3] else None,
                    })

            if not quarters:
                return DataResult(
                    success=False,
                    error=f"No quarterly data for {symbol}",
                    source=self.name,
                )

            # Calculate growth rates
            latest = quarters[0]
            yoy_quarter = quarters[4] if len(quarters) > 4 else None

            growth = {}
            if yoy_quarter and latest.get("revenue") and yoy_quarter.get("revenue"):
                growth["revenue_yoy"] = (latest["revenue"] - yoy_quarter["revenue"]) / abs(yoy_quarter["revenue"]) * 100
            if yoy_quarter and latest.get("net_income") and yoy_quarter.get("net_income"):
                if yoy_quarter["net_income"] > 0:
                    growth["net_income_yoy"] = (latest["net_income"] - yoy_quarter["net_income"]) / yoy_quarter["net_income"] * 100

            return DataResult(
                success=True,
                data={
                    "symbol": symbol,
                    "latest": latest,
                    "quarters": quarters,
                    "growth": growth,
                },
                source=self.name,
                quality=DataQuality.HIGH,
            )

        except Exception as e:
            logger.error(f"SEC quarterly fetch error: {e}")
            return DataResult(success=False, error=str(e), source=self.name)
