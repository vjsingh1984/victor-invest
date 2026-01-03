#!/usr/bin/env python3
"""
End-to-End Sector Coverage Test

Tests extraction for FAANG + one representative stock from each major sector.
Logs detailed output for analysis and issue identification.
"""

import json
import logging
import sys
from datetime import datetime
from decimal import Decimal
from dao.sec_bulk_dao import SECBulkDAO

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f'end_to_end_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

# Test companies: FAANG + one from each sector
TEST_COMPANIES = {
    # FAANG (Technology)
    "AAPL": "Technology - Consumer Electronics",
    "AMZN": "Consumer Discretionary - E-Commerce",
    "META": "Technology - Social Media",
    "GOOGL": "Technology - Internet Services",
    "NFLX": "Communication Services - Streaming",
    # Sector Representatives
    "XOM": "Energy - Oil & Gas",
    "JNJ": "Healthcare - Pharmaceuticals",
    "JPM": "Financials - Banking",
    "PG": "Consumer Staples - Household Products",
    "BA": "Industrials - Aerospace",
    "CAT": "Industrials - Machinery",
    "AMT": "Real Estate - REIT (Telecom Towers)",
    "NEE": "Utilities - Electric Utility",
    "LIN": "Materials - Industrial Gases",
}


def decimal_to_float(obj):
    """Convert Decimal to float for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError


def test_company(dao, symbol, sector, results):
    """Test extraction for a single company."""
    logger.info(f"=" * 80)
    logger.info(f"Testing {symbol} - {sector}")
    logger.info(f"=" * 80)

    try:
        metrics = dao.fetch_financial_metrics(symbol, 2024, "FY")

        if "error" in metrics:
            logger.error(f"{symbol}: Extraction failed with error: {metrics['error']}")
            results[symbol] = {"status": "ERROR", "sector": sector, "error": metrics["error"]}
            return

        # Count extracted metrics
        metric_count = len(
            [k for k in metrics.keys() if k not in ["symbol", "fiscal_year", "fiscal_period"] and not k.startswith("_")]
        )

        # Check critical metrics
        critical_metrics = {
            "total_revenue": metrics.get("total_revenue"),
            "net_income": metrics.get("net_income"),
            "total_assets": metrics.get("total_assets"),
            "total_liabilities": metrics.get("total_liabilities"),
        }

        critical_count = sum(1 for v in critical_metrics.values() if v is not None)

        # Check for calculated metrics
        calculated = []
        if metrics.get("_total_revenue_calculated"):
            calculated.append("revenue")
        if metrics.get("_net_income_calculated"):
            calculated.append("net_income")
        if metrics.get("_total_liabilities_calculated"):
            calculated.append("liabilities")

        logger.info(f"{symbol} Results:")
        logger.info(f"  Total metrics extracted: {metric_count}")
        logger.info(f"  Critical metrics: {critical_count}/4")
        logger.info(
            f"  Revenue: ${critical_metrics['total_revenue']:,.0f}"
            if critical_metrics["total_revenue"]
            else "  Revenue: MISSING"
        )
        logger.info(
            f"  Net Income: ${critical_metrics['net_income']:,.0f}"
            if critical_metrics["net_income"]
            else "  Net Income: MISSING"
        )
        logger.info(
            f"  Assets: ${critical_metrics['total_assets']:,.0f}"
            if critical_metrics["total_assets"]
            else "  Assets: MISSING"
        )
        logger.info(
            f"  Liabilities: ${critical_metrics['total_liabilities']:,.0f}"
            if critical_metrics["total_liabilities"]
            else "  Liabilities: MISSING"
        )

        if calculated:
            logger.info(f"  Calculated metrics: {', '.join(calculated)}")

        # Validate balance sheet identity
        if all(
            k in metrics and metrics[k] is not None
            for k in ["total_assets", "total_liabilities", "stockholders_equity"]
        ):
            assets = float(metrics["total_assets"])
            liab = float(metrics["total_liabilities"])
            equity = float(metrics["stockholders_equity"])
            calc_assets = liab + equity
            diff_pct = abs(assets - calc_assets) / assets * 100

            if diff_pct > 15:
                logger.warning(f"  Balance sheet identity check: FAIL ({diff_pct:.1f}% difference)")
                logger.warning(f"    Assets={assets:,.0f}, Liab+Equity={calc_assets:,.0f}")
                logger.warning(
                    f"    Note: >15% difference may indicate data quality issues or unsupported equity structures"
                )
            elif diff_pct > 2:
                logger.info(f"  Balance sheet identity: ACCEPTABLE ({diff_pct:.2f}% difference)")
                logger.info(
                    f"    Note: Utilities/REITs often have temporary equity/NCI not captured in simple stockholders_equity tag"
                )
            else:
                logger.info(f"  Balance sheet identity: OK ({diff_pct:.2f}% difference)")

        results[symbol] = {
            "status": "SUCCESS",
            "sector": sector,
            "metric_count": metric_count,
            "critical_count": critical_count,
            "calculated_metrics": calculated,
            "metrics": metrics,
        }

        logger.info(f"{symbol}: ✓ SUCCESS")

    except Exception as e:
        logger.exception(f"{symbol}: Exception occurred during extraction")
        results[symbol] = {"status": "EXCEPTION", "sector": sector, "error": str(e), "exception_type": type(e).__name__}


def main():
    logger.info("=" * 80)
    logger.info("END-TO-END SECTOR COVERAGE TEST")
    logger.info("=" * 80)
    logger.info(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Testing {len(TEST_COMPANIES)} companies across all major sectors")
    logger.info("=" * 80)
    logger.info("")

    dao = SECBulkDAO()
    results = {}

    for symbol, sector in TEST_COMPANIES.items():
        test_company(dao, symbol, sector, results)
        logger.info("")  # Blank line between companies

    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    success = [s for s, r in results.items() if r["status"] == "SUCCESS"]
    errors = [s for s, r in results.items() if r["status"] == "ERROR"]
    exceptions = [s for s, r in results.items() if r["status"] == "EXCEPTION"]

    logger.info(f"Total companies tested: {len(TEST_COMPANIES)}")
    logger.info(
        f"Successful extractions: {len(success)}/{len(TEST_COMPANIES)} ({len(success)*100/len(TEST_COMPANIES):.0f}%)"
    )
    logger.info(f"Errors: {len(errors)}")
    logger.info(f"Exceptions: {len(exceptions)}")
    logger.info("")

    if success:
        avg_metrics = sum(r["metric_count"] for s, r in results.items() if r["status"] == "SUCCESS") / len(success)
        avg_critical = sum(r["critical_count"] for s, r in results.items() if r["status"] == "SUCCESS") / len(success)
        logger.info(f"Average metrics per company: {avg_metrics:.1f}")
        logger.info(f"Average critical metrics: {avg_critical:.1f}/4")
        logger.info("")

        # Count calculated metrics usage
        calc_revenue = sum(1 for r in results.values() if "revenue" in r.get("calculated_metrics", []))
        calc_income = sum(1 for r in results.values() if "net_income" in r.get("calculated_metrics", []))
        calc_liab = sum(1 for r in results.values() if "liabilities" in r.get("calculated_metrics", []))

        logger.info("Financial Calculator Usage:")
        logger.info(f"  Revenue calculated: {calc_revenue} companies")
        logger.info(f"  Net income calculated: {calc_income} companies")
        logger.info(f"  Liabilities calculated: {calc_liab} companies")
        logger.info("")

    if errors or exceptions:
        logger.warning("FAILED COMPANIES:")
        for symbol in errors + exceptions:
            r = results[symbol]
            logger.warning(f"  {symbol} ({r['sector']}): {r.get('error', r.get('exception_type'))}")
        logger.warning("")

    # Sector breakdown
    logger.info("SECTOR BREAKDOWN:")
    for symbol, data in sorted(results.items()):
        status_icon = "✓" if data["status"] == "SUCCESS" else "✗"
        logger.info(f"  {status_icon} {symbol:6s} - {data['sector']}")

    # Save detailed results
    output_file = f'end_to_end_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=decimal_to_float)
    logger.info("")
    logger.info(f"Detailed results saved to: {output_file}")

    logger.info("=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)

    return 0 if len(success) == len(TEST_COMPANIES) else 1


if __name__ == "__main__":
    sys.exit(main())
