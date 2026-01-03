#!/usr/bin/env python3
"""
Verify normalized SEC CompanyFacts data for a given symbol.

This helper script reprocesses the RAW CompanyFacts JSON stored in
`sec_companyfacts_raw`, compares the normalized point-in-time values with the
flattened rows inside `sec_companyfacts_processed`, and optionally fixes legacy
`*_statement_qtrs` flags that still indicate YTD values after normalization.

Usage examples (run from repository root):

    # Reprocess and validate the latest 8 quarters (writes back to DB)
    PYTHONPATH=src source ../investment_ai_env/bin/activate
    python scripts/verify_sec_processed.py --symbol AAPL --quarters 8

    # Inspect without writing changes
    python scripts/verify_sec_processed.py --symbol AAPL --quarters 8 --dry-run

The script expects database credentials to be available via the standard config
(config.py) or environment variables consumed by `config.get_config()`.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sqlalchemy import text

from investigator.infrastructure.sec.data_processor import SECDataProcessor
from investigator.infrastructure.database.db import get_db_manager

logger = logging.getLogger("verify_sec_processed")


@dataclass
class FilingSnapshot:
    fiscal_year: int
    fiscal_period: str
    form_type: str
    income_qtrs: int
    cashflow_qtrs: int
    revenue: float
    ocf: float
    capex: float
    fcf: float

    @property
    def key(self) -> Tuple[int, str]:
        return (self.fiscal_year, self.fiscal_period)


PERIOD_SORT_ORDER = {
    "FY": 5,
    "Q4": 4,
    "Q3": 3,
    "Q2": 2,
    "Q1": 1,
}


def _period_rank(period: str) -> int:
    period = (period or "").upper()
    return PERIOD_SORT_ORDER.get(period, 0)


def _collect_processed_rows(conn, symbol: str, limit: int) -> Dict[Tuple[int, str], FilingSnapshot]:
    """Fetch the latest rows from sec_companyfacts_processed for a symbol."""
    query = text(
        """
        SELECT fiscal_year,
               fiscal_period,
               form_type,
               COALESCE(income_statement_qtrs, 1) AS income_qtrs,
               COALESCE(cash_flow_statement_qtrs, 1) AS cashflow_qtrs,
               COALESCE(total_revenue, 0) AS revenue,
               COALESCE(operating_cash_flow, 0) AS operating_cash_flow,
               COALESCE(capital_expenditures, 0) AS capital_expenditures,
               COALESCE(free_cash_flow, 0) AS free_cash_flow
          FROM sec_companyfacts_processed
         WHERE symbol = :symbol
         ORDER BY fiscal_year DESC,
                  CASE UPPER(fiscal_period)
                      WHEN 'FY' THEN 5
                      WHEN 'Q4' THEN 4
                      WHEN 'Q3' THEN 3
                      WHEN 'Q2' THEN 2
                      WHEN 'Q1' THEN 1
                      ELSE 0
                  END DESC
         LIMIT :limit
        """
    )
    rows = conn.execute(query, {"symbol": symbol.upper(), "limit": limit}).fetchall()

    snapshots: Dict[Tuple[int, str], FilingSnapshot] = {}
    for row in rows:
        snapshot = FilingSnapshot(
            fiscal_year=row.fiscal_year,
            fiscal_period=row.fiscal_period,
            form_type=row.form_type,
            income_qtrs=row.income_qtrs,
            cashflow_qtrs=row.cashflow_qtrs,
            revenue=float(row.revenue or 0),
            ocf=float(row.operating_cash_flow or 0),
            capex=float(row.capital_expenditures or 0),
            fcf=float(row.free_cash_flow or 0),
        )
        snapshots[snapshot.key] = snapshot
    return snapshots


def _collect_reprocessed_rows(
    symbol: str,
    raw_companyfacts: Dict,
    raw_data_id: int,
    processor: SECDataProcessor,
    persist: bool,
) -> Dict[Tuple[int, str], FilingSnapshot]:
    """
    Re-run the SECDataProcessor for the raw JSON to obtain normalized data
    optionally writing back to the database.
    """
    filings = processor.process_raw_data(
        symbol=symbol,
        raw_data=raw_companyfacts,
        raw_data_id=raw_data_id,
        extraction_version="verification",
        persist=persist,
    )

    snapshots: Dict[Tuple[int, str], FilingSnapshot] = {}
    for filing in filings:
        period = str(filing.get("fiscal_period")).upper()
        year = int(filing.get("fiscal_year") or 0)

        # Skip malformed entries
        if not year or not period:
            continue

        data = filing.get("data", {})
        snapshots[(year, period)] = FilingSnapshot(
            fiscal_year=year,
            fiscal_period=period,
            form_type=str(filing.get("form_type", "")),
            income_qtrs=int(filing.get("income_statement_qtrs") or 0),
            cashflow_qtrs=int(filing.get("cash_flow_statement_qtrs") or 0),
            revenue=float(data.get("total_revenue") or 0),
            ocf=float(data.get("operating_cash_flow") or 0),
            capex=float(data.get("capital_expenditures") or 0),
            fcf=float(data.get("free_cash_flow") or 0),
        )

    return snapshots


def _format_snapshot(snapshot: FilingSnapshot) -> str:
    return (
        f"{snapshot.fiscal_year}-{snapshot.fiscal_period:<2} "
        f"| Rev ${snapshot.revenue/1e9:6.2f}B "
        f"| OCF ${snapshot.ocf/1e9:6.2f}B "
        f"| CapEx ${snapshot.capex/1e9:6.2f}B "
        f"| FCF ${snapshot.fcf/1e9:6.2f}B "
        f"| inc_qtrs={snapshot.income_qtrs} "
        f"| cf_qtrs={snapshot.cashflow_qtrs}"
    )


def _print_comparison(
    processed: Dict[Tuple[int, str], FilingSnapshot],
    reprocessed: Dict[Tuple[int, str], FilingSnapshot],
    limit: int,
) -> None:
    combined_keys = sorted(
        set(processed.keys()).union(reprocessed.keys()),
        key=lambda key: (key[0], _period_rank(key[1])),
        reverse=True,
    )[:limit]

    print("\n=== Normalized Quarter Validation ===")
    for key in combined_keys:
        proc = processed.get(key)
        repro = reprocessed.get(key)

        if repro:
            print(f"[RAW→PIT] {_format_snapshot(repro)}")
        else:
            print(f"[RAW→PIT] {key} missing")

        if proc:
            marker = ""
            if repro and (
                abs(proc.revenue - repro.revenue) > 1e-2
                or abs(proc.ocf - repro.ocf) > 1e-2
                or abs(proc.capex - repro.capex) > 1e-2
                or abs(proc.fcf - repro.fcf) > 1e-2
            ):
                marker = "  ⚠︎ value mismatch"
            elif repro and (proc.income_qtrs != repro.income_qtrs or proc.cashflow_qtrs != repro.cashflow_qtrs):
                marker = "  ⚠︎ qtrs mismatch"

            print(f"[DB      ] {_format_snapshot(proc)}{marker}")
        else:
            print(f"[DB      ] {key} missing")

        print("-")


def _apply_flag_fixes(conn, symbol: str, fix_fcf: bool) -> None:
    """Normalize qtrs flags and optionally backfill FCF inside the processed table."""
    fixes = [
        (
            "UPDATE sec_companyfacts_processed "
            "SET cash_flow_statement_qtrs = 1 "
            "WHERE symbol = :symbol "
            "  AND fiscal_period IN ('Q2', 'Q3') "
            "  AND cash_flow_statement_qtrs IS NOT NULL "
            "  AND cash_flow_statement_qtrs > 1",
            "cash_flow_statement_qtrs → 1",
        ),
        (
            "UPDATE sec_companyfacts_processed "
            "SET income_statement_qtrs = 1 "
            "WHERE symbol = :symbol "
            "  AND fiscal_period IN ('Q2', 'Q3') "
            "  AND income_statement_qtrs IS NOT NULL "
            "  AND income_statement_qtrs > 1",
            "income_statement_qtrs → 1",
        ),
    ]

    if fix_fcf:
        fixes.append(
            (
                "UPDATE sec_companyfacts_processed "
                "SET free_cash_flow = operating_cash_flow - ABS(capital_expenditures) "
                "WHERE symbol = :symbol "
                "  AND (free_cash_flow IS NULL OR ABS(free_cash_flow) < 1e-6) "
                "  AND operating_cash_flow IS NOT NULL "
                "  AND capital_expenditures IS NOT NULL",
                "free_cash_flow derived from OCF - |CapEx|",
            )
        )

    for sql, description in fixes:
        result = conn.execute(text(sql), {"symbol": symbol.upper()})
        print(f"Applied fix [{description}] - rows affected: {result.rowcount}")

    conn.commit()


def run(symbol: str, quarters: int, apply_flag_fix: bool, fix_fcf: bool, persist: bool) -> None:
    db_manager = get_db_manager()
    engine = db_manager.engine
    processor = SECDataProcessor(db_engine=engine)

    with engine.connect() as conn:
        raw_row = conn.execute(
            text(
                """
                SELECT id, companyfacts
                  FROM sec_companyfacts_raw
                 WHERE symbol = :symbol
                 LIMIT 1
                """
            ),
            {"symbol": symbol.upper()},
        ).fetchone()

        if not raw_row:
            raise RuntimeError(f"No RAW companyfacts found for {symbol}")

        processed = _collect_processed_rows(conn, symbol, quarters * 2)
        reprocessed = _collect_reprocessed_rows(symbol, raw_row.companyfacts, raw_row.id, processor, persist)

        _print_comparison(processed, reprocessed, quarters)

        if apply_flag_fix:
            _apply_flag_fixes(conn, symbol, fix_fcf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate sec_companyfacts_processed normalized values.")
    parser.add_argument("--symbol", required=True, help="Ticker symbol to validate (e.g., AAPL)")
    parser.add_argument(
        "--quarters",
        type=int,
        default=8,
        help="Number of recent periods to display (default: 8)",
    )
    parser.add_argument(
        "--apply-flag-fix",
        action="store_true",
        help="Update *_statement_qtrs flags to 1 for normalized quarters in the database",
    )
    parser.add_argument(
        "--fix-fcf",
        action="store_true",
        help="When combined with --apply-flag-fix, derive missing free_cash_flow values from OCF - |CapEx|.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not persist reprocessed data back to sec_companyfacts_processed; compare only.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    run(
        symbol=args.symbol,
        quarters=args.quarters,
        apply_flag_fix=args.apply_flag_fix,
        fix_fcf=args.fix_fcf,
        persist=not args.dry_run,
    )


if __name__ == "__main__":
    main()
