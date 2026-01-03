#!/usr/bin/env python3
"""
RL Backtest Workflow Runner - StateGraph-based RL backtesting CLI

This script provides a CLI interface to the RL backtest workflow which uses
victor-core's StateGraph pattern and shared services for consistent behavior.

Features:
- Uses victor_invest.workflows.run_rl_backtest for execution
- Consistent with batch_analysis_runner.py and victor_invest CLI
- Shared services: SharesService, PriceService, TechnicalAnalysisService
- Multi-period reward calculation stored in JSONB

Usage:
    # Single symbol backtest
    python3 scripts/rl_backtest_workflow.py --symbols AAPL --max-lookback 120

    # Multiple symbols with parallel processing
    python3 scripts/rl_backtest_workflow.py --symbols AAPL MSFT GOOGL --parallel 3

    # All symbols with quarterly interval
    python3 scripts/rl_backtest_workflow.py --all-symbols --max-lookback 60 --interval quarterly

Author: Victor-Invest Team
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text

# Victor-Invest workflow imports
from victor_invest.workflows import (
    run_rl_backtest,
    run_rl_backtest_batch,
    generate_lookback_list,
    RLBacktestWorkflowState,
)

# Create logs directory
Path("logs").mkdir(exist_ok=True)

# Configure logging
log_filename = f"logs/rl_backtest_workflow_{datetime.now():%Y%m%d_%H%M%S}.log"

logger = logging.getLogger("rl_backtest_workflow")
logger.setLevel(logging.INFO)
logger.handlers = []

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Suppress noisy loggers
for noisy in ["investigator", "victor_invest", "sqlalchemy", "urllib3", "httpx"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger.info(f"Logging to: {log_filename}")


def get_db_engine():
    """Get database engine."""
    return create_engine(
        "postgresql://investigator:${SEC_DB_PASSWORD}@${SEC_DB_HOST}/sec_database"
    )


def get_all_eligible_symbols() -> List[str]:
    """Get all eligible symbols from database."""
    engine = get_db_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT DISTINCT symbol
            FROM stock_symbols
            WHERE is_active = true
              AND (is_sp500 = true OR is_russell1000 = true)
            ORDER BY symbol
        """))
        return [row[0] for row in result.fetchall()]


def get_top_n_symbols(n: int) -> List[str]:
    """Get top N symbols by market cap."""
    engine = get_db_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT symbol
            FROM stock_symbols
            WHERE is_active = true
              AND market_cap IS NOT NULL
            ORDER BY market_cap DESC
            LIMIT :n
        """), {"n": n})
        return [row[0] for row in result.fetchall()]


async def run_backtest_for_symbol(
    symbol: str,
    max_lookback_months: int,
    interval: str,
) -> RLBacktestWorkflowState:
    """Run backtest for a single symbol."""
    logger.info(f"Starting backtest for {symbol}")
    try:
        result = await run_rl_backtest(
            symbol=symbol,
            max_lookback_months=max_lookback_months,
            interval=interval,
        )
        summary = result.metadata.get("summary", {})
        logger.info(
            f"{symbol}: {summary.get('successful_predictions', 0)} recorded, "
            f"{summary.get('failed_predictions', 0)} failed"
        )
        return result
    except Exception as e:
        logger.error(f"{symbol}: Error - {e}")
        # Return empty state with error
        state = RLBacktestWorkflowState(symbol=symbol)
        state.add_error(str(e))
        return state


async def run_batch_backtest(
    symbols: List[str],
    max_lookback_months: int,
    interval: str,
    parallel_limit: int,
) -> List[RLBacktestWorkflowState]:
    """Run backtest for multiple symbols with parallelism control."""
    logger.info(f"Starting batch backtest for {len(symbols)} symbols")
    logger.info(f"Max lookback: {max_lookback_months} months, Interval: {interval}")
    logger.info(f"Parallel limit: {parallel_limit}")

    semaphore = asyncio.Semaphore(parallel_limit)

    async def limited_backtest(symbol: str) -> RLBacktestWorkflowState:
        async with semaphore:
            return await run_backtest_for_symbol(
                symbol, max_lookback_months, interval
            )

    tasks = [limited_backtest(s) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    successful = 0
    failed = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"{symbols[i]}: Exception - {result}")
            failed += 1
        elif isinstance(result, RLBacktestWorkflowState):
            summary = result.metadata.get("summary", {})
            if summary.get("successful_predictions", 0) > 0:
                successful += 1
            else:
                failed += 1
        else:
            failed += 1

    logger.info(f"Batch complete: {successful} successful, {failed} failed")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run RL backtest using victor workflow (StateGraph)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="List of stock symbols to backtest"
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Process all eligible symbols (SP500 + Russell 1000)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        help="Process top N symbols by market cap"
    )
    parser.add_argument(
        "--max-lookback",
        type=int,
        default=120,
        help="Maximum lookback in months (default: 120 = 10 years)"
    )
    parser.add_argument(
        "--interval",
        choices=["quarterly", "monthly"],
        default="quarterly",
        help="Interval between lookback periods (default: quarterly)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)"
    )

    args = parser.parse_args()

    # Determine symbols to process
    if args.all_symbols:
        symbols = get_all_eligible_symbols()
        logger.info(f"Processing all {len(symbols)} eligible symbols")
    elif args.top_n:
        symbols = get_top_n_symbols(args.top_n)
        logger.info(f"Processing top {len(symbols)} symbols by market cap")
    elif args.symbols:
        symbols = args.symbols
        logger.info(f"Processing {len(symbols)} specified symbols")
    else:
        parser.error("Must specify --symbols, --all-symbols, or --top-n")
        return

    # Calculate expected data points
    lookback_list = generate_lookback_list(args.max_lookback, args.interval)
    logger.info(
        f"Lookback periods: {len(lookback_list)} points per symbol "
        f"(3 to {args.max_lookback} months, {args.interval})"
    )

    # Run backtest
    start_time = datetime.now()
    try:
        results = asyncio.run(
            run_batch_backtest(
                symbols=symbols,
                max_lookback_months=args.max_lookback,
                interval=args.interval,
                parallel_limit=args.parallel,
            )
        )

        # Summary
        total_predictions = 0
        total_errors = 0
        for result in results:
            if isinstance(result, RLBacktestWorkflowState):
                summary = result.metadata.get("summary", {})
                total_predictions += summary.get("successful_predictions", 0)
                total_errors += len(result.errors)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info("")
        logger.info("=" * 60)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Symbols processed: {len(symbols)}")
        logger.info(f"Total predictions: {total_predictions}")
        logger.info(f"Total errors: {total_errors}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Rate: {len(symbols) / duration * 60:.1f} symbols/minute")
        logger.info("=" * 60)

        return 0 if total_errors == 0 else 1

    except KeyboardInterrupt:
        logger.warning("Backtest interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
