"""
RL Backtest and training commands for InvestiGator CLI
"""

import asyncio
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import click

from ..utils import MutuallyExclusiveOption, validate_date


@click.group()
@click.pass_context
def backtest(ctx):
    """Reinforcement Learning backtest and training commands

    Run backtests on historical data and train RL models.

    Examples:
        investigator backtest run --lookback 365 --parallel 10
        investigator backtest train --epochs 100
        investigator backtest status
    """
    pass


@backtest.command("run")
@click.option("--symbols", "-s", help="Comma-separated list of symbols (default: all in rl_decisions)")
@click.option("--lookback", "-l", default=365, type=int, help="Lookback period in days")
@click.option("--start-date", callback=validate_date, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", callback=validate_date, help="End date (YYYY-MM-DD)")
@click.option("--parallel", "-p", default=5, type=int, help="Number of parallel workers")
@click.option("--min-confidence", default=0.6, type=float, help="Minimum confidence threshold (0.0-1.0)")
@click.option("--holding-days", default=30, type=int, help="Holding period in days")
@click.option("--output", "-o", type=click.Path(), help="Output file for results")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def run_backtest(ctx, symbols, lookback, start_date, end_date, parallel, min_confidence, holding_days, output, verbose):
    """Run RL backtest on historical data

    Evaluates RL model decisions against historical price movements.

    Examples:
        investigator backtest run --lookback 365
        investigator backtest run --symbols AAPL,MSFT --lookback 180
        investigator backtest run --start-date 2024-01-01 --end-date 2024-12-31
    """
    click.echo("Running RL backtest...")

    # Parse symbols
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        click.echo(f"Symbols: {', '.join(symbol_list)}")

    click.echo(f"Lookback: {lookback} days")
    click.echo(f"Parallel workers: {parallel}")
    click.echo(f"Min confidence: {min_confidence}")
    click.echo(f"Holding period: {holding_days} days")

    try:
        # Import the backtest runner
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))
        from scripts.rl_backtest import RLBacktester

        backtester = RLBacktester(
            lookback_days=lookback,
            min_confidence=min_confidence,
            holding_days=holding_days,
            parallel_workers=parallel,
        )

        if symbol_list:
            results = asyncio.run(backtester.run_symbols(symbol_list))
        else:
            results = asyncio.run(backtester.run_all())

        # Display results
        click.echo("\n" + "=" * 60)
        click.echo("BACKTEST RESULTS")
        click.echo("=" * 60)

        if isinstance(results, dict):
            summary = results.get("summary", {})
            click.echo(f"Total decisions: {summary.get('total_decisions', 0)}")
            click.echo(f"Correct: {summary.get('correct', 0)}")
            click.echo(f"Accuracy: {summary.get('accuracy', 0):.2%}")
            click.echo(f"Avg return: {summary.get('avg_return', 0):.2%}")

            if output:
                with open(output, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                click.echo(f"\nResults saved to {output}")

    except ImportError as e:
        click.echo(f"Error: Could not import backtest module: {e}", err=True)
        click.echo("Run from project root directory", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Backtest failed: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@backtest.command("train")
@click.option("--epochs", "-e", default=100, type=int, help="Number of training epochs")
@click.option("--batch-size", "-b", default=32, type=int, help="Training batch size")
@click.option("--learning-rate", "-lr", default=0.001, type=float, help="Learning rate")
@click.option("--min-samples", default=100, type=int, help="Minimum samples required for training")
@click.option(
    "--policy",
    type=click.Choice(["contextual_bandit", "hybrid", "fundamental", "technical"]),
    default="hybrid",
    help="Policy type to train",
)
@click.option("--output-dir", "-o", default="models", help="Output directory for trained model")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def train(ctx, epochs, batch_size, learning_rate, min_samples, policy, output_dir, verbose):
    """Train RL model on historical data

    Trains the reinforcement learning model using historical decisions and outcomes.

    Examples:
        investigator backtest train --epochs 100
        investigator backtest train --policy hybrid --epochs 200
    """
    click.echo("Training RL model...")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Learning rate: {learning_rate}")
    click.echo(f"Policy: {policy}")

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))
        from scripts.rl_train import RLTrainer

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        trainer = RLTrainer(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            min_samples=min_samples,
            policy_type=policy,
        )

        click.echo("\nStarting training...")
        results = asyncio.run(trainer.train())

        click.echo("\n" + "=" * 60)
        click.echo("TRAINING RESULTS")
        click.echo("=" * 60)

        if isinstance(results, dict):
            click.echo(f"Samples used: {results.get('samples', 0)}")
            click.echo(f"Final loss: {results.get('final_loss', 'N/A')}")
            click.echo(f"Best accuracy: {results.get('best_accuracy', 0):.2%}")

            model_path = Path(output_dir) / f"rl_model_{policy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            click.echo(f"\nModel saved to: {model_path}")

    except ImportError as e:
        click.echo(f"Error: Could not import training module: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Training failed: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@backtest.command("status")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed statistics")
@click.pass_context
def status(ctx, detailed):
    """Show RL model and backtest status

    Displays current model state, recent decisions, and performance metrics.
    """
    click.echo("RL Model Status")
    click.echo("=" * 60)

    try:
        from sqlalchemy import text

        from investigator.infrastructure.database.db import get_engine

        engine = get_engine()

        with engine.connect() as conn:
            # Count decisions
            result = conn.execute(text("SELECT COUNT(*) FROM rl_decisions"))
            total_decisions = result.scalar()

            # Count outcomes
            result = conn.execute(text("SELECT COUNT(*) FROM rl_decisions WHERE actual_return IS NOT NULL"))
            with_outcomes = result.scalar()

            # Recent accuracy
            result = conn.execute(
                text(
                    """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct
                FROM rl_decisions
                WHERE actual_return IS NOT NULL
                AND decision_timestamp > NOW() - INTERVAL '30 days'
            """
                )
            )
            row = result.fetchone()
            recent_total = row[0] if row else 0
            recent_correct = row[1] if row else 0

        click.echo(f"\nDecisions:")
        click.echo(f"  Total: {total_decisions}")
        click.echo(f"  With outcomes: {with_outcomes}")
        click.echo(f"  Pending: {total_decisions - with_outcomes}")

        if recent_total > 0:
            accuracy = recent_correct / recent_total
            click.echo(f"\nLast 30 days:")
            click.echo(f"  Decisions: {recent_total}")
            click.echo(f"  Accuracy: {accuracy:.2%}")

        if detailed:
            click.echo("\nDetailed breakdown by action:")
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                    SELECT
                        action,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence,
                        AVG(actual_return) as avg_return
                    FROM rl_decisions
                    WHERE actual_return IS NOT NULL
                    GROUP BY action
                """
                    )
                )
                for row in result:
                    click.echo(f"  {row[0]}: {row[1]} decisions, " f"conf={row[2]:.2f}, ret={row[3]:.2%}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@backtest.command("outcomes")
@click.option("--days", "-d", default=30, type=int, help="Number of days to process")
@click.option("--force", is_flag=True, help="Reprocess existing outcomes")
@click.pass_context
def update_outcomes(ctx, days, force):
    """Update decision outcomes from price data

    Calculates actual returns for decisions that have completed their holding period.
    """
    click.echo(f"Updating outcomes for last {days} days...")

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))
        from scripts.rl_outcome_updater import update_outcomes

        results = asyncio.run(update_outcomes(days=days, force=force))

        click.echo("\n" + "=" * 40)
        click.echo(f"Processed: {results.get('processed', 0)}")
        click.echo(f"Updated: {results.get('updated', 0)}")
        click.echo(f"Skipped: {results.get('skipped', 0)}")
        click.echo(f"Errors: {results.get('errors', 0)}")

    except ImportError as e:
        click.echo(f"Error: Could not import outcome updater: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Update failed: {e}", err=True)
        sys.exit(1)


@backtest.command("analyze")
@click.option("--period", type=click.Choice(["7d", "30d", "90d", "365d", "all"]), default="30d", help="Analysis period")
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def analyze_results(ctx, period, output):
    """Analyze backtest results

    Provides detailed analysis of backtest performance metrics.
    """
    click.echo(f"Analyzing results for period: {period}")

    period_days = {"7d": 7, "30d": 30, "90d": 90, "365d": 365, "all": 9999}

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))
        from scripts.analyze_backtest import analyze_backtest_results

        results = analyze_backtest_results(days=period_days[period])

        click.echo("\n" + "=" * 60)
        click.echo("BACKTEST ANALYSIS")
        click.echo("=" * 60)

        if isinstance(results, dict):
            click.echo(f"\nPeriod: {period}")
            click.echo(f"Total trades: {results.get('total_trades', 0)}")
            click.echo(f"Win rate: {results.get('win_rate', 0):.2%}")
            click.echo(f"Avg return: {results.get('avg_return', 0):.2%}")
            click.echo(f"Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}")
            click.echo(f"Max drawdown: {results.get('max_drawdown', 0):.2%}")

            if output:
                with open(output, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                click.echo(f"\nResults saved to {output}")

    except ImportError as e:
        click.echo(f"Error: Analysis module not found: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Analysis failed: {e}", err=True)
        sys.exit(1)
