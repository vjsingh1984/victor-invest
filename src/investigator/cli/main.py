#!/usr/bin/env python3
"""
InvestiGator CLI - Main Entry Point

Provides a unified command-line interface for investment analysis,
RL backtesting, macro data, and system management.

Usage:
    investigator [OPTIONS] COMMAND [ARGS]...

Examples:
    investigator analyze single AAPL
    investigator backtest run --lookback 365
    investigator macro summary
    investigator cache clean --symbol AAPL
    investigator system status
"""

import sys
from pathlib import Path

import click

# Ensure src/ is in path for package imports
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from .utils import setup_logging, load_config
from .groups import analyze, backtest, cache, data, macro, system


CONTEXT_SETTINGS = dict(
    help_option_names=["-h", "--help"],
    max_content_width=120,
)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--config", "-c",
    default="config.yaml",
    envvar="INVESTIGATOR_CONFIG",
    help="Configuration file path"
)
@click.option(
    "--log-level", "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    envvar="INVESTIGATOR_LOG_LEVEL",
    help="Logging level"
)
@click.option(
    "--log-file",
    type=click.Path(),
    envvar="INVESTIGATOR_LOG_FILE",
    help="Log file path"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output (same as --log-level DEBUG)"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Suppress non-essential output"
)
@click.version_option(
    version="0.1.0",
    prog_name="investigator"
)
@click.pass_context
def cli(ctx, config, log_level, log_file, verbose, quiet):
    """InvestiGator - AI-Powered Investment Analysis

    A comprehensive investment research tool combining SEC fundamentals,
    technical indicators, and LLM synthesis for institutional-grade analysis.

    \b
    COMMAND GROUPS:
      analyze   Stock analysis (fundamental, technical, synthesis)
      backtest  RL model backtesting and training
      data      Data source management
      macro     Macroeconomic data and indicators
      cache     Cache management
      system    System administration

    \b
    EXAMPLES:
      $ investigator analyze single AAPL --mode comprehensive
      $ investigator backtest run --lookback 365 --parallel 10
      $ investigator macro summary
      $ investigator system status

    Run 'investigator COMMAND --help' for more information on a command.
    """
    # Determine effective log level
    effective_level = "DEBUG" if verbose else log_level
    if quiet:
        effective_level = "WARNING"

    setup_logging(effective_level, log_file)

    # Initialize context
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


# Register command groups
cli.add_command(analyze)
cli.add_command(backtest)
cli.add_command(cache)
cli.add_command(data)
cli.add_command(macro)
cli.add_command(system)


# Quick access commands (shortcuts)
@cli.command("quick")
@click.argument("symbol")
@click.pass_context
def quick_analysis(ctx, symbol):
    """Quick analysis shortcut (same as: analyze single SYMBOL --mode quick)"""
    ctx.invoke(
        analyze.commands["single"],
        symbol=symbol,
        mode="quick",
        output=None,
        format="json",
        detail="minimal",
        report=False,
        force_refresh=False,
    )


@cli.command("status")
@click.pass_context
def quick_status(ctx):
    """System status shortcut (same as: system status)"""
    ctx.invoke(system.commands["status"], verbose=False)


def main():
    """Main entry point for the CLI"""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        click.echo("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        if "--verbose" in sys.argv or "-v" in sys.argv:
            raise
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
