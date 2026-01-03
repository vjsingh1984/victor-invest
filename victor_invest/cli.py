# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Victor-based CLI for investment analysis.

This module provides the command-line interface for running investment
analysis using the Victor framework with StateGraph workflows.

Usage:
    python -m victor_invest.cli analyze AAPL --mode standard
    python -m victor_invest.cli analyze MSFT --mode comprehensive --output results/
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Victor framework imports (from local wheel)
try:
    from victor.framework import Agent, Event, EventType
except ImportError:
    # Fallback for development without victor installed
    Agent = None
    Event = None
    EventType = None

from victor_invest.vertical import InvestmentVertical
from victor_invest.workflows import AnalysisMode, run_analysis as workflow_run_analysis

console = Console()


def validate_victor_installed():
    """Check if victor-core is installed."""
    if Agent is None:
        console.print(
            "[red]Error: victor-core not installed.[/red]\n"
            "Install with: pip install ../codingagent/dist/victor-0.2.0-py3-none-any.whl"
        )
        sys.exit(1)


@click.group()
@click.version_option(version="0.1.0", prog_name="victor-invest")
def cli():
    """Victor Investment Analysis CLI - Institutional-grade equity research."""
    pass


@cli.command()
@click.argument("symbol")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["quick", "standard", "comprehensive"]),
    default="standard",
    help="Analysis mode: quick (technical only), standard (technical+fundamental), comprehensive (all)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory for results (default: stdout)",
)
@click.option(
    "--provider",
    "-p",
    type=str,
    default="ollama",
    help="LLM provider (ollama, anthropic, openai)",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model name (default: provider-specific)",
)
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Stream output as analysis progresses",
)
def analyze(
    symbol: str,
    mode: str,
    output: Optional[str],
    provider: str,
    model: Optional[str],
    stream: bool,
):
    """Run investment analysis on a stock symbol.

    Example:
        victor-invest analyze AAPL --mode comprehensive
        victor-invest analyze MSFT -m standard -o results/
    """
    validate_victor_installed()

    console.print(f"\n[bold blue]Victor Investment Analysis[/bold blue]")
    console.print(f"Symbol: [green]{symbol.upper()}[/green]")
    console.print(f"Mode: [yellow]{mode}[/yellow]")
    console.print(f"Provider: [cyan]{provider}[/cyan]")
    console.print()

    asyncio.run(_run_analysis(symbol, mode, output, provider, model, stream))


async def _run_analysis(
    symbol: str,
    mode: str,
    output: Optional[str],
    provider: str,
    model: Optional[str],
    stream: bool,
):
    """Execute the analysis workflow."""
    analysis_mode = AnalysisMode(mode)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Analyzing {symbol.upper()}...", total=None)

        try:
            # Execute workflow using the convenience function
            result = await workflow_run_analysis(symbol.upper(), analysis_mode)

            progress.update(task, description="Analysis complete!")

        except Exception as e:
            console.print(f"[red]Error during analysis: {e}[/red]")
            import traceback
            traceback.print_exc()
            return

    # Display results
    _display_results(result, symbol)

    # Save to file if output specified
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"{symbol.upper()}_analysis.json"

        with open(result_file, "w") as f:
            # Convert dataclass to dict for JSON serialization
            result_dict = {
                "symbol": result.symbol,
                "mode": result.mode.value,
                "fundamental_analysis": result.fundamental_analysis,
                "technical_analysis": result.technical_analysis,
                "market_context": result.market_context,
                "synthesis": result.synthesis,
                "recommendation": result.recommendation,
                "errors": result.errors,
            }
            json.dump(result_dict, f, indent=2, default=str)

        console.print(f"\n[green]Results saved to: {result_file}[/green]")


def _display_results(result, symbol: str):
    """Display analysis results in a formatted table."""
    console.print("\n[bold]Analysis Results[/bold]\n")

    # Summary table
    table = Table(title=f"{symbol.upper()} Analysis Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Key Findings", style="white")

    # Add rows based on what was analyzed
    if result.fundamental_analysis:
        findings = result.fundamental_analysis.get("summary", "Completed")
        table.add_row("Fundamental", "✓", str(findings)[:80])

    if result.technical_analysis:
        findings = result.technical_analysis.get("summary", "Completed")
        table.add_row("Technical", "✓", str(findings)[:80])

    if result.market_context:
        findings = result.market_context.get("summary", "Completed")
        table.add_row("Market Context", "✓", str(findings)[:80])

    if result.synthesis:
        findings = result.synthesis.get("recommendation", "See details")
        table.add_row("Synthesis", "✓", str(findings)[:80])

    if result.errors:
        for error in result.errors:
            table.add_row("Error", "✗", str(error)[:80])

    console.print(table)

    # Recommendation
    if result.recommendation:
        rec = result.recommendation
        console.print("\n[bold]Recommendation[/bold]")
        console.print(f"  Action: [bold]{rec.get('action', 'N/A')}[/bold]")
        console.print(f"  Confidence: {rec.get('confidence', 'N/A')}")
        if "price_target" in rec:
            console.print(f"  Price Target: ${rec['price_target']}")
        if "thesis" in rec:
            console.print(f"  Thesis: {rec['thesis']}")


@cli.command()
def status():
    """Check system status and dependencies."""
    console.print("\n[bold]Victor Investment System Status[/bold]\n")

    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")

    # Check victor-core
    try:
        from victor.framework import Agent

        table.add_row("victor-core", "✓ Installed", "Framework available")
    except ImportError:
        table.add_row("victor-core", "✗ Missing", "pip install victor wheel")

    # Check ollama
    try:
        import aiohttp

        table.add_row("aiohttp", "✓ Installed", "HTTP client available")
    except ImportError:
        table.add_row("aiohttp", "✗ Missing", "pip install aiohttp")

    # Check yfinance
    try:
        import yfinance

        table.add_row("yfinance", "✓ Installed", "Market data available")
    except ImportError:
        table.add_row("yfinance", "✗ Missing", "pip install yfinance")

    # Check pandas
    try:
        import pandas

        table.add_row("pandas", "✓ Installed", f"v{pandas.__version__}")
    except ImportError:
        table.add_row("pandas", "✗ Missing", "pip install pandas")

    console.print(table)


@cli.command()
@click.option("--port", "-p", default=8000, help="Port to run the API server")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
def serve(port: int, host: str):
    """Start the FastAPI server for API access."""
    try:
        import uvicorn

        from victor_invest.api.app import app

        console.print(f"\n[bold blue]Starting Victor Investment API[/bold blue]")
        console.print(f"Server: http://{host}:{port}")
        console.print(f"Docs: http://{host}:{port}/docs\n")

        uvicorn.run(app, host=host, port=port)
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Install with: pip install uvicorn fastapi")


@cli.command("clean-cache")
@click.option("--all", "clean_all", is_flag=True, help="Clean all caches")
@click.option("--db", "clean_db", is_flag=True, help="Clean database cache only")
@click.option("--disk", "clean_disk", is_flag=True, help="Clean disk cache only")
@click.option("--symbol", help="Clean cache for specific symbol")
def clean_cache(clean_all, clean_db, clean_disk, symbol):
    """Clean analysis caches.

    Example:
        victor-invest clean-cache --symbol AAPL
        victor-invest clean-cache --all
    """
    try:
        from investigator.infrastructure.cache import get_cache_manager
        from investigator.infrastructure.cache.cache_types import CacheType
        from investigator.infrastructure.cache.rdbms_cache_handler import RdbmsCacheStorageHandler
        from investigator.infrastructure.cache.file_cache_handler import FileCacheStorageHandler
    except ImportError as e:
        console.print(f"[red]Error: Cache infrastructure not available: {e}[/red]")
        return

    cache_manager = get_cache_manager()

    try:
        if clean_all:
            console.print("Cleaning all caches...")
            for cache_type in CacheType:
                try:
                    cache_manager.clear(cache_type)
                except Exception:
                    pass
            console.print("[green]✅ All caches cleared[/green]")

        elif clean_db:
            if symbol:
                console.print(f"Cleaning database cache for {symbol}...")
                deleted = 0
                for handlers in cache_manager.handlers.values():
                    for handler in handlers:
                        if isinstance(handler, RdbmsCacheStorageHandler):
                            try:
                                deleted += handler.delete_by_symbol(symbol)
                            except Exception as exc:
                                console.print(f"[red]❌ Error: {exc}[/red]")
                console.print(f"[green]✅ Database cache cleared for {symbol} (entries: {deleted})[/green]")
            else:
                console.print("Cleaning database cache...")
                for ct in [CacheType.LLM_RESPONSE, CacheType.COMPANY_FACTS, CacheType.SEC_RESPONSE]:
                    try:
                        cache_manager.clear(ct, storage_type='rdbms')
                    except Exception:
                        pass
                console.print("[green]✅ Database cache cleared[/green]")

        elif clean_disk:
            if symbol:
                console.print(f"Cleaning disk cache for {symbol}...")
                deleted = 0
                for handlers in cache_manager.handlers.values():
                    for handler in handlers:
                        if isinstance(handler, FileCacheStorageHandler):
                            try:
                                deleted += handler.delete_by_symbol(symbol)
                            except Exception as exc:
                                console.print(f"[red]❌ Error: {exc}[/red]")
                console.print(f"[green]✅ Disk cache cleared for {symbol} (entries: {deleted})[/green]")
            else:
                console.print("Cleaning disk cache...")
                for ct in [CacheType.LLM_RESPONSE, CacheType.TECHNICAL_DATA, CacheType.SEC_RESPONSE]:
                    try:
                        cache_manager.clear(ct, storage_type='disk')
                    except Exception:
                        pass
                console.print("[green]✅ Disk cache cleared[/green]")

        elif symbol:
            console.print(f"Cleaning all caches for {symbol}...")
            result = cache_manager.delete_by_symbol(symbol)
            total_deleted = sum(result.values()) if isinstance(result, dict) else result
            console.print(f"[green]✅ Cache cleared for {symbol} (entries: {total_deleted})[/green]")

        else:
            console.print("Cleaning default caches (LLM responses)...")
            cache_manager.clear(CacheType.LLM_RESPONSE)
            console.print("[green]✅ LLM response cache cleared[/green]")

    except Exception as e:
        console.print(f"[red]❌ Error cleaning cache: {e}[/red]")
        sys.exit(1)


@cli.command("cache-sizes")
def cache_sizes():
    """Show cache directory sizes.

    Example:
        victor-invest cache-sizes
    """
    console.print("\n[bold]Cache Directory Sizes[/bold]\n")

    cache_dirs = {
        "SEC Cache": "data/sec_cache",
        "LLM Cache": "data/llm_cache",
        "Technical Cache": "data/technical_cache",
        "Vector DB": "data/vector_db",
    }

    table = Table()
    table.add_column("Cache Type", style="cyan")
    table.add_column("Size (MB)", justify="right", style="green")
    table.add_column("Files", justify="right", style="yellow")

    total_size = 0
    for name, path in cache_dirs.items():
        cache_path = Path(path)
        if cache_path.exists():
            files = list(cache_path.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            size = sum(f.stat().st_size for f in files if f.is_file())
            total_size += size
            size_mb = size / (1024 * 1024)
            table.add_row(name, f"{size_mb:.2f}", str(file_count))
        else:
            table.add_row(name, "0.00", "0")

    table.add_row("─" * 20, "─" * 10, "─" * 10)
    table.add_row("[bold]Total[/bold]", f"[bold]{total_size / (1024 * 1024):.2f}[/bold]", "")

    console.print(table)


@cli.command("inspect-cache")
@click.option("--symbol", help="Inspect cache for specific symbol")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information")
def inspect_cache(symbol, verbose):
    """Inspect cache contents for a symbol.

    Example:
        victor-invest inspect-cache --symbol AAPL
        victor-invest inspect-cache --symbol AAPL --verbose
    """
    try:
        from investigator.infrastructure.cache import get_cache_manager
        from investigator.infrastructure.cache.cache_types import CacheType
    except ImportError as e:
        console.print(f"[red]Error: Cache infrastructure not available: {e}[/red]")
        return

    cache_manager = get_cache_manager()

    console.print("\n[bold]Cache Inspection Report[/bold]\n")

    if symbol:
        console.print(f"Symbol: [cyan]{symbol.upper()}[/cyan]\n")

        table = Table()
        table.add_column("Cache Type", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Size", justify="right", style="yellow")

        for cache_type in CacheType:
            try:
                key = {"symbol": symbol.upper()}
                data = cache_manager.get(cache_type, key)
                if data:
                    size_str = f"{len(str(data))} bytes" if verbose else "✓"
                    table.add_row(cache_type.value, "[green]Cached[/green]", size_str)
                else:
                    table.add_row(cache_type.value, "[dim]Not cached[/dim]", "-")
            except Exception:
                table.add_row(cache_type.value, "[dim]Error[/dim]", "-")

        console.print(table)
    else:
        console.print("Cache Statistics:")
        stats = cache_manager.get_stats() if hasattr(cache_manager, "get_stats") else {}
        if stats:
            for key, value in stats.items():
                console.print(f"  {key}: {value}")
        else:
            console.print("  No statistics available (specify --symbol for symbol-specific info)")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
