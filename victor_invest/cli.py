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
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Victor framework imports (from local wheel)
try:
    from victor.core.protocols import OrchestratorProtocol
    from victor.framework import Agent, Event, EventType
    from victor.workflows.streaming import WorkflowEventType
except ImportError:
    # Fallback for development without victor installed
    Agent = None
    Event = None
    EventType = None
    OrchestratorProtocol = None
    WorkflowEventType = None

from victor_invest.vertical import InvestmentVertical
from victor_invest.workflows import (
    AnalysisMode,
    AnalysisWorkflowState,
    InvestmentWorkflowProvider,
    run_analysis,
)

console = Console()


def _convert_workflow_result_to_state(workflow_result, symbol: str, mode: str) -> AnalysisWorkflowState:
    """Convert Victor WorkflowResult to AnalysisWorkflowState for CLI compatibility.

    Args:
        workflow_result: WorkflowResult from YAML workflow execution
        symbol: Stock symbol
        mode: Analysis mode string

    Returns:
        AnalysisWorkflowState with extracted data from workflow context
    """
    ctx = workflow_result.context

    # Extract outputs from workflow context
    fundamental = ctx.get("fundamental_analysis", {})
    technical = ctx.get("technical_analysis", {})
    market_context = ctx.get("market_context", {})
    synthesis = ctx.get("synthesis", {})

    # Build recommendation from synthesis
    recommendation = {}
    if synthesis:
        recommendation = {
            "action": synthesis.get("recommendation", "HOLD"),
            "confidence": synthesis.get("confidence", "MEDIUM"),
            "price_target": synthesis.get("price_target"),
            "thesis": synthesis.get("executive_summary", ""),
        }

    # Collect any errors from failed nodes
    errors = []
    for node_id, node_result in ctx.node_results.items():
        if node_result.error:
            errors.append(f"{node_id}: {node_result.error}")

    return AnalysisWorkflowState(
        symbol=symbol.upper(),
        mode=AnalysisMode(mode),
        fundamental_analysis=fundamental,
        technical_analysis=technical,
        market_context=market_context,
        synthesis=synthesis,
        recommendation=recommendation,
        errors=errors,
    )


def _convert_to_investment_recommendation(result, symbol: str):
    """Convert AnalysisWorkflowState to InvestmentRecommendation for PDF generation.

    Args:
        result: AnalysisWorkflowState from workflow execution
        symbol: Stock symbol

    Returns:
        InvestmentRecommendation instance for PDF report generation
    """
    from investigator.domain.models import InvestmentRecommendation

    # Extract data from actual workflow structure
    synthesis = result.synthesis or {}
    recommendation_data = result.recommendation or {}
    technical = result.technical_analysis or {}
    fundamental = result.fundamental_analysis or {}
    market_context = result.market_context or {}

    # Extract composite score from synthesis (0-100 scale)
    composite_score = synthesis.get("composite_score", 50.0)
    individual_scores = synthesis.get("individual_scores", {})

    # Map to expected score format
    overall_score = composite_score
    technical_score = individual_scores.get("technical", composite_score)
    fundamental_score = individual_scores.get("fundamental", composite_score)
    market_context_score = individual_scores.get("market_context", 50.0)

    # Sub-scores default to composite if not available
    income_score = fundamental_score
    cashflow_score = fundamental_score
    balance_score = fundamental_score
    growth_score = fundamental_score
    value_score = fundamental_score
    business_quality_score = fundamental_score

    # Extract recommendation details
    final_recommendation = recommendation_data.get("action", "HOLD")
    conviction_level = recommendation_data.get("confidence", "MEDIUM").upper()

    # Extract price data from technical analysis (nested in trend or support_resistance)
    trend_data = technical.get("trend", {})
    sr_data = technical.get("support_resistance", {})

    current_price = trend_data.get("current_price") or sr_data.get("current_price")

    # Extract support/resistance levels for stop loss and price target
    support_levels = sr_data.get("support_levels", {})
    resistance_levels = sr_data.get("resistance_levels", {})
    week_52 = sr_data.get("52_week", {})

    stop_loss = support_levels.get("support_1")
    price_target = resistance_levels.get("resistance_1")

    # Check if LLM synthesis is available (new format with executive_summary, key_catalysts, etc.)
    synthesis_method = synthesis.get("synthesis_method", "rule_based")
    llm_executive_summary = synthesis.get("executive_summary", "")
    llm_key_catalysts = synthesis.get("key_catalysts", [])
    llm_key_risks = synthesis.get("key_risks", [])
    llm_reasoning = synthesis.get("reasoning", "")

    # Build investment thesis from available data
    trend_signal = trend_data.get("overall_signal", "neutral")
    signal_counts = trend_data.get("signal_counts", {})
    bullish_pct = trend_data.get("signal_percentages", {}).get("bullish_pct", 0)
    bearish_pct = trend_data.get("signal_percentages", {}).get("bearish_pct", 0)

    # Use LLM executive summary if available, otherwise build from technical data
    if llm_executive_summary:
        investment_thesis = llm_executive_summary
        if llm_reasoning:
            investment_thesis += f" {llm_reasoning}"
    else:
        # Build meaningful thesis from technical data
        thesis_parts = []
        if current_price:
            thesis_parts.append(f"{symbol} is currently trading at ${current_price:.2f}")
        if week_52:
            high_52 = week_52.get("high")
            low_52 = week_52.get("low")
            if high_52 and low_52 and current_price:
                range_position = (current_price - low_52) / (high_52 - low_52) * 100
                thesis_parts.append(f"at {range_position:.0f}% of its 52-week range (${low_52:.2f} - ${high_52:.2f})")

        thesis_parts.append(
            f"The technical outlook is {trend_signal} with {bullish_pct:.0f}% bullish and {bearish_pct:.0f}% bearish signals."
        )
        thesis_parts.append(f"Composite analysis score: {composite_score:.1f}/100.")

        if final_recommendation == "BUY":
            thesis_parts.append("The analysis suggests accumulating shares at current levels.")
        elif final_recommendation == "SELL":
            thesis_parts.append("The analysis suggests reducing exposure.")
        else:
            thesis_parts.append("The analysis suggests maintaining current positions.")

        investment_thesis = " ".join(thesis_parts)

    # Extract key insights from technical signals
    signals = trend_data.get("signals", {})
    key_insights = []
    if signals:
        for category, indicators in signals.items():
            if isinstance(indicators, dict):
                for indicator, signal in indicators.items():
                    if signal in ["bullish", "bearish"]:
                        key_insights.append(f"{indicator.upper()}: {signal}")

    # Use LLM catalysts/risks if available, otherwise build from technical levels
    if llm_key_catalysts:
        key_catalysts = llm_key_catalysts
    else:
        key_catalysts = []
        if price_target and current_price:
            upside = ((price_target - current_price) / current_price) * 100
            key_catalysts.append(f"Near-term resistance at ${price_target:.2f} ({upside:.1f}% upside)")
        if week_52.get("high") and current_price:
            upside_52 = ((week_52["high"] - current_price) / current_price) * 100
            key_catalysts.append(f"52-week high of ${week_52['high']:.2f} ({upside_52:.1f}% from current)")

    if llm_key_risks:
        key_risks = llm_key_risks
    else:
        key_risks = []
        if stop_loss and current_price:
            downside = ((current_price - stop_loss) / current_price) * 100
            key_risks.append(f"Support at ${stop_loss:.2f} ({downside:.1f}% downside)")
        if week_52.get("low") and current_price:
            downside_52 = ((current_price - week_52["low"]) / current_price) * 100
            key_risks.append(f"52-week low of ${week_52['low']:.2f} ({downside_52:.1f}% below current)")

    # Entry/exit strategies based on levels
    key_levels = trend_data.get("key_levels", {})
    pivot = key_levels.get("pivot")
    support = key_levels.get("support")
    resistance = key_levels.get("resistance")

    if support and current_price:
        entry_strategy = f"Consider entries near support at ${support:.2f} (current: ${current_price:.2f})"
    else:
        entry_strategy = "Scale into position at current market levels"

    if resistance:
        exit_strategy = f"Consider taking profits near resistance at ${resistance:.2f}"
    else:
        exit_strategy = "Target-based exit with trailing stop loss"

    # Time horizon and position size based on score
    if composite_score >= 70:
        time_horizon = "MEDIUM-TERM"
        position_size = "MODERATE"
    elif composite_score >= 50:
        time_horizon = "LONG-TERM"
        position_size = "SMALL"
    else:
        time_horizon = "LONG-TERM"
        position_size = "SMALL"

    # Data quality - check what analyses were included
    analyses_included = synthesis.get("analyses_included", [])
    data_completeness = len(analyses_included) / 4.0  # 4 possible: fundamental, technical, market_context, valuation
    data_quality_score = min(data_completeness, 1.0)

    # Build synthesis details with all available data
    synthesis_details = {
        "synthesis": synthesis,
        "recommendation": recommendation_data,
        "technical_trend": trend_data,
        "support_resistance": sr_data,
        "market_context": market_context,
    }

    return InvestmentRecommendation(
        symbol=symbol.upper(),
        overall_score=overall_score,
        fundamental_score=fundamental_score,
        technical_score=technical_score,
        income_score=income_score,
        cashflow_score=cashflow_score,
        balance_score=balance_score,
        growth_score=growth_score,
        value_score=value_score,
        business_quality_score=business_quality_score,
        recommendation=final_recommendation,
        confidence=conviction_level,
        price_target=price_target,
        current_price=current_price,
        investment_thesis=investment_thesis,
        time_horizon=time_horizon,
        position_size=position_size,
        key_catalysts=key_catalysts[:5],
        key_risks=key_risks[:5],
        key_insights=key_insights[:10],
        entry_strategy=entry_strategy,
        exit_strategy=exit_strategy,
        stop_loss=stop_loss,
        analysis_timestamp=datetime.now(),
        data_quality_score=data_quality_score,
        analysis_thinking=synthesis.get("reasoning"),
        synthesis_details=json.dumps(synthesis_details, default=str),
        # Include technical levels for report
        support_resistance={
            "current_price": current_price,
            "support_1": support_levels.get("support_1"),
            "support_2": support_levels.get("support_2"),
            "resistance_1": resistance_levels.get("resistance_1"),
            "resistance_2": resistance_levels.get("resistance_2"),
            "52_week_high": week_52.get("high"),
            "52_week_low": week_52.get("low"),
            "pivot": pivot,
        },
    )


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
@click.option(
    "--report",
    is_flag=True,
    default=False,
    help="Generate PDF investment report using LLM synthesis",
)
def analyze(
    symbol: str,
    mode: str,
    output: Optional[str],
    provider: str,
    model: Optional[str],
    stream: bool,
    report: bool,
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
    if report:
        console.print(f"Report: [magenta]PDF generation enabled[/magenta]")
    console.print()

    asyncio.run(_run_analysis(symbol, mode, output, provider, model, stream, report))


async def _run_analysis(
    symbol: str,
    mode: str,
    output: Optional[str],
    provider: str,
    model: Optional[str],
    stream: bool,
    report: bool = False,
):
    """Execute the analysis workflow using InvestmentWorkflowProvider.

    Uses Victor's agentic workflow execution for proper LLM integration:
    - Compute handlers for data collection (SEC, market data, technicals)
    - Agent nodes for LLM synthesis via Victor's SubAgentOrchestrator
    - Proper provider/model abstraction through Victor framework
    """
    # Initialize workflow provider (loads YAML workflows, registers handlers)
    workflow_provider = InvestmentWorkflowProvider()

    # Map mode to workflow name
    workflow_name = workflow_provider.get_workflow_for_task_type(mode) or mode

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Analyzing {symbol.upper()}...", total=None)

        try:
            # Execute YAML workflow with full agent node support
            # Uses Victor's SubAgentOrchestrator for LLM synthesis
            workflow_result = await workflow_provider.run_agentic_workflow(
                workflow_name,
                context={"symbol": symbol.upper()},
                provider=provider,
                model=model,
                timeout=300.0,
            )
            progress.update(task, description="Analysis complete!")

            if not workflow_result.success:
                console.print(f"[red]Workflow failed: {workflow_result.error}[/red]")
                return

            # Convert WorkflowResult to AnalysisWorkflowState for compatibility
            result = _convert_workflow_result_to_state(workflow_result, symbol, mode)

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

    # Generate PDF report if requested
    if report:
        try:
            console.print("\n[bold]Generating PDF report...[/bold]")

            # Convert workflow result to InvestmentRecommendation format
            recommendation = _convert_to_investment_recommendation(result, symbol)

            # Initialize synthesizer for PDF generation
            from investigator.application import InvestmentSynthesizer

            synthesizer = InvestmentSynthesizer()

            # Generate PDF report
            report_path = synthesizer.generate_report([recommendation], report_type="synthesis")

            console.print(f"[green]✅ PDF report generated: {report_path}[/green]")

        except ImportError as e:
            console.print(f"[red]❌ PDF generation requires investigator package: {e}[/red]")
        except Exception as e:
            console.print(f"[red]❌ Failed to generate PDF report: {e}[/red]")
            import traceback

            traceback.print_exc()


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
        from investigator.infrastructure.cache.file_cache_handler import FileCacheStorageHandler
        from investigator.infrastructure.cache.rdbms_cache_handler import RdbmsCacheStorageHandler
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
                        cache_manager.clear(ct, storage_type="rdbms")
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
                        cache_manager.clear(ct, storage_type="disk")
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


@cli.command("from-batch")
@click.argument("jsonl_path", type=click.Path(exists=True))
@click.option(
    "--symbols",
    "-s",
    help="Comma-separated symbols to process (default: all successful)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="reports/batch",
    help="Output directory for reports",
)
@click.option(
    "--min-upside",
    type=float,
    default=None,
    help="Only process symbols with upside >= this percentage",
)
@click.option(
    "--tier",
    type=click.Choice(["BUY", "HOLD", "SELL"]),
    default=None,
    help="Only process symbols with this tier classification",
)
def from_batch(jsonl_path: str, symbols: Optional[str], output: str, min_upside: Optional[float], tier: Optional[str]):
    """Generate professional reports from batch analysis results.

    Reads cached batch results and generates PDF reports without re-running analysis.

    Example:
        victor-invest from-batch batch_results/batch_analysis_results.jsonl
        victor-invest from-batch results.jsonl --symbols AAPL,MSFT,GOOGL
        victor-invest from-batch results.jsonl --min-upside 10 --tier BUY
    """
    import json
    from pathlib import Path

    console.print(f"\n[bold blue]Generate Reports from Batch Results[/bold blue]")
    console.print(f"Source: [green]{jsonl_path}[/green]")

    # Load batch results
    results = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    console.print(f"Loaded: {len(results)} results")

    # Filter results
    filtered = []
    symbol_filter = set(s.upper() for s in symbols.split(",")) if symbols else None

    for r in results:
        # Must be successful
        if not r.get("success", False):
            continue

        # Must have required data for report
        if not r.get("fair_value") or not r.get("current_price"):
            continue

        # Apply symbol filter
        if symbol_filter and r.get("symbol") not in symbol_filter:
            continue

        # Apply upside filter
        if min_upside is not None:
            upside = r.get("upside_pct", 0) or 0
            if upside < min_upside:
                continue

        # Apply tier filter
        if tier and r.get("tier", "").upper() != tier:
            continue

        filtered.append(r)

    console.print(f"Filtered: {len(filtered)} symbols for report generation")

    if not filtered:
        console.print("[yellow]No symbols match the criteria. No reports generated.[/yellow]")
        return

    # Generate reports
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from investigator.infrastructure.reporting.professional_report import ProfessionalReportGenerator

        generator = ProfessionalReportGenerator(output_dir=output_path)

        success_count = 0
        for r in filtered:
            symbol = r.get("symbol", "UNKNOWN")
            console.print(f"  Generating report for [cyan]{symbol}[/cyan]...")

            try:
                # Convert batch result to report data format
                report_data = _convert_batch_result_to_report_data(r)
                report_path = generator.generate_report(report_data)

                if report_path:
                    success_count += 1
                    console.print(f"    [green]✓ {report_path}[/green]")
                else:
                    console.print(f"    [yellow]⚠ No report generated[/yellow]")

            except Exception as e:
                console.print(f"    [red]✗ Error: {e}[/red]")

        console.print(f"\n[bold green]Reports generated: {success_count}/{len(filtered)}[/bold green]")
        console.print(f"Output directory: {output_path}")

    except ImportError as e:
        console.print(f"[red]Error: Report generator not available: {e}[/red]")
        console.print("Ensure investigator package is installed.")


def _convert_batch_result_to_report_data(batch_result: dict) -> dict:
    """Convert batch analysis result to report-ready data format.

    Args:
        batch_result: SymbolResult from batch JSONL

    Returns:
        Dict compatible with ProfessionalReportGenerator
    """
    symbol = batch_result.get("symbol", "UNKNOWN")
    fair_value = batch_result.get("fair_value")
    current_price = batch_result.get("current_price")
    upside_pct = batch_result.get("upside_pct", 0)
    tier = batch_result.get("tier", "HOLD")
    model_fair_values = batch_result.get("model_fair_values", {})
    model_weights = batch_result.get("model_weights", {})
    sector = batch_result.get("sector", "")
    market_cap = batch_result.get("market_cap")

    # Map tier to recommendation
    rec_map = {
        "BUY": ("BUY", "HIGH"),
        "STRONG_BUY": ("STRONG BUY", "HIGH"),
        "HOLD": ("HOLD", "MEDIUM"),
        "SELL": ("SELL", "LOW"),
        "STRONG_SELL": ("STRONG SELL", "LOW"),
    }
    recommendation, confidence = rec_map.get(tier.upper(), ("HOLD", "MEDIUM"))

    # Calculate scores from upside
    if upside_pct is not None and upside_pct > 0:
        overall_score = min(50 + upside_pct * 2, 95)  # Scale upside to score
    else:
        overall_score = max(50 + (upside_pct or 0) * 2, 10)

    fundamental_score = overall_score + 5 if upside_pct and upside_pct > 10 else overall_score
    technical_score = overall_score - 5 if upside_pct and upside_pct < 0 else overall_score

    # Calculate stop loss (10% below current)
    stop_loss = current_price * 0.90 if current_price else None

    # Build valuation models section
    valuation_models = {}
    for model, fv in (model_fair_values or {}).items():
        if fv and current_price:
            model_upside = ((fv / current_price) - 1) * 100
            weight = (model_weights or {}).get(model, 0.33)
            valuation_models[model] = {
                "fair_value_per_share": fv,
                "upside_downside_pct": model_upside,
                "confidence": weight * 100,
            }

    # Build thesis from batch data
    thesis_parts = []
    thesis_parts.append(f"{symbol} in {sector}." if sector else f"{symbol}.")
    if upside_pct is not None:
        if upside_pct > 15:
            thesis_parts.append(
                f"Analysis indicates significant undervaluation with {upside_pct:.1f}% upside to fair value."
            )
        elif upside_pct > 5:
            thesis_parts.append(f"Moderate upside of {upside_pct:.1f}% to fair value estimate.")
        elif upside_pct > 0:
            thesis_parts.append(f"Trading near fair value with {upside_pct:.1f}% potential upside.")
        else:
            thesis_parts.append(f"Currently trading at {abs(upside_pct):.1f}% premium to fair value.")
    if market_cap:
        thesis_parts.append(f"Market cap: ${market_cap/1e9:.1f}B.")

    # Key catalysts/risks based on tier
    if tier.upper() in ["BUY", "STRONG_BUY"]:
        key_catalysts = [
            f"Fair value estimate of ${fair_value:.2f} suggests upside potential",
            "Valuation models show favorable risk/reward",
        ]
        key_risks = [
            "Market volatility could impact near-term price action",
            "Model assumptions may not reflect current market conditions",
        ]
    else:
        key_catalysts = ["Potential rerating if fundamentals improve"]
        key_risks = [
            "Limited upside at current valuation levels",
            "Market conditions may pressure valuations further",
        ]

    return {
        "symbol": symbol,
        "recommendation": recommendation,
        "confidence": confidence,
        "overall_score": overall_score,
        "fundamental_score": min(fundamental_score, 100),
        "technical_score": max(min(technical_score, 100), 10),
        "current_price": current_price,
        "target_price": fair_value,
        "stop_loss": stop_loss,
        "investment_thesis": " ".join(thesis_parts),
        "key_catalysts": key_catalysts,
        "key_risks": key_risks,
        "time_horizon": "MEDIUM-TERM",
        "position_size": "MODERATE" if tier.upper() in ["BUY", "STRONG_BUY"] else "SMALL",
        "valuation_models": valuation_models,
        "score_breakdown": {
            "value": min(50 + (upside_pct or 0) * 2, 100),
            "growth": 60,  # Default without detailed data
            "business_quality": 70,
            "data_quality": 80,
        },
        # Market regime placeholder
        "market_regime": {"regime": "Normal"},
        # Peer data placeholder
        "peer_comparison": {"peers": [], "metrics": {}},
    }


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
