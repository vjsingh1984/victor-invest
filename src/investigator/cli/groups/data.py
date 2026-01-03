"""
Data source management commands for InvestiGator CLI
"""

import json
import sys
from datetime import date
from typing import Optional

import click


@click.group()
@click.pass_context
def data(ctx):
    """Data source management

    View, test, and manage data sources for analysis.

    Examples:
        investigator data list
        investigator data fetch AAPL --source price_history
        investigator data health
        investigator data summary AAPL
    """
    pass


@data.command("list")
@click.option("--category", "-c",
    type=click.Choice(["market", "fundamental", "macro", "sentiment", "volatility", "fixed_income"]),
    help="Filter by category"
)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def list_sources(ctx, category, json_output):
    """List all available data sources

    Shows registered data sources with their status and metadata.
    """
    from investigator.domain.services.data_sources.manager import get_data_source_manager
    from investigator.domain.services.data_sources.base import DataCategory

    manager = get_data_source_manager()
    sources = manager.list_sources()

    # Filter by category if specified
    if category:
        cat_map = {
            "market": "MARKET_DATA",
            "fundamental": "FUNDAMENTAL",
            "macro": "MACRO",
            "sentiment": "SENTIMENT",
            "volatility": "VOLATILITY",
            "fixed_income": "FIXED_INCOME",
        }
        target_cat = cat_map.get(category)
        sources = [s for s in sources if s.get("category") == target_cat]

    if json_output:
        click.echo(json.dumps(sources, indent=2, default=str))
        return

    click.echo("\n" + "=" * 70)
    click.echo("DATA SOURCES")
    click.echo("=" * 70)

    # Group by category
    by_category = {}
    for source in sources:
        cat = source.get("category", "UNKNOWN")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(source)

    for cat, cat_sources in sorted(by_category.items()):
        click.echo(f"\n{cat}")
        click.echo("-" * 40)
        for s in cat_sources:
            enabled = "ON" if s.get("enabled", True) else "OFF"
            freq = s.get("frequency", "?")
            click.echo(f"  {s['name']:25s} [{enabled:3s}] {freq}")

    click.echo("\n" + "=" * 70)
    click.echo(f"Total: {len(sources)} sources")


@data.command("fetch")
@click.argument("symbol")
@click.option("--source", "-s", required=True, help="Source name to fetch from")
@click.option("--date", "-d", "as_of_date", help="Historical date (YYYY-MM-DD)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def fetch(ctx, symbol, source, as_of_date, json_output):
    """Fetch data from a specific source

    Examples:
        investigator data fetch AAPL --source price_history
        investigator data fetch _MACRO --source fred_macro
        investigator data fetch AAPL --source insider_transactions --json
    """
    from investigator.domain.services.data_sources.manager import get_data_source_manager
    from datetime import datetime

    manager = get_data_source_manager()
    data_source = manager.get_source(source)

    if not data_source:
        click.echo(f"Source not found: {source}", err=True)
        click.echo("Use 'investigator data list' to see available sources")
        sys.exit(1)

    target_date = None
    if as_of_date:
        target_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()

    click.echo(f"Fetching from {source}...")
    result = data_source.fetch(symbol, target_date)

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2, default=str))
        return

    if result.success:
        click.echo(f"\nSuccess: {result.source}")
        click.echo(f"Quality: {result.quality.name}")
        click.echo(f"Cache hit: {result.cache_hit}")
        click.echo("\nData:")
        click.echo(json.dumps(result.data, indent=2, default=str)[:2000])
        if len(json.dumps(result.data, default=str)) > 2000:
            click.echo("... (truncated)")
    else:
        click.echo(f"Failed: {result.error}", err=True)
        sys.exit(1)


@data.command("summary")
@click.argument("symbol")
@click.option("--date", "-d", "as_of_date", help="Historical date (YYYY-MM-DD)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def summary(ctx, symbol, as_of_date, json_output):
    """Get consolidated data summary for a symbol

    Fetches data from all relevant sources and shows a summary.

    Examples:
        investigator data summary AAPL
        investigator data summary MSFT --json
    """
    from investigator.domain.services.data_sources.manager import get_data_source_manager
    from datetime import datetime

    manager = get_data_source_manager()

    target_date = None
    if as_of_date:
        target_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()

    click.echo(f"Fetching data for {symbol}...")
    data = manager.get_data(symbol, target_date)

    if json_output:
        click.echo(json.dumps(data.to_dict(), indent=2, default=str))
        return

    click.echo("\n" + "=" * 60)
    click.echo(f"DATA SUMMARY: {symbol}")
    click.echo("=" * 60)
    click.echo(f"As of: {data.as_of_date}")
    click.echo(f"Quality: {data.overall_quality.name}")

    # Price
    if data.price:
        current = data.price.get("current", {})
        click.echo(f"\nPRICE")
        click.echo(f"  Current: ${current.get('close', 'N/A')}")
        returns = data.price.get("returns", {})
        if returns:
            click.echo(f"  1D Return: {returns.get('1d', 0):.2f}%")
            click.echo(f"  5D Return: {returns.get('5d', 0):.2f}%")

    # Technical
    if data.technical:
        indicators = data.technical.get("indicators", {})
        click.echo(f"\nTECHNICAL")
        click.echo(f"  RSI: {indicators.get('rsi_14', 'N/A')}")
        click.echo(f"  Above SMA20: {indicators.get('above_sma_20', 'N/A')}")

    # Sentiment
    if data.insider:
        summary = data.insider.get("summary", {})
        click.echo(f"\nINSIDER SENTIMENT")
        click.echo(f"  Buys: {summary.get('buys', 0)}")
        click.echo(f"  Sells: {summary.get('sells', 0)}")
        click.echo(f"  Sentiment: {summary.get('sentiment', 'N/A')}")

    if data.short_interest:
        current = data.short_interest.get("current", {})
        click.echo(f"\nSHORT INTEREST")
        click.echo(f"  Days to Cover: {current.get('days_to_cover', 'N/A')}")
        click.echo(f"  Short % Float: {current.get('short_pct_float', 'N/A')}%")

    # Macro
    if data.volatility:
        click.echo(f"\nVOLATILITY")
        click.echo(f"  VIX: {data.volatility.get('vix', 'N/A')}")
        click.echo(f"  Regime: {data.volatility.get('volatility_regime', 'N/A')}")

    # Sources
    click.echo(f"\nSOURCES")
    click.echo(f"  Succeeded: {len(data.sources_succeeded)}")
    click.echo(f"  Failed: {len(data.sources_failed)}")
    if data.sources_failed:
        click.echo(f"  Failed sources: {', '.join(data.sources_failed)}")

    click.echo("\n" + "=" * 60)


@data.command("health")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def health(ctx, json_output):
    """Check health of all data sources

    Tests each source and reports status.
    """
    from investigator.domain.services.data_sources.manager import get_data_source_manager

    click.echo("Checking data source health...")
    manager = get_data_source_manager()
    results = manager.health_check()

    if json_output:
        click.echo(json.dumps(results, indent=2, default=str))
        return

    click.echo("\n" + "=" * 60)
    click.echo("DATA SOURCE HEALTH")
    click.echo("=" * 60)

    healthy = 0
    degraded = 0
    failed = 0

    for source in results.get("sources", []):
        status = source.get("status", "unknown")
        name = source.get("name", "?")

        if status == "healthy":
            icon = click.style("OK", fg="green")
            healthy += 1
        elif status == "degraded":
            icon = click.style("WARN", fg="yellow")
            degraded += 1
        else:
            icon = click.style("FAIL", fg="red")
            failed += 1

        click.echo(f"  [{icon:4s}] {name:25s} {source.get('category', '')}")

        if source.get("error"):
            click.echo(f"         Error: {source['error'][:50]}")

    click.echo("\n" + "-" * 60)
    click.echo(f"Total: {results.get('total', 0)} sources")
    click.echo(f"Healthy: {healthy} | Degraded: {degraded} | Failed: {failed}")


@data.command("rl-features")
@click.argument("symbol")
@click.option("--date", "-d", "as_of_date", help="Historical date (YYYY-MM-DD)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def rl_features(ctx, symbol, as_of_date, json_output):
    """Extract RL features for a symbol

    Gets the feature vector used by RL models.

    Examples:
        investigator data rl-features AAPL
    """
    from investigator.domain.services.data_sources.manager import get_rl_features
    from datetime import datetime

    target_date = None
    if as_of_date:
        target_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()

    features = get_rl_features(symbol, target_date)

    if json_output:
        click.echo(json.dumps(features, indent=2))
        return

    click.echo(f"\nRL Features for {symbol}")
    click.echo("=" * 40)
    for name, value in sorted(features.items()):
        click.echo(f"  {name:20s}: {value:.4f}")


@data.command("refresh")
@click.option("--source", "-s", help="Specific source to refresh")
@click.option("--symbol", help="Specific symbol to refresh")
@click.option("--all", "refresh_all", is_flag=True, help="Refresh all sources")
@click.pass_context
def refresh(ctx, source, symbol, refresh_all):
    """Refresh/invalidate data source caches

    Examples:
        investigator data refresh --source price_history --symbol AAPL
        investigator data refresh --all
    """
    from investigator.domain.services.data_sources.manager import get_data_source_manager

    manager = get_data_source_manager()

    if refresh_all:
        from investigator.domain.services.data_sources.registry import get_registry
        registry = get_registry()
        registry.invalidate_all_caches()
        click.echo("All caches invalidated")

    elif source:
        if manager.refresh_source(source, symbol):
            msg = f"Cache invalidated for {source}"
            if symbol:
                msg += f" ({symbol})"
            click.echo(msg)
        else:
            click.echo(f"Source not found: {source}", err=True)
            sys.exit(1)

    else:
        click.echo("Specify --source, --symbol, or --all", err=True)
        sys.exit(1)
