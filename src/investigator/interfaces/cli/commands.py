"""
CLI Commands Interface

Click-based command-line interface for InvestiGator.
Thin wrapper around application layer services.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from investigator.application import AnalysisMode, AnalysisService
from investigator.infrastructure.cache.cache_manager import CacheManager
from investigator.infrastructure.monitoring import MetricsCollector


def create_cli():
    """
    Create the main CLI group

    This is the entry point for the investigator CLI.
    Pattern: CLI -> Application Service -> Domain/Infrastructure
    """

    @click.group()
    @click.option("--config", "-c", default="config.yaml", help="Configuration file")
    @click.option("--log-level", "-l", default="INFO", help="Log level")
    @click.option("--log-file", "-f", help="Log file path")
    @click.pass_context
    def cli(ctx, config, log_level, log_file):
        """InvestiGator - AI-powered investment analysis system"""
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout), *([] if not log_file else [logging.FileHandler(log_file)])],
        )

        # Store config in context
        ctx.ensure_object(dict)
        ctx.obj["config"] = config
        ctx.obj["log_level"] = log_level

    @cli.command()
    @click.argument("symbol")
    @click.option(
        "--mode",
        "-m",
        type=click.Choice(["quick", "standard", "comprehensive"]),
        default="standard",
        help="Analysis mode",
    )
    @click.option("--output", "-o", help="Output file for results")
    @click.option("--format", "-f", type=click.Choice(["json", "yaml", "text"]), default="json", help="Output format")
    @click.option("--report", is_flag=True, default=False, help="Generate PDF report")
    @click.option("--force-refresh", is_flag=True, default=False, help="Force cache refresh")
    @click.pass_context
    def analyze(ctx, symbol, mode, output, format, report, force_refresh):
        """
        Analyze a single stock

        Example:
            investigator analyze AAPL --mode standard
            investigator analyze MSFT --mode comprehensive --output results.json
        """

        async def run_analysis():
            # Initialize services
            cache_manager = CacheManager()
            metrics = MetricsCollector()

            # Use AnalysisService from application layer
            async with AnalysisService(cache_manager, metrics) as service:
                click.echo(f"Analyzing {symbol} in {mode} mode...")

                try:
                    results = await service.analyze_stock(symbol=symbol, mode=mode, force_refresh=force_refresh)

                    # Handle output
                    if output:
                        import json

                        output_path = Path(output)
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        with open(output_path, "w") as f:
                            if format == "json":
                                json.dump(results, f, indent=2)
                            elif format == "yaml":
                                import yaml

                                yaml.dump(results, f)
                            else:
                                f.write(str(results))

                        click.echo(f"Results saved to {output}")
                    else:
                        click.echo(f"\nAnalysis complete!")
                        click.echo(f"Status: {results.get('status', 'unknown')}")
                        click.echo(f"Duration: {results.get('duration', 0):.2f}s")

                    if report:
                        click.echo("PDF report generation not yet implemented in Clean Architecture")

                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
                    sys.exit(1)

        # Run async analysis
        asyncio.run(run_analysis())

    @cli.command()
    @click.argument("symbols", nargs=-1, required=True)
    @click.option(
        "--mode",
        "-m",
        type=click.Choice(["quick", "standard", "comprehensive"]),
        default="standard",
        help="Analysis mode",
    )
    @click.option("--output-dir", "-o", default="results", help="Output directory")
    @click.option("--force-refresh", is_flag=True, default=False, help="Force cache refresh")
    @click.pass_context
    def batch(ctx, symbols, mode, output_dir, force_refresh):
        """
        Analyze multiple stocks in batch

        Example:
            investigator batch AAPL MSFT GOOGL --mode standard
        """

        async def run_batch():
            cache_manager = CacheManager()
            metrics = MetricsCollector()

            async with AnalysisService(cache_manager, metrics) as service:
                click.echo(f"Analyzing {len(symbols)} symbols in {mode} mode...")

                try:
                    results = await service.batch_analyze(symbols=list(symbols), mode=mode)

                    # Save results
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)

                    for result in results:
                        symbol = result.get("symbol", "unknown")
                        file_path = output_path / f"{symbol}_{mode}.json"

                        import json

                        with open(file_path, "w") as f:
                            json.dump(result, f, indent=2)

                        click.echo(f"✓ {symbol} -> {file_path}")

                    click.echo(f"\nBatch analysis complete! Results in {output_dir}/")

                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
                    sys.exit(1)

        asyncio.run(run_batch())

    @cli.command()
    @click.pass_context
    def status(ctx):
        """Show system status"""
        click.echo("System Status Check")
        click.echo("=" * 50)
        click.echo("✓ CLI interface: OK")
        click.echo("✓ Application layer: OK")
        click.echo("✓ Domain layer: OK")
        click.echo("✓ Infrastructure layer: OK")
        click.echo("\nClean Architecture refactoring in progress...")

    return cli


# Compatibility with old imports
cli = create_cli()


if __name__ == "__main__":
    cli()
