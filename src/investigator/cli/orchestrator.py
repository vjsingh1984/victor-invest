#!/usr/bin/env python3
"""
InvestiGator Main Application
Agentic AI-powered investment analysis system
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import click
import yaml

# Add src/ to Python path for investigator package
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from investigator.application import (
    AgentOrchestrator,
    AnalysisMode,
    OutputDetailLevel,
    Priority,
    format_analysis_output,
)
from investigator.domain.agents.sec import SECAnalysisAgent
from investigator.infrastructure.cache import CacheManager
from investigator.infrastructure.events import EventBus
from investigator.infrastructure.llm import OllamaClient
from investigator.infrastructure.monitoring import AlertManager, MetricsCollector

# from api.main import create_app  # Will fix this separately

try:
    import uvicorn  # Optional; only needed for API mode
except ImportError:
    uvicorn = None

# PDF Report generation
from investigator.application import InvestmentSynthesizer
from investigator.domain.models import InvestmentRecommendation


# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure application logging with production-friendly defaults.

    Use INVESTIGATOR_LOG_PROFILE=debug (or --log-level DEBUG) when you need
    verbose component tracing. Otherwise we keep the runtime succinct by
    promoting chatty subsystems to WARNING.
    """

    profile = os.getenv("INVESTIGATOR_LOG_PROFILE", "prod").strip().lower()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers,
        force=True,  # Ensure repeated CLI invocations reset handlers cleanly
    )

    # Always quiet extremely noisy third-party loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    # Promote high-volume internal modules to WARNING in production-style runs
    if profile != "debug" and numeric_level >= logging.INFO:
        noisy_loggers = [
            "investigator.infrastructure.llm.pool",
            "investigator.infrastructure.llm.semaphore",
            "investigator.infrastructure.cache.cache_manager",
            "investigator.infrastructure.cache.cache_cleaner",
            "agent.synth_agent_1",
            "agent.fund_agent_1",
            "utils.market_data_fetcher",
            "utils.sec_data_strategy",
        ]
        for name in noisy_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)

    # Canonical mapper debug tracing can be toggled independently
    if profile == "debug" or numeric_level <= logging.DEBUG or os.getenv("INVESTIGATOR_DEBUG_CANONICAL") == "1":
        logging.getLogger("utils.canonical_key_mapper").setLevel(logging.DEBUG)


# Load configuration
def load_config(config_file: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    config_path = Path(config_file)

    if not config_path.exists():
        # Create default config
        default_config = {
            "ollama": {"base_url": "http://localhost:11434", "timeout": 300, "max_retries": 3},
            "cache": {"redis_url": "redis://localhost:6379", "file_cache_path": "data/cache", "ttl_default": 3600},
            "orchestrator": {"max_concurrent_analyses": 5, "max_concurrent_agents": 10},
            "api": {"host": "0.0.0.0", "port": 8000, "workers": 4},
            "monitoring": {"export_interval": 60, "metrics_port": 9090},
        }

        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

        return default_config

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_executive_summary(full_analysis: dict) -> dict:
    """
    Extract user-friendly highlights from full analysis.

    FEATURE: Executive Summary (user suggestion #7)
    Creates a concise, actionable summary from detailed analysis JSON.

    Args:
        full_analysis: Full analysis dict from orchestrator

    Returns:
        Dict with executive summary containing key highlights
    """
    try:
        fundamental = full_analysis.get("fundamental", {}).get("analysis", {})
        technical = full_analysis.get("technical", {}).get("analysis", {})
        synthesis = full_analysis.get("synthesis", {}).get("synthesis", {})

        # Extract fundamental data safely
        fund_response = fundamental.get("response", {})
        health_analysis = fund_response.get("health_analysis", {}).get("response", {})
        valuation = fund_response.get("valuation", {}).get("response", {})
        ratios = fund_response.get("ratios", {})

        # Extract technical data safely
        tech_signals = technical.get("signals", {}).get("response", {})

        # Extract synthesis data safely
        synth_response = synthesis.get("response", {})
        investment_thesis = synth_response.get("investment_thesis", {})
        risk_analysis = synth_response.get("risk_analysis", {})
        recommendation = synth_response.get("recommendation_and_action_plan", {})

        # Build executive summary
        summary = {
            "symbol": full_analysis.get("symbol", "N/A"),
            "timestamp": full_analysis.get("timestamp", datetime.now().isoformat()),
            # Top-line recommendation
            "recommendation": recommendation.get("recommendation", "N/A"),
            "confidence_level": full_analysis.get("fundamental", {})
            .get("confidence", {})
            .get("confidence_level", "N/A"),
            # Key metrics
            "current_price": ratios.get("current_price", "N/A"),
            "price_target_12m": valuation.get("price_target_12_month", "N/A"),
            "expected_return": f"{((valuation.get('price_target_12_month', 0) - ratios.get('current_price', 0)) / ratios.get('current_price', 1) * 100) if ratios.get('current_price', 0) > 0 else 0:.1f}%",
            # Investment grade
            "investment_grade": valuation.get("investment_grade", "N/A"),
            "financial_health_score": health_analysis.get("overall_health_score", "N/A"),
            # Key strengths (top 3)
            "key_strengths": (
                investment_thesis.get("value_drivers", [])[:3]
                if investment_thesis.get("value_drivers")
                else ["Data not available"]
            ),
            # Key risks (top 3)
            "key_risks": (
                risk_analysis.get("primary_risks", [])[:3]
                if risk_analysis.get("primary_risks")
                else ["Data not available"]
            ),
            # Technical signal
            "technical_signal": tech_signals.get("entry_signal", "N/A"),
            "technical_confidence": (
                tech_signals.get("reasoning", {}).get("confidence_level", "N/A")
                if isinstance(tech_signals.get("reasoning"), dict)
                else "N/A"
            ),
            # Data quality
            "data_quality": full_analysis.get("fundamental", {}).get("data_quality", {}).get("quality_grade", "N/A"),
            "data_quality_score": full_analysis.get("fundamental", {})
            .get("data_quality", {})
            .get("data_quality_score", "N/A"),
            # Next steps (top 3)
            "next_steps": (
                recommendation.get("specific_actions", [])[:3]
                if recommendation.get("specific_actions")
                else ["Review full analysis"]
            ),
            # Market context (brief)
            "market_regime": full_analysis.get("market_context", {})
            .get("market_performance", {})
            .get("market_regime", "N/A"),
        }

        return summary

    except Exception as e:
        logging.error(f"Error generating executive summary: {e}")
        return {
            "error": f"Failed to generate summary: {e}",
            "symbol": full_analysis.get("symbol", "Unknown"),
            "message": "Please review full analysis JSON",
        }


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Configuration file")
@click.option("--log-level", "-l", default="INFO", help="Log level")
@click.option("--log-file", "-f", help="Log file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging (equivalent to --log-level DEBUG)")
@click.pass_context
def cli(ctx, config, log_level, log_file, verbose):
    """InvestiGator - Agentic AI Investment Analysis System"""
    effective_level = "DEBUG" if verbose else log_level
    setup_logging(effective_level, log_file)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@cli.command()
@click.argument("symbol")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["quick", "standard", "comprehensive"]),
    default="comprehensive",
    help="Analysis mode",
)
@click.option("--output", "-o", help="Output file for results")
@click.option("--format", "-f", type=click.Choice(["json", "yaml", "text"]), default="json", help="Output format")
@click.option(
    "--detail-level",
    "-d",
    type=click.Choice(["minimal", "standard", "verbose"]),
    default="standard",
    help="Output detail level: minimal (summary only), standard (investor decision-making, default), verbose (full with metadata)",
)
@click.option("--report", is_flag=True, default=False, help="Generate PDF investment report")
@click.option(
    "--force-refresh",
    is_flag=True,
    default=False,
    help="Bypass cache for this run (clears cached data before analysis)",
)
@click.option("--refresh", "refresh_alias", is_flag=True, default=False, help="Alias for --force-refresh")
@click.pass_context
def analyze(ctx, symbol, mode, output, format, detail_level, report, force_refresh, refresh_alias):
    """Analyze a single stock symbol"""
    config = ctx.obj["config"]
    force_refresh = force_refresh or refresh_alias

    async def run_analysis():
        # Initialize components
        from investigator.config import get_config
        from investigator.infrastructure.cache import get_cache_manager

        cfg = get_config()
        original_force_refresh = getattr(cfg.cache_control, "force_refresh", False)
        original_force_symbols = getattr(cfg.cache_control, "force_refresh_symbols", None)

        if force_refresh:
            cfg.cache_control.force_refresh = True
            cfg.cache_control.force_refresh_symbols = [symbol]

            # Actually clear cache files for this symbol
            click.echo(f"🔄 Force refresh enabled - clearing all caches for {symbol}")
            cache_manager = get_cache_manager()
            cache_manager.config = cfg

            # Clear file and database caches
            from investigator.infrastructure.cache.cache_types import CacheType

            cache_types_to_clear = [
                CacheType.LLM_RESPONSE,
                CacheType.TECHNICAL_DATA,
                CacheType.COMPANY_FACTS,
                CacheType.SEC_RESPONSE,
                CacheType.MARKET_CONTEXT,
                CacheType.QUARTERLY_METRICS,
            ]

            for cache_type in cache_types_to_clear:
                try:
                    cache_manager.delete(cache_type, {"symbol": symbol})
                except Exception as e:
                    # Silently continue if cache doesn't exist
                    pass

            click.echo(f"✅ Cache cleared for {symbol}")
        else:
            cfg.cache_control.force_refresh = original_force_refresh
            cfg.cache_control.force_refresh_symbols = original_force_symbols
            cache_manager = get_cache_manager()
            cache_manager.config = cfg

        metrics_collector = MetricsCollector()
        orchestrator = AgentOrchestrator(cache_manager, metrics_collector)

        try:
            # Start services
            await metrics_collector.start()
            await orchestrator.start()

            # Submit analysis
            analysis_mode = AnalysisMode[mode.upper()]
            task_id = await orchestrator.analyze(symbol, analysis_mode)

            click.echo(f"Analysis started for {symbol} (Task ID: {task_id})")
            click.echo("Processing...")

            # Wait for results with progress indicator
            with click.progressbar(length=100, label="Analyzing") as bar:
                elapsed = 0
                while elapsed < 900:  # 15 minute timeout (exceeds agent timeout in agents/base.py:270)
                    status = await orchestrator.get_status(task_id)

                    if status["status"] == "completed":
                        bar.update(100 - bar.pos)
                        break
                    elif status["status"] == "processing":
                        progress = (status.get("agents_completed", 0) / status.get("total_agents", 1)) * 100
                        bar.update(progress - bar.pos)

                    await asyncio.sleep(2)
                    elapsed += 2

            # Get results
            results = await orchestrator.get_results(task_id)

            # Validate critical agent success
            if results and "agents" in results:
                agents = results["agents"]
                failures = []

                # Check critical agents based on mode
                critical_agents = []
                if analysis_mode in [AnalysisMode.STANDARD, AnalysisMode.COMPREHENSIVE]:
                    critical_agents = ["technical", "fundamental"]
                elif analysis_mode == AnalysisMode.QUICK:
                    critical_agents = ["technical"]

                for agent_name in critical_agents:
                    if agent_name in agents:
                        agent_result = agents[agent_name]
                        if isinstance(agent_result, dict):
                            status = agent_result.get("status")
                            if status == "error":
                                error_msg = agent_result.get("error", "Unknown error")
                                failures.append(f"{agent_name}: {error_msg}")

                if failures:
                    click.echo("\n❌ Analysis failed - critical agent errors:", err=True)
                    for failure in failures:
                        click.echo(f"  - {failure}", err=True)
                    sys.exit(1)

            if results:
                # Apply detail level formatting
                detail_level_enum = OutputDetailLevel(detail_level)
                formatted_results = format_analysis_output(results, detail_level_enum)

                # Generate executive summary using MINIMAL detail level
                exec_summary = format_analysis_output(results, OutputDetailLevel.MINIMAL)

                # Display executive summary to console (always shown)
                # Extract nested values from MINIMAL format
                rec = exec_summary.get("recommendation", {})
                val = exec_summary.get("valuation", {})
                thesis = exec_summary.get("thesis", {})
                dq = exec_summary.get("data_quality", {})

                click.echo("\n" + "=" * 60)
                click.echo("EXECUTIVE SUMMARY")
                click.echo("=" * 60)
                click.echo(f"Symbol: {exec_summary.get('symbol')}")
                click.echo(f"Recommendation: {rec.get('action', 'N/A')}")
                click.echo(f"Confidence: {rec.get('confidence', 'N/A')}")

                # Format price and target
                curr_price = val.get("current_price", "N/A")
                target = val.get("price_target_12m", "N/A")
                exp_ret = val.get("expected_return_pct", "N/A")
                if exp_ret != "N/A" and exp_ret is not None:
                    exp_ret_str = f"{exp_ret:.1f}%"
                else:
                    exp_ret_str = "N/A"
                click.echo(f"Price: ${curr_price} → Target: ${target} ({exp_ret_str})")

                click.echo(f"Investment Grade: {val.get('investment_grade', 'N/A')}")

                # Format data quality
                dq_score = dq.get("overall_score", "N/A")
                dq_assess = dq.get("assessment", "N/A")
                if dq_score != "N/A" and dq_score is not None:
                    click.echo(f"Data Quality: {dq_assess} ({dq_score:.1f}%)")
                else:
                    click.echo(f"Data Quality: {dq_assess}")

                click.echo(f"\nKey Strengths:")
                for strength in thesis.get("key_strengths", []):
                    click.echo(f"  • {strength}")

                click.echo(f"\nKey Risks:")
                for risk in thesis.get("key_risks", []):
                    click.echo(f"  • {risk}")

                click.echo("=" * 60 + "\n")

                # Format output using formatted results
                if format == "json":
                    output_data = json.dumps(formatted_results, indent=2, default=str)
                elif format == "yaml":
                    output_data = yaml.dump(formatted_results, default_flow_style=False)
                else:
                    output_data = format_results_text(formatted_results)

                # Save or print full analysis
                if output:
                    # Create output directory if it doesn't exist
                    output_path = Path(output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save full analysis
                    with open(output, "w") as f:
                        f.write(output_data)

                    # Also save executive summary separately
                    summary_path = str(output_path).replace(".json", "_summary.json")
                    with open(summary_path, "w") as f:
                        json.dump(exec_summary, f, indent=2, default=str)

                    click.echo(f"Full analysis saved to {output}")
                    click.echo(f"Executive summary saved to {summary_path}")
                else:
                    click.echo("\n[Full Analysis]")
                    click.echo(output_data)

                # Generate PDF report if requested
                if report:
                    try:
                        click.echo("\nGenerating PDF report...")

                        # Convert results to InvestmentRecommendation format
                        recommendation = convert_to_investment_recommendation(results, symbol)

                        # Initialize synthesizer for report generation
                        synthesizer = InvestmentSynthesizer()

                        # Generate PDF report
                        report_path = synthesizer.generate_report([recommendation], report_type="synthesis")

                        click.echo(f"✅ PDF report generated: {report_path}")

                    except Exception as e:
                        click.echo(f"❌ Failed to generate PDF report: {str(e)}", err=True)
                        # Don't fail the entire command if PDF generation fails

            else:
                click.echo("Analysis timed out or failed", err=True)
                sys.exit(1)

        finally:
            # Cleanup
            await orchestrator.stop()
            await metrics_collector.stop()

            # Restore cache configuration
            cfg.cache_control.force_refresh = original_force_refresh
            cfg.cache_control.force_refresh_symbols = original_force_symbols

    # Run async function
    asyncio.run(run_analysis())


@cli.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--mode", "-m", type=click.Choice(["quick", "standard", "comprehensive"]), default="standard", help="Analysis mode"
)
@click.option("--output-dir", "-o", default="results", help="Output directory")
@click.option(
    "--detail-level",
    "-d",
    type=click.Choice(["minimal", "standard", "verbose"]),
    default="standard",
    help="Output detail level: minimal (summary only), standard (investor decision-making, default), verbose (full with metadata)",
)
@click.option("--force-refresh", is_flag=True, default=False, help="Bypass cache for all symbols in this batch run")
@click.option("--refresh", "refresh_alias", is_flag=True, default=False, help="Alias for --force-refresh")
@click.pass_context
def batch(ctx, symbols, mode, output_dir, detail_level, force_refresh, refresh_alias):
    """Analyze multiple symbols in batch"""
    config = ctx.obj["config"]
    force_refresh = force_refresh or refresh_alias

    async def run_batch():
        # Initialize components
        from investigator.config import get_config
        from investigator.infrastructure.cache import get_cache_manager

        cfg = get_config()
        original_force_refresh = getattr(cfg.cache_control, "force_refresh", False)
        original_force_symbols = getattr(cfg.cache_control, "force_refresh_symbols", None)

        if force_refresh:
            cfg.cache_control.force_refresh = True
            cfg.cache_control.force_refresh_symbols = list(symbols)
        else:
            cfg.cache_control.force_refresh = original_force_refresh
            cfg.cache_control.force_refresh_symbols = original_force_symbols

        cache_manager = get_cache_manager()
        cache_manager.config = cfg
        metrics_collector = MetricsCollector()
        orchestrator = AgentOrchestrator(cache_manager, metrics_collector)

        try:
            # Start services
            await metrics_collector.start()
            await orchestrator.start()

            # Submit all analyses
            analysis_mode = AnalysisMode[mode.upper()]
            task_ids = await orchestrator.analyze_batch(list(symbols), analysis_mode)

            click.echo(f"Batch analysis started for {len(symbols)} symbols")

            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Wait for results
            results = {}
            detail_level_enum = OutputDetailLevel(detail_level)
            with click.progressbar(symbols, label="Processing") as bar:
                for symbol, task_id in zip(bar, task_ids):
                    result = await orchestrator.get_results(task_id, wait=True, timeout=300)
                    if result:
                        results[symbol] = result

                        # Apply detail level formatting
                        formatted_result = format_analysis_output(result, detail_level_enum)

                        # Save individual result
                        output_file = Path(output_dir) / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(output_file, "w") as f:
                            json.dump(formatted_result, f, indent=2, default=str)

            # Save summary
            summary_file = Path(output_dir) / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, "w") as f:
                json.dump(
                    {
                        "symbols": list(symbols),
                        "mode": mode,
                        "completed": len(results),
                        "failed": len(symbols) - len(results),
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            click.echo(f"\nCompleted {len(results)}/{len(symbols)} analyses")
            click.echo(f"Results saved to {output_dir}")

        finally:
            # Cleanup
            await orchestrator.stop()
            await metrics_collector.stop()
            cfg.cache_control.force_refresh = original_force_refresh
            cfg.cache_control.force_refresh_symbols = original_force_symbols

    asyncio.run(run_batch())


@cli.command()
@click.argument("target")
@click.argument("peers", nargs=-1, required=True)
@click.option("--output", "-o", help="Output file for comparison")
@click.pass_context
def compare(ctx, target, peers, output):
    """Compare target symbol with peer companies"""
    config = ctx.obj["config"]

    async def run_comparison():
        # Initialize components
        from investigator.infrastructure.cache import get_cache_manager

        cache_manager = get_cache_manager()
        metrics_collector = MetricsCollector()
        orchestrator = AgentOrchestrator(cache_manager, metrics_collector)

        try:
            # Start services
            await metrics_collector.start()
            await orchestrator.start()

            # Submit peer comparison
            task_id = await orchestrator.analyze_peer_group(target, list(peers))

            click.echo(f"Peer comparison started: {target} vs {', '.join(peers)}")
            click.echo("This may take several minutes...")

            # Wait for results
            results = await orchestrator.get_results(task_id, wait=True, timeout=600)

            if results:
                # Format comparison report
                report = format_peer_comparison(results)

                if output:
                    with open(output, "w") as f:
                        json.dump(results, f, indent=2, default=str)
                    click.echo(f"Comparison saved to {output}")

                click.echo("\n" + report)
            else:
                click.echo("Comparison timed out or failed", err=True)
                sys.exit(1)

        finally:
            # Cleanup
            await orchestrator.stop()
            await metrics_collector.stop()

    asyncio.run(run_comparison())


@cli.command()
@click.option("--host", "-h", default="0.0.0.0", help="API host")
@click.option("--port", "-p", default=8000, help="API port")
@click.option("--workers", "-w", default=4, help="Number of workers")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.pass_context
def serve(ctx, host, port, workers, reload):
    """Start the REST API server"""
    config = ctx.obj["config"]

    # Update config with CLI options
    config["api"]["host"] = host
    config["api"]["port"] = port
    config["api"]["workers"] = workers

    # Temporarily disable API server until fixed
    click.echo("API server is temporarily disabled while being refactored.")
    click.echo("Please use the 'analyze' command for analysis.")
    return

    # TODO: Fix API server imports
    # app = create_app(config)
    # uvicorn.run(
    #     app,
    #     host=host,
    #     port=port,
    #     workers=workers if not reload else 1,
    #     reload=reload,
    #     log_level="info"
    # )


@cli.command()
@click.pass_context
def status(ctx):
    """Check system status and health"""
    config = ctx.obj["config"]

    async def check_status():
        # Initialize components
        from investigator.infrastructure.cache import get_cache_manager

        cache_manager = get_cache_manager()
        metrics_collector = MetricsCollector()
        ollama_client = OllamaClient(config["ollama"]["base_url"])

        click.echo("InvestiGator System Status")
        click.echo("=" * 40)

        # Check Ollama
        async with ollama_client:
            if await ollama_client.health_check():
                models = await ollama_client.list_models()
                click.echo(f"✓ Ollama: Online ({len(models)} models available)")
                for model in models[:5]:  # Show first 5 models
                    click.echo(f"  - {model['name']}")
            else:
                click.echo("✗ Ollama: Offline")

        # Check Cache
        if await cache_manager.ping():
            click.echo("✓ Cache: Connected")
        else:
            click.echo("✗ Cache: Disconnected")

        # Check Metrics
        await metrics_collector.start()
        health = metrics_collector.get_system_health()
        await metrics_collector.stop()

        # Safely access health dict with defaults
        status = health.get("status", "unknown")
        click.echo(f"\nSystem Health: {status.upper()}")

        if health.get("issues"):
            click.echo("Issues:")
            for issue in health["issues"]:
                click.echo(f"  - {issue}")

        if health.get("metrics"):
            click.echo(f"\nMetrics:")
            for key, value in health["metrics"].items():
                if isinstance(value, float):
                    click.echo(f"  {key}: {value:.2f}")
                else:
                    click.echo(f"  {key}: {value}")

    asyncio.run(check_status())


@cli.command()
@click.option("--days", "-d", default=7, help="Days of history to show")
@click.pass_context
def metrics(ctx, days):
    """View system metrics and performance"""
    config = ctx.obj["config"]

    async def show_metrics():
        # Initialize metrics collector
        metrics_collector = MetricsCollector()

        # Load historical metrics
        import glob
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=days)
        metrics_files = glob.glob("metrics/metrics_*.json")

        all_metrics = []
        for filepath in sorted(metrics_files):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    timestamp = datetime.fromisoformat(data["timestamp"])
                    if timestamp >= cutoff_date:
                        all_metrics.append(data)
            except:
                continue

        if not all_metrics:
            click.echo("No metrics data available")
            return

        # Display summary
        click.echo(f"\nMetrics Summary (Last {days} days)")
        click.echo("=" * 50)

        latest = all_metrics[-1] if all_metrics else {}

        if "system_metrics" in latest:
            sys_metrics = latest["system_metrics"]
            click.echo("\nSystem Metrics:")
            click.echo(f"  Total Analyses: {sys_metrics.get('total_analyses', 0)}")
            click.echo(
                f"  Success Rate: {(sys_metrics.get('successful_analyses', 0) / max(sys_metrics.get('total_analyses', 1), 1) * 100):.1f}%"
            )
            click.echo(
                f"  Cache Hit Rate: {(sys_metrics.get('cache_hits', 0) / max(sys_metrics.get('cache_hits', 0) + sys_metrics.get('cache_misses', 0), 1) * 100):.1f}%"
            )

        if "agent_metrics" in latest:
            click.echo("\nAgent Performance:")
            for agent, metrics in latest["agent_metrics"].items():
                click.echo(f"  {agent}:")
                click.echo(f"    Executions: {metrics.get('executions', 0)}")
                click.echo(f"    Avg Duration: {metrics.get('average_duration', 0):.2f}s")
                click.echo(
                    f"    Success Rate: {((metrics.get('executions', 0) - metrics.get('failures', 0)) / max(metrics.get('executions', 1), 1) * 100):.1f}%"
                )

    asyncio.run(show_metrics())


@cli.command()
@click.argument("model")
@click.pass_context
def pull(ctx, model):
    """Pull an Ollama model"""
    config = ctx.obj["config"]

    async def pull_model():
        ollama_client = OllamaClient(config["ollama"]["base_url"])

        async with ollama_client:
            click.echo(f"Pulling model: {model}")

            with click.progressbar(length=100, label="Downloading") as bar:
                last_percent = 0

                async for status in ollama_client.pull_model(model, stream=True):
                    if "completed" in status and "total" in status:
                        percent = (status["completed"] / status["total"]) * 100
                        bar.update(percent - last_percent)
                        last_percent = percent

                    if status.get("status") == "success":
                        bar.update(100 - bar.pos)
                        break

            click.echo(f"Model {model} pulled successfully")

    asyncio.run(pull_model())


@cli.command()
@click.option("--source", "-s", default=None, help="Filter by source (e.g., atlanta_fed, cboe)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def economic(ctx, source, json_output):
    """View current economic indicators (Regional Fed + CBOE)

    Shows real-time economic data including:
    - Atlanta Fed GDPNow estimate
    - Chicago Fed CFNAI, NFCI
    - Cleveland Fed inflation expectations
    - Dallas Fed Trimmed Mean PCE
    - NY Fed recession probability
    - CBOE VIX, SKEW, term structure
    """
    from datetime import date

    from investigator.domain.services.data_sources.facade import get_data_source_facade

    facade = get_data_source_facade()
    analysis_data = facade.get_historical_data_sync(
        symbol="_MACRO",
        as_of_date=date.today(),
    )

    regional_fed = analysis_data.regional_fed_indicators or {}
    cboe = analysis_data.cboe_data or {}

    if json_output:
        import json as json_lib

        output = {
            "regional_fed": regional_fed,
            "cboe": cboe,
            "as_of_date": date.today().isoformat(),
        }
        click.echo(json_lib.dumps(output, indent=2, default=str))
        return

    # Format as readable text
    click.echo("\n" + "=" * 60)
    click.echo("ECONOMIC INDICATORS DASHBOARD")
    click.echo("=" * 60)
    click.echo(f"As of: {date.today().isoformat()}")

    # Show CBOE volatility data first
    if cboe and (source is None or source == "cboe"):
        click.echo("\n📈 VOLATILITY (CBOE)")
        click.echo("-" * 40)
        vix = cboe.get("vix")
        skew = cboe.get("skew")
        vix3m = cboe.get("vix3m")
        regime = cboe.get("volatility_regime", "unknown")

        if vix:
            click.echo(f"  VIX:              {vix:.2f}")
        if vix3m:
            click.echo(f"  VIX3M:            {vix3m:.2f}")
        if skew:
            skew_status = "⚠️  ELEVATED" if skew > 130 else "✓ Normal"
            click.echo(f"  SKEW:             {skew:.2f} {skew_status}")
        click.echo(f"  Volatility Regime: {regime.upper()}")
        if cboe.get("term_structure"):
            click.echo(f"  Term Structure:   {cboe['term_structure']}")
        if cboe.get("is_backwardation"):
            click.echo("  ⚠️  VIX in BACKWARDATION (fear signal)")

    # Show Regional Fed summary
    fed_summary = regional_fed.get("summary", {}) if isinstance(regional_fed, dict) else {}
    if fed_summary and (source is None or source not in ["cboe"]):
        click.echo("\n📊 ECONOMIC ACTIVITY")
        click.echo("-" * 40)

        if fed_summary.get("gdpnow") is not None:
            gdp = fed_summary["gdpnow"]
            gdp_status = "🔴" if gdp < 0 else "🟢" if gdp > 2 else "🟡"
            click.echo(f"  {gdp_status} GDPNow (Atlanta Fed):  {gdp:.2f}%")

        if fed_summary.get("cfnai") is not None:
            cfnai = fed_summary["cfnai"]
            cfnai_status = "🔴" if cfnai < -0.7 else "🟢" if cfnai > 0 else "🟡"
            click.echo(f"  {cfnai_status} CFNAI (Chicago Fed):  {cfnai:.3f}")

        if fed_summary.get("empire_state_mfg") is not None:
            emp = fed_summary["empire_state_mfg"]
            emp_status = "🔴" if emp < -10 else "🟢" if emp > 10 else "🟡"
            click.echo(f"  {emp_status} Empire State Mfg:     {emp:.1f}")

        click.echo("\n💰 FINANCIAL CONDITIONS")
        click.echo("-" * 40)

        if fed_summary.get("nfci") is not None:
            nfci = fed_summary["nfci"]
            nfci_status = "🔴" if nfci > 0.5 else "🟢" if nfci < 0 else "🟡"
            click.echo(f"  {nfci_status} NFCI (Chicago Fed):   {nfci:.3f} (0=avg, +=tight)")

        if fed_summary.get("kcfsi") is not None:
            kcfsi = fed_summary["kcfsi"]
            kcfsi_status = "🔴" if kcfsi > 0.5 else "🟢" if kcfsi < 0 else "🟡"
            click.echo(f"  {kcfsi_status} KCFSI (Kansas City):  {kcfsi:.3f} (0=avg, +=stress)")

        click.echo("\n📈 INFLATION & RECESSION")
        click.echo("-" * 40)

        if fed_summary.get("inflation_expectations") is not None:
            infl = fed_summary["inflation_expectations"]
            infl_status = "🔴" if infl > 4 else "🟢" if infl < 2.5 else "🟡"
            click.echo(f"  {infl_status} Inflation Expectations: {infl:.2f}%")

        if fed_summary.get("recession_probability") is not None:
            rec = fed_summary["recession_probability"]
            # Scale appropriately - NY Fed reports as percentage
            rec_pct = rec * 100 if rec < 1 else rec
            rec_status = "🔴" if rec_pct > 30 else "🟢" if rec_pct < 15 else "🟡"
            click.echo(f"  {rec_status} Recession Probability:  {rec_pct:.1f}%")

    # Show detailed data by district if requested
    by_district = regional_fed.get("by_district", {}) if isinstance(regional_fed, dict) else {}
    if source and source != "cboe" and by_district:
        if source in by_district:
            click.echo(f"\n📋 {source.upper()} DETAILS")
            click.echo("-" * 40)
            for indicator, data in by_district[source].items():
                val = data.get("value")
                if val is not None:
                    click.echo(f"  {indicator}: {val}")
        else:
            click.echo(f"\n⚠️  Source '{source}' not found")
            click.echo(f"Available sources: {', '.join(by_district.keys())}")

    click.echo("\n" + "=" * 60)


def format_results_text(results: dict) -> str:
    """Format results as readable text"""
    output = []
    output.append(f"Analysis Results for {results.get('symbol', 'Unknown')}")
    output.append("=" * 50)

    if "agents" in results:
        for agent_name, agent_data in results["agents"].items():
            output.append(f"\n{agent_name.upper()} Analysis:")
            output.append("-" * 30)

            if agent_name == "synthesis" and "synthesis" in agent_data:
                synthesis = agent_data["synthesis"]
                if "executive_summary" in synthesis:
                    output.append("\nExecutive Summary:")
                    output.append(synthesis["executive_summary"])

                if "recommendation" in synthesis:
                    rec = synthesis["recommendation"]
                    output.append(f"\nRecommendation: {rec.get('final_recommendation', 'N/A').upper()}")
                    output.append(f"Conviction: {rec.get('conviction_level', 'N/A')}")
                    output.append(f"Expected Return: {rec.get('expected_return', 0):.1%}")

            elif "analysis" in agent_data:
                analysis = agent_data["analysis"]
                if isinstance(analysis, dict):
                    for key, value in list(analysis.items())[:3]:  # Show first 3 items
                        output.append(f"{key}: {value}")

    return "\n".join(output)


def convert_to_investment_recommendation(results: dict, symbol: str) -> InvestmentRecommendation:
    """
    Convert AgentOrchestrator results to InvestmentRecommendation format

    Args:
        results: Results from orchestrator.get_results()
        symbol: Stock symbol

    Returns:
        InvestmentRecommendation dataclass instance
    """
    # Extract synthesis results
    synthesis_data = results.get("agents", {}).get("synthesis", {})
    synthesis = synthesis_data.get("synthesis", {})
    recommendation_data = synthesis.get("recommendation", {})

    # Extract scores
    scores = synthesis.get("scores", {})
    overall_score = scores.get("overall_score", 5.0)
    fundamental_score = scores.get("fundamental_score", 5.0)
    technical_score = scores.get("technical_score", 5.0)

    # Extract sub-scores (with defaults)
    income_score = scores.get("income_score", fundamental_score)
    cashflow_score = scores.get("cashflow_score", fundamental_score)
    balance_score = scores.get("balance_score", fundamental_score)
    growth_score = scores.get("growth_score", fundamental_score)
    value_score = scores.get("value_score", fundamental_score)
    business_quality_score = scores.get("business_quality_score", fundamental_score)

    # Extract recommendation details
    final_recommendation = recommendation_data.get("final_recommendation", "HOLD")
    conviction_level = recommendation_data.get("conviction_level", "MEDIUM")

    # Extract investment thesis and strategies
    investment_thesis = synthesis.get("executive_summary", "Analysis completed")
    key_catalysts = synthesis.get("key_catalysts", [])
    key_risks = synthesis.get("key_risks", [])
    key_insights = synthesis.get("key_insights", [])

    # Extract technical data
    technical_data = results.get("agents", {}).get("technical", {}).get("analysis", {})
    current_price = technical_data.get("current_price")
    price_target = recommendation_data.get("price_target")
    stop_loss = recommendation_data.get("stop_loss")

    # Extract entry/exit strategies
    entry_strategy = recommendation_data.get("entry_strategy", "Market order at current levels")
    exit_strategy = recommendation_data.get("exit_strategy", "Target-based or stop-loss exit")

    # Determine time horizon and position size
    expected_return = recommendation_data.get("expected_return", 0.0)
    if expected_return > 0.3:
        time_horizon = "SHORT-TERM"
        position_size = "MODERATE"
    elif expected_return > 0.15:
        time_horizon = "MEDIUM-TERM"
        position_size = "MODERATE"
    else:
        time_horizon = "LONG-TERM"
        position_size = "SMALL"

    # Handle data quality
    data_quality_score = synthesis.get("data_quality", {}).get("overall_quality", 0.8)

    # Create InvestmentRecommendation
    return InvestmentRecommendation(
        symbol=symbol,
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
        key_catalysts=key_catalysts if isinstance(key_catalysts, list) else [],
        key_risks=key_risks if isinstance(key_risks, list) else [],
        key_insights=key_insights if isinstance(key_insights, list) else [],
        entry_strategy=entry_strategy,
        exit_strategy=exit_strategy,
        stop_loss=stop_loss,
        analysis_timestamp=datetime.now(),
        data_quality_score=data_quality_score,
        analysis_thinking=synthesis.get("reasoning"),
        synthesis_details=json.dumps(synthesis, default=str),
    )


def format_peer_comparison(results: dict) -> str:
    """Format peer comparison results"""
    output = []
    output.append(f"Peer Comparison: {results.get('target', 'Unknown')}")
    output.append("=" * 50)

    if "comparison" in results:
        comp = results["comparison"]

        if "ranking" in comp:
            output.append("\nRanking:")
            for i, company in enumerate(comp["ranking"], 1):
                output.append(f"  {i}. {company}")

        if "investment_preference" in comp:
            output.append(f"\nBest Investment: {comp['investment_preference']}")

        if "relative_valuation" in comp:
            output.append("\nRelative Valuation:")
            output.append(comp["relative_valuation"])

    return "\n".join(output)


# ============================================================================
# Cache Management Commands
# ============================================================================


@cli.command("clean-cache")
@click.option("--all", "clean_all", is_flag=True, help="Clean all caches")
@click.option("--db", "clean_db", is_flag=True, help="Clean database cache only")
@click.option("--disk", "clean_disk", is_flag=True, help="Clean disk cache only")
@click.option("--symbol", help="Clean cache for specific symbol")
@click.pass_context
def clean_cache(ctx, clean_all, clean_db, clean_disk, symbol):
    """Clean analysis caches"""
    from investigator.infrastructure.cache import get_cache_manager
    from investigator.infrastructure.cache.cache_types import CacheType
    from investigator.infrastructure.cache.file_cache_handler import FileCacheStorageHandler
    from investigator.infrastructure.cache.rdbms_cache_handler import RdbmsCacheStorageHandler

    cache_manager = get_cache_manager()

    try:
        if clean_all:
            click.echo("Cleaning all caches...")
            # Clear all cache types
            for cache_type in CacheType:
                cache_manager.clear(cache_type)
            click.echo("✅ All caches cleared")

        elif clean_db:
            if symbol:
                click.echo(f"Cleaning database cache for {symbol}...")
                deleted = 0
                for handlers in cache_manager.handlers.values():
                    for handler in handlers:
                        if isinstance(handler, RdbmsCacheStorageHandler):
                            try:
                                deleted += handler.delete_by_symbol(symbol)
                            except Exception as exc:
                                click.echo(f"❌ Error cleaning DB handler {handler}: {exc}", err=True)
                click.echo(f"✅ Database cache cleared for {symbol} (entries removed: {deleted})")
            else:
                click.echo("Cleaning database cache...")
                cache_manager.clear(CacheType.LLM_RESPONSE, storage_type="rdbms")
                cache_manager.clear(CacheType.COMPANY_FACTS, storage_type="rdbms")
                cache_manager.clear(CacheType.SEC_RESPONSE, storage_type="rdbms")
                click.echo("✅ Database cache cleared")

        elif clean_disk:
            if symbol:
                click.echo(f"Cleaning disk cache for {symbol}...")
                deleted = 0
                for handlers in cache_manager.handlers.values():
                    for handler in handlers:
                        if isinstance(handler, FileCacheStorageHandler):
                            try:
                                deleted += handler.delete_by_symbol(symbol)
                            except Exception as exc:
                                click.echo(f"❌ Error cleaning disk handler {handler}: {exc}", err=True)
                click.echo(f"✅ Disk cache cleared for {symbol} (entries removed: {deleted})")
            else:
                click.echo("Cleaning disk cache...")
                cache_manager.clear(CacheType.LLM_RESPONSE, storage_type="disk")
                cache_manager.clear(CacheType.TECHNICAL_DATA, storage_type="disk")
                cache_manager.clear(CacheType.SEC_RESPONSE, storage_type="disk")
                click.echo("✅ Disk cache cleared")

        elif symbol:
            click.echo(f"Cleaning cache for {symbol}...")
            cache_manager.delete_by_symbol(symbol)
            click.echo(f"✅ Cache cleared for {symbol}")

        else:
            click.echo("Cleaning default caches...")
            cache_manager.clear(CacheType.LLM_RESPONSE)
            click.echo("✅ LLM response cache cleared")

    except Exception as e:
        click.echo(f"❌ Error cleaning cache: {e}", err=True)
        sys.exit(1)


@cli.command("inspect-cache")
@click.option("--symbol", help="Inspect cache for specific symbol")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information")
@click.pass_context
def inspect_cache(ctx, symbol, verbose):
    """Inspect cache contents and statistics"""
    from investigator.infrastructure.cache import get_cache_manager
    from investigator.infrastructure.cache.cache_types import CacheType

    cache_manager = get_cache_manager()

    click.echo("Cache Inspection Report")
    click.echo("=" * 60)

    if symbol:
        click.echo(f"\nSymbol: {symbol}")
        for cache_type in CacheType:
            try:
                key = {"symbol": symbol}
                data = cache_manager.get(cache_type, key)
                if data:
                    click.echo(f"  ✅ {cache_type.value}: Cached")
                    if verbose:
                        click.echo(f"     Size: {len(str(data))} bytes")
                else:
                    click.echo(f"  ❌ {cache_type.value}: Not cached")
            except:
                pass
    else:
        click.echo("\nCache Statistics:")
        # Show overall cache stats
        stats = cache_manager.get_stats() if hasattr(cache_manager, "get_stats") else {}
        if stats:
            for key, value in stats.items():
                click.echo(f"  {key}: {value}")
        else:
            click.echo("  No statistics available")


@cli.command("cache-sizes")
@click.pass_context
def cache_sizes(ctx):
    """Show cache sizes by type"""
    import os
    from pathlib import Path

    click.echo("Cache Directory Sizes")
    click.echo("=" * 60)

    cache_dirs = {
        "SEC Cache": "data/sec_cache",
        "LLM Cache": "data/llm_cache",
        "Technical Cache": "data/technical_cache",
        "Vector DB": "data/vector_db",
    }

    total_size = 0
    for name, path in cache_dirs.items():
        if Path(path).exists():
            size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
            total_size += size
            size_mb = size / (1024 * 1024)
            click.echo(f"{name:20s}: {size_mb:10.2f} MB")
        else:
            click.echo(f"{name:20s}: Not found")

    click.echo("-" * 60)
    click.echo(f"{'Total':20s}: {total_size / (1024 * 1024):10.2f} MB")


# ============================================================================
# Testing Commands
# ============================================================================


@cli.command("test-system")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def test_system(ctx, verbose):
    """Run system health tests"""
    import subprocess

    click.echo("Running system health tests...")
    click.echo("=" * 60)

    tests = [
        ("Python version", ["python3", "--version"]),
        ("Ollama connection", ["curl", "-s", "http://localhost:11434/api/tags"]),
        (
            "Database connection",
            [
                "python3",
                "-c",
                "from investigator.infrastructure.database.db import get_engine; get_engine(); print('OK')",
            ],
        ),
        (
            "Cache system",
            [
                "python3",
                "-c",
                "from investigator.infrastructure.cache import get_cache_manager; get_cache_manager(); print('OK')",
            ],
        ),
    ]

    results = []
    for test_name, cmd in tests:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                results.append((test_name, "✅ PASS"))
                if verbose:
                    click.echo(f"\n{test_name}:")
                    click.echo(f"  {result.stdout.strip()}")
            else:
                results.append((test_name, "❌ FAIL"))
                if verbose:
                    click.echo(f"\n{test_name}: FAILED")
                    click.echo(f"  {result.stderr.strip()}")
        except Exception as e:
            results.append((test_name, f"❌ ERROR: {e}"))

    click.echo("\nTest Results:")
    click.echo("-" * 60)
    for test_name, status in results:
        click.echo(f"{test_name:30s}: {status}")

    passed = sum(1 for _, status in results if "✅" in status)
    total = len(results)
    click.echo(f"\nPassed: {passed}/{total}")

    if passed < total:
        sys.exit(1)


@cli.command("run-tests")
@click.option("--pattern", "-p", help="Test file pattern")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def run_tests(ctx, pattern, verbose):
    """Run pytest test suite"""
    import subprocess

    click.echo("Running test suite...")

    cmd = ["python3", "-m", "pytest", "tests/"]

    if pattern:
        cmd.append(f"-k {pattern}")

    if verbose:
        cmd.append("-v")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


@cli.command("cache-facts")
@click.option(
    "--symbols-file",
    type=click.Path(exists=True, dir_okay=False),
    default="sp100_extraction_results.json",
    show_default=True,
    help="JSON or newline-separated list of symbols to cache.",
)
@click.option(
    "--symbol",
    "symbol_list",
    multiple=True,
    help="Optional symbol override (can be passed multiple times).",
)
@click.option(
    "--parallel",
    default=5,
    show_default=True,
    help="Number of concurrent SEC fetches.",
)
@click.option(
    "--process-raw/--raw-only",
    default=False,
    show_default=True,
    help="Also trigger processed-table ingestion (default raw-only).",
)
@click.option(
    "--hydrate-from-db",
    is_flag=True,
    help="Write cached raw files from sec_companyfacts_raw without calling the SEC API.",
)
def cache_facts(symbols_file, symbol_list, parallel, process_raw, hydrate_from_db):
    """Fetch CompanyFacts for SP100 (or supplied) symbols without running full analysis."""
    import hashlib

    from investigator.config import get_config
    from investigator.infrastructure.database.db import get_db_manager

    cfg = get_config()
    cache_manager = CacheManager(cfg)
    event_bus = EventBus()
    sec_agent = SECAnalysisAgent(
        "sec_bulk_cache",
        ollama_client=None,
        event_bus=event_bus,
        cache_manager=cache_manager,
    )

    def _load_symbols() -> List[str]:
        if symbol_list:
            return sorted({sym.upper() for sym in symbol_list})
        path = Path(symbols_file)
        if path.suffix.lower() == ".txt":
            try:
                with open(path) as handle:
                    symbols = [
                        line.strip().upper() for line in handle if line.strip() and not line.lstrip().startswith("#")
                    ]
            except OSError as exc:
                raise click.ClickException(f"Unable to read {symbols_file}: {exc}") from exc
            return sorted(set(symbols))

        try:
            with open(path) as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Unable to parse {symbols_file}: {exc}") from exc
        except OSError as exc:
            raise click.ClickException(f"Unable to read {symbols_file}: {exc}") from exc

        if isinstance(payload, dict):
            symbols = list(payload.keys())
        elif isinstance(payload, list):
            symbols = [str(item) for item in payload]
        else:
            raise click.ClickException(
                f"{symbols_file} must contain either newline symbols, an object mapping, or an array of symbols."
            )
        return sorted({sym.upper() for sym in symbols})

    symbols = _load_symbols()
    if not symbols:
        click.echo("No symbols supplied. Nothing to do.")
        return

    if hydrate_from_db:
        manager = get_db_manager()
        written = 0
        missing = []
        from sqlalchemy import text

        with manager.engine.connect() as conn:
            for symbol in symbols:
                row = conn.execute(
                    text(
                        """
                        SELECT cik, companyfacts
                        FROM sec_companyfacts_raw
                        WHERE symbol = :symbol
                        ORDER BY fetched_at DESC
                        LIMIT 1
                        """
                    ),
                    {"symbol": symbol},
                ).fetchone()
                if not row:
                    missing.append(symbol)
                    continue
                companyfacts = row.companyfacts
                if isinstance(companyfacts, str):
                    companyfacts = json.loads(companyfacts)
                hash_suffix = hashlib.sha256(json.dumps(companyfacts, sort_keys=True).encode()).hexdigest()[:12]
                sec_agent._persist_raw_companyfacts(symbol, row.cik, companyfacts, hash_suffix)
                written += 1
                click.echo(f"🗃️  {symbol} raw cache hydrated from DB")

        click.echo(f"\nHydrated {written}/{len(symbols)} symbols from sec_companyfacts_raw")
        if missing:
            click.echo(f"Symbols missing in DB: {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")
        return

    click.echo(
        f"Fetching raw CompanyFacts for {len(symbols)} symbols " f"(parallel={parallel}, process_raw={process_raw})"
    )

    async def _runner():
        sem = asyncio.Semaphore(max(1, parallel))
        results = []

        async def _fetch(symbol: str):
            async with sem:
                try:
                    await sec_agent._fetch_and_cache_companyfacts(symbol, process_raw=process_raw)
                    results.append((symbol, True, "cached"))
                    click.echo(f"✅ {symbol} cached{' (processed)' if process_raw else ' (raw-only)'}")
                except Exception as exc:
                    results.append((symbol, False, str(exc)))
                    click.echo(f"❌ {symbol} failed: {exc}")

        await asyncio.gather(*(_fetch(sym) for sym in symbols))
        success = sum(1 for _, ok, _ in results if ok)
        click.echo(f"\nCompleted {success}/{len(results)} symbols.")
        failures = [(sym, err) for sym, ok, err in results if not ok]
        if failures:
            click.echo("\nFailures:")
            for sym, err in failures:
                click.echo(f"  - {sym}: {err}")

    asyncio.run(_runner())


# ============================================================================
# Setup Commands
# ============================================================================


@cli.command("setup-database")
@click.option("--recreate", is_flag=True, help="Drop and recreate tables")
@click.pass_context
def setup_database(ctx):
    """Initialize database schema"""
    from sqlalchemy import text

    from investigator.infrastructure.database.db import get_engine

    click.echo("Setting up database...")

    try:
        engine = get_engine()

        # Run schema file if exists
        schema_file = Path("schema/consolidated_schema.sql")
        if schema_file.exists():
            with open(schema_file) as f:
                schema_sql = f.read()

            with engine.connect() as conn:
                for statement in schema_sql.split(";"):
                    if statement.strip():
                        conn.execute(text(statement))
                conn.commit()

            click.echo("✅ Database schema initialized")
        else:
            click.echo("❌ Schema file not found")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Database setup failed: {e}", err=True)
        sys.exit(1)


@cli.command("setup-system")
@click.option("--skip-deps", is_flag=True, help="Skip dependency installation")
@click.pass_context
def setup_system(ctx):
    """Setup system dependencies and configuration"""
    import subprocess

    click.echo("Setting up InvestiGator system...")
    click.echo("=" * 60)

    steps = []

    # Check Python version
    click.echo("\n1. Checking Python version...")
    result = subprocess.run(["python3", "--version"], capture_output=True, text=True)
    click.echo(f"   {result.stdout.strip()}")
    steps.append(("Python", result.returncode == 0))

    # Install dependencies
    if not ctx.params.get("skip_deps"):
        click.echo("\n2. Installing dependencies...")
        result = subprocess.run(["pip", "install", "-r", "requirements.txt"], capture_output=True)
        if result.returncode == 0:
            click.echo("   ✅ Dependencies installed")
            steps.append(("Dependencies", True))
        else:
            click.echo("   ❌ Failed to install dependencies")
            steps.append(("Dependencies", False))

    # Create directories
    click.echo("\n3. Creating data directories...")
    dirs = ["data/sec_cache", "data/llm_cache", "data/technical_cache", "data/vector_db", "logs", "reports", "results"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    click.echo("   ✅ Directories created")
    steps.append(("Directories", True))

    # Summary
    click.echo("\n" + "=" * 60)
    click.echo("Setup Summary:")
    for step_name, success in steps:
        status = "✅" if success else "❌"
        click.echo(f"  {status} {step_name}")

    if all(success for _, success in steps):
        click.echo("\n✅ System setup complete!")
    else:
        click.echo("\n⚠️  Some steps failed")
        sys.exit(1)


@cli.command("system-stats")
@click.pass_context
def system_stats(ctx):
    """Show system statistics and information"""
    import platform
    from pathlib import Path

    import psutil

    click.echo("InvestiGator System Information")
    click.echo("=" * 60)

    # System info
    click.echo("\nSystem:")
    click.echo(f"  Platform: {platform.system()} {platform.release()}")
    click.echo(f"  Python: {platform.python_version()}")
    click.echo(f"  Architecture: {platform.machine()}")

    # Resource usage
    click.echo("\nResources:")
    click.echo(f"  CPU Cores: {psutil.cpu_count()}")
    click.echo(f"  CPU Usage: {psutil.cpu_percent()}%")
    memory = psutil.virtual_memory()
    click.echo(f"  Memory: {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent}%)")

    # Disk usage
    click.echo("\nDisk:")
    disk = psutil.disk_usage(".")
    click.echo(f"  Total: {disk.total / (1024**3):.1f}GB")
    click.echo(f"  Used: {disk.used / (1024**3):.1f}GB ({disk.percent}%)")
    click.echo(f"  Free: {disk.free / (1024**3):.1f}GB")

    # Ollama status
    click.echo("\nOllama:")
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            click.echo(f"  Status: ✅ Running")
            click.echo(f"  Models: {len(models)}")
            for model in models[:5]:  # Show first 5
                click.echo(f"    - {model['name']}")
        else:
            click.echo(f"  Status: ❌ Not responding")
    except:
        click.echo(f"  Status: ❌ Not available")

    # Database status
    click.echo("\nDatabase:")
    try:
        from investigator.infrastructure.database.db import get_engine

        engine = get_engine()
        click.echo(f"  Status: ✅ Connected")
        click.echo(f"  URL: {engine.url}")
    except Exception as e:
        click.echo(f"  Status: ❌ Not connected")


if __name__ == "__main__":
    cli()
