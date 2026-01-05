"""
Stock analysis commands for InvestiGator CLI
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import yaml

from ..utils import MutuallyExclusiveOption, async_command, validate_symbols


@click.group()
@click.pass_context
def analyze(ctx):
    """Stock analysis commands

    Run fundamental, technical, and synthesis analysis on stocks.

    Examples:
        investigator analyze single AAPL
        investigator analyze single AAPL --mode comprehensive --report
        investigator analyze batch AAPL MSFT GOOGL --output-dir results/
        investigator analyze compare AAPL MSFT GOOGL
    """
    pass


@analyze.command("single")
@click.argument("symbol")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["quick", "standard", "comprehensive"]),
    default="comprehensive",
    help="Analysis depth: quick (technical only), standard (tech+fund), comprehensive (full synthesis)",
)
@click.option("--output", "-o", type=click.Path(), help="Output file path for results")
@click.option("--format", "-f", type=click.Choice(["json", "yaml", "text"]), default="json", help="Output format")
@click.option(
    "--detail",
    "-d",
    type=click.Choice(["minimal", "standard", "verbose"]),
    default="standard",
    help="Output detail level",
)
@click.option("--report", is_flag=True, help="Generate PDF investment report")
@click.option("--force-refresh", "--refresh", is_flag=True, help="Bypass cache and fetch fresh data")
@click.pass_context
def single(ctx, symbol, mode, output, format, detail, report, force_refresh):
    """Analyze a single stock symbol

    Runs fundamental, technical, and optionally synthesis analysis
    depending on the mode selected.

    Examples:
        investigator analyze single AAPL
        investigator analyze single MSFT --mode quick
        investigator analyze single TSLA --mode comprehensive --report
    """
    config = ctx.obj.get("config", {})
    symbol = symbol.upper()

    async def run_analysis():
        from investigator.application import AgentOrchestrator, AnalysisMode, OutputDetailLevel, format_analysis_output
        from investigator.config import get_config
        from investigator.infrastructure.cache import get_cache_manager
        from investigator.infrastructure.monitoring import MetricsCollector

        cfg = get_config()
        original_force_refresh = getattr(cfg.cache_control, "force_refresh", False)
        original_force_symbols = getattr(cfg.cache_control, "force_refresh_symbols", None)

        if force_refresh:
            cfg.cache_control.force_refresh = True
            cfg.cache_control.force_refresh_symbols = [symbol]
            click.echo(f"Force refresh enabled for {symbol}")

        cache_manager = get_cache_manager()
        cache_manager.config = cfg
        metrics_collector = MetricsCollector()
        orchestrator = AgentOrchestrator(cache_manager, metrics_collector)

        try:
            await metrics_collector.start()
            await orchestrator.start()

            analysis_mode = AnalysisMode[mode.upper()]
            task_id = await orchestrator.analyze(symbol, analysis_mode)

            click.echo(f"Analysis started for {symbol} (Task ID: {task_id})")

            # Wait for results
            with click.progressbar(length=100, label="Analyzing") as bar:
                elapsed = 0
                while elapsed < 900:
                    status = await orchestrator.get_status(task_id)

                    if status["status"] == "completed":
                        bar.update(100 - bar.pos)
                        break
                    elif status["status"] == "processing":
                        progress = (status.get("agents_completed", 0) / status.get("total_agents", 1)) * 100
                        bar.update(progress - bar.pos)

                    await asyncio.sleep(2)
                    elapsed += 2

            results = await orchestrator.get_results(task_id)

            if results:
                detail_level_enum = OutputDetailLevel(detail)
                formatted = format_analysis_output(results, detail_level_enum)

                # Show executive summary
                exec_summary = format_analysis_output(results, OutputDetailLevel.MINIMAL)
                _print_executive_summary(exec_summary)

                # Format output
                if format == "json":
                    output_data = json.dumps(formatted, indent=2, default=str)
                elif format == "yaml":
                    output_data = yaml.dump(formatted, default_flow_style=False)
                else:
                    output_data = _format_text(formatted)

                # Save or print
                if output:
                    output_path = Path(output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output, "w") as f:
                        f.write(output_data)
                    click.echo(f"Results saved to {output}")
                else:
                    click.echo("\n[Full Analysis]")
                    click.echo(output_data)

                # Generate PDF report if requested
                if report:
                    try:
                        from investigator.application import InvestmentSynthesizer
                        from investigator.domain.models import InvestmentRecommendation

                        click.echo("\nGenerating PDF report...")
                        recommendation = _convert_to_recommendation(results, symbol)
                        synthesizer = InvestmentSynthesizer()
                        report_path = synthesizer.generate_report([recommendation], report_type="synthesis")
                        click.echo(f"PDF report generated: {report_path}")
                    except Exception as e:
                        click.echo(f"Failed to generate PDF: {e}", err=True)
            else:
                click.echo("Analysis timed out or failed", err=True)
                sys.exit(1)

        finally:
            await orchestrator.stop()
            await metrics_collector.stop()
            cfg.cache_control.force_refresh = original_force_refresh
            cfg.cache_control.force_refresh_symbols = original_force_symbols

    asyncio.run(run_analysis())


@analyze.command("batch")
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["quick", "standard", "comprehensive"]),
    default="standard",
    help="Analysis mode for all symbols",
)
@click.option("--output-dir", "-o", default="results", help="Output directory for results")
@click.option(
    "--detail",
    "-d",
    type=click.Choice(["minimal", "standard", "verbose"]),
    default="standard",
    help="Output detail level",
)
@click.option("--force-refresh", "--refresh", is_flag=True, help="Bypass cache")
@click.option("--parallel", "-p", default=3, type=int, help="Number of parallel analyses")
@click.pass_context
def batch(ctx, symbols, mode, output_dir, detail, force_refresh, parallel):
    """Analyze multiple symbols in batch

    Runs analysis on multiple symbols concurrently.

    Examples:
        investigator analyze batch AAPL MSFT GOOGL
        investigator analyze batch $(cat symbols.txt) --parallel 5
    """
    config = ctx.obj.get("config", {})
    symbols = [s.upper() for s in symbols]

    async def run_batch():
        from investigator.application import AgentOrchestrator, AnalysisMode, OutputDetailLevel, format_analysis_output
        from investigator.config import get_config
        from investigator.infrastructure.cache import get_cache_manager
        from investigator.infrastructure.monitoring import MetricsCollector

        cfg = get_config()

        if force_refresh:
            cfg.cache_control.force_refresh = True
            cfg.cache_control.force_refresh_symbols = list(symbols)

        cache_manager = get_cache_manager()
        cache_manager.config = cfg
        metrics_collector = MetricsCollector()
        orchestrator = AgentOrchestrator(cache_manager, metrics_collector)

        try:
            await metrics_collector.start()
            await orchestrator.start()

            analysis_mode = AnalysisMode[mode.upper()]
            task_ids = await orchestrator.analyze_batch(list(symbols), analysis_mode)

            click.echo(f"Batch analysis started for {len(symbols)} symbols")

            Path(output_dir).mkdir(parents=True, exist_ok=True)

            results = {}
            detail_level_enum = OutputDetailLevel(detail)

            with click.progressbar(symbols, label="Processing") as bar:
                for symbol, task_id in zip(bar, task_ids):
                    result = await orchestrator.get_results(task_id, wait=True, timeout=300)
                    if result:
                        results[symbol] = result
                        formatted = format_analysis_output(result, detail_level_enum)
                        output_file = Path(output_dir) / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(output_file, "w") as f:
                            json.dump(formatted, f, indent=2, default=str)

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
            await orchestrator.stop()
            await metrics_collector.stop()

    asyncio.run(run_batch())


@analyze.command("compare")
@click.argument("target")
@click.argument("peers", nargs=-1, required=True)
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def compare(ctx, target, peers, output):
    """Compare target symbol with peer companies

    Runs comparative analysis across multiple companies in the same sector.

    Examples:
        investigator analyze compare AAPL MSFT GOOGL AMZN
    """
    config = ctx.obj.get("config", {})
    target = target.upper()
    peers = [p.upper() for p in peers]

    async def run_comparison():
        from investigator.application import AgentOrchestrator
        from investigator.infrastructure.cache import get_cache_manager
        from investigator.infrastructure.monitoring import MetricsCollector

        cache_manager = get_cache_manager()
        metrics_collector = MetricsCollector()
        orchestrator = AgentOrchestrator(cache_manager, metrics_collector)

        try:
            await metrics_collector.start()
            await orchestrator.start()

            task_id = await orchestrator.analyze_peer_group(target, list(peers))

            click.echo(f"Peer comparison started: {target} vs {', '.join(peers)}")
            click.echo("This may take several minutes...")

            results = await orchestrator.get_results(task_id, wait=True, timeout=600)

            if results:
                report = _format_peer_comparison(results)

                if output:
                    with open(output, "w") as f:
                        json.dump(results, f, indent=2, default=str)
                    click.echo(f"Comparison saved to {output}")

                click.echo("\n" + report)
            else:
                click.echo("Comparison timed out or failed", err=True)
                sys.exit(1)

        finally:
            await orchestrator.stop()
            await metrics_collector.stop()

    asyncio.run(run_comparison())


# Helper functions


def _print_executive_summary(summary: dict):
    """Print formatted executive summary"""
    rec = summary.get("recommendation", {})
    val = summary.get("valuation", {})
    thesis = summary.get("thesis", {})
    dq = summary.get("data_quality", {})

    click.echo("\n" + "=" * 60)
    click.echo("EXECUTIVE SUMMARY")
    click.echo("=" * 60)
    click.echo(f"Symbol: {summary.get('symbol')}")
    click.echo(f"Recommendation: {rec.get('action', 'N/A')}")
    click.echo(f"Confidence: {rec.get('confidence', 'N/A')}")

    curr_price = val.get("current_price", "N/A")
    target = val.get("price_target_12m", "N/A")
    exp_ret = val.get("expected_return_pct", "N/A")
    exp_ret_str = f"{exp_ret:.1f}%" if exp_ret != "N/A" and exp_ret is not None else "N/A"
    click.echo(f"Price: ${curr_price} -> Target: ${target} ({exp_ret_str})")

    click.echo(f"Investment Grade: {val.get('investment_grade', 'N/A')}")

    dq_score = dq.get("overall_score", "N/A")
    dq_assess = dq.get("assessment", "N/A")
    if dq_score != "N/A" and dq_score is not None:
        click.echo(f"Data Quality: {dq_assess} ({dq_score:.1f}%)")
    else:
        click.echo(f"Data Quality: {dq_assess}")

    click.echo(f"\nKey Strengths:")
    for strength in thesis.get("key_strengths", []):
        click.echo(f"  * {strength}")

    click.echo(f"\nKey Risks:")
    for risk in thesis.get("key_risks", []):
        click.echo(f"  * {risk}")

    click.echo("=" * 60 + "\n")


def _format_text(results: dict) -> str:
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

    return "\n".join(output)


def _format_peer_comparison(results: dict) -> str:
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


def _convert_to_recommendation(results: dict, symbol: str):
    """Convert analysis results to InvestmentRecommendation

    Maps the synthesis agent output structure to InvestmentRecommendation model.

    Synthesis agent returns:
    {
        "status": "success",
        "symbol": symbol,
        "synthesis": report,  # Contains composite_scores, investment_thesis, etc.
        "recommendation": {...},  # LLM recommendation response
        "confidence": float,
        "risk_score": float,
        ...
    }
    """
    from investigator.domain.models import InvestmentRecommendation

    # Get synthesis agent results
    synthesis_data = results.get("agents", {}).get("synthesis", {})

    # The report is stored in 'synthesis' key
    report = synthesis_data.get("synthesis", {})

    # Helper to unwrap LLM responses (they're wrapped in {"response": {...}})
    def unwrap(data):
        if isinstance(data, dict) and "response" in data:
            return data.get("response", {})
        return data if isinstance(data, dict) else {}

    # Extract composite scores from report
    composite_scores = report.get("composite_scores", {})
    component_scores = composite_scores.get("component_scores", {})

    # Get recommendation from synthesis_data (direct) or report (nested)
    recommendation_data = unwrap(synthesis_data.get("recommendation", {}))
    if not recommendation_data.get("final_recommendation"):
        recommendation_data = unwrap(report.get("recommendation", {}))

    # Get investment thesis
    investment_thesis = unwrap(report.get("investment_thesis", {}))
    thesis_text = investment_thesis.get("core_investment_narrative", "")
    if not thesis_text:
        thesis_text = investment_thesis.get("executive_summary", "Analysis completed")

    # Get key insights
    key_insights = unwrap(report.get("key_insights", {}))
    insights_list = []
    for category in ["fundamental", "technical", "sec", "quantitative"]:
        category_insights = key_insights.get(category, [])
        if isinstance(category_insights, list):
            insights_list.extend(category_insights[:3])  # Top 3 from each category

    # Get risk assessment
    risk_assessment = unwrap(report.get("risk_assessment", {}))

    # Get action plan
    action_plan = unwrap(synthesis_data.get("action_plan", {}))

    # Extract technical data for current price
    technical_data = results.get("agents", {}).get("technical", {})
    current_price = (
        technical_data.get("current_price")
        or technical_data.get("analysis", {}).get("current_price")
        or technical_data.get("price_data", {}).get("current_price")
    )

    # Map recommendation string to uppercase
    rec_string = recommendation_data.get("final_recommendation", "hold")
    rec_map = {"strong_buy": "STRONG BUY", "buy": "BUY", "hold": "HOLD", "sell": "SELL", "strong_sell": "STRONG SELL"}
    recommendation = rec_map.get(rec_string.lower().replace(" ", "_"), "HOLD") if rec_string else "HOLD"

    # Map conviction to confidence
    conviction = recommendation_data.get("conviction_level", "medium")
    confidence_map = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
    confidence = confidence_map.get(conviction.lower(), "MEDIUM") if conviction else "MEDIUM"

    # Map time horizon
    time_horizon_raw = recommendation_data.get("time_horizon", "medium term")
    if "short" in str(time_horizon_raw).lower():
        time_horizon = "SHORT-TERM"
    elif "long" in str(time_horizon_raw).lower():
        time_horizon = "LONG-TERM"
    else:
        time_horizon = "MEDIUM-TERM"

    # Map position sizing
    position_pct = recommendation_data.get("position_sizing_suggestion", 3)
    if isinstance(position_pct, (int, float)):
        if position_pct >= 5:
            position_size = "LARGE"
        elif position_pct >= 2:
            position_size = "MODERATE"
        else:
            position_size = "SMALL"
    else:
        position_size = "MODERATE"

    # Get key catalysts from thesis or action plan
    key_catalysts = investment_thesis.get("growth_catalysts", []) or action_plan.get("catalysts", [])
    if not key_catalysts:
        key_catalysts = recommendation_data.get("key_reasons_for_recommendation", [])

    # Get key risks
    key_risks = recommendation_data.get("main_risks_to_monitor", [])
    if not key_risks:
        key_risks = investment_thesis.get("bear_case_considerations", [])

    # Get multi-year trends for enhanced analysis
    multi_year_trends = report.get("multi_year_trends", {})

    # Get scenarios for price targets
    scenarios = unwrap(report.get("scenarios", {}))
    base_case = scenarios.get("base_case", {})
    price_target = base_case.get("price_target")

    # Get entry/exit from action plan
    entry_strategy = action_plan.get("entry_strategy", "Market order on pullback to support")
    exit_strategy = action_plan.get("exit_strategy", "Scale out at price targets")
    if isinstance(exit_strategy, list):
        exit_strategy = "; ".join(exit_strategy[:2])

    # Get stop loss from action plan or scenarios
    stop_loss = action_plan.get("stop_loss") or scenarios.get("bear_case", {}).get("price_target")

    return InvestmentRecommendation(
        symbol=symbol,
        overall_score=composite_scores.get("overall_score", 50.0),
        fundamental_score=component_scores.get("fundamental", 50.0),
        technical_score=component_scores.get("technical", 50.0),
        income_score=component_scores.get("fundamental", 50.0),  # Use fundamental as proxy
        cashflow_score=component_scores.get("fundamental", 50.0),
        balance_score=component_scores.get("sec", 50.0),
        growth_score=component_scores.get("fundamental", 50.0),
        value_score=component_scores.get("fundamental", 50.0),
        business_quality_score=component_scores.get("sec", 50.0),
        recommendation=recommendation,
        confidence=confidence,
        price_target=price_target,
        current_price=current_price,
        investment_thesis=thesis_text,
        time_horizon=time_horizon,
        position_size=position_size,
        key_catalysts=key_catalysts if isinstance(key_catalysts, list) else [],
        key_risks=key_risks if isinstance(key_risks, list) else [],
        key_insights=insights_list[:10],  # Limit to 10 insights
        entry_strategy=entry_strategy if isinstance(entry_strategy, str) else "Market order",
        exit_strategy=exit_strategy if isinstance(exit_strategy, str) else "Target-based exit",
        stop_loss=stop_loss,
        analysis_timestamp=datetime.now(),
        data_quality_score=composite_scores.get("confidence", 80.0) / 100.0,
        analysis_thinking=investment_thesis.get("key_value_drivers"),
        synthesis_details=json.dumps(report, default=str),
        multi_year_trends=multi_year_trends if multi_year_trends else None,
        risk_scores=risk_assessment if risk_assessment else None,
    )
