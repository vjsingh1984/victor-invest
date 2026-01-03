#!/usr/bin/env python3
"""
Enhanced JNJ Analysis with Detailed Market Regime Logging
Includes ASCII art, tables, and PDF report generation
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

# Configure detailed logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Rich console for beautiful output
console = Console()


class MarketRegimeVisualizer:
    """Generate visual representations of market regime"""

    @staticmethod
    def generate_ascii_art(regime: str) -> str:
        """Generate ASCII art for market regime"""
        if regime == "risk_on":
            return """
    📈 RISK-ON MARKET REGIME 📈
    ╔══════════════════════════╗
    ║    🚀 BULLISH MODE 🚀    ║
    ║  ┌─────────────────────┐ ║
    ║  │ ▲ Stocks > Bonds    │ ║
    ║  │ ▲ Small > Large Cap │ ║
    ║  │ ▲ Commodities Up    │ ║
    ║  │ ▼ Gold Declining    │ ║
    ║  │ ▼ VIX < 20         │ ║
    ║  └─────────────────────┘ ║
    ╚══════════════════════════╝
            """
        elif regime == "risk_off":
            return """
    📉 RISK-OFF MARKET REGIME 📉
    ╔══════════════════════════╗
    ║    🛡️ DEFENSIVE MODE 🛡️   ║
    ║  ┌─────────────────────┐ ║
    ║  │ ▲ Bonds > Stocks    │ ║
    ║  │ ▲ Gold Rally        │ ║
    ║  │ ▲ VIX > 30         │ ║
    ║  │ ▼ Oil Declining     │ ║
    ║  │ ▼ Small Caps Weak   │ ║
    ║  └─────────────────────┘ ║
    ╚══════════════════════════╝
            """
        else:
            return """
    ⚖️ MIXED MARKET REGIME ⚖️
    ╔══════════════════════════╗
    ║    🔄 NEUTRAL MODE 🔄    ║
    ║  ┌─────────────────────┐ ║
    ║  │ ~ Mixed Signals     │ ║
    ║  │ ~ No Clear Trend    │ ║
    ║  │ ~ Sector Rotation   │ ║
    ║  │ ~ Await Catalyst    │ ║
    ║  └─────────────────────┘ ║
    ╚══════════════════════════╝
            """

    @staticmethod
    def create_market_table(market_data: Dict) -> Table:
        """Create rich table for market performance"""
        table = Table(title="📊 Market Performance Analysis", show_header=True, header_style="bold magenta")

        table.add_column("Asset Class", style="cyan", width=20)
        table.add_column("ETF", style="yellow", width=8)
        table.add_column("1M Return", justify="right", style="green")
        table.add_column("Volatility", justify="right", style="blue")
        table.add_column("Signal", justify="center", width=15)

        # Add rows based on market data
        medium_term = market_data.get("medium_term", {})

        # Equities
        spy_data = medium_term.get("broad_market", {})
        if spy_data:
            signal = (
                "🟢 Risk-On"
                if spy_data.get("return", 0) > 0.03
                else "🔴 Risk-Off" if spy_data.get("return", 0) < -0.03 else "⚪ Neutral"
            )
            table.add_row(
                "S&P 500", "SPY", f"{spy_data.get('return', 0):.2%}", f"{spy_data.get('volatility', 0):.2%}", signal
            )

        # Bonds
        bond_data = medium_term.get("bonds", {})
        if bond_data:
            signal = "🔴 Risk-Off" if bond_data.get("return", 0) > spy_data.get("return", 0) else "🟢 Risk-On"
            table.add_row(
                "Aggregate Bonds",
                "AGG",
                f"{bond_data.get('return', 0):.2%}",
                f"{bond_data.get('volatility', 0):.2%}",
                signal,
            )

        # Gold
        gold_data = medium_term.get("gold", {})
        if gold_data:
            signal = (
                "🔴 Safe Haven"
                if gold_data.get("return", 0) > 0.05
                else "🟢 Risk-On" if gold_data.get("return", 0) < -0.02 else "⚪ Neutral"
            )
            table.add_row(
                "Gold", "GLD", f"{gold_data.get('return', 0):.2%}", f"{gold_data.get('volatility', 0):.2%}", signal
            )

        # Oil
        oil_data = medium_term.get("oil", {})
        if oil_data:
            signal = (
                "🟢 Growth"
                if oil_data.get("return", 0) > 0.03
                else "🔴 Slowdown" if oil_data.get("return", 0) < -0.05 else "⚪ Neutral"
            )
            table.add_row(
                "Crude Oil", "USO", f"{oil_data.get('return', 0):.2%}", f"{oil_data.get('volatility', 0):.2%}", signal
            )

        # Commodities
        commodity_data = medium_term.get("commodities", {})
        if commodity_data:
            signal = (
                "🟢 Inflation"
                if commodity_data.get("return", 0) > 0.03
                else "🔴 Deflation" if commodity_data.get("return", 0) < -0.05 else "⚪ Neutral"
            )
            table.add_row(
                "Commodities",
                "DBC",
                f"{commodity_data.get('return', 0):.2%}",
                f"{commodity_data.get('volatility', 0):.2%}",
                signal,
            )

        return table


async def run_jnj_analysis_with_enhanced_logging():
    """Run comprehensive JNJ analysis with detailed market regime logging"""

    console.print("\n[bold cyan]=" * 80)
    console.print("[bold yellow]🚀 ENHANCED JNJ ANALYSIS WITH MARKET REGIME INTELLIGENCE[/bold yellow]")
    console.print("[bold cyan]=" * 80)
    console.print(f"[dim]📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    console.print(f"[bold green]🎯 Target: Johnson & Johnson (JNJ) - Healthcare Sector[/bold green]")
    console.print("[bold cyan]=" * 80 + "\n")

    try:
        # Import orchestrator components
        console.print("[yellow]📦 Loading agentic AI framework...[/yellow]")
        from agents.orchestrator import AgentOrchestrator, AnalysisMode, Priority
        from utils.cache.cache_manager import CacheManager
        from utils.monitoring import MetricsCollector
        from utils.report_generator import PDFReportGenerator, ReportConfig
        from pathlib import Path

        # Initialize components
        cache_manager = CacheManager()
        metrics_collector = MetricsCollector()

        # Create reports directory
        reports_dir = Path("reports") / datetime.now().strftime("%Y%m%d")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize PDF report generator
        pdf_generator = PDFReportGenerator(
            reports_dir,
            ReportConfig(
                title="JNJ Investment Analysis Report",
                subtitle="AI-Powered Market Context Analysis",
                include_charts=True,
                include_disclaimer=True,
            ),
        )

        # Create orchestrator
        orchestrator = AgentOrchestrator(
            cache_manager=cache_manager,
            metrics_collector=metrics_collector,
            max_concurrent_analyses=2,
            max_concurrent_agents=4,
        )

        console.print("[green]✅ Framework initialized successfully[/green]\n")

        # Start orchestrator
        await orchestrator.start()

        # Queue comprehensive analysis
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:

            task = progress.add_task("[cyan]Initiating comprehensive analysis...", total=None)

            task_id = await orchestrator.analyze(symbol="JNJ", mode=AnalysisMode.COMPREHENSIVE, priority=Priority.HIGH)

            progress.update(task, description=f"[green]Analysis queued: {task_id}")

        console.print(f"\n[bold green]✅ Analysis task initiated: {task_id}[/bold green]\n")

        # Monitor progress with detailed logging
        console.print("[yellow]📊 Monitoring analysis progress...[/yellow]\n")

        max_wait = 600  # 10 minutes
        wait_time = 0
        last_status = None

        with Progress(console=console) as progress:
            analysis_task = progress.add_task("[cyan]Analysis in progress...", total=100)

            while wait_time < max_wait:
                status = await orchestrator.get_task_status(task_id)

                if status != last_status:
                    last_status = status

                    if status["status"] == "processing":
                        agents_done = status.get("agents_completed", 0)
                        total_agents = status.get("total_agents", 5)
                        percent = (agents_done / max(total_agents, 1)) * 100

                        progress.update(analysis_task, completed=percent)
                        console.print(f"[dim]⏳ Progress: {agents_done}/{total_agents} agents completed[/dim]")

                if status["status"] == "completed":
                    progress.update(analysis_task, completed=100)
                    console.print("[bold green]✅ Analysis completed![/bold green]")
                    break
                elif status["status"] == "failed":
                    console.print(f"[bold red]❌ Analysis failed: {status}[/bold red]")
                    return False

                await asyncio.sleep(5)
                wait_time += 5

        # Get comprehensive results
        console.print("\n[yellow]📥 Retrieving analysis results...[/yellow]\n")
        results = await orchestrator.get_analysis_results(task_id)

        if not results:
            console.print("[bold red]❌ No results returned[/bold red]")
            return False

        # Extract and display market context with enhanced visualization
        agents_data = results.get("agents", {})

        if "market_context" in agents_data:
            market_data = agents_data["market_context"]

            console.print("[bold magenta]" + "=" * 80)
            console.print("🏦 MARKET CONTEXT ANALYSIS")
            console.print("=" * 80 + "[/bold magenta]\n")

            # Display market regime ASCII art
            market_regime = market_data.get("market_regime", "neutral")
            console.print(MarketRegimeVisualizer.generate_ascii_art(market_regime))

            # Display market performance table
            market_context = market_data.get("market_context", {})
            if market_context:
                table = MarketRegimeVisualizer.create_market_table(market_context)
                console.print(table)

            # Display sector analysis
            sector_context = market_data.get("sector_context", {})
            if sector_context:
                console.print("\n[bold cyan]🏥 Healthcare Sector Analysis:[/bold cyan]")
                panel_content = f"""
Primary ETF: [yellow]{sector_context.get('primary_etf', 'XLV')}[/yellow]
Sector Strength: [{'green' if sector_context.get('sector_strength') == 'strong' else 'yellow' if sector_context.get('sector_strength') == 'moderate' else 'red'}]{sector_context.get('sector_strength', 'N/A')}[/]
Description: {sector_context.get('description', 'Healthcare Select Sector SPDR Fund')}
                """
                console.print(Panel(panel_content, title="Sector Context", border_style="cyan"))

            # Display market sentiment analysis
            market_sentiment = market_data.get("market_sentiment", {})
            if market_sentiment:
                console.print("\n[bold yellow]🎯 Market Sentiment Analysis:[/bold yellow]")

                sentiment_table = Table(show_header=True, header_style="bold yellow")
                sentiment_table.add_column("Metric", style="cyan")
                sentiment_table.add_column("Value", style="green")

                sentiment_table.add_row("Overall Sentiment", market_sentiment.get("sentiment", "neutral"))
                sentiment_table.add_row("Market Regime", market_sentiment.get("market_regime", "neutral"))
                sentiment_table.add_row("Sector Rotation", market_sentiment.get("sector_rotation", "N/A"))

                if "key_drivers" in market_sentiment:
                    drivers = market_sentiment["key_drivers"]
                    if isinstance(drivers, list):
                        sentiment_table.add_row("Key Drivers", "\n".join(f"• {d}" for d in drivers[:3]))

                console.print(sentiment_table)

            # Display relative performance
            relative_perf = market_data.get("relative_performance", {})
            if relative_perf:
                console.print("\n[bold green]📈 JNJ Relative Performance:[/bold green]")

                perf_table = Table(show_header=True, header_style="bold green")
                perf_table.add_column("Comparison", style="cyan")
                perf_table.add_column("1 Month", justify="right")
                perf_table.add_column("Beta", justify="right")

                vs_market = relative_perf.get("vs_market", {}).get("medium_term", {})
                if vs_market:
                    perf_table.add_row(
                        "vs S&P 500", f"{vs_market.get('relative_return', 0):.2%}", f"{vs_market.get('beta', 1.0):.2f}"
                    )

                vs_sector = relative_perf.get("vs_sector", {}).get("medium_term", {})
                if vs_sector:
                    perf_table.add_row(
                        "vs Healthcare",
                        f"{vs_sector.get('relative_return', 0):.2%}",
                        f"{vs_sector.get('relative_strength', 1.0):.2f}",
                    )

                console.print(perf_table)

        # Display synthesis results
        if "synthesis" in agents_data:
            synthesis_data = agents_data["synthesis"]

            console.print("\n[bold magenta]" + "=" * 80)
            console.print("💰 INVESTMENT SYNTHESIS")
            console.print("=" * 80 + "[/bold magenta]\n")

            # Investment recommendation
            investment_appeal = synthesis_data.get("investment_appeal", "N/A")
            recommendation = synthesis_data.get("recommendation", {})

            rec_color = (
                "green"
                if "BUY" in investment_appeal.upper()
                else "yellow" if "HOLD" in investment_appeal.upper() else "red"
            )
            console.print(f"[bold {rec_color}]📊 Investment Appeal: {investment_appeal}[/bold {rec_color}]")

            # Price targets
            price_targets = synthesis_data.get("price_targets", {})
            if price_targets:
                current_price = price_targets.get("current_price", 0)
                target_price = price_targets.get("adjusted_target", 0)
                upside = ((target_price - current_price) / current_price * 100) if current_price else 0

                price_panel = f"""
Current Price: [yellow]${current_price:.2f}[/yellow]
Target Price: [green]${target_price:.2f}[/green]
Upside Potential: [{'green' if upside > 0 else 'red'}]{upside:+.1f}%[/]
Market Regime Adjustment: {price_targets.get('market_adjustment', 'Applied')}
                """
                console.print(Panel(price_panel, title="💵 Price Analysis", border_style="green"))

            # Risk assessment
            risk_assessment = synthesis_data.get("risk_assessment", {})
            if risk_assessment:
                risk_level = risk_assessment.get("overall_risk_level", "medium")
                risk_color = "green" if risk_level == "low" else "yellow" if risk_level == "medium" else "red"

                console.print(f"\n[bold]⚠️ Risk Level: [{risk_color}]{risk_level.upper()}[/{risk_color}][/bold]")

                if "key_risks" in risk_assessment:
                    console.print("[dim]Key Risks:[/dim]")
                    for risk in risk_assessment["key_risks"][:3]:
                        console.print(f"  [dim]• {risk}[/dim]")

        # Generate comprehensive PDF report
        console.print("\n[yellow]📄 Generating PDF report...[/yellow]")

        # Prepare report data
        report_data = [
            {
                "symbol": "JNJ",
                "company_name": "Johnson & Johnson",
                "sector": "Healthcare",
                "current_price": synthesis_data.get("price_targets", {}).get("current_price", 0),
                "target_price": synthesis_data.get("price_targets", {}).get("adjusted_target", 0),
                "recommendation": synthesis_data.get("investment_appeal", "Hold"),
                "confidence_score": synthesis_data.get("confidence_score", 0),
                "market_regime": market_data.get("market_regime", "neutral"),
                "sector_strength": market_data.get("sector_context", {}).get("sector_strength", "neutral"),
                "analysis_date": datetime.now().isoformat(),
                "risk_level": synthesis_data.get("risk_assessment", {}).get("overall_risk_level", "medium"),
                "key_insights": synthesis_data.get("key_insights", []),
                "technical_signals": agents_data.get("technical", {}).get("signals", {}),
                "fundamental_metrics": agents_data.get("fundamental", {}).get("metrics", {}),
                "market_context": market_data,
            }
        ]

        try:
            report_path = pdf_generator.generate_report(
                recommendations=report_data, report_type="comprehensive", include_charts=True
            )
            console.print(f"[green]✅ PDF report generated: {report_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ PDF generation failed (ReportLab may not be installed): {e}[/yellow]")

        # Save detailed JSON results
        json_path = reports_dir / f"jnj_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"[green]✅ Detailed results saved: {json_path}[/green]")

        # Stop orchestrator
        await orchestrator.stop()

        console.print("\n[bold green]" + "=" * 80)
        console.print("🎉 ANALYSIS COMPLETE!")
        console.print("=" * 80 + "[/bold green]")

        # Final summary
        summary = f"""
[bold cyan]📊 Analysis Summary:[/bold cyan]
• Symbol: JNJ (Johnson & Johnson)
• Sector: Healthcare
• Market Regime: {market_data.get('market_regime', 'neutral').upper()}
• Sector Strength: {market_data.get('sector_context', {}).get('sector_strength', 'neutral').upper()}
• Investment Appeal: {synthesis_data.get('investment_appeal', 'N/A')}
• Upside Potential: {upside:+.1f}%
• Risk Level: {synthesis_data.get('risk_assessment', {}).get('overall_risk_level', 'medium').upper()}
        """
        console.print(Panel(summary, title="Final Summary", border_style="green"))

        return True

    except Exception as e:
        console.print(f"[bold red]💥 Analysis failed: {e}[/bold red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return False


async def main():
    """Main entry point"""
    success = await run_jnj_analysis_with_enhanced_logging()
    return 0 if success else 1


if __name__ == "__main__":
    # Install rich if not available
    try:
        from rich import console
    except ImportError:
        print("Installing rich for enhanced output...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])

    exit_code = asyncio.run(main())
    exit(exit_code)
