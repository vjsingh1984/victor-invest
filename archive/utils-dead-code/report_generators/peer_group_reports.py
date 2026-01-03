#!/usr/bin/env python3
"""
Generate synthesis PDF reports for all peer group symbols
Creates comprehensive investment reports for each analyzed symbol
"""

import json
import time
import os
import sys
from datetime import datetime
import logging
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthesizer import InvestmentSynthesizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_peer_group_results() -> Dict[str, Any]:
    """Load the latest peer group analysis results"""
    # Find the most recent peer group analysis file
    reports_dir = "reports/peer_group"
    files = [f for f in os.listdir(reports_dir) if f.startswith("major_peer_groups_fast_") and f.endswith(".json")]

    if not files:
        raise FileNotFoundError("No peer group analysis results found")

    # Get the most recent file
    latest_file = sorted(files)[-1]
    file_path = os.path.join(reports_dir, latest_file)

    logger.info(f"Loading peer group results from: {file_path}")

    with open(file_path, "r") as f:
        return json.load(f)


def generate_reports_for_peer_group(
    group_data: Dict[str, Any], synthesizer: InvestmentSynthesizer
) -> Dict[str, Dict[str, Any]]:
    """Generate PDF reports for all symbols in a peer group"""
    sector = group_data["sector"]
    industry = group_data["industry"]
    symbols = group_data["symbols"]
    results = group_data["results"]

    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATING REPORTS: {sector.upper()} - {industry.upper()}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"{'='*80}")

    group_results = {}

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Generating report for {symbol}...")
        start_time = time.time()

        try:
            # Check if synthesis was successful
            if results[symbol]["status"] != "success":
                logger.warning(f"   ⚠️ Skipping {symbol} - synthesis failed")
                group_results[symbol] = {"status": "skipped", "reason": "synthesis_failed", "duration": 0}
                continue

            # Get the recommendation for this symbol (from cache)
            recommendation = synthesizer.synthesize_analysis(symbol)

            # Generate the PDF report
            report_path = synthesizer.generate_report([recommendation])

            duration = time.time() - start_time

            if report_path and os.path.exists(report_path):
                logger.info(f"   ✅ Report generated: {os.path.basename(report_path)} ({duration:.1f}s)")
                group_results[symbol] = {
                    "status": "success",
                    "report_path": report_path,
                    "duration": duration,
                    "peer_count": results[symbol]["peer_count"],
                    "score": results[symbol]["recommendation"]["overall_score"],
                }
            else:
                logger.error(f"   ❌ Report generation failed for {symbol}")
                group_results[symbol] = {"status": "failed", "reason": "report_generation_failed", "duration": duration}

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"   ❌ Error generating report for {symbol}: {e}")
            group_results[symbol] = {"status": "error", "reason": str(e), "duration": duration}

        # Brief pause between reports
        time.sleep(1)

    # Group summary
    successful = sum(1 for r in group_results.values() if r["status"] == "success")
    total_time = sum(r["duration"] for r in group_results.values())

    logger.info(f"\n📊 GROUP REPORT SUMMARY: {sector}/{industry}")
    logger.info(f"   Reports Generated: {successful}/{len(symbols)}")
    logger.info(f"   Group Time: {total_time:.1f}s")

    return group_results


def create_summary_report(all_results: List[Dict[str, Any]], report_results: List[Dict[str, Any]]) -> str:
    """Create a summary report of all generated PDF reports"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = "reports/peer_group"
    summary_file = f"{report_dir}/peer_group_pdf_reports_summary_{timestamp}.md"

    total_symbols = sum(len(group["symbols"]) for group in all_results)
    total_successful_reports = sum(
        sum(1 for r in group_reports.values() if r["status"] == "success") for group_reports in report_results
    )
    total_time = sum(sum(r["duration"] for r in group_reports.values()) for group_reports in report_results)

    with open(summary_file, "w") as f:
        f.write("# Peer Group PDF Reports Summary\n\n")
        f.write(f"**Generation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")

        f.write("## Overview\n")
        f.write(f"- **Peer Groups:** {len(all_results)}\\n")
        f.write(f"- **Total Symbols:** {total_symbols}\\n")
        f.write(
            f"- **PDF Reports Generated:** {total_successful_reports}/{total_symbols} ({100*total_successful_reports/total_symbols:.1f}%)\\n"
        )
        f.write(f"- **Total Generation Time:** {total_time:.1f} seconds\\n")
        f.write(f"- **Avg per Report:** {total_time/total_symbols:.1f} seconds\\n\\n")

        f.write("## Report Generation Results\\n\\n")

        for i, (group_data, group_reports) in enumerate(zip(all_results, report_results)):
            sector = group_data["sector"]
            industry = group_data["industry"]
            f.write(f"### {sector.title()} - {industry.replace('_', ' ').title()}\\n\\n")

            # Results table
            f.write("| Symbol | Status | Score | Peers | Report File | Time |\\n")
            f.write("|--------|--------|-------|-------|-------------|------|\\n")

            for symbol in group_data["symbols"]:
                result = group_reports[symbol]
                if result["status"] == "success":
                    report_name = os.path.basename(result["report_path"])
                    f.write(
                        f"| {symbol} | ✅ | {result['score']:.1f} | {result['peer_count']} | {report_name} | {result['duration']:.1f}s |\\n"
                    )
                else:
                    f.write(f"| {symbol} | ❌ | - | - | - | {result['duration']:.1f}s |\\n")

            successful = sum(1 for r in group_reports.values() if r["status"] == "success")
            total = len(group_data["symbols"])
            group_time = sum(r["duration"] for r in group_reports.values())
            f.write(f"\\n**Group Summary:** {successful}/{total} reports generated ({group_time:.1f}s)\\n\\n")

        f.write("## PDF Report Locations\\n\\n")
        f.write(
            "All PDF reports are generated in the `reports/synthesis/` directory with the following naming convention:\\n"
        )
        f.write("- `{symbol}_synthesis_report_{timestamp}.pdf`\\n\\n")

        f.write("## System Performance\\n\\n")
        f.write("✅ **Peer Group Integration:** All reports include peer comparison context\\n\\n")
        f.write("✅ **Batch Generation:** Automated PDF creation across multiple sectors\\n\\n")
        f.write("✅ **Error Handling:** Graceful handling of synthesis and report generation failures\\n\\n")
        f.write("✅ **Performance Tracking:** Detailed timing and success metrics\\n\\n")

    return summary_file


def main():
    """Main function to generate PDF reports for all peer group symbols"""
    print("📄 PEER GROUP PDF REPORT GENERATION")
    print("=" * 80)
    print("Generating synthesis PDF reports for all analyzed peer groups...")

    try:
        # Load peer group analysis results
        analysis_data = load_peer_group_results()
        peer_groups = analysis_data["peer_groups"]

        print(
            f"\\nFound {len(peer_groups)} peer groups with {analysis_data['overall_summary']['total_symbols_analyzed']} symbols"
        )

        # Initialize synthesizer
        synthesizer = InvestmentSynthesizer()

        # Generate reports for each peer group
        all_report_results = []
        overall_start = time.time()

        for i, group_data in enumerate(peer_groups, 1):
            print(f"\\n🔍 Processing Group {i}/{len(peer_groups)}...")
            group_reports = generate_reports_for_peer_group(group_data, synthesizer)
            all_report_results.append(group_reports)

            # Brief pause between groups
            if i < len(peer_groups):
                time.sleep(2)

        overall_duration = time.time() - overall_start

        # Create summary report
        print(f"\\n📄 Creating summary report...")
        summary_file = create_summary_report(peer_groups, all_report_results)

        # Final summary
        total_symbols = analysis_data["overall_summary"]["total_symbols_analyzed"]
        total_successful = sum(
            sum(1 for r in group_reports.values() if r["status"] == "success") for group_reports in all_report_results
        )

        print(f"\\n🎉 PDF REPORT GENERATION COMPLETE!")
        print("=" * 60)
        print(f"📊 Peer Groups Processed: {len(peer_groups)}")
        print(f"📈 Total Symbols: {total_symbols}")
        print(
            f"📄 PDF Reports Generated: {total_successful}/{total_symbols} ({100*total_successful/total_symbols:.1f}%)"
        )
        print(f"⏱️  Total Duration: {overall_duration:.1f} seconds")
        print(f"📄 Summary Report: {summary_file}")
        print(f"📁 PDF Reports Location: reports/synthesis/")

        # Print group-by-group summary
        print(f"\\n📊 REPORTS BY PEER GROUP:")
        for group_data, group_reports in zip(peer_groups, all_report_results):
            successful = sum(1 for r in group_reports.values() if r["status"] == "success")
            total = len(group_data["symbols"])
            print(
                f"• {group_data['sector'].title()} - {group_data['industry'].replace('_', ' ').title()}: {successful}/{total}"
            )

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"\\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
