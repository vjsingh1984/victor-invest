#!/usr/bin/env python3
"""
InvestiGator - Comprehensive Peer Group Report Generator
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Generates a single comprehensive PDF report containing all peer group analysis
with 2D/3D charts, peer comparisons, and individual symbol analysis
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY

from utils.chart_generator import ChartGenerator
from utils.peer_group_report_generator import PeerGroupPDFReportGenerator
from run_peer_group_with_metrics import run_peer_group_analysis_with_metrics
from utils.ascii_art import ASCIIArt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ComprehensivePeerGroupReportGenerator:
    """Generates comprehensive peer group reports combining all sectors and symbols"""

    def __init__(self, output_dir: Path):
        """Initialize the comprehensive report generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chart_generator = ChartGenerator(self.output_dir.parent / "charts")
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

        # Store all peer group data
        self.all_peer_groups = {}
        self.all_symbol_data = {}
        self.sector_summaries = {}

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Main title
        self.styles.add(
            ParagraphStyle(
                name="MainTitle",
                parent=self.styles["Title"],
                fontSize=28,
                textColor=colors.HexColor("#1a1a1a"),
                spaceAfter=20,
                alignment=TA_CENTER,
            )
        )

        # Section title
        self.styles.add(
            ParagraphStyle(
                name="SectionTitle",
                parent=self.styles["Heading1"],
                fontSize=20,
                textColor=colors.HexColor("#2c3e50"),
                spaceBefore=20,
                spaceAfter=12,
                borderWidth=2,
                borderColor=colors.HexColor("#2c3e50"),
                borderPadding=6,
            )
        )

        # Subsection title
        self.styles.add(
            ParagraphStyle(
                name="SubsectionTitle",
                parent=self.styles["Heading2"],
                fontSize=16,
                textColor=colors.HexColor("#34495e"),
                spaceBefore=15,
                spaceAfter=8,
            )
        )

    def load_peer_groups_data(self) -> Dict[str, Any]:
        """Load peer groups configuration"""
        with open("data/russell_1000_peer_groups.json", "r") as f:
            return json.load(f)

    def get_peer_groups_to_analyze(
        self, target_sector: Optional[str] = None, target_industry: Optional[str] = None
    ) -> List[tuple]:
        """Get list of peer groups to analyze"""
        data = self.load_peer_groups_data()
        peer_groups = data.get("peer_groups", {})

        groups_to_analyze = []

        for sector, industries in peer_groups.items():
            if target_sector and sector != target_sector:
                continue

            for industry, companies in industries.items():
                if target_industry and industry != target_industry:
                    continue

                # Get up to 7 symbols for analysis (large cap + mid cap)
                symbols = companies.get("large_cap", [])[:7]
                if len(symbols) < 7:
                    symbols.extend(companies.get("mid_cap", [])[: 7 - len(symbols)])

                if len(symbols) >= 2:  # Need at least 2 symbols for comparison
                    groups_to_analyze.append((sector, industry, symbols))

        return groups_to_analyze

    def run_comprehensive_analysis(self, target_sector: Optional[str] = None, target_industry: Optional[str] = None):
        """Run comprehensive analysis for all specified peer groups"""
        logger.info("🔍 Starting comprehensive peer group analysis...")

        groups_to_analyze = self.get_peer_groups_to_analyze(target_sector, target_industry)

        logger.info(f"Found {len(groups_to_analyze)} peer groups to analyze")

        for i, (sector, industry, symbols) in enumerate(groups_to_analyze, 1):
            logger.info(f"\n[{i}/{len(groups_to_analyze)}] Analyzing {sector}/{industry}: {symbols}")

            try:
                # Run enhanced metrics analysis for this peer group
                run_peer_group_analysis_with_metrics(sector, industry, symbols)

                # Store peer group info
                peer_group_key = f"{sector}_{industry}"
                self.all_peer_groups[peer_group_key] = {"sector": sector, "industry": industry, "symbols": symbols}

                # Load the generated metrics
                metrics_file = Path(
                    f"reports/peer_group_comprehensive/{sector}_{industry.replace('_', '_')}_metrics.json"
                )
                if metrics_file.exists():
                    with open(metrics_file, "r") as f:
                        metrics_data = json.load(f)

                    # Store symbol data
                    for symbol, symbol_metrics in metrics_data.get("symbol_metrics", {}).items():
                        self.all_symbol_data[symbol] = symbol_metrics

                    # Store sector summary
                    if sector not in self.sector_summaries:
                        self.sector_summaries[sector] = []
                    self.sector_summaries[sector].append(
                        {"industry": industry, "symbols": symbols, "metrics": metrics_data.get("peer_averages", {})}
                    )

            except Exception as e:
                logger.error(f"Error analyzing {sector}/{industry}: {e}")
                continue

    def generate_comprehensive_report(
        self, target_sector: Optional[str] = None, target_industry: Optional[str] = None
    ) -> str:
        """Generate the comprehensive PDF report"""

        # Run analysis first
        self.run_comprehensive_analysis(target_sector, target_industry)

        if not self.all_symbol_data:
            logger.error("No symbol data available for report generation")
            return None

        # Generate filename
        if target_sector and target_industry:
            filename = f"comprehensive_peer_group_{target_sector}_{target_industry}.pdf"
        elif target_sector:
            filename = f"comprehensive_peer_group_{target_sector}.pdf"
        else:
            filename = "comprehensive_peer_group_all_sectors.pdf"

        filepath = self.output_dir / filename

        # Create document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        # Build content
        story = []

        # Title page
        story.extend(self._create_title_page(target_sector, target_industry))

        # Executive summary
        story.extend(self._create_executive_summary())

        # Universe overview with 2D/3D charts
        story.extend(self._create_universe_overview())

        # Sector analysis
        story.extend(self._create_sector_analysis())

        # Individual symbol analysis
        story.extend(self._create_individual_symbol_analysis())

        # Performance rankings
        story.extend(self._create_performance_rankings())

        # Disclaimer
        story.extend(self._create_disclaimer())

        # Build PDF
        doc.build(story)

        logger.info(f"📄 Generated comprehensive peer group report: {filename}")
        return str(filepath)

    def _create_title_page(self, target_sector: Optional[str], target_industry: Optional[str]) -> List:
        """Create comprehensive report title page"""
        elements = []

        # Main title
        if target_sector and target_industry:
            title = f"Comprehensive Peer Group Analysis\n{target_sector.title()} - {target_industry.replace('_', ' ').title()}"
        elif target_sector:
            title = f"Comprehensive Peer Group Analysis\n{target_sector.title()} Sector"
        else:
            title = "Comprehensive Peer Group Analysis\nMulti-Sector Investment Research"

        elements.append(Paragraph(title, self.styles["MainTitle"]))
        elements.append(Spacer(1, 0.3 * inch))

        # Summary stats
        total_symbols = len(self.all_symbol_data)
        total_sectors = len(self.sector_summaries)
        total_industries = sum(len(industries) for industries in self.sector_summaries.values())

        summary_text = f"""
        <b>Analysis Scope:</b><br/>
        • Symbols Analyzed: {total_symbols}<br/>
        • Sectors Covered: {total_sectors}<br/>
        • Industries Analyzed: {total_industries}<br/>
        • Analysis Date: {datetime.now().strftime('%B %d, %Y')}<br/>
        • Generated by: InvestiGator AI System
        """

        elements.append(Paragraph(summary_text, self.styles["BodyText"]))
        elements.append(PageBreak())

        return elements

    def _create_executive_summary(self) -> List:
        """Create executive summary section"""
        elements = []
        elements.append(Paragraph("Executive Summary", self.styles["SectionTitle"]))

        # Calculate overall statistics
        scores = []
        recommendations = {"BUY": 0, "HOLD": 0, "SELL": 0}

        for symbol_data in self.all_symbol_data.values():
            if "scores" in symbol_data and "overall_score" in symbol_data["scores"]:
                scores.append(symbol_data["scores"]["overall_score"])

            if "recommendation" in symbol_data:
                rec = symbol_data["recommendation"]
                if rec in recommendations:
                    recommendations[rec] += 1

        avg_score = sum(scores) / len(scores) if scores else 0

        summary_text = f"""
        <b>Portfolio Overview:</b><br/>
        Our comprehensive analysis covers {len(self.all_symbol_data)} companies across 
        {len(self.sector_summaries)} sectors, providing institutional-grade peer group 
        comparative analysis with AI-powered insights.<br/><br/>
        
        <b>Key Findings:</b><br/>
        • Average Investment Score: {avg_score:.1f}/10<br/>
        • Investment Recommendations: {recommendations['BUY']} BUY, {recommendations['HOLD']} HOLD, {recommendations['SELL']} SELL<br/>
        • Sectors: {', '.join(sector.title() for sector in self.sector_summaries.keys())}<br/><br/>
        
        <b>Methodology:</b><br/>
        Each company undergoes comprehensive analysis including SEC fundamental analysis, 
        technical market analysis, and AI-powered investment synthesis. Peer comparisons 
        use Russell 1000 industry classifications for accurate competitive positioning.
        """

        elements.append(Paragraph(summary_text, self.styles["BodyText"]))
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_universe_overview(self) -> List:
        """Create universe overview with 2D/3D positioning charts"""
        elements = []
        elements.append(Paragraph("Investment Universe Overview", self.styles["SectionTitle"]))

        # Prepare data for comprehensive charts
        chart_data = []
        for symbol, data in self.all_symbol_data.items():
            scores = data.get("scores", {})
            financials = data.get("financial_metrics", {})

            chart_data.append(
                {
                    "symbol": symbol,
                    "overall_score": scores.get("overall_score", 0),
                    "technical_score": scores.get("technical_score", 0),
                    "fundamental_score": scores.get("overall_score", 0) * 0.6,  # Estimate
                    "pe_ratio": financials.get("pe_ratio", 0),
                    "roe": financials.get("roe", 0),
                    "revenue": financials.get("revenue", 0),
                    "sector": next(
                        (pg["sector"] for pg in self.all_peer_groups.values() if symbol in pg["symbols"]), "Unknown"
                    ),
                }
            )

        if chart_data:
            # Generate comprehensive 3D positioning chart
            chart_3d = self.chart_generator.generate_comprehensive_3d_plot(chart_data)
            if chart_3d and os.path.exists(chart_3d):
                elements.append(Paragraph("<b>3D Universe Positioning Analysis</b>", self.styles["SubsectionTitle"]))
                elements.append(Spacer(1, 0.1 * inch))
                try:
                    img = Image(chart_3d, width=6.5 * inch, height=4.5 * inch)
                    elements.append(img)
                except Exception as e:
                    logger.warning(f"Could not load 3D chart: {e}")

                elements.append(Spacer(1, 0.1 * inch))
                chart_desc = """
                This 3D chart positions all analyzed companies across fundamental strength, 
                technical momentum, and valuation metrics. Companies in the upper-right 
                represent the optimal combination of strong fundamentals and positive technicals.
                """
                elements.append(Paragraph(chart_desc, self.styles["BodyText"]))
                elements.append(Spacer(1, 0.2 * inch))

            # Generate sector comparison chart
            sector_2d = self.chart_generator.generate_sector_comparison_plot(chart_data)
            if sector_2d and os.path.exists(sector_2d):
                elements.append(Paragraph("<b>Sector Comparison Analysis</b>", self.styles["SubsectionTitle"]))
                elements.append(Spacer(1, 0.1 * inch))
                try:
                    img = Image(sector_2d, width=6.5 * inch, height=4.5 * inch)
                    elements.append(img)
                except Exception as e:
                    logger.warning(f"Could not load sector chart: {e}")

                chart_desc = """
                This chart compares investment opportunities across sectors, highlighting 
                relative performance and positioning within the analyzed universe.
                """
                elements.append(Paragraph(chart_desc, self.styles["BodyText"]))

        elements.append(Spacer(1, 0.3 * inch))
        return elements

    def _create_sector_analysis(self) -> List:
        """Create detailed sector analysis"""
        elements = []
        elements.append(Paragraph("Sector Analysis", self.styles["SectionTitle"]))

        for sector, industries in self.sector_summaries.items():
            elements.append(Paragraph(f"{sector.title()} Sector", self.styles["SubsectionTitle"]))

            # Sector overview
            sector_symbols = []
            for industry_data in industries:
                sector_symbols.extend(industry_data["symbols"])

            sector_text = f"""
            <b>Industries Analyzed:</b> {len(industries)}<br/>
            <b>Companies:</b> {len(sector_symbols)} ({', '.join(sector_symbols)})<br/>
            <b>Focus:</b> {self._get_sector_focus_description(sector)}
            """

            elements.append(Paragraph(sector_text, self.styles["BodyText"]))
            elements.append(Spacer(1, 0.1 * inch))

            # Industry breakdown table
            table_data = [["Industry", "Companies", "Avg Score", "Top Recommendation"]]

            for industry_data in industries:
                industry = industry_data["industry"].replace("_", " ").title()
                symbols = industry_data["symbols"]

                # Calculate industry averages
                industry_scores = []
                industry_recs = []
                for symbol in symbols:
                    if symbol in self.all_symbol_data:
                        symbol_data = self.all_symbol_data[symbol]
                        if "scores" in symbol_data and "overall_score" in symbol_data["scores"]:
                            industry_scores.append(symbol_data["scores"]["overall_score"])
                        if "recommendation" in symbol_data:
                            industry_recs.append(symbol_data["recommendation"])

                avg_score = sum(industry_scores) / len(industry_scores) if industry_scores else 0
                top_rec = max(set(industry_recs), key=industry_recs.count) if industry_recs else "N/A"

                table_data.append([industry, ", ".join(symbols), f"{avg_score:.1f}/10", top_rec])

            if len(table_data) > 1:
                table = Table(table_data, colWidths=[2 * inch, 2 * inch, 1 * inch, 1.5 * inch])
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 10),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#ecf0f1")),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            ("FONTSIZE", (0, 1), (-1, -1), 9),
                        ]
                    )
                )
                elements.append(table)

            elements.append(Spacer(1, 0.2 * inch))

        return elements

    def _create_individual_symbol_analysis(self) -> List:
        """Create individual symbol analysis section"""
        elements = []
        elements.append(Paragraph("Individual Company Analysis", self.styles["SectionTitle"]))

        # Create comprehensive ranking table
        ranked_symbols = []
        for symbol, data in self.all_symbol_data.items():
            scores = data.get("scores", {})
            financials = data.get("financial_metrics", {})
            rec_data = data.get("recommendation_data", {})

            ranked_symbols.append(
                {
                    "symbol": symbol,
                    "overall_score": scores.get("overall_score", 0),
                    "recommendation": data.get("recommendation", "HOLD"),
                    "current_price": rec_data.get("current_price", 0),
                    "price_target": rec_data.get("price_target", 0),
                    "pe_ratio": financials.get("pe_ratio", 0),
                    "roe": financials.get("roe", 0),
                    "sector": next(
                        (pg["sector"] for pg in self.all_peer_groups.values() if symbol in pg["symbols"]), "Unknown"
                    ),
                }
            )

        # Sort by overall score
        ranked_symbols.sort(key=lambda x: x["overall_score"], reverse=True)

        # Create table
        table_data = [["Rank", "Symbol", "Sector", "Score", "Rec", "Current", "Target", "P/E", "ROE"]]

        for i, symbol_data in enumerate(ranked_symbols[:20], 1):  # Top 20
            table_data.append(
                [
                    str(i),
                    symbol_data["symbol"],
                    symbol_data["sector"].title()[:8],
                    f"{symbol_data['overall_score']:.1f}",
                    symbol_data["recommendation"][:4],
                    f"${symbol_data['current_price']:.0f}" if symbol_data["current_price"] > 0 else "N/A",
                    f"${symbol_data['price_target']:.0f}" if symbol_data["price_target"] > 0 else "N/A",
                    f"{symbol_data['pe_ratio']:.1f}" if symbol_data["pe_ratio"] > 0 else "N/A",
                    f"{symbol_data['roe']:.1f}%" if symbol_data["roe"] > 0 else "N/A",
                ]
            )

        if len(table_data) > 1:
            table = Table(
                table_data,
                colWidths=[
                    0.4 * inch,
                    0.6 * inch,
                    0.8 * inch,
                    0.6 * inch,
                    0.5 * inch,
                    0.7 * inch,
                    0.7 * inch,
                    0.5 * inch,
                    0.6 * inch,
                ],
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ]
                )
            )
            elements.append(table)

        elements.append(Spacer(1, 0.3 * inch))
        return elements

    def _create_performance_rankings(self) -> List:
        """Create performance rankings section"""
        elements = []
        elements.append(Paragraph("Performance Rankings", self.styles["SectionTitle"]))

        # Top performers by score
        top_performers = sorted(
            [(symbol, data.get("scores", {}).get("overall_score", 0)) for symbol, data in self.all_symbol_data.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        elements.append(Paragraph("<b>Top 10 Investment Opportunities:</b>", self.styles["SubsectionTitle"]))

        for i, (symbol, score) in enumerate(top_performers, 1):
            symbol_data = self.all_symbol_data[symbol]
            rec = symbol_data.get("recommendation", "HOLD")
            sector = next((pg["sector"] for pg in self.all_peer_groups.values() if symbol in pg["symbols"]), "Unknown")

            ranking_text = f"{i}. <b>{symbol}</b> ({sector.title()}) - Score: {score:.1f}/10, Recommendation: {rec}"
            elements.append(Paragraph(ranking_text, self.styles["BodyText"]))

        elements.append(Spacer(1, 0.3 * inch))
        return elements

    def _create_disclaimer(self) -> List:
        """Create disclaimer section"""
        elements = []
        elements.append(PageBreak())
        elements.append(Paragraph("Important Disclaimers", self.styles["SectionTitle"]))

        disclaimer_text = """
        <b>Investment Risk Disclaimer:</b> This comprehensive peer group analysis is for 
        informational purposes only and does not constitute investment advice, recommendation, 
        or solicitation. All investments carry risk of loss. Past performance does not guarantee 
        future results.<br/><br/>
        
        <b>Data Sources:</b> Analysis is based on SEC filings, market data, and AI-powered 
        analysis. While we strive for accuracy, data may contain errors or omissions. Always 
        verify information independently before making investment decisions.<br/><br/>
        
        <b>AI Analysis Limitations:</b> This report contains AI-generated analysis and insights. 
        AI models may have biases, limitations, or errors. Human oversight and additional 
        research are recommended for investment decisions.<br/><br/>
        
        <b>Peer Group Analysis:</b> Comparisons are based on Russell 1000 industry 
        classifications and current market data. Relative positioning may change rapidly 
        with market conditions.
        """
        elements.append(Paragraph(disclaimer_text, self.styles["BodyText"]))

        return elements

    def _get_sector_focus_description(self, sector: str) -> str:
        """Get sector-specific focus description"""
        descriptions = {
            "financials": "Interest rates, credit quality, regulatory environment, capital adequacy",
            "technology": "Innovation cycles, market disruption, R&D efficiency, competitive moats",
            "healthcare": "Drug pipelines, regulatory approvals, demographic trends, patent protection",
            "consumer_discretionary": "Consumer spending, economic sensitivity, brand strength",
            "consumer_staples": "Defensive characteristics, dividend yields, market share stability",
            "industrials": "Economic cycles, infrastructure spending, operational efficiency",
            "energy": "Commodity prices, ESG factors, operational efficiency, reserves quality",
            "materials": "Commodity cycles, supply chain dynamics, cost structure",
            "real_estate": "Interest rate sensitivity, occupancy rates, property values",
            "communication_services": "Digital transformation, content consumption, infrastructure",
            "utilities": "Regulatory environment, dividend sustainability, ESG transition",
        }
        return descriptions.get(sector, "Sector-specific competitive dynamics and market positioning")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate comprehensive peer group report")
    parser.add_argument("--sector", help="Target specific sector")
    parser.add_argument("--industry", help="Target specific industry within sector")
    args = parser.parse_args()

    # Display peer analysis banner
    ASCIIArt.print_banner("peer")

    print("📊 COMPREHENSIVE PEER GROUP REPORT GENERATION")
    print("=" * 80)

    if args.sector:
        if args.industry:
            print(f"Generating comprehensive report for {args.sector}/{args.industry}")
        else:
            print(f"Generating comprehensive report for {args.sector} sector")
    else:
        print("Generating comprehensive report for all sectors")

    # Initialize generator
    output_dir = Path("reports/peer_group_comprehensive")
    generator = ComprehensivePeerGroupReportGenerator(output_dir)

    try:
        # Generate comprehensive report
        report_path = generator.generate_comprehensive_report(args.sector, args.industry)

        if report_path:
            print(f"\n✅ Comprehensive peer group report generated successfully!")
            print(f"📄 Report location: {report_path}")
            print(f"📊 Symbols analyzed: {len(generator.all_symbol_data)}")
            print(f"🏭 Sectors covered: {len(generator.sector_summaries)}")
        else:
            print("\n❌ Failed to generate comprehensive report")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error generating comprehensive report: {e}")
        logger.error(f"Comprehensive report generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
