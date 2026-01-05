#!/usr/bin/env python3
"""
InvestiGator - PDF Report Generation Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

PDF Report Generation Module for InvestiGator
Handles creation of investment analysis reports
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import markdown

    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    logging.warning("markdown not available - some report features will be limited")

try:
    from reportlab.graphics import renderPDF
    from reportlab.graphics.shapes import Circle, Drawing, Rect
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.platypus import (
        HRFlowable,
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus.flowables import Flowable

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("reportlab not available - PDF report generation will be disabled")

logger = logging.getLogger(__name__)


class ScoreCard(Flowable):
    """Visual scorecard component for financial metrics"""

    def __init__(self, width, height, score, max_score=10, label="Score", color_scheme="default"):
        self.width = width
        self.height = height
        self.score = score
        self.max_score = max_score
        self.label = label
        self.color_scheme = color_scheme

    def draw(self):
        # Background
        self.canv.setFillColor(colors.HexColor("#f8f9fa"))
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)

        # Progress bar background
        bar_width = self.width * 0.6
        bar_height = 12
        bar_x = (self.width - bar_width) / 2
        bar_y = self.height * 0.3

        self.canv.setFillColor(colors.HexColor("#e9ecef"))
        self.canv.rect(bar_x, bar_y, bar_width, bar_height, fill=1, stroke=0)

        # Progress bar fill
        progress = min(self.score / self.max_score, 1.0)
        if progress >= 0.8:
            fill_color = colors.HexColor("#28a745")  # Green
        elif progress >= 0.6:
            fill_color = colors.HexColor("#ffc107")  # Yellow
        elif progress >= 0.4:
            fill_color = colors.HexColor("#fd7e14")  # Orange
        else:
            fill_color = colors.HexColor("#dc3545")  # Red

        self.canv.setFillColor(fill_color)
        self.canv.rect(bar_x, bar_y, bar_width * progress, bar_height, fill=1, stroke=0)

        # Score text
        self.canv.setFillColor(colors.black)
        self.canv.setFont("Helvetica-Bold", 16)
        self.canv.drawCentredText(self.width / 2, self.height * 0.7, f"{self.score:.1f}/{self.max_score}")

        # Label
        self.canv.setFont("Helvetica", 10)
        self.canv.drawCentredText(self.width / 2, self.height * 0.1, self.label)


class RecommendationBadge(Flowable):
    """Visual badge for investment recommendations"""

    def __init__(self, width, height, recommendation, confidence):
        self.width = width
        self.height = height
        self.recommendation = recommendation
        self.confidence = confidence

    def draw(self):
        # Badge color based on recommendation
        if self.recommendation in ["BUY", "STRONG_BUY"]:
            badge_color = colors.HexColor("#28a745")
        elif self.recommendation in ["SELL", "STRONG_SELL"]:
            badge_color = colors.HexColor("#dc3545")
        else:
            badge_color = colors.HexColor("#6c757d")

        # Draw badge background
        self.canv.setFillColor(badge_color)
        self.canv.roundRect(0, 0, self.width, self.height, 8, fill=1, stroke=0)

        # Recommendation text
        self.canv.setFillColor(colors.white)
        self.canv.setFont("Helvetica-Bold", 14)
        self.canv.drawCentredText(self.width / 2, self.height * 0.6, self.recommendation)

        # Confidence text
        self.canv.setFont("Helvetica", 10)
        self.canv.drawCentredText(self.width / 2, self.height * 0.25, f"{self.confidence} CONFIDENCE")


class EntryExitZone(Flowable):
    """Visual representation of entry/exit zones with price levels"""

    def __init__(self, width, height, current_price, entry_zone, support_levels=None, resistance_levels=None):
        """
        Args:
            width: Flowable width
            height: Flowable height
            current_price: Current stock price
            entry_zone: Dict with lower_bound, upper_bound, ideal_entry
            support_levels: List of support price levels
            resistance_levels: List of resistance price levels
        """
        self.width = width
        self.height = height
        self.current_price = current_price
        self.entry_zone = entry_zone or {}
        self.support_levels = support_levels or []
        self.resistance_levels = resistance_levels or []

    def draw(self):
        # Calculate price range for visualization
        all_prices = [self.current_price]
        if self.entry_zone:
            all_prices.extend(
                [
                    self.entry_zone.get("lower_bound", self.current_price),
                    self.entry_zone.get("upper_bound", self.current_price),
                ]
            )
        all_prices.extend(self.support_levels[:3])
        all_prices.extend(self.resistance_levels[:3])

        min_price = min(all_prices) * 0.98
        max_price = max(all_prices) * 1.02
        price_range = max_price - min_price

        if price_range <= 0:
            return

        # Helper to convert price to Y coordinate
        def price_to_y(price):
            return 20 + ((price - min_price) / price_range) * (self.height - 40)

        # Background
        self.canv.setFillColor(colors.HexColor("#fafafa"))
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=1)

        # Draw entry zone (green rectangle)
        if self.entry_zone:
            lower = self.entry_zone.get("lower_bound", self.current_price * 0.97)
            upper = self.entry_zone.get("upper_bound", self.current_price * 1.02)
            zone_y1 = price_to_y(lower)
            zone_y2 = price_to_y(upper)

            self.canv.setFillColor(colors.HexColor("#d4edda"))
            self.canv.setStrokeColor(colors.HexColor("#28a745"))
            self.canv.rect(20, zone_y1, self.width - 100, zone_y2 - zone_y1, fill=1, stroke=1)

            # Ideal entry line
            if ideal := self.entry_zone.get("ideal_entry"):
                ideal_y = price_to_y(ideal)
                self.canv.setStrokeColor(colors.HexColor("#155724"))
                self.canv.setDash([4, 2])
                self.canv.line(20, ideal_y, self.width - 80, ideal_y)
                self.canv.setDash([])
                self.canv.setFont("Helvetica-Bold", 8)
                self.canv.setFillColor(colors.HexColor("#155724"))
                self.canv.drawRightString(self.width - 10, ideal_y - 3, f"Ideal: ${ideal:.2f}")

        # Draw support levels (red dashed lines)
        self.canv.setStrokeColor(colors.HexColor("#dc3545"))
        for i, support in enumerate(self.support_levels[:3]):
            y = price_to_y(support)
            self.canv.setDash([2, 2])
            self.canv.line(20, y, self.width - 80, y)
            self.canv.setDash([])
            self.canv.setFont("Helvetica", 7)
            self.canv.setFillColor(colors.HexColor("#dc3545"))
            self.canv.drawRightString(self.width - 10, y - 3, f"S{i+1}: ${support:.2f}")

        # Draw resistance levels (blue dashed lines)
        self.canv.setStrokeColor(colors.HexColor("#0052cc"))
        for i, resistance in enumerate(self.resistance_levels[:3]):
            y = price_to_y(resistance)
            self.canv.setDash([2, 2])
            self.canv.line(20, y, self.width - 80, y)
            self.canv.setDash([])
            self.canv.setFont("Helvetica", 7)
            self.canv.setFillColor(colors.HexColor("#0052cc"))
            self.canv.drawRightString(self.width - 10, y - 3, f"R{i+1}: ${resistance:.2f}")

        # Draw current price line (thick black)
        current_y = price_to_y(self.current_price)
        self.canv.setStrokeColor(colors.black)
        self.canv.setLineWidth(2)
        self.canv.line(10, current_y, self.width - 80, current_y)
        self.canv.setLineWidth(1)
        self.canv.setFont("Helvetica-Bold", 9)
        self.canv.setFillColor(colors.black)
        self.canv.drawRightString(self.width - 10, current_y - 3, f"Now: ${self.current_price:.2f}")

        # Legend
        self.canv.setFont("Helvetica", 7)
        self.canv.setFillColor(colors.HexColor("#28a745"))
        self.canv.drawString(5, 5, "Entry Zone")
        self.canv.setFillColor(colors.HexColor("#dc3545"))
        self.canv.drawString(60, 5, "Support")
        self.canv.setFillColor(colors.HexColor("#0052cc"))
        self.canv.drawString(105, 5, "Resistance")


class SignalStrengthBar(Flowable):
    """Horizontal bar showing signal strength (0-100)"""

    def __init__(self, width, height, score, label="Signal Strength", show_value=True):
        """
        Args:
            width: Flowable width
            height: Flowable height
            score: Score value (0-100)
            label: Label text
            show_value: Whether to show numeric value
        """
        self.width = width
        self.height = height
        self.score = min(max(score, 0), 100)  # Clamp to 0-100
        self.label = label
        self.show_value = show_value

    def draw(self):
        bar_height = self.height * 0.4
        bar_y = self.height * 0.3
        bar_width = self.width * 0.7

        # Background bar
        self.canv.setFillColor(colors.HexColor("#e9ecef"))
        self.canv.roundRect(0, bar_y, bar_width, bar_height, 3, fill=1, stroke=0)

        # Progress bar with color gradient
        progress = self.score / 100
        if self.score >= 70:
            fill_color = colors.HexColor("#28a745")  # Green
        elif self.score >= 50:
            fill_color = colors.HexColor("#ffc107")  # Yellow
        elif self.score >= 30:
            fill_color = colors.HexColor("#fd7e14")  # Orange
        else:
            fill_color = colors.HexColor("#dc3545")  # Red

        self.canv.setFillColor(fill_color)
        self.canv.roundRect(0, bar_y, bar_width * progress, bar_height, 3, fill=1, stroke=0)

        # Label
        self.canv.setFillColor(colors.black)
        self.canv.setFont("Helvetica", 8)
        self.canv.drawString(0, self.height * 0.8, self.label)

        # Value
        if self.show_value:
            self.canv.setFont("Helvetica-Bold", 10)
            self.canv.drawRightString(self.width, bar_y + bar_height / 2 - 3, f"{self.score:.0f}%")


class StopLossIndicator(Flowable):
    """Visual stop loss level relative to entry"""

    def __init__(self, width, height, entry_price, stop_loss, target_price=None):
        """
        Args:
            width: Flowable width
            height: Flowable height
            entry_price: Entry price level
            stop_loss: Stop loss price level
            target_price: Optional target price level
        """
        self.width = width
        self.height = height
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.target_price = target_price

    def draw(self):
        # Calculate risk/reward
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target_price - self.entry_price) if self.target_price else risk * 2
        risk_reward = reward / risk if risk > 0 else 0
        stop_loss_pct = abs((self.entry_price - self.stop_loss) / self.entry_price) * 100

        # Draw bar
        bar_height = 15
        bar_y = self.height / 2 - bar_height / 2

        # Risk zone (red)
        risk_width = self.width * 0.3
        self.canv.setFillColor(colors.HexColor("#f8d7da"))
        self.canv.rect(0, bar_y, risk_width, bar_height, fill=1, stroke=0)

        # Entry point
        entry_x = risk_width
        self.canv.setFillColor(colors.HexColor("#28a745"))
        self.canv.circle(entry_x, bar_y + bar_height / 2, 5, fill=1, stroke=0)

        # Target zone (green)
        target_width = self.width * 0.5
        self.canv.setFillColor(colors.HexColor("#d4edda"))
        self.canv.rect(entry_x, bar_y, target_width, bar_height, fill=1, stroke=0)

        # Labels
        self.canv.setFont("Helvetica", 8)
        self.canv.setFillColor(colors.HexColor("#dc3545"))
        self.canv.drawString(5, bar_y - 12, f"Stop: ${self.stop_loss:.2f} (-{stop_loss_pct:.1f}%)")

        self.canv.setFillColor(colors.HexColor("#155724"))
        self.canv.drawString(entry_x + 10, bar_y - 12, f"Entry: ${self.entry_price:.2f}")

        if self.target_price:
            target_pct = abs((self.target_price - self.entry_price) / self.entry_price) * 100
            self.canv.setFillColor(colors.HexColor("#28a745"))
            self.canv.drawRightString(self.width, bar_y - 12, f"Target: ${self.target_price:.2f} (+{target_pct:.1f}%)")

        # Risk/Reward ratio
        self.canv.setFont("Helvetica-Bold", 9)
        self.canv.setFillColor(colors.black)
        self.canv.drawCentredString(self.width / 2, self.height - 5, f"Risk/Reward: 1:{risk_reward:.1f}")


@dataclass
class ReportConfig:
    """Configuration for report generation"""

    title: str = "InvestiGator Investment Analysis"
    subtitle: str = "AI-Powered Investment Research Report"
    author: str = "InvestiGator AI System"
    include_charts: bool = True
    include_disclaimer: bool = True
    page_size: str = "letter"
    margin: float = 0.75 * inch


class NumberedCanvas(canvas.Canvas):
    """Custom canvas for page numbering"""

    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """Add page numbers to all pages"""
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        """Draw page number and disclaimer at bottom of page"""
        # Draw page number
        self.setFont("Helvetica", 9)
        self.drawRightString(self._pagesize[0] - 0.75 * inch, 0.5 * inch, f"Page {self._pageNumber} of {page_count}")

        # Draw disclaimer footer on every page
        self.setFont("Helvetica-Oblique", 8)
        self.setFillColor(colors.HexColor("#cc0000"))
        disclaimer_text = "AI-Generated Report - Educational Testing Only - NOT Investment Advice - See Full Disclaimer"
        self.drawCentredString(self._pagesize[0] / 2, 0.5 * inch, disclaimer_text)
        self.setFillColor(colors.black)  # Reset color


class PDFReportGenerator:
    """Generates PDF investment reports"""

    def __init__(self, output_dir: Path, config: Optional[ReportConfig] = None):
        """
        Initialize PDF report generator

        Args:
            output_dir: Directory for output reports
            config: Report configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ReportConfig()

        if not REPORTLAB_AVAILABLE:
            logger.warning("reportlab not available - PDF generation disabled")
            return

        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""

        # Helper function to safely add styles
        def safe_add_style(name, style):
            if name not in self.styles:
                self.styles.add(style)

        # Enhanced title style
        safe_add_style(
            "CustomTitle",
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Title"],
                fontSize=24,
                textColor=colors.HexColor("#1a1a1a"),
                spaceAfter=12,
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
            ),
        )

        # Executive summary header
        safe_add_style(
            "ExecutiveHeader",
            ParagraphStyle(
                name="ExecutiveHeader",
                parent=self.styles["Heading1"],
                fontSize=18,
                textColor=colors.HexColor("#0052cc"),
                spaceBefore=18,
                spaceAfter=12,
                fontName="Helvetica-Bold",
                borderWidth=2,
                borderColor=colors.HexColor("#0052cc"),
                borderPadding=8,
            ),
        )

        # Section header
        safe_add_style(
            "SectionHeader",
            ParagraphStyle(
                name="SectionHeader",
                parent=self.styles["Heading2"],
                fontSize=14,
                textColor=colors.HexColor("#2c3e50"),
                spaceBefore=12,
                spaceAfter=6,
                fontName="Helvetica-Bold",
                leftIndent=0,
                borderWidth=1,
                borderColor=colors.HexColor("#ecf0f1"),
                borderPadding=4,
            ),
        )

        # Highlight box style
        safe_add_style(
            "HighlightBox",
            ParagraphStyle(
                name="HighlightBox",
                parent=self.styles["Normal"],
                fontSize=11,
                textColor=colors.HexColor("#2c3e50"),
                spaceBefore=6,
                spaceAfter=6,
                leftIndent=12,
                rightIndent=12,
                borderWidth=1,
                borderColor=colors.HexColor("#3498db"),
                borderPadding=8,
                backColor=colors.HexColor("#ebf3fd"),
            ),
        )

        # Risk warning style
        safe_add_style(
            "RiskWarning",
            ParagraphStyle(
                name="RiskWarning",
                parent=self.styles["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#c0392b"),
                spaceBefore=6,
                spaceAfter=6,
                leftIndent=12,
                rightIndent=12,
                borderWidth=1,
                borderColor=colors.HexColor("#e74c3c"),
                borderPadding=6,
                backColor=colors.HexColor("#fadbd8"),
            ),
        )

        # Metrics style
        safe_add_style(
            "MetricsText",
            ParagraphStyle(
                name="MetricsText",
                parent=self.styles["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#34495e"),
                spaceBefore=3,
                spaceAfter=3,
                leftIndent=6,
            ),
        )

        # Subtitle style
        safe_add_style(
            "CustomSubtitle",
            ParagraphStyle(
                name="CustomSubtitle",
                parent=self.styles["Heading2"],
                fontSize=16,
                textColor=colors.HexColor("#444444"),
                spaceBefore=6,
                spaceAfter=12,
                alignment=TA_CENTER,
            ),
        )

        # Analysis text style
        safe_add_style(
            "AnalysisText",
            ParagraphStyle(
                name="AnalysisText",
                parent=self.styles["BodyText"],
                fontSize=11,
                alignment=TA_JUSTIFY,
                spaceBefore=6,
                spaceAfter=6,
            ),
        )

    def generate_report(
        self, recommendations: List[Dict], report_type: str = "synthesis", include_charts: Optional[List[str]] = None
    ) -> str:
        """
        Generate PDF report from recommendations

        Args:
            recommendations: List of investment recommendations
            report_type: Type of report (synthesis, weekly, etc.)
            include_charts: List of chart paths to include

        Returns:
            Path to generated PDF report or empty string if reportlab unavailable
        """
        if not REPORTLAB_AVAILABLE:
            logger.warning("Cannot generate PDF report - reportlab not available")
            return ""

        # Create filename with symbol-based naming
        filename = self._generate_filename(recommendations, report_type)
        filepath = self.output_dir / filename

        # Create document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter if self.config.page_size == "letter" else A4,
            rightMargin=self.config.margin,
            leftMargin=self.config.margin,
            topMargin=self.config.margin,
            bottomMargin=self.config.margin,
        )

        # Build content
        story = []

        # Add title page
        story.extend(self._create_title_page(report_type))

        # Add executive summary
        story.extend(self._create_executive_summary(recommendations))

        # Add detailed analysis for each symbol
        for rec in recommendations:
            story.append(PageBreak())
            story.extend(self._create_symbol_analysis(rec, include_charts))

        # Add portfolio summary
        if len(recommendations) > 1:
            story.append(PageBreak())
            story.extend(self._create_portfolio_summary(recommendations))

        # Add charts section if provided
        if include_charts and self.config.include_charts:
            story.append(PageBreak())
            story.extend(self._create_charts_section(include_charts))

        # Add Tier 2 interpretation appendix
        story.append(PageBreak())
        story.extend(self._create_tier2_interpretation_appendix())

        # Add disclaimer
        if self.config.include_disclaimer:
            story.append(PageBreak())
            story.extend(self._create_disclaimer())

        # Add appendix section with definitions
        story.append(PageBreak())
        story.append(Paragraph("Appendix B - Investment Analysis Definitions", self.styles["Heading1"]))
        story.append(Spacer(1, 12))

        # Time Horizon Definitions
        story.append(Paragraph("Time Horizon Definitions", self.styles["Heading2"]))
        story.append(Spacer(1, 6))

        time_horizon_data = [
            ["Time Horizon", "Duration", "Description"],
            [
                "SHORT_TERM",
                "3-12 months",
                "Tactical positions based on near-term catalysts, earnings, or technical patterns. Suitable for active trading strategies.",
            ],
            [
                "MEDIUM_TERM",
                "1-3 years",
                "Strategic positions based on business fundamentals, industry trends, and moderate growth expectations. Balanced risk-reward profile.",
            ],
            [
                "LONG_TERM",
                "3+ years",
                "Investment positions based on long-term competitive advantages, secular trends, and compound growth potential. Lower volatility tolerance required.",
            ],
        ]

        time_horizon_table = Table(time_horizon_data, colWidths=[1.5 * inch, 1.2 * inch, 4 * inch])
        time_horizon_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                ]
            )
        )

        story.append(time_horizon_table)
        story.append(Spacer(1, 12))

        # Position Size Definitions
        story.append(Paragraph("Position Size Definitions", self.styles["Heading2"]))
        story.append(Spacer(1, 6))

        position_size_data = [
            ["Position Size", "Portfolio Weight", "Risk Profile"],
            ["LARGE", "5-10%", "High conviction positions with strong fundamental and technical alignment"],
            ["MODERATE", "2-5%", "Standard positions with good risk-adjusted return potential"],
            ["SMALL", "0.5-2%", "Speculative or higher-risk positions with asymmetric upside"],
            ["AVOID", "0%", "Positions with significant downside risk or poor risk-reward ratio"],
        ]

        position_size_table = Table(position_size_data, colWidths=[1.5 * inch, 1.5 * inch, 3.5 * inch])
        position_size_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                ]
            )
        )

        story.append(position_size_table)
        story.append(Spacer(1, 12))

        # Additional Reference Definitions
        story.append(Paragraph("Recommendation Definitions", self.styles["Heading2"]))
        story.append(Spacer(1, 6))

        recommendation_data = [
            ["Recommendation", "Description"],
            ["STRONG_BUY", "Exceptional opportunity with significant upside potential and strong conviction"],
            ["BUY", "Attractive investment with positive risk-adjusted return expectations"],
            ["HOLD", "Fair value with limited upside/downside, suitable for existing positions"],
            ["SELL", "Overvalued or deteriorating fundamentals warrant position reduction"],
            ["STRONG_SELL", "Significant downside risk or fundamental concerns warrant avoidance"],
        ]

        recommendation_table = Table(recommendation_data, colWidths=[1.5 * inch, 5 * inch])
        recommendation_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                ]
            )
        )

        story.append(recommendation_table)

        # Build PDF with custom canvas for page numbers
        doc.build(story, canvasmaker=NumberedCanvas)

        logger.info(f"📄 Generated PDF report: {filepath}")
        return str(filepath)

    def _create_title_page(self, report_type: str) -> List:
        """Create title page with comprehensive legal disclaimer"""
        elements = []

        # Add title
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(self.config.title, self.styles["CustomTitle"]))
        elements.append(Paragraph(self.config.subtitle, self.styles["CustomSubtitle"]))

        # Add report type
        report_type_text = report_type.replace("_", " ").title()
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"{report_type_text} Report", self.styles["Heading2"]))

        # Add date
        elements.append(Spacer(1, 0.2 * inch))
        date_text = datetime.now().strftime("%B %d, %Y")
        elements.append(Paragraph(date_text, self.styles["Normal"]))

        # Add comprehensive legal disclaimer
        elements.append(Spacer(1, 0.4 * inch))

        # Create prominent disclaimer style
        disclaimer_style = ParagraphStyle(
            "TitleDisclaimer",
            parent=self.styles["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#cc0000"),
            borderWidth=2,
            borderColor=colors.HexColor("#cc0000"),
            borderPadding=10,
            borderRadius=4,
            backColor=colors.HexColor("#fff5f5"),
            spaceAfter=12,
            alignment=TA_CENTER,
        )

        legal_disclaimer = """
        <b>IMPORTANT LEGAL DISCLAIMER</b><br/><br/>
        
        This report is generated entirely by artificial intelligence (AI) using Large Language Models (LLMs) 
        and is provided for <b>EDUCATIONAL and TESTING PURPOSES ONLY</b>.<br/><br/>
        
        <b>NOT INVESTMENT ADVICE:</b> The author is NOT a licensed investment advisor, financial planner, 
        broker-dealer, or any other financial professional. This report does NOT constitute investment advice, 
        financial advice, trading advice, or any other type of professional advice.<br/><br/>
        
        <b>AI-GENERATED CONTENT:</b> All analysis, recommendations, and insights in this report are 
        generated by AI systems which may contain errors, inaccuracies, hallucinations, or biases. 
        The AI has no fiduciary duty to you and cannot guarantee accuracy.<br/><br/>
        
        <b>NO WARRANTIES:</b> This report is provided "AS IS" without any warranties of any kind, 
        either express or implied, including but not limited to warranties of accuracy, completeness, 
        merchantability, or fitness for a particular purpose.<br/><br/>
        
        <b>USE AT YOUR OWN RISK:</b> Any investment decisions made based on this report are entirely at 
        your own risk. You could lose all of your invested capital. Past performance is not indicative 
        of future results. Always consult with qualified, licensed financial professionals before making 
        any investment decisions.<br/><br/>
        
        <b>NO LIABILITY:</b> The creators, developers, and operators of this AI system assume no liability 
        for any losses, damages, or consequences arising from the use of this report.
        """

        elements.append(Paragraph(legal_disclaimer, disclaimer_style))

        # Add author with clarification
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"AI System: {self.config.author}", self.styles["Normal"]))
        elements.append(
            Paragraph("<b>For Educational Testing Only - Not Professional Investment Advice</b>", self.styles["Normal"])
        )

        return elements

    def _create_executive_summary(self, recommendations: List[Dict]) -> List:
        """Create enhanced executive summary with visual elements"""
        elements = []

        elements.append(Paragraph("Executive Summary", self.styles["ExecutiveHeader"]))
        elements.append(Spacer(1, 0.3 * inch))

        # Portfolio Overview Section
        total_symbols = len(recommendations)
        buy_count = sum(1 for r in recommendations if "BUY" in r.get("recommendation", "").upper())
        sell_count = sum(1 for r in recommendations if "SELL" in r.get("recommendation", "").upper())
        hold_count = total_symbols - buy_count - sell_count

        avg_score = (
            sum(r.get("overall_score", 5.0) for r in recommendations) / total_symbols if total_symbols > 0 else 0
        )

        # Create portfolio overview table with visual elements
        overview_data = [
            ["Portfolio Snapshot", "", ""],
            ["Total Securities Analyzed", str(total_symbols), ""],
            [
                "Buy Recommendations",
                str(buy_count),
                f"{(buy_count/total_symbols*100):.0f}%" if total_symbols > 0 else "0%",
            ],
            [
                "Hold Recommendations",
                str(hold_count),
                f"{(hold_count/total_symbols*100):.0f}%" if total_symbols > 0 else "0%",
            ],
            [
                "Sell Recommendations",
                str(sell_count),
                f"{(sell_count/total_symbols*100):.0f}%" if total_symbols > 0 else "0%",
            ],
            ["Average Investment Score", f"{avg_score:.1f}/10", self._get_score_rating(avg_score)],
        ]

        overview_table = Table(overview_data, colWidths=[2.5 * inch, 1 * inch, 1 * inch])
        overview_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0052cc")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 10),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                ]
            )
        )

        elements.append(overview_table)
        elements.append(Spacer(1, 0.3 * inch))

        # Top Recommendations Section with Visual Cards
        if recommendations:
            elements.append(Paragraph("Top Investment Opportunities", self.styles["SectionHeader"]))
            elements.append(Spacer(1, 0.2 * inch))

            # Sort by overall score and get top 3
            top_recs = sorted(recommendations, key=lambda x: x.get("overall_score", 0), reverse=True)[:3]

            # Create visual cards for top recommendations
            for i, rec in enumerate(top_recs):
                symbol = rec.get("symbol", "N/A")
                overall_score = rec.get("overall_score", 0)
                recommendation = rec.get("recommendation", "N/A")
                confidence = rec.get("confidence", "MEDIUM")
                current_price = rec.get("current_price", 0)
                price_target = rec.get("price_target", 0)

                # Create recommendation card data
                card_data = [
                    [f"#{i+1}", symbol, recommendation, f"{overall_score:.1f}/10"],
                    [
                        "Price",
                        f"${current_price:.2f}" if current_price else "N/A",
                        "Target",
                        f"${price_target:.2f}" if price_target else "N/A",
                    ],
                    [
                        "Confidence",
                        confidence,
                        "Upside",
                        f"{((price_target/current_price-1)*100):.1f}%" if current_price and price_target else "N/A",
                    ],
                ]

                card_table = Table(card_data, colWidths=[0.5 * inch, 1 * inch, 1 * inch, 1 * inch])

                # Style based on recommendation
                if "BUY" in recommendation.upper():
                    header_color = colors.HexColor("#28a745")
                elif "SELL" in recommendation.upper():
                    header_color = colors.HexColor("#dc3545")
                else:
                    header_color = colors.HexColor("#6c757d")

                card_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), header_color),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 11),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                            ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
                            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                            ("FONTSIZE", (0, 1), (-1, -1), 9),
                        ]
                    )
                )

                elements.append(card_table)
                elements.append(Spacer(1, 0.1 * inch))

        # Risk Assessment Summary
        elements.append(Spacer(1, 0.2 * inch))
        high_risk_count = sum(1 for r in recommendations if r.get("overall_score", 5) < 4)

        if high_risk_count > 0:
            risk_text = f"⚠️ <b>Risk Alert:</b> {high_risk_count} securities show elevated risk profiles. Review detailed analysis before investment decisions."
            elements.append(Paragraph(risk_text, self.styles["RiskWarning"]))
        else:
            elements.append(
                Paragraph(
                    "✅ <b>Portfolio Risk:</b> All analyzed securities meet acceptable risk thresholds.",
                    self.styles["HighlightBox"],
                )
            )

        return elements

    def _get_score_rating(self, score: float) -> str:
        """Convert numeric score to rating"""
        if score >= 8:
            return "Excellent"
        elif score >= 6:
            return "Good"
        elif score >= 4:
            return "Fair"
        else:
            return "Poor"

    def _create_symbol_analysis(self, recommendation: Dict, include_charts: Optional[List[str]] = None) -> List:
        """Create detailed analysis for a single symbol"""
        elements = []

        symbol = recommendation.get("symbol", "N/A")

        # Header
        elements.append(Paragraph(f"{symbol} Analysis", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.1 * inch))

        # Data Quality Badge
        if recommendation.get("data_quality_detailed"):
            quality = recommendation["data_quality_detailed"]
            grade = quality["grade"]

            # Color-coded badge
            grade_colors = {"A": "green", "B": "blue", "C": "orange", "D": "red", "F": "darkred"}
            color = grade_colors.get(grade, "black")

            quality_text = f"""
            <para align="right">
            <b><font color="{color}" size="14">Data Quality: {grade}</font></b><br/>
            <font size="10">{quality['overall_score']:.1f}% complete</font>
            </para>
            """
            elements.append(Paragraph(quality_text, self.styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))

        # Get comprehensive analysis data first for enhanced scoring
        comprehensive_data = self._get_comprehensive_analysis_data(symbol)

        # Scores table - Enhanced with SEC comprehensive scores when available
        scores_data = [["Metric", "Score", "Rating"]]

        # Core synthesis scores
        scores_data.extend(
            [
                [
                    "Overall Score",
                    f"{recommendation.get('overall_score', 0):.1f}/10",
                    self._get_rating(recommendation.get("overall_score", 0)),
                ],
                [
                    "Fundamental Score",
                    f"{recommendation.get('fundamental_score', 0):.1f}/10",
                    self._get_rating(recommendation.get("fundamental_score", 0)),
                ],
                [
                    "Technical Score",
                    f"{recommendation.get('technical_score', 0):.1f}/10",
                    self._get_rating(recommendation.get("technical_score", 0)),
                ],
            ]
        )

        # Financial statement component scores
        scores_data.extend(
            [
                [
                    "Income Statement",
                    f"{recommendation.get('income_score', 0):.1f}/10",
                    self._get_rating(recommendation.get("income_score", 0)),
                ],
                [
                    "Cash Flow",
                    f"{recommendation.get('cashflow_score', 0):.1f}/10",
                    self._get_rating(recommendation.get("cashflow_score", 0)),
                ],
                [
                    "Balance Sheet",
                    f"{recommendation.get('balance_score', 0):.1f}/10",
                    self._get_rating(recommendation.get("balance_score", 0)),
                ],
            ]
        )

        # Investment characteristic scores
        scores_data.extend(
            [
                [
                    "Growth Score",
                    f"{recommendation.get('growth_score', 0):.1f}/10",
                    self._get_rating(recommendation.get("growth_score", 0)),
                ],
                [
                    "Value Score",
                    f"{recommendation.get('value_score', 0):.1f}/10",
                    self._get_rating(recommendation.get("value_score", 0)),
                ],
            ]
        )

        # Quality scores - prioritize comprehensive analysis when available, remove duplicates
        if comprehensive_data.get("business_quality_score") is not None:
            bq_score = comprehensive_data["business_quality_score"]
            # Handle dict format for scores (e.g., {"score": 8.5, "explanation": "..."})
            if isinstance(bq_score, dict):
                bq_score = bq_score.get("score", 0)
            scores_data.append(["Business Quality", f"{float(bq_score):.1f}/10", self._get_rating(float(bq_score))])
        else:
            scores_data.append(
                [
                    "Business Quality",
                    f"{recommendation.get('business_quality_score', 0):.1f}/10",
                    self._get_rating(recommendation.get("business_quality_score", 0)),
                ]
            )

        if comprehensive_data.get("data_quality_score") is not None:
            dq_score = comprehensive_data["data_quality_score"]
            # Handle dict format for scores
            if isinstance(dq_score, dict):
                dq_score = dq_score.get("score", 0)
            scores_data.append(["Data Quality", f"{float(dq_score):.1f}/10", self._get_rating(float(dq_score))])
        else:
            scores_data.append(
                [
                    "Data Quality",
                    f"{recommendation.get('data_quality_score', 0):.1f}/10",
                    self._get_rating(recommendation.get("data_quality_score", 0)),
                ]
            )

        scores_table = Table(scores_data, colWidths=[2.5 * inch, 1.5 * inch, 1.5 * inch])
        scores_table.setStyle(
            TableStyle(
                [
                    # Header styling
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0052cc")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("TOPPADDING", (0, 0), (-1, 0), 8),
                    # Body styling
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 10),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                    # Score-based coloring for the Score column
                    ("TEXTCOLOR", (1, 1), (1, -1), colors.black),
                    ("TOPPADDING", (0, 1), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                ]
            )
        )

        # Add conditional formatting for scores
        for i, row in enumerate(scores_data[1:], 1):  # Skip header
            try:
                score_text = row[1]
                if "/" in score_text:
                    score = float(score_text.split("/")[0])
                    if score >= 8:
                        scores_table.setStyle(
                            TableStyle([("BACKGROUND", (1, i), (1, i), colors.HexColor("#d4edda"))])
                        )  # Light green
                    elif score >= 6:
                        scores_table.setStyle(
                            TableStyle([("BACKGROUND", (1, i), (1, i), colors.HexColor("#fff3cd"))])
                        )  # Light yellow
                    elif score < 4:
                        scores_table.setStyle(
                            TableStyle([("BACKGROUND", (1, i), (1, i), colors.HexColor("#f8d7da"))])
                        )  # Light red
            except (ValueError, IndexError):
                pass

        elements.append(scores_table)
        elements.append(Spacer(1, 0.2 * inch))

        # Technical Analysis Summary (using structured data from direct extraction)
        tech_summary = self._create_technical_summary(recommendation)
        if tech_summary:
            elements.extend(tech_summary)
            elements.append(Spacer(1, 0.2 * inch))

        # Entry/Exit Signal Analysis Section
        entry_exit_section = self._create_entry_exit_section(recommendation)
        if entry_exit_section:
            elements.extend(entry_exit_section)
            elements.append(Spacer(1, 0.2 * inch))

        # Investment recommendation
        elements.append(Paragraph("<b>Investment Recommendation</b>", self.styles["Heading3"]))
        rec_text = f"""
        <b>Recommendation:</b> {recommendation.get('recommendation', 'N/A')}<br/>
        <b>Confidence Level:</b> {recommendation.get('confidence', 'N/A')}<br/>
        <b>Time Horizon:</b> {recommendation.get('time_horizon', 'N/A')}<br/>
        <b>Position Size:</b> {recommendation.get('position_size', 'N/A')}<br/>
        """
        elements.append(Paragraph(rec_text, self.styles["AnalysisText"]))

        # Price targets
        if recommendation.get("price_target"):
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Price Analysis</b>", self.styles["Heading3"]))
            price_text = f"""
            <b>Current Price:</b> ${recommendation.get('current_price', 0):.2f}<br/>
            <b>Price Target:</b> ${recommendation.get('price_target', 0):.2f}<br/>
            <b>Upside Potential:</b> {((recommendation.get('price_target', 0) / max(recommendation.get('current_price', 1), 0.01) - 1) * 100) if recommendation.get('current_price', 0) > 0 else 0:.1f}%<br/>
            <b>Stop Loss:</b> ${recommendation.get('stop_loss', 0):.2f}<br/>
            """
            elements.append(Paragraph(price_text, self.styles["AnalysisText"]))

        # Support/Resistance Levels
        if sr_levels := recommendation.get("support_resistance"):
            if sr_levels.get("support_levels") or sr_levels.get("resistance_levels"):
                elements.append(Spacer(1, 0.1 * inch))
                elements.append(Paragraph("<b>Key Price Levels</b>", self.styles["Heading3"]))

                sr_text = []
                current_price = sr_levels.get("current_price", recommendation.get("current_price", 0))
                sr_text.append(f"<b>Current Price:</b> ${current_price:.2f}")

                if nearest_support := sr_levels.get("nearest_support"):
                    distance = sr_levels.get("distance_to_support", 0)
                    sr_text.append(f"<b>Nearest Support:</b> ${nearest_support:.2f} ({distance:.1f}% below)")

                if nearest_resistance := sr_levels.get("nearest_resistance"):
                    distance = sr_levels.get("distance_to_resistance", 0)
                    sr_text.append(f"<b>Nearest Resistance:</b> ${nearest_resistance:.2f} ({distance:.1f}% above)")

                # Show all levels
                if support_levels := sr_levels.get("support_levels"):
                    support_str = ", ".join([f"${s:.2f}" for s in support_levels])
                    sr_text.append(f"<b>Support Levels:</b> {support_str}")

                if resistance_levels := sr_levels.get("resistance_levels"):
                    resistance_str = ", ".join([f"${r:.2f}" for r in resistance_levels])
                    sr_text.append(f"<b>Resistance Levels:</b> {resistance_str}")

                elements.append(Paragraph("<br/>".join(sr_text), self.styles["AnalysisText"]))

        # Investment thesis - prioritize SEC comprehensive analysis
        investment_thesis = self._get_comprehensive_investment_thesis(
            symbol, recommendation.get("investment_thesis", "")
        )
        if investment_thesis:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Investment Thesis</b>", self.styles["Heading3"]))
            # Convert markdown to HTML for proper rendering
            investment_thesis_html = self._markdown_to_html(investment_thesis)
            elements.append(Paragraph(investment_thesis_html, self.styles["AnalysisText"]))

        # Key insights - prioritize SEC comprehensive analysis
        insights_to_show = comprehensive_data.get("key_insights") or recommendation.get("key_insights", [])
        insights_source = "SEC Comprehensive" if comprehensive_data.get("key_insights") else "Synthesis"

        if insights_to_show:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph(f"<b>Key Insights ({insights_source})</b>", self.styles["Heading3"]))
            for insight in insights_to_show[:5]:  # Show top 5 insights
                elements.append(Paragraph(f"• {insight}", self.styles["AnalysisText"]))

        # Red Flags - automated detection
        if recommendation.get("red_flags"):
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("⚠️ <b>Red Flags Detected</b>", self.styles["Heading3"]))

            for flag in recommendation["red_flags"]:
                severity_emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}
                emoji = severity_emoji.get(flag["severity"], "⚠️")

                flag_text = (
                    f"{emoji} <b>[{flag['severity'].upper()}]</b> {flag['description']}<br/><i>{flag['detail']}</i>"
                )
                elements.append(Paragraph(flag_text, self.styles["AnalysisText"]))
                elements.append(Spacer(1, 0.05 * inch))

        # Multi-Year Historical Trends
        if multi_year_trends := recommendation.get("multi_year_trends"):
            if multi_year_metrics := multi_year_trends.get("metrics"):
                elements.append(Spacer(1, 0.2 * inch))
                # Dynamic year count based on actual data
                years_analyzed = len(multi_year_trends.get("data", []))
                elements.append(
                    Paragraph(
                        f"📈 <b>Multi-Year Historical Trends ({years_analyzed} Years)</b>", self.styles["Heading3"]
                    )
                )

                trend_items = []

                # Revenue CAGR
                if revenue_cagr := multi_year_metrics.get("revenue_cagr"):
                    if isinstance(revenue_cagr, (int, float)):
                        color = "green" if revenue_cagr > 0 else "red"
                        trend_items.append(f"<b>Revenue CAGR:</b> <font color='{color}'>{revenue_cagr:.1f}%</font>")

                # Earnings CAGR
                if earnings_cagr := multi_year_metrics.get("earnings_cagr"):
                    if isinstance(earnings_cagr, (int, float)):
                        color = "green" if earnings_cagr > 0 else "red"
                        trend_items.append(f"<b>Earnings CAGR:</b> <font color='{color}'>{earnings_cagr:.1f}%</font>")

                # Volatility
                if revenue_volatility := multi_year_metrics.get("revenue_volatility"):
                    if isinstance(revenue_volatility, (int, float)):
                        trend_items.append(f"<b>Revenue Volatility:</b> {revenue_volatility:.1f}%")

                # Business pattern
                if cyclical_pattern := multi_year_metrics.get("cyclical_pattern"):
                    pattern_emoji = {
                        "Highly Cyclical": "🔄",
                        "Moderately Cyclical": "↗️",
                        "Stable Growth": "📊",
                        "Volatile": "⚡",
                    }
                    emoji = pattern_emoji.get(cyclical_pattern, "📈")
                    trend_items.append(f"<b>Business Pattern:</b> {emoji} {cyclical_pattern}")

                # Trend direction
                if trend_direction := multi_year_metrics.get("trend_direction"):
                    direction_emoji = {"up": "↗️", "down": "↘️", "flat": "→"}
                    emoji = direction_emoji.get(trend_direction, "→")
                    trend_items.append(f"<b>Trend Direction:</b> {emoji} {trend_direction.upper()}")

                for item in trend_items:
                    elements.append(Paragraph(f"• {item}", self.styles["AnalysisText"]))

        # Tier 3: DCF Valuation Analysis
        if dcf_valuation := recommendation.get("dcf_valuation"):
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("💰 <b>DCF Valuation Analysis</b>", self.styles["Heading3"]))

            fair_value = dcf_valuation.get("fair_value_per_share", 0)
            current_price = dcf_valuation.get("current_price", 0)
            upside = dcf_valuation.get("upside_downside_pct", 0)
            assessment = dcf_valuation.get("valuation_assessment", "Unknown")

            # Color code based on valuation
            assessment_colors = {
                "Significantly Undervalued": "darkgreen",
                "Undervalued": "green",
                "Fairly Valued": "blue",
                "Overvalued": "orange",
                "Significantly Overvalued": "red",
            }
            color = assessment_colors.get(assessment, "black")

            # Main valuation summary
            dcf_items = [
                f"<b>Fair Value (DCF):</b> ${fair_value:.2f}",
                f"<b>Current Price:</b> ${current_price:.2f}",
                f"<b>Upside/Downside:</b> <font color='{color}' size='12'>{upside:+.1f}%</font>",
                f"<b>Assessment:</b> <font color='{color}'><b>{assessment}</b></font>",
            ]

            for item in dcf_items:
                elements.append(Paragraph(f"• {item}", self.styles["AnalysisText"]))

            elements.append(Spacer(1, 0.1 * inch))

            # Key Assumptions
            if assumptions := dcf_valuation.get("assumptions"):
                elements.append(Paragraph("<b>Key DCF Assumptions:</b>", self.styles["AnalysisText"]))
                assumption_items = [
                    f"WACC (Discount Rate): {assumptions.get('wacc', 'N/A')}%",
                    f"Terminal Growth Rate: {assumptions.get('terminal_growth_rate', 'N/A')}%",
                    f"Projection Period: {assumptions.get('projection_years', 'N/A')} years",
                    f"Latest Free Cash Flow: ${assumptions.get('latest_fcf', 'N/A'):.1f}M",
                ]
                for item in assumption_items:
                    elements.append(Paragraph(f"  • {item}", self.styles["AnalysisText"]))

            elements.append(Spacer(1, 0.1 * inch))

            # Enterprise & Equity Value
            if enterprise_value := dcf_valuation.get("enterprise_value"):
                equity_value = dcf_valuation.get("equity_value", 0)
                elements.append(Paragraph("<b>Valuation Components:</b>", self.styles["AnalysisText"]))
                elements.append(
                    Paragraph(f"  • Enterprise Value: ${enterprise_value:.2f}B", self.styles["AnalysisText"])
                )
                elements.append(Paragraph(f"  • Equity Value: ${equity_value:.2f}B", self.styles["AnalysisText"]))

            # Sensitivity Analysis Table
            if sensitivity := dcf_valuation.get("sensitivity_analysis"):
                elements.append(Spacer(1, 0.1 * inch))
                elements.append(
                    Paragraph("<b>Sensitivity Analysis - Fair Value Table:</b>", self.styles["AnalysisText"])
                )
                elements.append(Paragraph("<i>(Varies WACC and Terminal Growth Rate)</i>", self.styles["AnalysisText"]))
                elements.append(Spacer(1, 0.05 * inch))

                # Get sensitivity data
                wacc_values = sensitivity.get("wacc_values", [])
                tgr_values = sensitivity.get("terminal_growth_rates", [])
                fair_values = sensitivity.get("fair_values", [])

                # Build table data
                table_data = [["TGR \\ WACC"] + wacc_values]
                for i, tgr in enumerate(tgr_values):
                    row = [tgr] + [f"${v:.2f}" if v > 0 else "N/A" for v in fair_values[i]]
                    table_data.append(row)

                # Create table
                sens_table = Table(table_data)
                sens_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ]
                    )
                )
                elements.append(sens_table)

        # Tier 3: Recession Performance Analysis
        if recession_perf := recommendation.get("recession_performance"):
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("📉 <b>Recession Performance Analysis</b>", self.styles["Heading3"]))

            defensive_score = recession_perf.get("defensive_score", 5.0)
            defensive_rating = recession_perf.get("defensive_rating", "Unknown")

            # Color code based on defensive characteristics
            rating_colors = {
                "Very Defensive": "darkgreen",
                "Defensive": "green",
                "Neutral": "blue",
                "Cyclical": "orange",
                "Highly Cyclical": "red",
            }
            color = rating_colors.get(defensive_rating, "black")

            # Main defensive rating
            elements.append(
                Paragraph(
                    f"<b>Defensive Rating:</b> <font color='{color}' size='12'><b>{defensive_rating}</b></font> ({defensive_score}/10)",
                    self.styles["AnalysisText"],
                )
            )
            elements.append(Spacer(1, 0.1 * inch))

            # Historical crisis performance
            if crisis_performance := recession_perf.get("crisis_performance"):
                elements.append(Paragraph("<b>Historical Crisis Performance:</b>", self.styles["AnalysisText"]))
                elements.append(Spacer(1, 0.05 * inch))

                for crisis_id, crisis_data in crisis_performance.items():
                    perf = crisis_data.get("performance", {})
                    crisis_name = crisis_data.get("name", crisis_id)

                    elements.append(Paragraph(f"<b>{crisis_name}:</b>", self.styles["AnalysisText"]))

                    # Revenue decline
                    if revenue_decline := perf.get("revenue_decline_pct"):
                        decline_color = (
                            "green" if revenue_decline >= -5 else ("orange" if revenue_decline >= -15 else "red")
                        )
                        elements.append(
                            Paragraph(
                                f"  • Revenue Decline: <font color='{decline_color}'>{revenue_decline:+.1f}%</font>",
                                self.styles["AnalysisText"],
                            )
                        )

                    # Earnings stability
                    if earnings_stability := perf.get("earnings_stability"):
                        stability_emoji = {"high": "✅", "medium": "⚠️", "low": "❌"}
                        emoji = stability_emoji.get(earnings_stability, "•")
                        elements.append(
                            Paragraph(
                                f"  • Earnings Stability: {emoji} {earnings_stability.title()}",
                                self.styles["AnalysisText"],
                            )
                        )
                        if neg_quarters := perf.get("negative_earnings_quarters"):
                            elements.append(
                                Paragraph(
                                    f"    ({neg_quarters} loss quarter{'s' if neg_quarters != 1 else ''})",
                                    self.styles["AnalysisText"],
                                )
                            )

                    # Cash position
                    if cash_change := perf.get("cash_position_change"):
                        cash_color = "green" if cash_change >= 0 else "red"
                        elements.append(
                            Paragraph(
                                f"  • Cash Position Change: <font color='{cash_color}'>{cash_change:+.1f}%</font>",
                                self.styles["AnalysisText"],
                            )
                        )

                    # Recovery speed
                    quarters = perf.get("quarters_to_recover", "N/A")
                    if quarters != "N/A":
                        recovery_color = "green" if quarters <= 4 else ("orange" if quarters <= 8 else "red")
                        elements.append(
                            Paragraph(
                                f"  • Recovery Time: <font color='{recovery_color}'>{quarters} quarters</font>",
                                self.styles["AnalysisText"],
                            )
                        )
                    else:
                        elements.append(Paragraph(f"  • Recovery Time: Data incomplete", self.styles["AnalysisText"]))

                    elements.append(Spacer(1, 0.05 * inch))

            # Interpretation
            elements.append(Paragraph("<b>Interpretation:</b>", self.styles["AnalysisText"]))
            if defensive_score >= 8.0:
                interpretation = "This company demonstrates exceptional resilience during economic downturns. Strong defensive characteristics make it suitable for conservative portfolios and recessionary hedging."
            elif defensive_score >= 6.5:
                interpretation = "This company shows solid defensive qualities. It maintains stability during crises better than most peers, suitable for balanced portfolios."
            elif defensive_score >= 5.0:
                interpretation = "This company exhibits neutral defensive characteristics. Performance during recessions aligns with market averages."
            elif defensive_score >= 3.0:
                interpretation = "This company is cyclical and vulnerable to economic downturns. Consider reducing exposure during recession fears."
            else:
                interpretation = "This company is highly cyclical with significant recession vulnerability. Revenue and earnings are heavily impacted by economic cycles. Best suited for risk-tolerant investors during expansionary periods."

            elements.append(Paragraph(f"<i>{interpretation}</i>", self.styles["AnalysisText"]))

        # Tier 3: Insider Trading Analysis
        if insider_trading := recommendation.get("insider_trading"):
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("📊 <b>Insider Trading Analysis</b>", self.styles["Heading3"]))

            sentiment_score = insider_trading.get("sentiment_score", 5.0)
            sentiment_rating = insider_trading.get("sentiment_rating", "Neutral")
            buy_count = insider_trading.get("buy_count", 0)
            sell_count = insider_trading.get("sell_count", 0)
            total_buy_value = insider_trading.get("total_buy_value", 0.0)
            total_sell_value = insider_trading.get("total_sell_value", 0.0)
            period_days = insider_trading.get("period_days", 180)

            # Check if data is available
            data_available = insider_trading.get("data_available", False)

            if data_available:
                # Color code based on insider sentiment
                sentiment_colors = {
                    "Very Bullish": "darkgreen",
                    "Bullish": "green",
                    "Neutral": "blue",
                    "Bearish": "orange",
                    "Very Bearish": "red",
                }
                color = sentiment_colors.get(sentiment_rating, "black")

                # Main sentiment rating
                elements.append(
                    Paragraph(
                        f"<b>Insider Sentiment (Last {period_days} days):</b> <font color='{color}' size='12'><b>{sentiment_rating}</b></font> ({sentiment_score}/10)",
                        self.styles["AnalysisText"],
                    )
                )
                elements.append(Spacer(1, 0.1 * inch))

                # Transaction summary
                elements.append(Paragraph("<b>Transaction Summary:</b>", self.styles["AnalysisText"]))
                elements.append(Spacer(1, 0.05 * inch))

                # Buy/Sell counts
                total_transactions = buy_count + sell_count
                if total_transactions > 0:
                    buy_pct = (buy_count / total_transactions) * 100
                    buy_color = "green" if buy_count > sell_count else ("orange" if buy_count == sell_count else "red")

                    elements.append(
                        Paragraph(
                            f"  • Purchases: <font color='{buy_color}'><b>{buy_count}</b></font> ({buy_pct:.0f}%)",
                            self.styles["AnalysisText"],
                        )
                    )
                    elements.append(
                        Paragraph(f"  • Sales: {sell_count} ({100-buy_pct:.0f}%)", self.styles["AnalysisText"])
                    )
                    elements.append(Spacer(1, 0.05 * inch))

                    # Transaction values
                    if total_buy_value > 0 or total_sell_value > 0:
                        elements.append(
                            Paragraph(f"  • Total Buy Value: ${total_buy_value:,.0f}", self.styles["AnalysisText"])
                        )
                        elements.append(
                            Paragraph(f"  • Total Sell Value: ${total_sell_value:,.0f}", self.styles["AnalysisText"])
                        )

                        # Net insider activity
                        net_value = total_buy_value - total_sell_value
                        net_color = "green" if net_value > 0 else "red"
                        elements.append(
                            Paragraph(
                                f"  • Net Insider Activity: <font color='{net_color}'>${net_value:+,.0f}</font>",
                                self.styles["AnalysisText"],
                            )
                        )
                        elements.append(Spacer(1, 0.05 * inch))

                # Unusual patterns
                if unusual_patterns := insider_trading.get("unusual_patterns"):
                    if unusual_patterns:
                        elements.append(Paragraph("<b>Unusual Patterns Detected:</b>", self.styles["AnalysisText"]))
                        for pattern in unusual_patterns[:5]:  # Limit to 5 patterns
                            elements.append(Paragraph(f"  • {pattern}", self.styles["AnalysisText"]))
                        elements.append(Spacer(1, 0.05 * inch))

                # Interpretation
                elements.append(Paragraph("<b>Interpretation:</b>", self.styles["AnalysisText"]))
                if sentiment_score >= 7.5:
                    interpretation = "Strong insider buying activity suggests high confidence from company executives. This bullish signal often precedes positive developments."
                elif sentiment_score >= 6.0:
                    interpretation = "Moderate insider buying indicates positive sentiment from management. Insiders appear to view the current price as attractive."
                elif sentiment_score >= 4.0:
                    interpretation = "Balanced insider activity with no clear directional bias. Insiders appear neutral on current valuation."
                elif sentiment_score >= 2.5:
                    interpretation = "Elevated insider selling activity suggests reduced confidence or profit-taking. Monitor for potential concerns."
                else:
                    interpretation = "Heavy insider selling is a bearish signal. Executives may have concerns about near-term prospects or believe shares are overvalued."

                elements.append(Paragraph(f"<i>{interpretation}</i>", self.styles["AnalysisText"]))
            else:
                # Data not yet available
                note = insider_trading.get("note", "Insider trading data not yet available")
                elements.append(Paragraph(f"<i>{note}</i>", self.styles["AnalysisText"]))
                elements.append(
                    Paragraph(
                        "<i>Note: Full insider trading analysis requires SEC Form 4 filing integration, which will be available in a future update.</i>",
                        self.styles["AnalysisText"],
                    )
                )

        # Tier 3: News Sentiment Analysis
        if news_sentiment := recommendation.get("news_sentiment"):
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("📰 <b>News Sentiment Analysis</b>", self.styles["Heading3"]))

            sentiment_score = news_sentiment.get("sentiment_score", 5.0)
            sentiment_rating = news_sentiment.get("sentiment_rating", "Neutral")
            article_count = news_sentiment.get("article_count", 0)
            period_days = news_sentiment.get("period_days", 7)

            # Check if data is available
            data_available = news_sentiment.get("data_available", False)

            if data_available:
                # Color code based on news sentiment
                sentiment_colors = {
                    "Very Positive": "darkgreen",
                    "Positive": "green",
                    "Neutral": "blue",
                    "Negative": "orange",
                    "Very Negative": "red",
                }
                color = sentiment_colors.get(sentiment_rating, "black")

                # Main sentiment rating
                elements.append(
                    Paragraph(
                        f"<b>Sentiment (Last {period_days} days):</b> <font color='{color}' size='12'><b>{sentiment_rating}</b></font> ({sentiment_score}/10)",
                        self.styles["AnalysisText"],
                    )
                )
                elements.append(Spacer(1, 0.1 * inch))

                # Article breakdown
                positive_count = news_sentiment.get("positive_count", 0)
                negative_count = news_sentiment.get("negative_count", 0)
                neutral_count = news_sentiment.get("neutral_count", 0)

                if article_count > 0:
                    elements.append(
                        Paragraph(
                            f"<b>Article Breakdown</b> ({article_count} total articles):", self.styles["AnalysisText"]
                        )
                    )
                    elements.append(Spacer(1, 0.05 * inch))

                    # Calculate percentages
                    pos_pct = (positive_count / article_count) * 100 if article_count > 0 else 0
                    neg_pct = (negative_count / article_count) * 100 if article_count > 0 else 0
                    neu_pct = (neutral_count / article_count) * 100 if article_count > 0 else 0

                    elements.append(
                        Paragraph(
                            f"  • Positive: <font color='green'><b>{positive_count}</b></font> ({pos_pct:.0f}%)",
                            self.styles["AnalysisText"],
                        )
                    )
                    elements.append(
                        Paragraph(
                            f"  • Negative: <font color='red'><b>{negative_count}</b></font> ({neg_pct:.0f}%)",
                            self.styles["AnalysisText"],
                        )
                    )
                    elements.append(
                        Paragraph(f"  • Neutral: {neutral_count} ({neu_pct:.0f}%)", self.styles["AnalysisText"])
                    )
                    elements.append(Spacer(1, 0.05 * inch))

                # Sentiment trend
                if sentiment_trend := news_sentiment.get("sentiment_trend"):
                    direction = sentiment_trend.get("direction", "stable")
                    strength = sentiment_trend.get("strength", 0)

                    trend_emojis = {"improving": "📈", "declining": "📉", "stable": "➡️"}
                    trend_emoji = trend_emojis.get(direction, "➡️")

                    elements.append(
                        Paragraph(
                            f"<b>Trend:</b> {trend_emoji} {direction.title()} (strength: {strength}/5.0)",
                            self.styles["AnalysisText"],
                        )
                    )
                    elements.append(Spacer(1, 0.05 * inch))

                # Interpretation
                elements.append(Paragraph("<b>Interpretation:</b>", self.styles["AnalysisText"]))
                if sentiment_score >= 8.0:
                    interpretation = "Very positive news coverage suggests strong market confidence. This optimistic sentiment often supports price momentum and can attract new investors."
                elif sentiment_score >= 6.5:
                    interpretation = "Positive news sentiment indicates favorable market perception. The company appears to be executing well and generating positive media attention."
                elif sentiment_score >= 4.5:
                    interpretation = "Neutral news sentiment with balanced coverage. No clear directional bias from recent news flow."
                elif sentiment_score >= 2.5:
                    interpretation = "Negative news sentiment may create near-term headwinds. Monitor for potential concerns mentioned in recent coverage."
                else:
                    interpretation = "Very negative news coverage is a significant concern. Recent developments appear to have damaged market sentiment. Caution advised."

                elements.append(Paragraph(f"<i>{interpretation}</i>", self.styles["AnalysisText"]))
            else:
                # Data not yet available
                note = news_sentiment.get("note", "News sentiment data not yet available")
                elements.append(Paragraph(f"<i>{note}</i>", self.styles["AnalysisText"]))
                elements.append(
                    Paragraph(
                        "<i>Note: Full news sentiment analysis requires NewsAPI integration, which will be available in a future update.</i>",
                        self.styles["AnalysisText"],
                    )
                )

        # Multi-Dimensional Risk Assessment
        if risk_scores := recommendation.get("risk_scores"):
            elements.append(Spacer(1, 0.2 * inch))
            overall_risk = risk_scores.get("overall_risk", "N/A")
            risk_rating = risk_scores.get("risk_rating", "Unknown")

            # Color code the risk rating
            rating_colors = {
                "Very Low": "green",
                "Low": "green",
                "Medium": "orange",
                "High": "red",
                "Very High": "darkred",
            }
            color = rating_colors.get(risk_rating, "black")

            elements.append(Paragraph(f"⚠️ <b>Multi-Dimensional Risk Assessment</b>", self.styles["Heading3"]))
            elements.append(
                Paragraph(
                    f"<b>Overall Risk Score:</b> <font color='{color}' size='12'>{overall_risk}/10 ({risk_rating})</font>",
                    self.styles["AnalysisText"],
                )
            )
            elements.append(Spacer(1, 0.1 * inch))

            risk_items = []

            # Financial Health Risk
            if financial_risk := risk_scores.get("financial_health_risk"):
                risk_color = "green" if financial_risk < 4 else ("orange" if financial_risk < 7 else "red")
                risk_items.append(
                    f"<b>Financial Health:</b> <font color='{risk_color}'>{financial_risk}/10</font> - Leverage, liquidity, solvency"
                )

            # Market Risk
            if market_risk := risk_scores.get("market_risk"):
                risk_color = "green" if market_risk < 4 else ("orange" if market_risk < 7 else "red")
                risk_items.append(
                    f"<b>Market Risk:</b> <font color='{risk_color}'>{market_risk}/10</font> - Volatility, beta, drawdowns"
                )

            # Operational Risk
            if operational_risk := risk_scores.get("operational_risk"):
                risk_color = "green" if operational_risk < 4 else ("orange" if operational_risk < 7 else "red")
                risk_items.append(
                    f"<b>Operational Risk:</b> <font color='{risk_color}'>{operational_risk}/10</font> - Cash flow quality, working capital"
                )

            # Business Model Risk
            if business_risk := risk_scores.get("business_model_risk"):
                risk_color = "green" if business_risk < 4 else ("orange" if business_risk < 7 else "red")
                risk_items.append(
                    f"<b>Business Model Risk:</b> <font color='{risk_color}'>{business_risk}/10</font> - Margin stability, competitive moat"
                )

            # Growth Risk
            if growth_risk := risk_scores.get("growth_risk"):
                risk_color = "green" if growth_risk < 4 else ("orange" if growth_risk < 7 else "red")
                risk_items.append(
                    f"<b>Growth Risk:</b> <font color='{risk_color}'>{growth_risk}/10</font> - Growth sustainability, capital efficiency"
                )

            for item in risk_items:
                elements.append(Paragraph(f"• {item}", self.styles["AnalysisText"]))

        # Peer Performance Leaderboard
        if peer_leaderboard := recommendation.get("peer_leaderboard"):
            if peers := peer_leaderboard.get("peers"):
                elements.append(Spacer(1, 0.2 * inch))
                industry = peer_leaderboard.get("industry", "Industry")
                elements.append(
                    Paragraph(f"📊 <b>Peer Performance Leaderboard - {industry}</b>", self.styles["Heading3"])
                )

                # Find target company
                target_peer = next((p for p in peers if p.get("is_target")), None)
                if target_peer:
                    overall_rank = target_peer.get("overall_rank", "N/A")
                    total_peers = len(peers)
                    percentile = round((1 - (overall_rank / total_peers)) * 100) if overall_rank != "N/A" else 0
                    rank_color = "green" if percentile >= 75 else ("orange" if percentile >= 50 else "red")
                    elements.append(
                        Paragraph(
                            f"<b>Overall Rank:</b> <font color='{rank_color}'>#{overall_rank} of {total_peers}</font> ({percentile}th percentile)",
                            self.styles["AnalysisText"],
                        )
                    )
                    elements.append(Spacer(1, 0.1 * inch))

                # Create leaderboard table (show top 10)
                table_data = [["Rank", "Symbol", "Rev Growth", "Profit Margin", "ROE", "Market Cap"]]

                for peer in peers[:10]:  # Show top 10 peers
                    rank = peer.get("overall_rank", "-")
                    symbol = peer.get("symbol", "")
                    growth = peer.get("revenue_growth", 0)
                    margin = peer.get("profit_margin", 0)
                    roe = peer.get("roe", 0)
                    mcap = peer.get("market_cap", 0)

                    # Format values
                    growth_str = f"{growth:.1f}%" if growth else "N/A"
                    margin_str = f"{margin:.1f}%" if margin else "N/A"
                    roe_str = f"{roe:.1f}%" if roe else "N/A"
                    mcap_str = f"${mcap/1e9:.1f}B" if mcap and mcap > 1e9 else (f"${mcap/1e6:.0f}M" if mcap else "N/A")

                    # Highlight target company
                    if peer.get("is_target"):
                        symbol = f"**{symbol}**"

                    table_data.append([str(rank), symbol, growth_str, margin_str, roe_str, mcap_str])

                # Create table with styling
                leaderboard_table = Table(
                    table_data, colWidths=[0.6 * inch, 0.9 * inch, 1.0 * inch, 1.1 * inch, 0.8 * inch, 1.1 * inch]
                )
                leaderboard_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E86AB")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 10),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("FONTSIZE", (0, 1), (-1, -1), 9),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
                        ]
                    )
                )

                elements.append(leaderboard_table)

        # Key risks - prioritize SEC comprehensive analysis
        risks_to_show = comprehensive_data.get("key_risks") or recommendation.get("key_risks", [])
        risks_source = "SEC Comprehensive" if comprehensive_data.get("key_risks") else "Synthesis"

        if risks_to_show:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph(f"<b>Key Risks ({risks_source})</b>", self.styles["Heading3"]))
            for risk in risks_to_show[:5]:  # Show top 5 risks
                elements.append(Paragraph(f"• {risk}", self.styles["AnalysisText"]))

        # SEC Trend Analysis (if available)
        if comprehensive_data.get("trend_analysis"):
            trend_analysis = comprehensive_data["trend_analysis"]
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Trend Analysis (SEC)</b>", self.styles["Heading3"]))

            trend_items = []
            if trend_analysis.get("revenue_trend"):
                trend_items.append(f"Revenue Trend: {trend_analysis['revenue_trend']}")
            if trend_analysis.get("margin_trend"):
                trend_items.append(f"Margin Trend: {trend_analysis['margin_trend']}")
            if trend_analysis.get("cash_flow_trend"):
                trend_items.append(f"Cash Flow Trend: {trend_analysis['cash_flow_trend']}")

            for item in trend_items:
                elements.append(Paragraph(f"• {item}", self.styles["AnalysisText"]))

        # LLM Thinking and Details Sections
        sec_thinking, tech_thinking = self._get_llm_thinking_details(symbol)

        # SEC Fundamental Analysis Thinking
        if sec_thinking:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>SEC Fundamental Analysis Thinking</b>", self.styles["Heading3"]))
            # Convert markdown to HTML for proper rendering
            sec_thinking_html = self._markdown_to_html(sec_thinking)
            elements.append(Paragraph(sec_thinking_html, self.styles["AnalysisText"]))

        # Technical Analysis Thinking
        if tech_thinking:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Technical Analysis Thinking</b>", self.styles["Heading3"]))
            # Convert markdown to HTML for proper rendering
            tech_thinking_html = self._markdown_to_html(tech_thinking)
            elements.append(Paragraph(tech_thinking_html, self.styles["AnalysisText"]))

        # Analysis thinking process
        if recommendation.get("analysis_thinking"):
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Synthesis Analysis Reasoning</b>", self.styles["Heading3"]))
            # Convert markdown to HTML for proper rendering
            analysis_thinking_html = self._markdown_to_html(recommendation.get("analysis_thinking", ""))
            elements.append(Paragraph(analysis_thinking_html, self.styles["AnalysisText"]))

        # Synthesis details
        if recommendation.get("synthesis_details"):
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Synthesis Methodology</b>", self.styles["Heading3"]))
            # Convert markdown to HTML for proper rendering
            synthesis_details_html = self._markdown_to_html(recommendation.get("synthesis_details", ""))
            elements.append(Paragraph(synthesis_details_html, self.styles["AnalysisText"]))

        # Tier 4: Monte Carlo Probabilistic Forecasting
        if monte_carlo := recommendation.get("monte_carlo_results"):
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("📊 <b>Monte Carlo Probabilistic Forecast (1 Year)</b>", self.styles["Heading3"]))
            elements.append(Spacer(1, 0.1 * inch))

            # Summary statistics
            mean_price = monte_carlo.mean_price
            median_price = monte_carlo.median_price
            current_price_mc = monte_carlo.current_price
            prob_profit = monte_carlo.probability_profit
            var_95 = monte_carlo.var_95

            # Color code based on probability
            prob_color = "green" if prob_profit >= 0.6 else ("orange" if prob_profit >= 0.4 else "red")

            mc_summary = f"""
            Based on 10,000 simulations using Geometric Brownian Motion:<br/>
            <b>Expected Price:</b> ${mean_price:.2f} (Median: ${median_price:.2f})<br/>
            <b>Probability of Profit:</b> <font color='{prob_color}'>{prob_profit:.1%}</font><br/>
            <b>Value at Risk (95%):</b> ${var_95:.2f} maximum expected loss<br/>
            """
            elements.append(Paragraph(mc_summary, self.styles["AnalysisText"]))

            # Scenario analysis table
            if hasattr(monte_carlo, "scenarios") and monte_carlo.scenarios:
                elements.append(Spacer(1, 0.1 * inch))
                elements.append(Paragraph("<b>Price Target Scenarios:</b>", self.styles["AnalysisText"]))

                scenario_data = [["Scenario", "Probability", "Price Target", "Return"]]

                for name, scenario in monte_carlo.scenarios.items():
                    scenario_name = scenario.get("scenario", name.replace("_", " ").title())
                    prob = f"{scenario.get('probability', 0):.0f}%"
                    price = f"${scenario.get('price_target', 0):.2f}"
                    return_pct = scenario.get("return_pct", 0)
                    return_str = f"{return_pct:+.1f}%"

                    # Color code returns
                    if return_pct > 0:
                        return_str = f'<font color="green">{return_str}</font>'
                    elif return_pct < 0:
                        return_str = f'<font color="red">{return_str}</font>'

                    scenario_data.append([scenario_name, prob, price, Paragraph(return_str, self.styles["Normal"])])

                scenario_table = Table(scenario_data, colWidths=[1.8 * inch, 1.0 * inch, 1.2 * inch, 1.0 * inch])
                scenario_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E86AB")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 10),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("FONTSIZE", (0, 1), (-1, -1), 9),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
                        ]
                    )
                )

                elements.append(scenario_table)

            # Interpretation
            elements.append(Spacer(1, 0.1 * inch))
            if prob_profit >= 0.65:
                interpretation = "The probabilistic model suggests favorable odds for positive returns, with most simulated paths ending above the current price. This indicates bullish momentum potential."
            elif prob_profit >= 0.50:
                interpretation = "The simulation shows balanced probabilities, suggesting a neutral outlook. Price movement could go either direction with roughly equal likelihood."
            elif prob_profit >= 0.35:
                interpretation = "The model indicates elevated downside risk, with more simulated paths ending below current levels. Consider defensive positioning or risk management strategies."
            else:
                interpretation = "Significant downside risk is indicated by the simulation results. The majority of probabilistic outcomes suggest price decline. Caution strongly advised."

            elements.append(Paragraph(f"<i>{interpretation}</i>", self.styles["AnalysisText"]))

        # Tier 4: Chart Pattern Recognition
        if chart_patterns := recommendation.get("chart_patterns"):
            if chart_patterns.get("pattern_count", 0) > 0:
                elements.append(Spacer(1, 0.2 * inch))
                elements.append(Paragraph("📈 <b>Chart Pattern Analysis</b>", self.styles["Heading3"]))
                elements.append(Spacer(1, 0.1 * inch))

                patterns = chart_patterns.get("patterns", [])
                days_analyzed = chart_patterns.get("days_analyzed", 0)

                elements.append(
                    Paragraph(
                        f"<b>Detected {len(patterns)} chart pattern(s)</b> from {days_analyzed} days of price data:",
                        self.styles["AnalysisText"],
                    )
                )

                # Pattern details table
                pattern_data = [["Pattern", "Type", "Confidence", "Price Target", "Direction"]]

                for pattern in patterns[:5]:  # Show top 5 patterns
                    pattern_name = pattern.pattern_type.value.replace("_", " ").title()
                    confidence = f"{pattern.confidence * 100:.0f}%"
                    price_target = f"${pattern.price_target:.2f}"
                    direction = pattern.direction.capitalize()

                    # Color code direction
                    if direction == "Bullish":
                        direction_colored = '<font color="green">●</font> Bullish'
                    elif direction == "Bearish":
                        direction_colored = '<font color="red">●</font> Bearish'
                    else:
                        direction_colored = '<font color="gray">●</font> Neutral'

                    pattern_data.append(
                        [
                            pattern_name,
                            pattern.pattern_type.value.split("_")[0].title(),  # First word
                            confidence,
                            price_target,
                            Paragraph(direction_colored, self.styles["Normal"]),
                        ]
                    )

                pattern_table = Table(
                    pattern_data, colWidths=[1.5 * inch, 1.0 * inch, 0.9 * inch, 1.1 * inch, 1.0 * inch]
                )
                pattern_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0052cc")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 10),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("FONTSIZE", (0, 1), (-1, -1), 9),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
                        ]
                    )
                )

                elements.append(pattern_table)

                # Pattern interpretation
                elements.append(Spacer(1, 0.1 * inch))
                bullish_count = sum(1 for p in patterns if p.direction == "bullish")
                bearish_count = sum(1 for p in patterns if p.direction == "bearish")

                if bullish_count > bearish_count:
                    interpretation = f"Pattern analysis shows a bullish bias with {bullish_count} bullish pattern(s) detected. These patterns suggest potential upward price movement and may indicate accumulation."
                elif bearish_count > bullish_count:
                    interpretation = f"Pattern analysis shows a bearish bias with {bearish_count} bearish pattern(s) detected. These patterns suggest potential downward pressure and may indicate distribution."
                else:
                    interpretation = "Pattern analysis shows mixed signals with balanced bullish and bearish formations. This suggests consolidation or indecision in the market."

                elements.append(Paragraph(f"<i>{interpretation}</i>", self.styles["AnalysisText"]))

                # Individual pattern descriptions
                for pattern in patterns[:3]:  # Detail top 3 patterns
                    elements.append(Spacer(1, 0.05 * inch))
                    elements.append(
                        Paragraph(
                            f"• <b>{pattern.pattern_type.value.replace('_', ' ').title()}:</b> {pattern.description}",
                            self.styles["AnalysisText"],
                        )
                    )

        # Add technical chart if available
        if include_charts and self.config.include_charts:
            tech_chart = f"{symbol}_technical_analysis.png"
            for chart_path in include_charts:
                if tech_chart in chart_path and Path(chart_path).exists():
                    elements.append(Spacer(1, 0.2 * inch))
                    elements.append(Paragraph("<b>Technical Analysis Chart</b>", self.styles["Heading3"]))
                    img = Image(chart_path, width=6 * inch, height=4 * inch)
                    elements.append(img)
                    break

        return elements

    def _create_portfolio_summary(self, recommendations: List[Dict]) -> List:
        """Create portfolio summary section"""
        elements = []

        elements.append(Paragraph("Portfolio Summary", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Sort by score
        sorted_recs = sorted(recommendations, key=lambda x: x.get("overall_score", 0), reverse=True)

        # Create summary table
        table_data = [["Symbol", "Recommendation", "Overall Score", "Target Return", "Position Size"]]

        for rec in sorted_recs:
            symbol = rec.get("symbol", "N/A")
            recommendation = rec.get("recommendation", "N/A")
            score = f"{rec.get('overall_score', 0):.1f}"

            # Calculate target return
            current = rec.get("current_price", 0) or 0
            target = rec.get("price_target", 0) or 0
            target_return = ((target / current - 1) * 100) if current and current > 0 and target and target > 0 else 0
            target_return_str = f"{target_return:+.1f}%" if target and target > 0 and current and current > 0 else "N/A"

            position = rec.get("position_size", "N/A")

            table_data.append([symbol, recommendation, score, target_return_str, position])

        # Create table
        summary_table = Table(table_data, colWidths=[1.2 * inch, 1.8 * inch, 1.2 * inch, 1.3 * inch, 1.5 * inch])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 11),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("FONTSIZE", (0, 1), (-1, -1), 10),
                ]
            )
        )

        elements.append(summary_table)

        return elements

    def _create_charts_section(self, chart_paths: List[str]) -> List:
        """Create section with additional charts"""
        elements = []

        elements.append(Paragraph("Analysis Charts", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Add 3D fundamental chart
        for chart_path in chart_paths:
            if "3d_fundamental" in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>3D Fundamental Analysis</b>", self.styles["Heading3"]))
                img = Image(chart_path, width=6 * inch, height=4.5 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break

        # Add 2D technical vs fundamental chart
        for chart_path in chart_paths:
            if "2d_technical_fundamental" in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>Technical vs Fundamental Analysis</b>", self.styles["Heading3"]))
                img = Image(chart_path, width=6 * inch, height=4.5 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break

        # Add growth vs value chart
        for chart_path in chart_paths:
            if "growth_value" in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>Growth vs Value Positioning</b>", self.styles["Heading3"]))
                img = Image(chart_path, width=6 * inch, height=4.5 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break

        # Add score history chart
        for chart_path in chart_paths:
            if "score_history" in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>Investment Score History</b>", self.styles["Heading3"]))
                elements.append(
                    Paragraph("Track how our investment assessment has evolved over time", self.styles["Normal"])
                )
                img = Image(chart_path, width=6.5 * inch, height=5 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break

        # Add valuation comparison chart
        for chart_path in chart_paths:
            if "valuation_comparison" in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>Valuation vs Peers</b>", self.styles["Heading3"]))
                elements.append(
                    Paragraph("Comparison of valuation multiples against peer group median", self.styles["Normal"])
                )
                img = Image(chart_path, width=6 * inch, height=4 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break

        # Add quarterly revenue trend chart
        for chart_path in chart_paths:
            if "revenue_trend" in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>Quarterly Revenue Trend</b>", self.styles["Heading3"]))
                elements.append(
                    Paragraph("Revenue trend showing quarter-over-quarter growth rates", self.styles["Normal"])
                )
                img = Image(chart_path, width=6.5 * inch, height=4 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break

        # Add quarterly profitability chart
        for chart_path in chart_paths:
            if "profitability" in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>Quarterly Profitability Analysis</b>", self.styles["Heading3"]))
                elements.append(Paragraph("Net income and profit margins by quarter", self.styles["Normal"]))
                img = Image(chart_path, width=6.5 * inch, height=5 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break

        # Add quarterly cash flow chart
        for chart_path in chart_paths:
            if "cash_flow" in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>Quarterly Operating Cash Flow</b>", self.styles["Heading3"]))
                elements.append(
                    Paragraph("Operating cash flow with quarter-over-quarter growth", self.styles["Normal"])
                )
                img = Image(chart_path, width=6.5 * inch, height=4 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break

        # Add multi-year trends chart
        for chart_path in chart_paths:
            if "multi_year_trends" in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>Multi-Year Historical Trends</b>", self.styles["Heading3"]))
                elements.append(
                    Paragraph("Multi-year revenue and earnings trends with CAGR analysis", self.styles["Normal"])
                )
                img = Image(chart_path, width=7 * inch, height=5 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break

        # Add risk radar chart
        for chart_path in chart_paths:
            if "risk_radar" in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>Multi-Dimensional Risk Assessment</b>", self.styles["Heading3"]))
                elements.append(
                    Paragraph(
                        "5-dimensional risk analysis across financial health, market, operational, business model, and growth factors",
                        self.styles["Normal"],
                    )
                )
                img = Image(chart_path, width=6 * inch, height=6 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break

        # Add competitive positioning matrix
        for chart_path in chart_paths:
            if "competitive_positioning" in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>Competitive Positioning Matrix</b>", self.styles["Heading3"]))
                elements.append(
                    Paragraph(
                        "Competitive position relative to industry peers based on revenue growth and profit margins",
                        self.styles["Normal"],
                    )
                )
                img = Image(chart_path, width=7 * inch, height=6 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break

        # Add volume profile chart
        for chart_path in chart_paths:
            if "volume_profile" in chart_path and Path(chart_path).exists():
                elements.append(Paragraph("<b>Volume Profile Analysis</b>", self.styles["Heading3"]))
                elements.append(
                    Paragraph(
                        "Volume distribution across price levels showing Point of Control (POC) and value area (70% of volume)",
                        self.styles["Normal"],
                    )
                )
                img = Image(chart_path, width=7 * inch, height=6 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
                break

        return elements

    def _create_disclaimer(self) -> List:
        """Create disclaimer section"""
        elements = []

        elements.append(Paragraph("Full Legal Disclaimer", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.1 * inch))

        disclaimer_text = """
        <b>AI-GENERATED REPORT - NOT INVESTMENT ADVICE</b><br/><br/>
        
        This report is generated entirely by artificial intelligence using Large Language Models (LLMs) and is 
        provided for <b>educational and testing purposes only</b>. The creators are NOT licensed investment advisors, 
        broker-dealers, or financial professionals.<br/><br/>
        
        <b>IMPORTANT WARNINGS:</b><br/>
        • This is NOT investment advice or a recommendation to buy, sell, or hold any securities<br/>
        • All content is AI-generated and may contain errors, inaccuracies, or hallucinations<br/>
        • Past performance is not indicative of future results<br/>
        • You could lose all invested capital - investments carry substantial risk<br/>
        • The AI system has no fiduciary duty and cannot guarantee accuracy<br/><br/>
        
        <b>REQUIRED ACTIONS BEFORE INVESTING:</b><br/>
        • Conduct your own thorough research and due diligence<br/>
        • Consult with licensed, qualified financial advisors<br/>
        • Verify all information independently<br/>
        • Consider your personal financial situation and risk tolerance<br/><br/>
        
        <b>NO LIABILITY:</b> The creators, developers, and operators of InvestiGator assume no liability whatsoever 
        for any losses, damages, or consequences arising from the use of this report. Use of this report is entirely 
        at your own risk.<br/><br/>
        
        <b>REGULATORY NOTICE:</b> This report has not been reviewed or approved by any regulatory authority. 
        It is not intended for distribution in jurisdictions where such distribution would be unlawful.<br/><br/>
        
        Generated by InvestiGator - AI-Powered Investment Research System<br/>
        For Educational Testing Only - Not Professional Investment Advice
        """

        elements.append(Paragraph(disclaimer_text, self.styles["AnalysisText"]))

        return elements

    def _create_tier2_interpretation_appendix(self) -> List:
        """Create Tier 2 enhancements interpretation appendix"""
        elements = []

        # Main heading
        elements.append(Paragraph("Appendix A - Tier 2 Enhancements Interpretation Guide", self.styles["Heading1"]))
        elements.append(Spacer(1, 0.15 * inch))

        # Introduction
        intro_text = """
        This appendix provides guidance on interpreting the <b>5 Tier 2 Enhancements</b> included in this report.
        These institutional-grade features provide deeper insights beyond traditional analysis.
        """
        elements.append(Paragraph(intro_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Enhancement #1: Multi-Year Historical Trends
        elements.append(Paragraph("📈 1. Multi-Year Historical Trends (5 Years)", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        myt_text = """
        <b>What It Shows:</b> Revenue and earnings compound annual growth rate (CAGR), volatility, and business patterns over 5 years.<br/><br/>

        <b>Revenue CAGR Interpretation:</b><br/>
        • <b>&gt;15%:</b> Exceptional growth - high-growth stock potential<br/>
        • <b>10-15%:</b> Strong growth - above market average<br/>
        • <b>5-10%:</b> Moderate growth - stable performer<br/>
        • <b>0-5%:</b> Slow growth - mature company<br/>
        • <b>&lt;0%:</b> Declining - potential value trap or turnaround<br/><br/>

        <b>Revenue Volatility:</b><br/>
        • <b>&lt;10%:</b> Stable, predictable (utilities, staples)<br/>
        • <b>10-20%:</b> Moderate volatility (most companies)<br/>
        • <b>&gt;20%:</b> Highly cyclical (energy, commodities)<br/><br/>

        <b>Investment Signal:</b><br/>
        • <font color="green"><b>Best Case:</b></font> Revenue CAGR 15%+, Earnings CAGR &gt; Revenue CAGR, Volatility &lt;10% → <b>STRONG BUY</b><br/>
        • <font color="red"><b>Caution:</b></font> Revenue CAGR &lt;3%, Negative Earnings CAGR, Volatility &gt;25% → <b>AVOID</b>
        """
        elements.append(Paragraph(myt_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Enhancement #2: Multi-Dimensional Risk Scoring
        elements.append(Paragraph("⚠️ 2. Multi-Dimensional Risk Scoring", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        risk_text = """
        <b>What It Shows:</b> Quantified risk across 5 dimensions on 0-10 scale (lower = safer).<br/><br/>

        <b>Five Risk Dimensions:</b><br/>
        • <b>Financial Health Risk:</b> Balance sheet strength, debt levels, liquidity<br/>
        • <b>Market Risk:</b> Stock volatility, beta, correlation to market<br/>
        • <b>Operational Risk:</b> Cash flow quality, earnings reliability<br/>
        • <b>Business Model Risk:</b> Margin stability, competitive advantages<br/>
        • <b>Growth Risk:</b> Growth sustainability, capital efficiency<br/><br/>

        <b>Overall Risk Rating → Investment Strategy:</b><br/>
        • <b>Very Low (0-3.0):</b> Conservative core holding, 8-12% position size, hold through volatility<br/>
        • <b>Low (3.0-4.5):</b> High-quality growth, 5-10% position, long-term hold<br/>
        • <b>Medium (4.5-6.0):</b> Moderate risk/reward, 3-7% position, monitor quarterly<br/>
        • <b>High (6.0-7.5):</b> Speculative, 1-3% position, tight stop loss<br/>
        • <b>Very High (7.5-10):</b> Avoid or &lt;1% position, high loss probability<br/><br/>

        <b>Radar Chart Usage:</b><br/>
        • <font color="green"><b>Green Zone (0-3):</b></font> Low risk - safe for conservative portfolios<br/>
        • <font color="orange"><b>Yellow Zone (3-6):</b></font> Medium risk - requires monitoring<br/>
        • <font color="red"><b>Red Zone (6-10):</b></font> High risk - requires higher return to justify
        """
        elements.append(Paragraph(risk_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Enhancement #3: Competitive Positioning Matrix
        elements.append(Paragraph("🎯 3. Competitive Positioning Matrix", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        matrix_text = """
        <b>What It Shows:</b> Company position vs peers on Revenue Growth (X-axis) and Profit Margin (Y-axis).<br/><br/>

        <b>Four Quadrants:</b><br/>
        • <b>Top Right (High Growth + High Margin):</b> ⭐ <font color="green"><b>Best positioned - Champions</b></font><br/>
          → Strong buy candidates, premium valuation justified, 7-12% position<br/>
          → Examples: AAPL, MSFT, GOOGL in their sectors<br/><br/>

        • <b>Top Left (Low Growth + High Margin):</b> <font color="blue"><b>Mature cash cows</b></font><br/>
          → Dividend candidates, value/income play, 5-8% position<br/>
          → Examples: Utilities, mature consumer staples<br/><br/>

        • <b>Bottom Right (High Growth + Low Margin):</b> <font color="orange"><b>Growth at any cost</b></font><br/>
          → Higher risk, speculative growth, 2-5% position<br/>
          → Monitor margin trajectory closely<br/><br/>

        • <b>Bottom Left (Low Growth + Low Margin):</b> 🚩 <font color="red"><b>Avoid zone</b></font><br/>
          → Structural challenges, 0% position unless deep value<br/>
          → High probability of value destruction<br/><br/>

        <b>How to Use:</b><br/>
        1. Locate target company (red star)<br/>
        2. Identify quadrant position<br/>
        3. Check distance from median lines (further = better if in top right)<br/>
        4. Isolated from peers = unique business model (investigate why)
        """
        elements.append(Paragraph(matrix_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Enhancement #4: Peer Performance Leaderboard
        elements.append(Paragraph("🏆 4. Peer Performance Leaderboard", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        leaderboard_text = """
        <b>What It Shows:</b> Ranked table of peers by composite score across revenue growth, profit margin, ROE, and market cap.<br/><br/>

        <b>Percentile Ranking → Quality Assessment:</b><br/>
        • <b>90th-100th (Top 10%):</b> Industry leader, premium valuation justified, core holding<br/>
        • <b>75th-90th (Top 25%):</b> Above-average performer, buy with conviction<br/>
        • <b>50th-75th (Above Median):</b> Solid performer, hold or accumulate on dips<br/>
        • <b>25th-50th (Below Median):</b> Underperformer, needs catalyst, wait for turnaround<br/>
        • <b>0-25th (Bottom 25%):</b> Laggard, avoid unless deep value play<br/><br/>

        <b>Composite Score Analysis:</b><br/>
        Low composite score (1-5) = Best overall performance across all metrics<br/>
        • <b>Balanced Leader:</b> Excellent across all dimensions → Core position<br/>
        • <b>Growth Without Profit:</b> #1 growth but #12 margin → Speculative only<br/>
        • <b>Profitable But Stagnant:</b> #2 margin but #14 growth → Income play only<br/><br/>

        <b>Strategy:</b><br/>
        1. Check overall rank - Top 25%? → Strong buy candidate<br/>
        2. Analyze rank consistency across metrics<br/>
        3. Compare to valuation - #1 rank at same P/E as #10? → UNDERVALUED<br/>
        4. Track momentum - Rank improving quarter-over-quarter? → Positive catalyst
        """
        elements.append(Paragraph(leaderboard_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Enhancement #5: Volume Profile Analysis
        elements.append(Paragraph("📊 5. Volume Profile Analysis", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        volume_text = """
        <b>What It Shows:</b> 90-day volume distribution at each price level, identifying fair value zones and support/resistance.<br/><br/>

        <b>Key Components:</b><br/>
        • <b>Point of Control (POC):</b> Price with highest volume = fair value consensus<br/>
        • <b>Value Area:</b> Price range containing 70% of volume = accepted value zone<br/>
        • <b>High Volume Nodes (HVN):</b> Strong support/resistance zones<br/>
        • <b>Low Volume Nodes (LVN):</b> Weak support/resistance, expect fast moves<br/><br/>

        <b>Current Price vs POC:</b><br/>
        • <b>Price ABOVE POC (+1% to +5%):</b> Expensive vs consensus, wait for pullback<br/>
        • <b>Price AT POC (±1%):</b> Fair value zone, good entry for long-term holds<br/>
        • <b>Price BELOW POC (-1% to -5%):</b> Cheap vs consensus, BUY opportunity<br/>
        • <b>Price FAR ABOVE POC (&gt;5%):</b> Extended/overbought, take profits<br/>
        • <b>Price FAR BELOW POC (&gt;5%):</b> Oversold, scale in with tight stops<br/><br/>

        <b>Value Area Strategy:</b><br/>
        • <b>Inside Value Area:</b> Fair value, wait for extremes (buy near low, sell near high)<br/>
        • <b>Above Value Area:</b> Extended/expensive, take profits or wait for pullback<br/>
        • <b>Below Value Area:</b> Discounted/attractive, scale in with targets at value area low<br/><br/>

        <b>Trading Applications:</b><br/>
        • <b>Mean Reversion:</b> When price &gt;3% from POC, expect move back toward POC<br/>
        • <b>Breakout Confirmation:</b> Volume increasing at new price levels validates move<br/>
        • <b>Support/Resistance:</b> HVN acts as strong support/resistance, LVN offers weak resistance
        """
        elements.append(Paragraph(volume_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Summary recommendation
        elements.append(Paragraph("Summary: Combining All Enhancements", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        summary_text = """
        <b>Ideal Investment Scenario:</b><br/>
        ✅ Multi-Year: Revenue CAGR 15%+, low volatility, stable growth pattern<br/>
        ✅ Risk: Overall risk score &lt;4.5 (Low), all dimensions in green/yellow zones<br/>
        ✅ Positioning: Top-right quadrant (high growth + high margin)<br/>
        ✅ Leaderboard: Top 25% rank (75th percentile+), improving momentum<br/>
        ✅ Volume: Price at or below POC, inside value area for entry<br/>
        <b>→ Action: STRONG BUY with 7-12% position size</b><br/><br/>

        <b>Red Flag Scenario:</b><br/>
        🚩 Multi-Year: Negative CAGR, high volatility (&gt;25%), declining trend<br/>
        🚩 Risk: Overall risk score &gt;7.5 (Very High), multiple dimensions in red zone<br/>
        🚩 Positioning: Bottom-left quadrant (low growth + low margin)<br/>
        🚩 Leaderboard: Bottom 25% rank, declining momentum<br/>
        🚩 Volume: Price far above POC (&gt;5%), extended move<br/>
        <b>→ Action: AVOID - High probability of capital loss</b><br/><br/>

        <i>Use these enhancements together to build conviction and size positions appropriately.
        Higher quality across all dimensions justifies larger position sizes and higher valuations.</i>
        """
        elements.append(Paragraph(summary_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.3 * inch))

        # ==========================
        # TIER 1 ENHANCEMENTS
        # ==========================

        elements.append(Paragraph("Tier 1 Enhancements Interpretation", self.styles["Heading1"]))
        elements.append(Spacer(1, 0.15 * inch))

        intro_tier1 = """
        These foundational enhancements provide critical context and warning signals for investment decisions.
        """
        elements.append(Paragraph(intro_tier1, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Enhancement #1: Historical Score Trends
        elements.append(Paragraph("📈 1. Historical Score Trends", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        score_trends_text = """
        <b>What It Shows:</b> Investment score evolution over the past 4-8 quarters.<br/><br/>

        <b>Score Interpretation (0-10 scale):</b><br/>
        • <b>8.0-10.0:</b> <font color="green"><b>Excellent</b></font> - Strong buy territory, high conviction<br/>
        • <b>6.0-8.0:</b> <font color="blue"><b>Good</b></font> - Buy candidate, moderate conviction<br/>
        • <b>4.0-6.0:</b> <font color="orange"><b>Neutral</b></font> - Hold or wait for better entry<br/>
        • <b>2.0-4.0:</b> <font color="red"><b>Poor</b></font> - Avoid or reduce position<br/>
        • <b>0.0-2.0:</b> <font color="darkred"><b>Critical</b></font> - Sell or avoid completely<br/><br/>

        <b>Trend Direction:</b><br/>
        • <b>↗️ Improving:</b> Quality improving, fundamentals strengthening → <b>BUY signal</b><br/>
        • <b>→ Stable:</b> Consistent quality, predictable → <b>HOLD signal</b><br/>
        • <b>↘️ Declining:</b> Quality deteriorating, fundamentals weakening → <b>SELL signal</b><br/><br/>

        <b>Investment Strategy:</b><br/>
        • <b>Score &gt;7.0 + Upward trend:</b> Add to position, core holding candidate<br/>
        • <b>Score 5.0-7.0 + Upward trend:</b> Accumulate on dips<br/>
        • <b>Score &lt;5.0 + Downward trend:</b> Exit or avoid, quality concerns<br/>
        • <b>Sudden spike (2+ points):</b> Investigate catalyst, may be opportunity or false signal
        """
        elements.append(Paragraph(score_trends_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Enhancement #2: Relative Valuation
        elements.append(Paragraph("💰 2. Relative Valuation Analysis", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        valuation_text = """
        <b>What It Shows:</b> Company's valuation multiples (P/E, P/B, P/S, PEG) compared to industry peers.<br/><br/>

        <b>Valuation Multiples Explained:</b><br/>
        • <b>P/E Ratio (Price-to-Earnings):</b> How much you pay for $1 of earnings<br/>
          - Tech avg: 25-35x | Industrials avg: 15-20x | Utilities avg: 12-18x<br/>
        • <b>P/B Ratio (Price-to-Book):</b> Price vs book value (assets - liabilities)<br/>
          - High P/B (&gt;3x) = Asset-light or growth | Low P/B (&lt;1x) = Value or distress<br/>
        • <b>P/S Ratio (Price-to-Sales):</b> Valuation vs revenue (useful for unprofitable companies)<br/>
          - SaaS avg: 6-12x | Retail avg: 0.5-2x<br/>
        • <b>PEG Ratio (P/E-to-Growth):</b> P/E relative to growth rate<br/>
          - &lt;1.0 = Undervalued growth | &gt;2.0 = Expensive growth<br/><br/>

        <b>Relative Valuation Assessment:</b><br/>
        • <b>UNDERVALUED:</b> Trading below peer median across multiple metrics → <b>BUY opportunity</b><br/>
          - Look for: P/E &lt; peer median, PEG &lt; 1.0, P/B below industry average<br/>
        • <b>FAIRLY VALUED:</b> In-line with peers → <b>HOLD</b> or wait for pullback<br/>
        • <b>OVERVALUED:</b> Premium to peers without justification → <b>AVOID or SELL</b><br/>
          - Justified premium: If company is top quartile in growth, margins, and quality<br/>
          - Unjustified premium: If company is average or below on fundamentals<br/><br/>

        <b>Strategy:</b><br/>
        • Undervalued + High quality (Tier 2 top 25%) = <b>STRONG BUY</b><br/>
        • Overvalued + Declining quality = <b>STRONG SELL</b><br/>
        • Compare to historical average - Trading at 10-year low multiple? → Opportunity<br/>
        • Growth stocks: Focus on PEG and P/S | Value stocks: Focus on P/E and P/B
        """
        elements.append(Paragraph(valuation_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Enhancement #3: Red Flags Detection
        elements.append(Paragraph("🚩 3. Red Flags Detection", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        red_flags_text = """
        <b>What It Shows:</b> Automated detection of warning signs in financial data.<br/><br/>

        <b>Severity Levels:</b><br/>
        • <b><font color="red">HIGH:</font></b> Immediate concern, potential deal-breaker → <b>AVOID or EXIT</b><br/>
        • <b><font color="orange">MEDIUM:</font></b> Monitor closely, may deteriorate → <b>REDUCE position or wait</b><br/>
        • <b><font color="yellow">LOW:</font></b> Minor concern, keep watching → <b>Monitor quarterly</b><br/><br/>

        <b>Common Red Flags:</b><br/>
        1. <b>Declining Revenue (2+ quarters):</b> [HIGH] → Losing market share or demand weakness<br/>
        2. <b>Negative Cash Flow + Positive Earnings:</b> [HIGH] → Earnings quality concern, possible accounting issues<br/>
        3. <b>Debt Spike (50%+ increase):</b> [MEDIUM] → Leverage increasing, financial stress risk<br/>
        4. <b>Margin Compression (20%+ decline):</b> [MEDIUM] → Pricing pressure, competitive challenges<br/>
        5. <b>Accounts Receivable Days Increasing (30%+):</b> [MEDIUM] → Collection problems, revenue quality issues<br/>
        6. <b>Inventory Buildup:</b> [MEDIUM] → Demand slowing, obsolescence risk<br/>
        7. <b>Goodwill Impairment:</b> [HIGH] → Failed acquisition, asset writedown<br/><br/>

        <b>How to Respond:</b><br/>
        • <b>0 Red Flags:</b> Clean bill of health → Confidence booster for investment<br/>
        • <b>1-2 Medium Flags:</b> Investigate further, understand the context<br/>
        • <b>1+ High Flag:</b> Serious concern → Exit position or avoid entry<br/>
        • <b>3+ Flags (any severity):</b> Multiple problems → Strong avoidance signal<br/><br/>

        <b>Context Matters:</b><br/>
        • One-time events (acquisition causing debt spike) may be acceptable<br/>
        • Industry-wide issues (all competitors showing margin compression) less concerning<br/>
        • But: Multiple company-specific flags = structural problems → <b>AVOID</b>
        """
        elements.append(Paragraph(red_flags_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Enhancement #4: Data Quality Badge
        elements.append(Paragraph("🎖️ 4. Data Quality Badge", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        data_quality_text = """
        <b>What It Shows:</b> Completeness and reliability of data used in analysis (A-F grade).<br/><br/>

        <b>Grade Interpretation:</b><br/>
        • <b><font color="green">A (90-100%):</font></b> Excellent data - High confidence in analysis<br/>
          → Full quarters available, complete LLM analysis, fresh market data, peer comparisons<br/>
        • <b><font color="blue">B (80-89%):</font></b> Good data - Reliable analysis<br/>
          → Minor gaps but sufficient for quality recommendation<br/>
        • <b><font color="orange">C (70-79%):</font></b> Adequate data - Use with caution<br/>
          → Some data missing, confidence reduced, require larger margin of safety<br/>
        • <b><font color="red">D (60-69%):</font></b> Poor data - Low confidence<br/>
          → Significant gaps, recommendation may be unreliable<br/>
        • <b><font color="darkred">F (&lt;60%):</font></b> Insufficient data - Avoid<br/>
          → Too many gaps, analysis unreliable, wait for better data<br/><br/>

        <b>Impact on Investment Decision:</b><br/>
        • <b>Grade A/B:</b> Trust the analysis, proceed with recommended position size<br/>
        • <b>Grade C:</b> Reduce position size by 30-50%, require 20%+ margin of safety<br/>
        • <b>Grade D/F:</b> Skip this stock, find alternative with better data quality<br/><br/>

        <b>Why Data Quality Matters:</b><br/>
        "Garbage in, garbage out" - Even sophisticated AI analysis requires quality inputs.
        A STRONG BUY with Grade F data is less reliable than a HOLD with Grade A data.
        """
        elements.append(Paragraph(data_quality_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Enhancement #5: Support/Resistance Levels
        elements.append(Paragraph("📍 5. Support/Resistance Levels", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        sr_levels_text = """
        <b>What It Shows:</b> Key price levels where stock historically bounces (support) or stalls (resistance).<br/><br/>

        <b>Support Levels (Green lines on charts):</b><br/>
        • Prices where buyers historically step in<br/>
        • Stock tends to bounce off these levels<br/>
        • <b>Strategy:</b> Buy near support, set stop loss just below it<br/><br/>

        <b>Resistance Levels (Red lines on charts):</b><br/>
        • Prices where sellers historically emerge<br/>
        • Stock tends to stall or reverse at these levels<br/>
        • <b>Strategy:</b> Take profits near resistance, wait for breakout confirmation<br/><br/>

        <b>Distance to Key Levels:</b><br/>
        • <b>&lt;2% from support:</b> Strong buy zone, good risk/reward<br/>
        • <b>Midway between support/resistance:</b> No edge, wait for extremes<br/>
        • <b>&lt;2% from resistance:</b> Take profits, expect pullback<br/><br/>

        <b>Breakout/Breakdown Trading:</b><br/>
        • <b>Break ABOVE resistance + volume increase:</b> Bullish breakout → BUY<br/>
          - Old resistance becomes new support<br/>
          - Target: Distance to next resistance level<br/>
        • <b>Break BELOW support + volume increase:</b> Bearish breakdown → SELL<br/>
          - Old support becomes new resistance<br/>
          - Target: Distance to next support level<br/><br/>

        <b>Multiple Tests:</b><br/>
        • Level tested 3+ times = Strong level (more reliable)<br/>
        • Break on 4th test = Powerful move (high probability of continuation)<br/><br/>

        <b>Combine with Fundamentals:</b><br/>
        • Undervalued company at support = <b>STRONG BUY</b><br/>
        • Overvalued company at resistance = <b>STRONG SELL</b><br/>
        • Use support/resistance for timing, fundamentals for conviction
        """
        elements.append(Paragraph(sr_levels_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.3 * inch))

        # ==========================
        # CORE METRICS INTERPRETATION
        # ==========================

        elements.append(Paragraph("Core Financial Metrics Interpretation", self.styles["Heading1"]))
        elements.append(Spacer(1, 0.15 * inch))

        # Fundamental Metrics
        elements.append(Paragraph("📊 Fundamental Metrics", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        fundamental_text = """
        <b>Profitability Metrics:</b><br/>
        • <b>Gross Margin:</b> (Revenue - COGS) / Revenue<br/>
          - Software/SaaS: 70-90% | Manufacturing: 20-40% | Retail: 25-35%<br/>
          - Higher = Better pricing power and operating leverage<br/>
        • <b>Operating Margin:</b> Operating Income / Revenue<br/>
          - Tech: 20-30% | Industrials: 8-15% | Retail: 5-10%<br/>
          - Measures operational efficiency<br/>
        • <b>Net Margin:</b> Net Income / Revenue<br/>
          - &gt;20% = Excellent | 10-20% = Good | &lt;5% = Low profitability<br/>
          - Bottom line profitability after all expenses<br/><br/>

        <b>Return Metrics:</b><br/>
        • <b>ROE (Return on Equity):</b> Net Income / Shareholders' Equity<br/>
          - &gt;20% = Excellent | 15-20% = Good | &lt;10% = Poor<br/>
          - Measures return on shareholder capital<br/>
        • <b>ROA (Return on Assets):</b> Net Income / Total Assets<br/>
          - &gt;10% = Excellent | 5-10% = Good | &lt;3% = Poor<br/>
          - Measures asset efficiency<br/>
        • <b>ROIC (Return on Invested Capital):</b> NOPAT / Invested Capital<br/>
          - &gt;15% = Creates value | &lt;WACC = Destroys value<br/>
          - Best measure of capital allocation quality<br/><br/>

        <b>Leverage Metrics:</b><br/>
        • <b>Debt-to-Equity:</b> Total Debt / Shareholders' Equity<br/>
          - &lt;0.5 = Conservative | 0.5-1.5 = Moderate | &gt;2.0 = Aggressive/Risky<br/>
        • <b>Interest Coverage:</b> EBIT / Interest Expense<br/>
          - &gt;5x = Safe | 2-5x = Adequate | &lt;2x = Distress risk<br/>
          - Can company afford its debt?<br/><br/>

        <b>Liquidity Metrics:</b><br/>
        • <b>Current Ratio:</b> Current Assets / Current Liabilities<br/>
          - &gt;2.0 = Strong | 1.0-2.0 = Adequate | &lt;1.0 = Liquidity risk<br/>
        • <b>Quick Ratio:</b> (Current Assets - Inventory) / Current Liabilities<br/>
          - &gt;1.5 = Strong | 0.8-1.5 = Adequate | &lt;0.8 = Concern<br/>
          - More conservative than current ratio<br/><br/>

        <b>Valuation Multiples:</b><br/>
        • <b>EV/EBITDA:</b> Enterprise Value / EBITDA<br/>
          - Better than P/E for companies with different capital structures<br/>
          - Tech: 12-20x | Industrials: 8-12x | Utilities: 7-10x<br/>
        • <b>Price/FCF:</b> Market Cap / Free Cash Flow<br/>
          - Similar to P/E but uses cash flow (more reliable)<br/>
          - &lt;15x = Cheap | 15-25x = Fair | &gt;25x = Expensive
        """
        elements.append(Paragraph(fundamental_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Technical Indicators
        elements.append(Paragraph("📈 Technical Indicators", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        technical_text = """
        <b>Momentum Indicators:</b><br/>
        • <b>RSI (Relative Strength Index):</b> Measures overbought/oversold conditions<br/>
          - &gt;70 = Overbought → Take profits or wait for pullback<br/>
          - 30-70 = Neutral → No clear signal<br/>
          - &lt;30 = Oversold → Buy opportunity (if fundamentals strong)<br/>
        • <b>MACD (Moving Average Convergence Divergence):</b><br/>
          - MACD line crosses above signal line = <b>BUY signal</b><br/>
          - MACD line crosses below signal line = <b>SELL signal</b><br/>
          - Divergence from price = Potential reversal<br/><br/>

        <b>Trend Indicators:</b><br/>
        • <b>Moving Averages (50-day, 200-day):</b><br/>
          - Price above MA = Uptrend (bullish) → HOLD or BUY dips<br/>
          - Price below MA = Downtrend (bearish) → SELL or avoid<br/>
          - <b>Golden Cross:</b> 50-day MA crosses above 200-day = <b>Major BUY signal</b><br/>
          - <b>Death Cross:</b> 50-day MA crosses below 200-day = <b>Major SELL signal</b><br/>
        • <b>ADX (Average Directional Index):</b> Measures trend strength<br/>
          - &gt;25 = Strong trend (follow it)<br/>
          - &lt;20 = Weak/no trend (range-bound, use support/resistance)<br/><br/>

        <b>Volatility Indicators:</b><br/>
        • <b>Bollinger Bands:</b> Price envelope based on volatility<br/>
          - Price at upper band = Overbought → SELL or take profits<br/>
          - Price at lower band = Oversold → BUY opportunity<br/>
          - <b>Squeeze:</b> Bands narrow → Big move coming (direction unknown)<br/>
          - <b>Breakout:</b> Price breaks outside bands → Strong directional move<br/>
        • <b>ATR (Average True Range):</b> Measures volatility<br/>
          - Use for stop loss placement (e.g., 2x ATR below entry)<br/>
          - High ATR = More volatile, wider stops needed<br/><br/>

        <b>Volume Indicators:</b><br/>
        • <b>Volume Confirmation:</b><br/>
          - Price up + Volume up = <b>Bullish confirmation</b><br/>
          - Price down + Volume up = <b>Bearish confirmation</b><br/>
          - Price move + Volume down = Weak move (likely reverses)<br/>
        • <b>OBV (On-Balance Volume):</b> Cumulative volume flow<br/>
          - Rising OBV + Rising price = Strong uptrend<br/>
          - Divergence (price up, OBV down) = Weakness, potential reversal<br/><br/>

        <b>Combining Technical + Fundamental:</b><br/>
        • <b>Best Setup:</b> Undervalued fundamentals + oversold technicals (RSI &lt;30) + at support = <b>STRONG BUY</b><br/>
        • <b>Worst Setup:</b> Overvalued fundamentals + overbought technicals (RSI &gt;70) + at resistance = <b>STRONG SELL</b><br/>
        • Use fundamentals for WHAT to buy, technicals for WHEN to buy
        """
        elements.append(Paragraph(technical_text, self.styles["AnalysisText"]))
        elements.append(Spacer(1, 0.2 * inch))

        # Investment Score System
        elements.append(Paragraph("🎯 Investment Score System (0-10)", self.styles["Heading2"]))
        elements.append(Spacer(1, 6))

        score_system_text = """
        <b>How the Overall Score is Calculated:</b><br/>
        The overall investment score combines fundamental (60%) and technical (40%) factors weighted by quality and conviction.<br/><br/>

        <b>Score Components:</b><br/>
        • <b>Fundamental Score (60% weight):</b><br/>
          - Financial health: 25% (profitability, leverage, liquidity)<br/>
          - Growth quality: 20% (revenue/earnings growth, sustainability)<br/>
          - Valuation: 15% (relative to peers and history)<br/>
        • <b>Technical Score (40% weight):</b><br/>
          - Trend strength: 15% (momentum, moving averages)<br/>
          - Entry timing: 15% (oversold/overbought, support/resistance)<br/>
          - Volume confirmation: 10%<br/><br/>

        <b>Score Ranges &amp; Investment Actions:</b><br/>
        • <b>9.0-10.0:</b> <font color="green"><b>Exceptional</b></font> → STRONG BUY, 10-12% position, core holding<br/>
          - Rare opportunities, all factors aligned<br/>
        • <b>8.0-8.9:</b> <font color="green"><b>Excellent</b></font> → STRONG BUY, 7-10% position<br/>
          - High conviction, minor imperfections acceptable<br/>
        • <b>7.0-7.9:</b> <font color="blue"><b>Very Good</b></font> → BUY, 5-8% position<br/>
          - Solid opportunity, good risk/reward<br/>
        • <b>6.0-6.9:</b> <font color="blue"><b>Good</b></font> → BUY, 3-6% position<br/>
          - Above average, wait for better entry if patient<br/>
        • <b>5.0-5.9:</b> <font color="gray"><b>Neutral</b></font> → HOLD, maintain if owned<br/>
          - Fair value, no edge for new entry<br/>
        • <b>4.0-4.9:</b> <font color="orange"><b>Below Average</b></font> → REDUCE, 0-2% position max<br/>
          - Quality concerns, limited upside<br/>
        • <b>3.0-3.9:</b> <font color="red"><b>Poor</b></font> → SELL or AVOID<br/>
          - Multiple weaknesses, better opportunities exist<br/>
        • <b>0.0-2.9:</b> <font color="darkred"><b>Critical</b></font> → STRONG SELL, 0% position<br/>
          - Serious problems, high loss probability<br/><br/>

        <b>Context Adjustments:</b><br/>
        • <b>High data quality (Grade A):</b> Trust the score fully<br/>
        • <b>Low data quality (Grade C/D):</b> Reduce confidence, require 1-2 points higher for same action<br/>
        • <b>Red flags present:</b> Subtract 1-2 points for risk adjustment<br/>
        • <b>Improving trend:</b> Can accept slightly lower score (e.g., 7.5 improving vs 8.0 stable)<br/>
        • <b>Declining trend:</b> Require higher score for confidence (e.g., need 8.5+ vs usual 7.5+)
        """
        elements.append(Paragraph(score_system_text, self.styles["AnalysisText"]))

        return elements

    def _generate_filename(self, recommendations: List[Dict], report_type: str) -> str:
        """
        Generate filename based on symbols in recommendations

        Args:
            recommendations: List of investment recommendations
            report_type: Type of report

        Returns:
            Generated filename
        """
        if not recommendations:
            return f"{report_type}_report_no_symbols.pdf"

        # Extract symbols from recommendations
        symbols = [rec.get("symbol", "UNKNOWN") for rec in recommendations]

        # Create symbol part of filename
        if len(symbols) <= 4:
            # Use all symbols separated by hyphens
            symbol_part = "-".join(symbols)
        elif len(symbols) == 5:
            # Use all 5 symbols
            symbol_part = "-".join(symbols)
        else:
            # Use first 4 symbols + "OTHERS"
            symbol_part = "-".join(symbols[:4]) + "-OTHERS"

        # Create final filename
        filename = f"{report_type}_report_{symbol_part}.pdf"

        return filename

    def _get_comprehensive_analysis_data(self, symbol: str) -> dict:
        """
        Get comprehensive analysis data from SEC comprehensive analysis cache

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary containing all comprehensive analysis data including scores and insights
        """
        try:
            # Import cache manager and types
            from investigator.config import get_config
            from investigator.infrastructure.cache.cache_manager import CacheManager
            from investigator.infrastructure.cache.cache_types import CacheType

            # Initialize cache manager
            config = get_config()
            cache_manager = CacheManager(config)

            # Determine the most recent fiscal period (use current assumptions)
            from datetime import datetime

            current_date = datetime.now()
            current_year = current_date.year

            # Try to find SEC comprehensive analysis data
            # First try with current year comprehensive
            cache_key = {
                "symbol": symbol,
                "form_type": "COMPREHENSIVE",
                "period": f"{current_year}-FY",
                "llm_type": "sec",
            }

            cached_response = cache_manager.get(CacheType.LLM_RESPONSE, cache_key)

            # If not found, try previous year
            if not cached_response:
                cache_key["period"] = f"{current_year - 1}-FY"
                cached_response = cache_manager.get(CacheType.LLM_RESPONSE, cache_key)

            if cached_response:
                # Parse the comprehensive analysis response
                response_content = cached_response.get("response", cached_response.get("content", ""))

                if response_content:
                    try:
                        # Try to parse as JSON
                        import json

                        if isinstance(response_content, str):
                            comp_analysis = json.loads(response_content)
                        else:
                            comp_analysis = response_content

                        # Extract all valuable data
                        comprehensive_data = {
                            "financial_health_score": comp_analysis.get("financial_health_score"),
                            "business_quality_score": comp_analysis.get("business_quality_score"),
                            "growth_prospects_score": comp_analysis.get("growth_prospects_score"),
                            "data_quality_score": comp_analysis.get("data_quality_score"),
                            "overall_score": comp_analysis.get("overall_score"),
                            "trend_analysis": comp_analysis.get("trend_analysis", {}),
                            "key_insights": comp_analysis.get("key_insights", []),
                            "key_risks": comp_analysis.get("key_risks", []),
                            "investment_thesis": comp_analysis.get("investment_thesis", ""),
                            "confidence_level": comp_analysis.get("confidence_level", ""),
                            "analysis_summary": comp_analysis.get("analysis_summary", ""),
                            "quarterly_analyses": comp_analysis.get("quarterly_analyses", []),
                            "quarters_analyzed": comp_analysis.get("quarters_analyzed", 0),
                        }

                        logger.info(f"📊 Extracted comprehensive analysis data for {symbol}")
                        return comprehensive_data

                    except (json.JSONDecodeError, AttributeError, KeyError) as e:
                        logger.warning(f"Failed to parse SEC comprehensive analysis for {symbol}: {e}")

            # Return empty dict if no data found
            return {}

        except Exception as e:
            logger.error(f"Error extracting comprehensive analysis data for {symbol}: {e}")
            return {}

    def _get_comprehensive_investment_thesis(self, symbol: str, fallback_thesis: str = "") -> str:
        """
        Get investment thesis from SEC comprehensive analysis cache

        Args:
            symbol: Stock symbol
            fallback_thesis: Fallback thesis from synthesis if SEC data unavailable

        Returns:
            Investment thesis text from SEC comprehensive analysis or fallback
        """
        try:
            # Get comprehensive analysis data using new extraction method
            comp_data = self._get_comprehensive_analysis_data(symbol)

            if comp_data and comp_data.get("investment_thesis"):
                logger.info(f"📊 Using SEC comprehensive investment thesis for {symbol}")
                return comp_data["investment_thesis"]

            # Try to build thesis from key insights and risks
            if comp_data and (comp_data.get("key_insights") or comp_data.get("key_risks")):
                thesis_parts = []

                if comp_data.get("key_insights"):
                    thesis_parts.append("Key Investment Insights:")
                    for insight in comp_data["key_insights"][:3]:  # Top 3 insights
                        thesis_parts.append(f"• {insight}")

                if comp_data.get("key_risks"):
                    thesis_parts.append("\nKey Risk Factors:")
                    for risk in comp_data["key_risks"][:2]:  # Top 2 risks
                        thesis_parts.append(f"• {risk}")

                # Add analysis summary if available
                if comp_data.get("analysis_summary"):
                    thesis_parts.append(f"\nOverall Assessment: {comp_data['analysis_summary']}")

                if thesis_parts:
                    comprehensive_thesis = "\n".join(thesis_parts)
                    logger.info(f"📊 Built comprehensive investment thesis for {symbol} from insights")
                    return comprehensive_thesis

            # Fallback to synthesis thesis if SEC data unavailable
            if fallback_thesis:
                logger.info(f"📈 Using synthesis investment thesis for {symbol} (SEC comprehensive not available)")
                return fallback_thesis

            # Ultimate fallback
            logger.warning(f"No investment thesis available for {symbol}")
            return f"Investment analysis for {symbol} based on fundamental and technical factors."

        except Exception as e:
            logger.error(f"Error fetching comprehensive investment thesis for {symbol}: {e}")
            return fallback_thesis or f"Investment analysis for {symbol} based on available data."

    def _create_technical_summary(self, recommendation: Dict) -> List:
        """Create visual technical analysis summary using structured data"""
        elements = []

        # Extract technical indicators from the recommendation (set by direct extraction)
        support_levels = recommendation.get("support_levels", [])
        resistance_levels = recommendation.get("resistance_levels", [])
        trend_direction = recommendation.get("trend_direction", "NEUTRAL")
        technical_score = recommendation.get("technical_score", 0)

        # Only create section if we have technical data
        if not (support_levels or resistance_levels or trend_direction != "NEUTRAL"):
            return elements

        elements.append(Paragraph("Technical Analysis Summary", self.styles["SectionHeader"]))

        # Technical overview table
        tech_data = [["Technical Metric", "Value", "Interpretation"]]

        # Add technical score
        if technical_score > 0:
            tech_data.append(["Technical Score", f"{technical_score:.1f}/10", self._get_rating(technical_score)])

        # Add trend information
        if trend_direction and trend_direction != "NEUTRAL":
            trend_color = "🟢" if trend_direction == "BULLISH" else "🔴" if trend_direction == "BEARISH" else "🟡"
            tech_data.append(
                [
                    "Trend Direction",
                    f"{trend_color} {trend_direction}",
                    (
                        "Favorable"
                        if trend_direction == "BULLISH"
                        else "Concerning" if trend_direction == "BEARISH" else "Neutral"
                    ),
                ]
            )

        # Add support levels
        if support_levels:
            support_str = ", ".join([f"${level:.2f}" for level in support_levels[:3]])
            tech_data.append(["Key Support Levels", support_str, "Downside protection"])

        # Add resistance levels
        if resistance_levels:
            resistance_str = ", ".join([f"${level:.2f}" for level in resistance_levels[:3]])
            tech_data.append(["Key Resistance Levels", resistance_str, "Upside targets"])

        # Create table if we have data
        if len(tech_data) > 1:
            tech_table = Table(tech_data, colWidths=[2 * inch, 2 * inch, 2 * inch])
            tech_table.setStyle(
                TableStyle(
                    [
                        # Header styling
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17a2b8")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 11),
                        # Body styling
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 1), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                        ("TOPPADDING", (0, 1), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
                    ]
                )
            )

            elements.append(tech_table)

            # Add technical insights if available
            momentum_signals = recommendation.get("momentum_signals", [])
            if momentum_signals:
                elements.append(Spacer(1, 0.1 * inch))
                elements.append(Paragraph("<b>Technical Signals</b>", self.styles["Heading3"]))
                for signal in momentum_signals[:3]:  # Show top 3 signals
                    elements.append(Paragraph(f"• {signal}", self.styles["MetricsText"]))

        return elements

    def _get_rating(self, score: float) -> str:
        """Get rating text from score"""
        if score >= 8:
            return "Excellent"
        elif score >= 6:
            return "Good"
        elif score >= 4:
            return "Fair"
        else:
            return "Poor"

    def _create_entry_exit_section(self, recommendation: Dict) -> List:
        """
        Create entry/exit signal section with visual components.

        Args:
            recommendation: Dict with entry_signals, exit_signals, optimal_entry_zone

        Returns:
            List of ReportLab flowables
        """
        elements = []

        entry_signals = recommendation.get("entry_signals", [])
        exit_signals = recommendation.get("exit_signals", [])
        optimal_entry_zone = recommendation.get("optimal_entry_zone")
        current_price = recommendation.get("current_price", 0)

        # Only create section if we have entry/exit data
        if not (entry_signals or exit_signals or optimal_entry_zone):
            return elements

        elements.append(Paragraph("Entry/Exit Signal Analysis", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.1 * inch))

        # Entry Zone Visualization
        if optimal_entry_zone and current_price > 0:
            support_levels = recommendation.get("support_levels", [])
            resistance_levels = recommendation.get("resistance_levels", [])

            elements.append(Paragraph("<b>Optimal Entry Zone</b>", self.styles["Heading3"]))
            elements.append(Spacer(1, 0.05 * inch))

            # Add the visual entry zone flowable
            zone_chart = EntryExitZone(
                width=5.5 * inch,
                height=1.5 * inch,
                current_price=current_price,
                entry_zone=optimal_entry_zone,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
            )
            elements.append(zone_chart)
            elements.append(Spacer(1, 0.1 * inch))

            # Entry zone details table
            zone_data = [["Entry Zone Metric", "Value"]]
            zone_data.append(["Lower Bound", f"${optimal_entry_zone.get('lower_bound', 0):.2f}"])
            zone_data.append(["Ideal Entry", f"${optimal_entry_zone.get('ideal_entry', 0):.2f}"])
            zone_data.append(["Upper Bound", f"${optimal_entry_zone.get('upper_bound', 0):.2f}"])
            zone_data.append(["Timing", optimal_entry_zone.get("timing", "N/A").replace("_", " ")])
            zone_data.append(["Scaling Strategy", optimal_entry_zone.get("scaling_strategy", "N/A").replace("_", " ")])
            zone_data.append(["Confidence", optimal_entry_zone.get("confidence", "N/A")])
            zone_data.append(
                ["Recommended Allocation", f"{optimal_entry_zone.get('recommended_allocation_pct', 0):.1f}%"]
            )

            zone_table = Table(zone_data, colWidths=[2.5 * inch, 2.5 * inch])
            zone_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#28a745")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
                        ("FONTSIZE", (0, 1), (-1, -1), 9),
                        ("TOPPADDING", (0, 1), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
                    ]
                )
            )
            elements.append(zone_table)
            elements.append(Spacer(1, 0.15 * inch))

        # Entry Signals Table
        if entry_signals:
            elements.append(Paragraph("<b>Entry Signals</b>", self.styles["Heading3"]))
            elements.append(Spacer(1, 0.05 * inch))

            entry_data = [["Signal Type", "Price", "Stop Loss", "Target", "R/R", "Confidence"]]
            for signal in entry_signals[:5]:  # Top 5 signals
                entry_data.append(
                    [
                        signal.get("signal_type", "N/A").replace("_", " "),
                        f"${signal.get('price_level', 0):.2f}",
                        f"${signal.get('stop_loss', 0):.2f}",
                        f"${signal.get('target_price', 0):.2f}",
                        f"1:{signal.get('risk_reward_ratio', 0):.1f}",
                        signal.get("confidence", "N/A"),
                    ]
                )

            entry_table = Table(
                entry_data, colWidths=[1.3 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.7 * inch, 0.9 * inch]
            )
            entry_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#007bff")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 9),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
                        ("FONTSIZE", (0, 1), (-1, -1), 8),
                        ("TOPPADDING", (0, 1), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
                    ]
                )
            )
            elements.append(entry_table)

            # Add signal strength bars for top signals
            if entry_signals and len(entry_signals) > 0:
                elements.append(Spacer(1, 0.1 * inch))
                for signal in entry_signals[:3]:
                    # Calculate signal score based on confidence and R/R
                    confidence_map = {"HIGH": 80, "MEDIUM": 60, "LOW": 40}
                    base_score = confidence_map.get(signal.get("confidence", "MEDIUM"), 60)
                    rr_bonus = min(signal.get("risk_reward_ratio", 2) * 5, 20)
                    signal_score = min(base_score + rr_bonus, 100)

                    signal_type = signal.get("signal_type", "Signal").replace("_", " ")
                    strength_bar = SignalStrengthBar(
                        width=4 * inch,
                        height=0.35 * inch,
                        score=signal_score,
                        label=signal_type,
                    )
                    elements.append(strength_bar)
                    elements.append(Spacer(1, 0.05 * inch))

            elements.append(Spacer(1, 0.1 * inch))

        # Exit Signals Table
        if exit_signals:
            elements.append(Paragraph("<b>Exit Signals</b>", self.styles["Heading3"]))
            elements.append(Spacer(1, 0.05 * inch))

            exit_data = [["Signal Type", "Price Level", "Urgency", "Confidence", "Partial Exit"]]
            for signal in exit_signals[:5]:  # Top 5 signals
                exit_data.append(
                    [
                        signal.get("signal_type", "N/A").replace("_", " "),
                        f"${signal.get('price_level', 0):.2f}",
                        signal.get("urgency", "N/A").replace("_", " "),
                        signal.get("confidence", "N/A"),
                        f"{signal.get('partial_exit_pct', 100):.0f}%",
                    ]
                )

            exit_table = Table(exit_data, colWidths=[1.5 * inch, 1.2 * inch, 1.2 * inch, 1 * inch, 1 * inch])
            exit_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dc3545")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 9),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
                        ("FONTSIZE", (0, 1), (-1, -1), 8),
                        ("TOPPADDING", (0, 1), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
                    ]
                )
            )
            elements.append(exit_table)
            elements.append(Spacer(1, 0.1 * inch))

        # Stop Loss Visualization for primary entry signal
        if entry_signals and len(entry_signals) > 0:
            primary_signal = entry_signals[0]
            entry_price = primary_signal.get("price_level", current_price)
            stop_loss = primary_signal.get("stop_loss", entry_price * 0.95)
            target_price = primary_signal.get("target_price", entry_price * 1.10)

            if entry_price > 0 and stop_loss > 0:
                elements.append(Paragraph("<b>Primary Signal Risk/Reward</b>", self.styles["Heading3"]))
                elements.append(Spacer(1, 0.05 * inch))

                stop_loss_viz = StopLossIndicator(
                    width=5 * inch,
                    height=0.6 * inch,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_price=target_price,
                )
                elements.append(stop_loss_viz)
                elements.append(Spacer(1, 0.1 * inch))

        # Position Sizing Guidance
        if position_sizing := recommendation.get("position_sizing_guidance"):
            elements.append(Paragraph("<b>Position Sizing Guidance</b>", self.styles["Heading3"]))
            elements.append(Spacer(1, 0.05 * inch))

            sizing_text = f"""
            <b>Conviction Level:</b> {position_sizing.get('conviction_level', 'N/A')}<br/>
            <b>Volatility-Adjusted Size:</b> {position_sizing.get('volatility_adjusted_size', 'N/A')}<br/>
            <b>Recommended Position:</b> {position_sizing.get('recommended_position_pct', 0):.1f}% of portfolio<br/>
            <b>Max Risk:</b> {position_sizing.get('max_risk_pct', 0):.1f}%<br/>
            """
            elements.append(Paragraph(sizing_text, self.styles["AnalysisText"]))

            # Scaling approach
            if scaling := position_sizing.get("scaling_approach"):
                scaling_text = f"""
                <b>Scaling Strategy:</b><br/>
                Initial Entry: {scaling.get('initial_entry_pct', 0):.0f}% @ current price<br/>
                Second Tranche: {scaling.get('second_tranche_pct', 0):.0f}% @ ${scaling.get('second_tranche_price', 0):.2f}<br/>
                Third Tranche: {scaling.get('third_tranche_pct', 0):.0f}% @ ${scaling.get('third_tranche_price', 0):.2f}
                """
                elements.append(Paragraph(scaling_text, self.styles["MetricsText"]))

        return elements

    def _markdown_to_html(self, markdown_text: str) -> str:
        """
        Convert markdown text to HTML suitable for ReportLab Paragraph

        Args:
            markdown_text: Text with markdown formatting

        Returns:
            HTML-formatted text suitable for ReportLab
        """
        if not markdown_text:
            return ""

        try:
            # Convert markdown to HTML
            html = markdown.markdown(markdown_text, extensions=["nl2br"])

            # Clean up HTML for ReportLab compatibility
            # ReportLab uses a limited subset of HTML tags

            # Replace markdown bold (**text**) with HTML bold
            html = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", html)

            # Replace markdown italic (*text*) with HTML italic
            html = re.sub(r"\*(.*?)\*", r"<i>\1</i>", html)

            # Ensure bullet points are properly formatted
            html = re.sub(r"<li>(.*?)</li>", r"• \1<br/>", html)

            # Remove unsupported HTML tags but keep content
            html = re.sub(r"</?(?:ul|ol)>", "", html)
            html = re.sub(r"</?p>", "", html)

            # Convert line breaks
            html = html.replace("\n", "<br/>")

            # Clean up multiple consecutive line breaks
            html = re.sub(r"(<br/>){3,}", "<br/><br/>", html)

            return html.strip()

        except Exception as e:
            logger.warning(f"Failed to convert markdown to HTML: {e}")
            # Fallback: basic markdown conversion
            text = markdown_text
            text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
            text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
            text = text.replace("\n", "<br/>")
            return text

    def _get_llm_thinking_details(self, symbol: str) -> tuple:
        """
        Extract thinking and details from SEC fundamental and technical analysis LLM responses

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (sec_thinking, tech_thinking) strings
        """
        try:
            # Import cache manager and types
            from investigator.config import get_config
            from investigator.infrastructure.cache.cache_manager import CacheManager
            from investigator.infrastructure.cache.cache_types import CacheType

            # Initialize cache manager
            config = get_config()
            cache_manager = CacheManager(config)

            sec_thinking = ""
            tech_thinking = ""

            # Get SEC comprehensive analysis thinking
            from datetime import datetime

            current_date = datetime.now()
            current_year = current_date.year

            # Try to find SEC comprehensive analysis response
            cache_key = {
                "symbol": symbol,
                "form_type": "COMPREHENSIVE",
                "period": f"{current_year}-FY",
                "llm_type": "sec",
            }

            cached_response = cache_manager.get(CacheType.LLM_RESPONSE, cache_key)

            # If not found, try previous year
            if not cached_response:
                cache_key["period"] = f"{current_year - 1}-FY"
                cached_response = cache_manager.get(CacheType.LLM_RESPONSE, cache_key)

            if cached_response:
                response_content = cached_response.get("response", cached_response.get("content", ""))

                if response_content:
                    try:
                        # Try to parse as JSON and extract thinking/details
                        import json

                        if isinstance(response_content, str):
                            comp_analysis = json.loads(response_content)
                        else:
                            comp_analysis = response_content

                        # Extract thinking/reasoning fields
                        thinking_fields = []

                        # Extract analysis summary
                        if comp_analysis.get("analysis_summary"):
                            thinking_fields.append(f"**Analysis Summary**: {comp_analysis['analysis_summary']}")

                        # Extract investment thesis
                        if comp_analysis.get("investment_thesis"):
                            thinking_fields.append(f"**Investment Thesis**: {comp_analysis['investment_thesis']}")

                        # Extract top-level detail field (contextual summary)
                        if comp_analysis.get("detail"):
                            thinking_fields.append(f"**Detailed Analysis**: {comp_analysis['detail']}")

                        # Extract quarterly analysis details
                        quarterly_analyses = comp_analysis.get("quarterly_analyses", [])
                        if quarterly_analyses and len(quarterly_analyses) > 0:
                            # Get the most recent quarter's detailed analysis
                            recent_quarter = quarterly_analyses[0]
                            if recent_quarter.get("detail"):
                                thinking_fields.append(f"**Recent Quarter Details**: {recent_quarter['detail']}")

                        if thinking_fields:
                            sec_thinking = "\n\n".join(thinking_fields)

                    except (json.JSONDecodeError, AttributeError, KeyError) as e:
                        logger.warning(f"Failed to parse SEC thinking for {symbol}: {e}")

            # Get technical analysis thinking
            tech_cache_key = {"symbol": symbol, "analysis_type": "technical_indicators", "llm_type": "technical"}

            tech_cached_response = cache_manager.get(CacheType.LLM_RESPONSE, tech_cache_key)

            # Fallback: Check file-based technical analysis cache
            if not tech_cached_response:
                try:
                    from pathlib import Path

                    tech_file_path = f"data/llm_cache/{symbol}/response_technical_indicators.txt"
                    if Path(tech_file_path).exists():
                        with open(tech_file_path, "r") as f:
                            tech_file_content = f.read()
                        tech_cached_response = {
                            "response": tech_file_content,
                            "content": tech_file_content,
                            "metadata": {"source": "file_fallback"},
                        }
                        logger.info(f"📊 Using file fallback for technical analysis thinking: {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to read technical analysis file for {symbol}: {e}")

            if tech_cached_response:
                tech_content = tech_cached_response.get("response", tech_cached_response.get("content", ""))

                if tech_content:
                    try:
                        # Handle file format with headers - extract JSON part
                        json_content = tech_content
                        if "=== AI RESPONSE ===" in tech_content:
                            json_start = tech_content.find("=== AI RESPONSE ===") + len("=== AI RESPONSE ===")
                            json_content = tech_content[json_start:].strip()

                        # Try to parse as JSON and extract thinking/details
                        if isinstance(json_content, str):
                            tech_analysis = json.loads(json_content)
                        else:
                            tech_analysis = json_content

                        # Extract thinking/reasoning fields
                        tech_thinking_fields = []

                        if tech_analysis.get("thinking"):
                            # Clean up the thinking text (remove escaped newlines)
                            thinking_text = tech_analysis["thinking"].replace("\\n", "\n").strip()
                            tech_thinking_fields.append(f"**Technical Analysis Process**: {thinking_text}")

                        # Extract top-level detail field (contextual technical summary)
                        if tech_analysis.get("detail"):
                            detail_text = tech_analysis["detail"].replace("\\n", "\n").strip()
                            tech_thinking_fields.append(f"**Technical Detail Summary**: {detail_text}")

                        # Add technical signals summary
                        if tech_analysis.get("momentum_signals"):
                            signals_text = "\n".join([f"• {signal}" for signal in tech_analysis["momentum_signals"]])
                            tech_thinking_fields.append(f"**Key Technical Signals**:\n{signals_text}")

                        # Add risk factors
                        if tech_analysis.get("risk_factors"):
                            risks_text = "\n".join([f"• {risk}" for risk in tech_analysis["risk_factors"]])
                            tech_thinking_fields.append(f"**Technical Risk Factors**:\n{risks_text}")

                        if tech_thinking_fields:
                            tech_thinking = "\n\n".join(tech_thinking_fields)

                    except (json.JSONDecodeError, AttributeError, KeyError) as e:
                        logger.warning(f"Failed to parse technical thinking for {symbol}: {e}")
                        fallback_content = tech_content.strip() if isinstance(tech_content, str) else ""
                        if fallback_content.startswith("```"):
                            fallback_content = fallback_content.strip("`")
                        fallback_content = fallback_content.replace("\\n", "\n")
                        if fallback_content:
                            tech_thinking = fallback_content
                        else:
                            tech_thinking = ""

            return (sec_thinking, tech_thinking)

        except Exception as e:
            logger.error(f"Error extracting LLM thinking for {symbol}: {e}")
            return ("", "")
