#!/usr/bin/env python3
"""
Professional Investment Report Generator
Clean, concise, institutional-grade reports
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import math

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
        KeepTogether, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus.flowables import Flowable
    from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Wedge
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics import renderPDF
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("reportlab not available")

logger = logging.getLogger(__name__)


class GaugeChart(Flowable):
    """Semi-circular gauge chart for scores"""

    def __init__(self, width, height, value, max_value=100, label="Score"):
        self.width = width
        self.height = height
        self.value = min(max(value, 0), max_value)
        self.max_value = max_value
        self.label = label

    def draw(self):
        cx, cy = self.width / 2, self.height * 0.3
        radius = min(self.width, self.height) * 0.35

        # Background arc (gray)
        self.canv.setStrokeColor(colors.HexColor('#e0e0e0'))
        self.canv.setLineWidth(12)
        self.canv.arc(cx - radius, cy - radius, cx + radius, cy + radius, 0, 180)

        # Value arc (colored based on score)
        pct = self.value / self.max_value
        if pct >= 0.7:
            color = colors.HexColor('#22c55e')  # Green
        elif pct >= 0.5:
            color = colors.HexColor('#eab308')  # Yellow
        elif pct >= 0.3:
            color = colors.HexColor('#f97316')  # Orange
        else:
            color = colors.HexColor('#ef4444')  # Red

        self.canv.setStrokeColor(color)
        self.canv.arc(cx - radius, cy - radius, cx + radius, cy + radius, 0, 180 * pct)

        # Score text
        self.canv.setFillColor(colors.HexColor('#1f2937'))
        self.canv.setFont("Helvetica-Bold", 24)
        self.canv.drawCentredString(cx, cy + radius * 0.3, f"{self.value:.0f}")

        # Label
        self.canv.setFont("Helvetica", 10)
        self.canv.setFillColor(colors.HexColor('#6b7280'))
        self.canv.drawCentredString(cx, cy - 10, self.label)


class RecommendationBadge(Flowable):
    """Clean recommendation badge"""

    def __init__(self, width, height, recommendation, confidence):
        self.width = width
        self.height = height
        self.recommendation = recommendation.upper()
        self.confidence = confidence.upper()

    def draw(self):
        # Color based on recommendation
        colors_map = {
            'BUY': '#22c55e', 'STRONG_BUY': '#16a34a',
            'SELL': '#ef4444', 'STRONG_SELL': '#dc2626',
            'HOLD': '#6b7280'
        }
        bg_color = colors.HexColor(colors_map.get(self.recommendation, '#6b7280'))

        # Rounded rectangle
        self.canv.setFillColor(bg_color)
        self.canv.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=0)

        # Recommendation text
        self.canv.setFillColor(colors.white)
        self.canv.setFont("Helvetica-Bold", 16)
        self.canv.drawCentredString(self.width/2, self.height * 0.55, self.recommendation)

        # Confidence
        self.canv.setFont("Helvetica", 9)
        self.canv.drawCentredString(self.width/2, self.height * 0.2, f"{self.confidence} confidence")


class MetricBar(Flowable):
    """Horizontal bar for metric visualization"""

    def __init__(self, width, height, value, max_value=100, label="", show_value=True):
        self.width = width
        self.height = height
        self.value = min(max(value, 0), max_value)
        self.max_value = max_value
        self.label = label
        self.show_value = show_value

    def draw(self):
        bar_height = 8
        bar_y = (self.height - bar_height) / 2

        # Label
        self.canv.setFillColor(colors.HexColor('#374151'))
        self.canv.setFont("Helvetica", 9)
        self.canv.drawString(0, bar_y + bar_height + 3, self.label)

        # Background bar
        bar_width = self.width - 50
        self.canv.setFillColor(colors.HexColor('#e5e7eb'))
        self.canv.roundRect(0, bar_y, bar_width, bar_height, 4, fill=1, stroke=0)

        # Value bar
        pct = self.value / self.max_value
        if pct >= 0.7:
            color = colors.HexColor('#22c55e')
        elif pct >= 0.5:
            color = colors.HexColor('#eab308')
        elif pct >= 0.3:
            color = colors.HexColor('#f97316')
        else:
            color = colors.HexColor('#ef4444')

        self.canv.setFillColor(color)
        self.canv.roundRect(0, bar_y, bar_width * pct, bar_height, 4, fill=1, stroke=0)

        # Value text
        if self.show_value:
            self.canv.setFillColor(colors.HexColor('#1f2937'))
            self.canv.setFont("Helvetica-Bold", 10)
            self.canv.drawRightString(self.width, bar_y + 2, f"{self.value:.0f}")


class PeerComparisonChart(Flowable):
    """Horizontal bar chart showing peer comparison by upside potential"""

    def __init__(self, width, height, target_symbol, target_upside, peers):
        self.width = width
        self.height = height
        self.target_symbol = target_symbol
        self.target_upside = target_upside or 0
        self.peers = peers or []  # List of {"symbol": str, "upside": float}

    def draw(self):
        # Prepare data - target first, then peers
        all_data = [{"symbol": self.target_symbol, "upside": self.target_upside, "is_target": True}]
        for p in self.peers[:4]:  # Max 4 peers
            val = p.get("valuation") or {}
            upside = val.get("upside_pct")
            if upside is not None:
                all_data.append({"symbol": p.get("symbol", "?"), "upside": upside, "is_target": False})

        if len(all_data) < 2:
            return

        # Calculate scale
        upsides = [d["upside"] for d in all_data]
        min_up = min(-50, min(upsides) - 10)
        max_up = max(50, max(upsides) + 10)
        range_up = max_up - min_up

        bar_height = 16
        bar_spacing = 4
        total_height = (bar_height + bar_spacing) * len(all_data)
        chart_width = self.width - 100  # Reserve space for labels
        label_x = 0
        bar_x = 60
        zero_x = bar_x + chart_width * (-min_up / range_up)

        # Draw zero line
        self.canv.setStrokeColor(colors.HexColor('#9ca3af'))
        self.canv.setLineWidth(1)
        self.canv.line(zero_x, 0, zero_x, total_height)

        # Draw bars
        for i, data in enumerate(all_data):
            y = total_height - (i + 1) * (bar_height + bar_spacing)
            upside = data["upside"]
            is_target = data.get("is_target", False)

            # Symbol label
            self.canv.setFillColor(colors.HexColor('#1f2937' if is_target else '#6b7280'))
            self.canv.setFont("Helvetica-Bold" if is_target else "Helvetica", 9)
            self.canv.drawString(label_x, y + 4, data["symbol"][:6])

            # Calculate bar position and width
            pct = (upside - min_up) / range_up
            bar_start = bar_x + chart_width * (-min_up / range_up)  # zero point
            bar_end = bar_x + chart_width * pct
            bar_w = bar_end - bar_start

            # Color based on positive/negative
            if upside >= 0:
                color = colors.HexColor('#22c55e' if is_target else '#86efac')
            else:
                color = colors.HexColor('#ef4444' if is_target else '#fca5a5')

            self.canv.setFillColor(color)
            if bar_w >= 0:
                self.canv.roundRect(bar_start, y, bar_w, bar_height, 3, fill=1, stroke=0)
            else:
                self.canv.roundRect(bar_start + bar_w, y, -bar_w, bar_height, 3, fill=1, stroke=0)

            # Value text
            self.canv.setFillColor(colors.HexColor('#374151'))
            self.canv.setFont("Helvetica", 8)
            self.canv.drawString(bar_x + chart_width + 5, y + 4, f"{upside:+.1f}%")


class MarketRegimeIndicator(Flowable):
    """Visual indicator for market regime (bull/bear/neutral)"""

    def __init__(self, width, height, regime, vix=None, yield_curve=None):
        self.width = width
        self.height = height
        self.regime = (regime or "normal").lower()
        self.vix = vix
        self.yield_curve = yield_curve

    def draw(self):
        # Regime indicator circle
        cx = 30
        cy = self.height / 2
        radius = 12

        # Color based on regime
        regime_colors = {
            "bullish": "#22c55e",
            "bull": "#22c55e",
            "normal": "#eab308",
            "neutral": "#6b7280",
            "bearish": "#ef4444",
            "bear": "#ef4444",
            "risk_off": "#ef4444",
            "risk_on": "#22c55e",
        }
        color = colors.HexColor(regime_colors.get(self.regime, "#6b7280"))

        # Draw circle
        self.canv.setFillColor(color)
        self.canv.circle(cx, cy, radius, fill=1, stroke=0)

        # Regime text
        self.canv.setFillColor(colors.HexColor('#1f2937'))
        self.canv.setFont("Helvetica-Bold", 11)
        self.canv.drawString(cx + 20, cy + 4, self.regime.upper())

        # VIX indicator if available
        if self.vix:
            vix_x = 150
            vix_color = "#22c55e" if self.vix < 15 else "#eab308" if self.vix < 25 else "#ef4444"
            self.canv.setFillColor(colors.HexColor(vix_color))
            self.canv.circle(vix_x, cy, 8, fill=1, stroke=0)
            self.canv.setFillColor(colors.HexColor('#374151'))
            self.canv.setFont("Helvetica", 9)
            self.canv.drawString(vix_x + 15, cy + 3, f"VIX: {self.vix:.1f}")

        # Yield curve indicator if available
        if self.yield_curve is not None:
            yc_x = 260
            yc_color = "#22c55e" if self.yield_curve > 0 else "#ef4444"
            curve_text = "Normal" if self.yield_curve > 0 else "Inverted"
            self.canv.setFillColor(colors.HexColor(yc_color))
            self.canv.circle(yc_x, cy, 8, fill=1, stroke=0)
            self.canv.setFillColor(colors.HexColor('#374151'))
            self.canv.setFont("Helvetica", 9)
            self.canv.drawString(yc_x + 15, cy + 3, f"Yield: {curve_text}")


class GrowthValueScatterplot(Flowable):
    """Quadrant scatterplot showing growth vs value positioning"""

    def __init__(self, width, height, target_symbol, growth_score, value_score, peers=None):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.target_symbol = target_symbol
        self.growth_score = growth_score or 50
        self.value_score = value_score or 50
        self.peers = peers or []

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        # Chart dimensions
        chart_left = 40
        chart_bottom = 30
        chart_width = self.width - 60
        chart_height = self.height - 50

        # Draw axes
        self.canv.setStrokeColor(colors.HexColor('#9ca3af'))
        self.canv.setLineWidth(1)
        # X axis (Value)
        self.canv.line(chart_left, chart_bottom, chart_left + chart_width, chart_bottom)
        # Y axis (Growth)
        self.canv.line(chart_left, chart_bottom, chart_left, chart_bottom + chart_height)

        # Quadrant lines (at 50)
        self.canv.setStrokeColor(colors.HexColor('#e5e7eb'))
        self.canv.setDash([3, 3])
        mid_x = chart_left + chart_width * 0.5
        mid_y = chart_bottom + chart_height * 0.5
        self.canv.line(mid_x, chart_bottom, mid_x, chart_bottom + chart_height)
        self.canv.line(chart_left, mid_y, chart_left + chart_width, mid_y)
        self.canv.setDash([])

        # Quadrant labels
        self.canv.setFillColor(colors.HexColor('#9ca3af'))
        self.canv.setFont("Helvetica", 7)
        self.canv.drawString(chart_left + 5, chart_bottom + chart_height - 10, "High Growth")
        self.canv.drawString(chart_left + 5, chart_bottom + 5, "Low Growth")
        self.canv.drawRightString(chart_left + chart_width - 5, chart_bottom + 5, "Deep Value")
        self.canv.drawString(chart_left + 5, chart_bottom + 5, "Growth Trap")

        # Quadrant colors (light backgrounds)
        # Top-right: High Growth + Deep Value = Best
        self.canv.setFillColor(colors.HexColor('#dcfce7'))  # Light green
        self.canv.rect(mid_x, mid_y, chart_width * 0.5, chart_height * 0.5, fill=1, stroke=0)
        # Top-left: High Growth + Expensive
        self.canv.setFillColor(colors.HexColor('#fef3c7'))  # Light yellow
        self.canv.rect(chart_left, mid_y, chart_width * 0.5, chart_height * 0.5, fill=1, stroke=0)
        # Bottom-right: Low Growth + Deep Value = Value Trap
        self.canv.setFillColor(colors.HexColor('#fef3c7'))  # Light yellow
        self.canv.rect(mid_x, chart_bottom, chart_width * 0.5, chart_height * 0.5, fill=1, stroke=0)
        # Bottom-left: Low Growth + Expensive = Worst
        self.canv.setFillColor(colors.HexColor('#fee2e2'))  # Light red
        self.canv.rect(chart_left, chart_bottom, chart_width * 0.5, chart_height * 0.5, fill=1, stroke=0)

        # Re-draw axes on top
        self.canv.setStrokeColor(colors.HexColor('#6b7280'))
        self.canv.setLineWidth(1)
        self.canv.line(chart_left, chart_bottom, chart_left + chart_width, chart_bottom)
        self.canv.line(chart_left, chart_bottom, chart_left, chart_bottom + chart_height)

        # Plot peers as small dots
        for peer in self.peers[:5]:
            score_breakdown = peer.get('score_breakdown') or {}
            p_growth = score_breakdown.get('growth', 50)
            p_value = score_breakdown.get('value', 50)
            px = chart_left + (p_value / 100) * chart_width
            py = chart_bottom + (p_growth / 100) * chart_height
            self.canv.setFillColor(colors.HexColor('#94a3b8'))
            self.canv.circle(px, py, 4, fill=1, stroke=0)

        # Plot target company as larger dot
        tx = chart_left + (self.value_score / 100) * chart_width
        ty = chart_bottom + (self.growth_score / 100) * chart_height
        self.canv.setFillColor(colors.HexColor('#3b82f6'))
        self.canv.circle(tx, ty, 8, fill=1, stroke=0)
        self.canv.setFillColor(colors.white)
        self.canv.setFont("Helvetica-Bold", 6)
        self.canv.drawCentredString(tx, ty - 2, self.target_symbol[:4])

        # Axis labels
        self.canv.setFillColor(colors.HexColor('#374151'))
        self.canv.setFont("Helvetica-Bold", 8)
        self.canv.drawCentredString(chart_left + chart_width / 2, 5, "Value Score →")
        self.canv.saveState()
        self.canv.translate(12, chart_bottom + chart_height / 2)
        self.canv.rotate(90)
        self.canv.drawCentredString(0, 0, "Growth Score →")
        self.canv.restoreState()


class ScoreRadarChart(Flowable):
    """Radar/spider chart for score breakdown visualization"""

    def __init__(self, width, height, scores):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.scores = scores or {}

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        import math

        cx = self.width / 2
        cy = self.height / 2 + 5
        radius = min(self.width, self.height) * 0.35

        # Score categories
        categories = [
            ('income_statement', 'Income'),
            ('cash_flow', 'Cash Flow'),
            ('balance_sheet', 'Balance'),
            ('growth', 'Growth'),
            ('value', 'Value'),
            ('business_quality', 'Quality'),
        ]

        n = len(categories)
        if n == 0:
            return

        # Draw concentric circles (guidelines)
        for pct in [0.25, 0.5, 0.75, 1.0]:
            r = radius * pct
            self.canv.setStrokeColor(colors.HexColor('#e5e7eb'))
            self.canv.setLineWidth(0.5)
            self.canv.circle(cx, cy, r, fill=0, stroke=1)

        # Draw axes and labels
        angles = []
        for i, (key, label) in enumerate(categories):
            angle = (2 * math.pi * i / n) - math.pi / 2  # Start from top
            angles.append(angle)

            # Axis line
            self.canv.setStrokeColor(colors.HexColor('#d1d5db'))
            self.canv.line(cx, cy, cx + radius * math.cos(angle), cy + radius * math.sin(angle))

            # Label
            label_r = radius + 15
            lx = cx + label_r * math.cos(angle)
            ly = cy + label_r * math.sin(angle)
            self.canv.setFillColor(colors.HexColor('#4b5563'))
            self.canv.setFont("Helvetica", 7)
            self.canv.drawCentredString(lx, ly - 3, label)

        # Draw score polygon
        points = []
        for i, (key, _) in enumerate(categories):
            score = self.scores.get(key, 50) / 100
            r = radius * score
            angle = angles[i]
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

        if points:
            # Fill polygon
            self.canv.setFillColor(colors.HexColor('#3b82f6'))
            self.canv.setFillAlpha(0.3)
            path = self.canv.beginPath()
            path.moveTo(points[0][0], points[0][1])
            for px, py in points[1:]:
                path.lineTo(px, py)
            path.close()
            self.canv.drawPath(path, fill=1, stroke=0)
            self.canv.setFillAlpha(1.0)

            # Draw polygon outline
            self.canv.setStrokeColor(colors.HexColor('#3b82f6'))
            self.canv.setLineWidth(2)
            path = self.canv.beginPath()
            path.moveTo(points[0][0], points[0][1])
            for px, py in points[1:]:
                path.lineTo(px, py)
            path.close()
            self.canv.drawPath(path, fill=0, stroke=1)

            # Draw score dots
            for px, py in points:
                self.canv.setFillColor(colors.HexColor('#3b82f6'))
                self.canv.circle(px, py, 3, fill=1, stroke=0)


class ValuationBarChart(Flowable):
    """Horizontal bar chart comparing valuation model results"""

    def __init__(self, width, height, current_price, models):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.current_price = current_price or 0
        self.models = models or {}

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        if not self.models or not self.current_price:
            return

        # Filter valid models
        valid_models = []
        for model_name, model_data in self.models.items():
            if isinstance(model_data, dict):
                fv = model_data.get('fair_value_per_share')
                if fv and fv > 0:
                    valid_models.append((model_name.upper(), fv))

        if not valid_models:
            return

        # Calculate scale - use current price as max if higher than all fair values
        all_prices = [fv for _, fv in valid_models] + [self.current_price]
        max_price = max(all_prices) * 1.15

        bar_height = 20
        bar_spacing = 6
        chart_left = 55
        chart_width = self.width - 90
        total_height = (bar_height + bar_spacing) * len(valid_models)

        # Draw background
        self.canv.setFillColor(colors.HexColor('#f9fafb'))
        self.canv.rect(chart_left - 5, 0, chart_width + 50, total_height + 25, fill=1, stroke=0)

        # Draw current price line (vertical dashed line)
        current_x = chart_left + (self.current_price / max_price) * chart_width
        self.canv.setStrokeColor(colors.HexColor('#dc2626'))
        self.canv.setLineWidth(2)
        self.canv.setDash([4, 2])
        self.canv.line(current_x, 0, current_x, total_height + 5)
        self.canv.setDash([])

        # Label current price at top
        self.canv.setFillColor(colors.HexColor('#dc2626'))
        self.canv.setFont("Helvetica-Bold", 8)
        self.canv.drawCentredString(current_x, total_height + 10, f"Current ${self.current_price:.0f}")

        # Draw bars from top to bottom
        for i, (name, fv) in enumerate(valid_models):
            y = total_height - (i + 1) * (bar_height + bar_spacing) + bar_spacing

            # Model label on left
            self.canv.setFillColor(colors.HexColor('#374151'))
            self.canv.setFont("Helvetica-Bold", 9)
            self.canv.drawRightString(chart_left - 8, y + 6, name)

            # Calculate bar width
            bar_width = max((fv / max_price) * chart_width, 5)  # Minimum width of 5

            # Bar color: green if fair value > current (undervalued), orange if < current (overvalued)
            bar_color = '#22c55e' if fv > self.current_price else '#f97316'
            self.canv.setFillColor(colors.HexColor(bar_color))
            self.canv.roundRect(chart_left, y, bar_width, bar_height, 4, fill=1, stroke=0)

            # Value label at end of bar
            self.canv.setFillColor(colors.HexColor('#1f2937'))
            self.canv.setFont("Helvetica-Bold", 9)
            label_x = chart_left + bar_width + 8
            self.canv.drawString(label_x, y + 6, f"${fv:.0f}")


class PriceTargetChart(Flowable):
    """Visual price target with support/resistance"""

    def __init__(self, width, height, current_price, target_price=None,
                 stop_loss=None, support=None, resistance=None):
        self.width = width
        self.height = height
        self.current = current_price or 0
        self.target = target_price
        self.stop = stop_loss
        self.support = support
        self.resistance = resistance

    def draw(self):
        if not self.current:
            return

        # Calculate price range
        prices = [self.current]
        if self.target: prices.append(self.target)
        if self.stop: prices.append(self.stop)
        if self.support: prices.append(self.support)
        if self.resistance: prices.append(self.resistance)

        min_p = min(prices) * 0.95
        max_p = max(prices) * 1.05
        range_p = max_p - min_p

        if range_p <= 0:
            return

        def price_to_y(p):
            return 20 + ((p - min_p) / range_p) * (self.height - 40)

        # Background
        self.canv.setFillColor(colors.HexColor('#f9fafb'))
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)

        # Current price line
        y = price_to_y(self.current)
        self.canv.setStrokeColor(colors.HexColor('#3b82f6'))
        self.canv.setLineWidth(2)
        self.canv.line(30, y, self.width - 60, y)
        self.canv.setFillColor(colors.HexColor('#3b82f6'))
        self.canv.circle(30, y, 4, fill=1)
        self.canv.setFont("Helvetica-Bold", 9)
        self.canv.drawRightString(self.width - 5, y - 3, f"${self.current:.2f}")
        self.canv.setFont("Helvetica", 7)
        self.canv.drawString(5, y + 5, "Current")

        # Target price (green, dashed)
        if self.target:
            y = price_to_y(self.target)
            self.canv.setStrokeColor(colors.HexColor('#22c55e'))
            self.canv.setDash([4, 2])
            self.canv.line(30, y, self.width - 60, y)
            self.canv.setDash([])
            self.canv.setFillColor(colors.HexColor('#22c55e'))
            self.canv.drawRightString(self.width - 5, y - 3, f"${self.target:.2f}")
            self.canv.setFont("Helvetica", 7)
            self.canv.drawString(5, y + 5, "Target")

        # Stop loss (red, dashed)
        if self.stop:
            y = price_to_y(self.stop)
            self.canv.setStrokeColor(colors.HexColor('#ef4444'))
            self.canv.setDash([4, 2])
            self.canv.line(30, y, self.width - 60, y)
            self.canv.setDash([])
            self.canv.setFillColor(colors.HexColor('#ef4444'))
            self.canv.drawRightString(self.width - 5, y - 3, f"${self.stop:.2f}")
            self.canv.setFont("Helvetica", 7)
            self.canv.drawString(5, y + 5, "Stop")


class FinancialMetricsTable(Flowable):
    """Financial metrics comparison table: Company vs Sector vs Peers"""

    def __init__(self, width, height, metrics: Dict):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.metrics = metrics or {}

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        if not self.metrics:
            return

        # Table configuration
        col_widths = [120, 70, 70, 70]
        row_height = 18
        header_height = 22
        x_start = 10
        y_start = self.height - 5

        # Header
        headers = ['Metric', 'Company', 'Sector', 'Status']
        self.canv.setFillColor(colors.HexColor('#1f2937'))
        self.canv.rect(x_start, y_start - header_height, sum(col_widths), header_height, fill=1, stroke=0)

        self.canv.setFillColor(colors.white)
        self.canv.setFont("Helvetica-Bold", 9)
        x = x_start + 5
        for i, header in enumerate(headers):
            self.canv.drawString(x, y_start - 15, header)
            x += col_widths[i]

        # Rows
        metric_order = [
            ('pe_ratio', 'P/E Ratio', 'lower'),
            ('ev_ebitda', 'EV/EBITDA', 'lower'),
            ('roe', 'ROE', 'higher'),
            ('debt_to_equity', 'Debt/Equity', 'lower'),
            ('fcf_margin', 'FCF Margin', 'higher'),
            ('revenue_growth', 'Rev Growth', 'higher'),
        ]

        y = y_start - header_height - row_height
        for key, label, better in metric_order:
            data = self.metrics.get(key, {})
            company_val = data.get('company')
            sector_val = data.get('sector')

            if company_val is None:
                continue

            # Alternate row colors
            row_idx = metric_order.index((key, label, better))
            if row_idx % 2 == 0:
                self.canv.setFillColor(colors.HexColor('#f9fafb'))
                self.canv.rect(x_start, y, sum(col_widths), row_height, fill=1, stroke=0)

            # Metric label
            self.canv.setFillColor(colors.HexColor('#374151'))
            self.canv.setFont("Helvetica", 9)
            self.canv.drawString(x_start + 5, y + 5, label)

            # Company value
            x = x_start + col_widths[0] + 5
            self.canv.setFont("Helvetica-Bold", 9)
            if isinstance(company_val, (int, float)):
                if key in ['roe', 'fcf_margin', 'revenue_growth']:
                    val_str = f"{company_val:.1f}%"
                elif key in ['pe_ratio', 'ev_ebitda', 'debt_to_equity']:
                    val_str = f"{company_val:.1f}x" if key != 'debt_to_equity' else f"{company_val:.2f}"
                else:
                    val_str = f"{company_val:.1f}"
            else:
                val_str = str(company_val) if company_val else "N/A"
            self.canv.drawString(x, y + 5, val_str)

            # Sector value
            x = x_start + col_widths[0] + col_widths[1] + 5
            self.canv.setFont("Helvetica", 9)
            self.canv.setFillColor(colors.HexColor('#6b7280'))
            if isinstance(sector_val, (int, float)):
                if key in ['roe', 'fcf_margin', 'revenue_growth']:
                    val_str = f"{sector_val:.1f}%"
                elif key in ['pe_ratio', 'ev_ebitda', 'debt_to_equity']:
                    val_str = f"{sector_val:.1f}x" if key != 'debt_to_equity' else f"{sector_val:.2f}"
                else:
                    val_str = f"{sector_val:.1f}"
            else:
                val_str = str(sector_val) if sector_val else "N/A"
            self.canv.drawString(x, y + 5, val_str)

            # Status indicator
            x = x_start + col_widths[0] + col_widths[1] + col_widths[2] + 5
            if company_val is not None and sector_val is not None:
                if better == 'higher':
                    is_good = company_val >= sector_val
                else:
                    is_good = company_val <= sector_val

                indicator_color = colors.HexColor('#22c55e') if is_good else colors.HexColor('#ef4444')
                self.canv.setFillColor(indicator_color)
                self.canv.circle(x + 15, y + 8, 5, fill=1, stroke=0)

            y -= row_height


class TrendChart(Flowable):
    """Simple trend line chart for financial metrics over time"""

    def __init__(self, width, height, data_points: List, label: str, format_type: str = 'currency'):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.data_points = data_points or []  # List of (year, value) tuples
        self.label = label
        self.format_type = format_type  # 'currency', 'percent', 'ratio'

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        if len(self.data_points) < 2:
            # Not enough data
            self.canv.setFillColor(colors.HexColor('#9ca3af'))
            self.canv.setFont("Helvetica", 8)
            self.canv.drawCentredString(self.width/2, self.height/2, f"{self.label}: Insufficient data")
            return

        # Chart area
        margin_left = 45
        margin_right = 10
        margin_top = 20
        margin_bottom = 25
        chart_width = self.width - margin_left - margin_right
        chart_height = self.height - margin_top - margin_bottom

        # Background
        self.canv.setFillColor(colors.HexColor('#f9fafb'))
        self.canv.rect(margin_left, margin_bottom, chart_width, chart_height, fill=1, stroke=0)

        # Label
        self.canv.setFillColor(colors.HexColor('#374151'))
        self.canv.setFont("Helvetica-Bold", 9)
        self.canv.drawString(margin_left, self.height - 12, self.label)

        # Calculate scales
        values = [v for _, v in self.data_points if v is not None]
        if not values:
            return

        min_val = min(values) * 0.9
        max_val = max(values) * 1.1
        if max_val == min_val:
            max_val = min_val + 1

        # Draw trend line
        self.canv.setStrokeColor(colors.HexColor('#3b82f6'))
        self.canv.setLineWidth(2)

        points = []
        for i, (year, value) in enumerate(self.data_points):
            if value is None:
                continue
            x = margin_left + (i / (len(self.data_points) - 1)) * chart_width
            y = margin_bottom + ((value - min_val) / (max_val - min_val)) * chart_height
            points.append((x, y))

        if len(points) >= 2:
            path = self.canv.beginPath()
            path.moveTo(points[0][0], points[0][1])
            for px, py in points[1:]:
                path.lineTo(px, py)
            self.canv.drawPath(path, fill=0, stroke=1)

        # Draw points
        for px, py in points:
            self.canv.setFillColor(colors.HexColor('#3b82f6'))
            self.canv.circle(px, py, 3, fill=1, stroke=0)

        # X-axis labels (years)
        self.canv.setFillColor(colors.HexColor('#6b7280'))
        self.canv.setFont("Helvetica", 7)
        for i, (year, _) in enumerate(self.data_points):
            x = margin_left + (i / (len(self.data_points) - 1)) * chart_width
            self.canv.drawCentredString(x, margin_bottom - 12, str(year))

        # Y-axis labels (min/max)
        self.canv.setFont("Helvetica", 7)
        if self.format_type == 'currency':
            min_str = f"${min_val/1e9:.1f}B" if min_val >= 1e9 else f"${min_val/1e6:.0f}M"
            max_str = f"${max_val/1e9:.1f}B" if max_val >= 1e9 else f"${max_val/1e6:.0f}M"
        elif self.format_type == 'percent':
            min_str = f"{min_val:.1f}%"
            max_str = f"{max_val:.1f}%"
        else:
            min_str = f"{min_val:.1f}"
            max_str = f"{max_val:.1f}"

        self.canv.drawRightString(margin_left - 3, margin_bottom, min_str)
        self.canv.drawRightString(margin_left - 3, margin_bottom + chart_height - 5, max_str)


class InvestmentActionBox(Flowable):
    """Investment action plan visual box with entry/exit levels"""

    def __init__(self, width, height, action_data: Dict):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.action_data = action_data or {}

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        # Box background
        self.canv.setFillColor(colors.HexColor('#f0f9ff'))
        self.canv.roundRect(0, 0, self.width, self.height, 8, fill=1, stroke=0)

        # Border
        self.canv.setStrokeColor(colors.HexColor('#0ea5e9'))
        self.canv.setLineWidth(1)
        self.canv.roundRect(0, 0, self.width, self.height, 8, fill=0, stroke=1)

        # Title
        self.canv.setFillColor(colors.HexColor('#0369a1'))
        self.canv.setFont("Helvetica-Bold", 11)
        self.canv.drawString(12, self.height - 20, "Investment Action Plan")

        # Content
        y = self.height - 40
        col1_x = 12
        col2_x = self.width / 2 + 10

        # Entry zone
        entry_low = self.action_data.get('entry_low')
        entry_high = self.action_data.get('entry_high')
        self.canv.setFillColor(colors.HexColor('#374151'))
        self.canv.setFont("Helvetica-Bold", 9)
        self.canv.drawString(col1_x, y, "Entry Zone:")
        self.canv.setFont("Helvetica", 9)
        self.canv.setFillColor(colors.HexColor('#22c55e'))
        if entry_low and entry_high:
            self.canv.drawString(col1_x + 65, y, f"${entry_low:.2f} - ${entry_high:.2f}")
        elif entry_low:
            self.canv.drawString(col1_x + 65, y, f"Below ${entry_low:.2f}")

        # Target price
        target = self.action_data.get('target_price')
        self.canv.setFillColor(colors.HexColor('#374151'))
        self.canv.setFont("Helvetica-Bold", 9)
        self.canv.drawString(col2_x, y, "Target:")
        self.canv.setFont("Helvetica", 9)
        self.canv.setFillColor(colors.HexColor('#22c55e'))
        if target:
            self.canv.drawString(col2_x + 50, y, f"${target:.2f}")

        y -= 18

        # Stop loss
        stop = self.action_data.get('stop_loss')
        self.canv.setFillColor(colors.HexColor('#374151'))
        self.canv.setFont("Helvetica-Bold", 9)
        self.canv.drawString(col1_x, y, "Stop Loss:")
        self.canv.setFont("Helvetica", 9)
        self.canv.setFillColor(colors.HexColor('#ef4444'))
        if stop:
            self.canv.drawString(col1_x + 65, y, f"${stop:.2f}")

        # Risk/Reward ratio
        rr_ratio = self.action_data.get('risk_reward_ratio')
        self.canv.setFillColor(colors.HexColor('#374151'))
        self.canv.setFont("Helvetica-Bold", 9)
        self.canv.drawString(col2_x, y, "Risk/Reward:")
        self.canv.setFont("Helvetica", 9)
        if rr_ratio:
            rr_color = '#22c55e' if rr_ratio >= 2 else '#eab308' if rr_ratio >= 1 else '#ef4444'
            self.canv.setFillColor(colors.HexColor(rr_color))
            self.canv.drawString(col2_x + 80, y, f"1:{rr_ratio:.1f}")

        y -= 18

        # Position size
        position = self.action_data.get('position_size', 'MODERATE')
        self.canv.setFillColor(colors.HexColor('#374151'))
        self.canv.setFont("Helvetica-Bold", 9)
        self.canv.drawString(col1_x, y, "Position Size:")
        self.canv.setFont("Helvetica", 9)
        self.canv.setFillColor(colors.HexColor('#6b7280'))
        self.canv.drawString(col1_x + 80, y, position)

        # Time horizon
        horizon = self.action_data.get('time_horizon', 'MEDIUM-TERM')
        self.canv.setFillColor(colors.HexColor('#374151'))
        self.canv.setFont("Helvetica-Bold", 9)
        self.canv.drawString(col2_x, y, "Horizon:")
        self.canv.setFont("Helvetica", 9)
        self.canv.setFillColor(colors.HexColor('#6b7280'))
        self.canv.drawString(col2_x + 55, y, horizon)


class ValuationMethodologyBox(Flowable):
    """Callout box explaining valuation methodology and assumptions"""

    def __init__(self, width, height, methodology: Dict):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.methodology = methodology or {}

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        # Box background
        self.canv.setFillColor(colors.HexColor('#fefce8'))
        self.canv.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=0)

        # Border
        self.canv.setStrokeColor(colors.HexColor('#ca8a04'))
        self.canv.setLineWidth(0.5)
        self.canv.roundRect(0, 0, self.width, self.height, 6, fill=0, stroke=1)

        # Title
        self.canv.setFillColor(colors.HexColor('#854d0e'))
        self.canv.setFont("Helvetica-Bold", 9)
        self.canv.drawString(10, self.height - 15, "Valuation Assumptions")

        # Content
        y = self.height - 32
        self.canv.setFont("Helvetica", 8)
        self.canv.setFillColor(colors.HexColor('#713f12'))

        assumptions = [
            ('WACC', self.methodology.get('wacc'), '%'),
            ('Terminal Growth', self.methodology.get('terminal_growth'), '%'),
            ('Target P/E', self.methodology.get('target_pe'), 'x'),
            ('Target P/S', self.methodology.get('target_ps'), 'x'),
        ]

        x = 10
        for label, value, suffix in assumptions:
            if value:
                text = f"{label}: {value:.1f}{suffix}"
                self.canv.drawString(x, y, text)
                x += 95

        # Confidence note
        confidence = self.methodology.get('confidence', 'Medium')
        y -= 15
        self.canv.setFont("Helvetica-Oblique", 8)
        self.canv.drawString(10, y, f"Model confidence: {confidence}")


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    title: str = "Investment Analysis"
    margin: float = 0.6 * inch
    include_disclaimer: bool = True


class ProfessionalReportGenerator:
    """
    Clean, professional investment report generator.

    Produces concise 2-3 page reports with:
    - Executive summary with recommendation
    - Key metrics with visual gauges
    - Price targets visualization
    - Brief risk/catalyst summary
    """

    def __init__(self, output_dir: Path = None, config: ReportConfig = None):
        self.output_dir = Path(output_dir) if output_dir else Path("reports/professional")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ReportConfig()
        self.styles = self._create_styles()

    def _create_styles(self):
        """Create custom paragraph styles"""
        styles = getSampleStyleSheet()

        styles.add(ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f2937'),
            spaceAfter=6,
            alignment=TA_CENTER
        ))

        styles.add(ParagraphStyle(
            'Symbol',
            parent=styles['Heading1'],
            fontSize=36,
            textColor=colors.HexColor('#1f2937'),
            spaceBefore=12,
            spaceAfter=0,
            alignment=TA_CENTER
        ))

        styles.add(ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#374151'),
            spaceBefore=18,
            spaceAfter=8,
            borderWidth=0,
            borderPadding=0
        ))

        styles.add(ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#4b5563'),
            spaceBefore=4,
            spaceAfter=4,
            leading=14
        ))

        styles.add(ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=7,
            textColor=colors.HexColor('#9ca3af'),
            alignment=TA_CENTER
        ))

        return styles

    def generate_report(self, data: Dict[str, Any]) -> str:
        """
        Generate professional investment report.

        Args:
            data: Dict with keys:
                - symbol: Stock ticker
                - recommendation: BUY/HOLD/SELL
                - confidence: HIGH/MEDIUM/LOW
                - overall_score: 0-100
                - fundamental_score: 0-100
                - technical_score: 0-100
                - current_price: float
                - target_price: float (optional)
                - stop_loss: float (optional)
                - investment_thesis: str
                - key_catalysts: List[str]
                - key_risks: List[str]

        Returns:
            Path to generated PDF
        """
        if not REPORTLAB_AVAILABLE:
            logger.warning("reportlab not available")
            return ""

        symbol = data.get('symbol', 'UNKNOWN')
        filename = f"{symbol}_professional_report.pdf"
        filepath = self.output_dir / filename

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=self.config.margin,
            leftMargin=self.config.margin,
            topMargin=self.config.margin,
            bottomMargin=self.config.margin
        )

        story = self._build_story(data)
        doc.build(story)

        logger.info(f"Generated professional report: {filepath}")
        return str(filepath)

    def _build_story(self, data: Dict) -> List:
        """Build the report content"""
        story = []
        symbol = data.get('symbol', 'UNKNOWN')

        # Header
        story.append(Paragraph("Investment Analysis Report", self.styles['ReportTitle']))
        story.append(Paragraph(symbol, self.styles['Symbol']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            datetime.now().strftime('%B %d, %Y'),
            self.styles['ReportBody']
        ))
        story.append(Spacer(1, 12))

        # Recommendation badge
        rec = data.get('recommendation', 'HOLD')
        conf = data.get('confidence', 'MEDIUM')
        story.append(KeepTogether([
            RecommendationBadge(120, 50, rec, conf)
        ]))
        story.append(Spacer(1, 20))

        # Score gauges row
        overall = data.get('overall_score', 50)
        fundamental = data.get('fundamental_score', 50)
        technical = data.get('technical_score', 50)

        # Create gauge table
        gauge_data = [[
            GaugeChart(120, 80, overall, 100, "Overall"),
            GaugeChart(120, 80, fundamental, 100, "Fundamental"),
            GaugeChart(120, 80, technical, 100, "Technical")
        ]]
        gauge_table = Table(gauge_data, colWidths=[2.3*inch, 2.3*inch, 2.3*inch])
        gauge_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(gauge_table)
        story.append(Spacer(1, 20))

        # Price targets section
        current = data.get('current_price')
        target = data.get('target_price') or data.get('price_target')
        stop = data.get('stop_loss')

        if current:
            story.append(Paragraph("Price Levels", self.styles['SectionHeader']))
            story.append(PriceTargetChart(
                400, 100, current, target, stop
            ))

            # Upside/downside calculation
            if target and current:
                upside = ((target - current) / current) * 100
                upside_text = f"Upside: {upside:+.1f}%" if upside >= 0 else f"Downside: {upside:.1f}%"
                color = '#22c55e' if upside >= 0 else '#ef4444'
                story.append(Paragraph(
                    f'<font color="{color}"><b>{upside_text}</b></font>',
                    self.styles['ReportBody']
                ))
            story.append(Spacer(1, 16))

        # Investment Action Plan box
        if current and (target or stop):
            # Calculate risk/reward ratio
            rr_ratio = None
            if target and stop and current:
                potential_gain = target - current
                potential_loss = current - stop
                if potential_loss > 0:
                    rr_ratio = potential_gain / potential_loss

            action_data = {
                'entry_low': stop * 1.02 if stop else current * 0.95,  # Entry zone above stop loss
                'entry_high': current,
                'target_price': target,
                'stop_loss': stop,
                'risk_reward_ratio': rr_ratio,
                'position_size': data.get('position_size', 'MODERATE'),
                'time_horizon': data.get('time_horizon', 'MEDIUM-TERM'),
            }
            story.append(InvestmentActionBox(450, 90, action_data))
            story.append(Spacer(1, 16))

        # Investment thesis
        thesis = data.get('investment_thesis', data.get('executive_summary', ''))
        if thesis:
            story.append(Paragraph("Investment Thesis", self.styles['SectionHeader']))
            story.append(Paragraph(thesis[:800], self.styles['ReportBody']))
            story.append(Spacer(1, 12))

        # Overall reasoning (if available)
        reasoning = data.get('reasoning', '')
        if reasoning and isinstance(reasoning, str):
            story.append(Paragraph("Analysis Rationale", self.styles['SectionHeader']))
            story.append(Paragraph(reasoning, self.styles['ReportBody']))
            story.append(Spacer(1, 12))

        # Key catalysts and risks in two columns
        catalysts = data.get('key_catalysts', [])
        risks = data.get('key_risks', [])

        if catalysts or risks:
            # Catalysts column
            catalyst_content = []
            if catalysts:
                catalyst_content.append(Paragraph(
                    '<font color="#22c55e"><b>Key Catalysts</b></font>',
                    self.styles['ReportBody']
                ))
                for c in catalysts[:4]:
                    catalyst_content.append(Paragraph(
                        f'<bullet>&bull;</bullet> {c}',
                        self.styles['ReportBody']
                    ))

            # Risks column
            risk_content = []
            if risks:
                risk_content.append(Paragraph(
                    '<font color="#ef4444"><b>Key Risks</b></font>',
                    self.styles['ReportBody']
                ))
                for r in risks[:4]:
                    risk_content.append(Paragraph(
                        f'<bullet>&bull;</bullet> {r}',
                        self.styles['ReportBody']
                    ))

            if catalyst_content or risk_content:
                col_table = Table(
                    [[catalyst_content, risk_content]],
                    colWidths=[3.3*inch, 3.3*inch]
                )
                col_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ]))
                story.append(col_table)
                story.append(Spacer(1, 16))

        # Score Breakdown Table (detailed component scores)
        score_breakdown = data.get('score_breakdown', {})
        if score_breakdown:
            story.append(Paragraph("Detailed Score Breakdown", self.styles['SectionHeader']))

            score_rows = [['Component', 'Score', 'Assessment']]
            component_order = [
                ('income_statement', 'Income Statement'),
                ('cash_flow', 'Cash Flow'),
                ('balance_sheet', 'Balance Sheet'),
                ('growth', 'Growth'),
                ('value', 'Value'),
                ('business_quality', 'Business Quality'),
                ('data_quality', 'Data Quality'),
            ]

            for key, label in component_order:
                score = score_breakdown.get(key)
                if score is not None:
                    if score >= 70:
                        assessment = 'Strong'
                        color = '#22c55e'
                    elif score >= 50:
                        assessment = 'Moderate'
                        color = '#eab308'
                    elif score >= 30:
                        assessment = 'Weak'
                        color = '#f97316'
                    else:
                        assessment = 'Poor'
                        color = '#ef4444'
                    score_rows.append([
                        label,
                        f'{score:.0f}',
                        Paragraph(f'<font color="{color}"><b>{assessment}</b></font>', self.styles['ReportBody'])
                    ])

            if len(score_rows) > 1:
                score_table = Table(score_rows, colWidths=[2*inch, 0.8*inch, 1.5*inch])
                score_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#374151')),
                    ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(score_table)
                story.append(Spacer(1, 16))

        # Financial Metrics Dashboard - Company vs Sector comparison
        financial_metrics = data.get('financial_metrics', {})
        if financial_metrics:
            story.append(Paragraph("Financial Metrics: Company vs Sector", self.styles['SectionHeader']))
            # Calculate table height based on number of metrics
            num_metrics = len([k for k in financial_metrics.keys() if financial_metrics.get(k, {}).get('company') is not None])
            table_height = 30 + min(num_metrics, 6) * 20
            story.append(FinancialMetricsTable(400, table_height, financial_metrics))
            story.append(Spacer(1, 16))

        # Historical Trend Charts
        historical_data = data.get('historical_financials', {})
        if historical_data:
            story.append(Paragraph("Historical Trends", self.styles['SectionHeader']))

            # Build trend charts row
            trend_charts = []

            # Revenue trend
            revenue_history = historical_data.get('revenue', [])
            if revenue_history and len(revenue_history) >= 2:
                trend_charts.append(TrendChart(150, 80, revenue_history, "Revenue", 'currency'))

            # Free Cash Flow trend
            fcf_history = historical_data.get('free_cash_flow', [])
            if fcf_history and len(fcf_history) >= 2:
                trend_charts.append(TrendChart(150, 80, fcf_history, "Free Cash Flow", 'currency'))

            # ROE trend
            roe_history = historical_data.get('roe', [])
            if roe_history and len(roe_history) >= 2:
                trend_charts.append(TrendChart(150, 80, roe_history, "ROE", 'percent'))

            if trend_charts:
                # Arrange in table (up to 3 per row)
                while len(trend_charts) < 3:
                    trend_charts.append(Spacer(150, 80))  # Placeholder for empty cells
                trend_table = Table([trend_charts[:3]], colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
                trend_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(trend_table)
                story.append(Spacer(1, 16))

        # Score breakdown visual charts row: Growth/Value Scatterplot + Score Radar
        if score_breakdown:
            growth_score = score_breakdown.get('growth', 50)
            value_score = score_breakdown.get('value', 50)
            peer_comparison = data.get('peer_comparison', {})
            peers = peer_comparison.get('peers', [])

            chart_row = [[
                GrowthValueScatterplot(220, 150, symbol, growth_score, value_score, peers),
                ScoreRadarChart(220, 150, score_breakdown)
            ]]
            chart_table = Table(chart_row, colWidths=[3.3*inch, 3.3*inch])
            chart_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(Paragraph("Score Analysis Charts", self.styles['SectionHeader']))
            story.append(chart_table)
            story.append(Spacer(1, 16))

        # SEC Fundamental Analysis Thinking section
        fundamental_thinking = data.get('fundamental_analysis_thinking', '')
        if fundamental_thinking and isinstance(fundamental_thinking, str):
            story.append(Paragraph(
                '<font color="#1d4ed8"><b>SEC Fundamental Analysis Thinking</b></font>',
                self.styles['SectionHeader']
            ))
            # Split into paragraphs for better readability
            paragraphs = fundamental_thinking.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    story.append(Paragraph(para.strip(), self.styles['ReportBody']))
            story.append(Spacer(1, 16))

        # Technical Analysis Thinking section
        technical_thinking = data.get('technical_analysis_thinking', '')
        if technical_thinking and isinstance(technical_thinking, str):
            story.append(Paragraph(
                '<font color="#7c3aed"><b>Technical Analysis Thinking</b></font>',
                self.styles['SectionHeader']
            ))
            # Split into paragraphs for better readability
            paragraphs = technical_thinking.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    story.append(Paragraph(para.strip(), self.styles['ReportBody']))
            story.append(Spacer(1, 16))

        # Key Technical Signals
        key_signals = data.get('key_technical_signals', [])
        if key_signals and isinstance(key_signals, list):
            story.append(Paragraph(
                '<font color="#7c3aed"><b>Key Technical Signals</b></font>',
                self.styles['SectionHeader']
            ))
            for signal in key_signals[:5]:
                if isinstance(signal, str):
                    story.append(Paragraph(
                        f'<bullet>&bull;</bullet> {signal}',
                        self.styles['ReportBody']
                    ))
            story.append(Spacer(1, 12))

        # Detailed Risk Factors
        risk_factors = data.get('risk_factors_detailed', [])
        if risk_factors and isinstance(risk_factors, list):
            story.append(Paragraph(
                '<font color="#dc2626"><b>Detailed Risk Factors</b></font>',
                self.styles['SectionHeader']
            ))
            for risk in risk_factors[:5]:
                if isinstance(risk, str):
                    story.append(Paragraph(
                        f'<bullet>&bull;</bullet> {risk}',
                        self.styles['ReportBody']
                    ))
            story.append(Spacer(1, 12))

        # Valuation Models table (if available)
        valuation_models = data.get('valuation_models', {})
        if valuation_models:
            story.append(Paragraph("Valuation Models", self.styles['SectionHeader']))

            # Build valuation table data
            val_header = ['Model', 'Fair Value', 'Upside', 'Confidence']
            val_rows = [val_header]
            for model_name, model_data in valuation_models.items():
                if isinstance(model_data, dict):
                    fv = model_data.get('fair_value_per_share')
                    # Handle both key names: upside_percent (sector router) and upside_downside_pct (DCF, GGM)
                    upside = model_data.get('upside_percent') or model_data.get('upside_downside_pct') or model_data.get('upside_pct')
                    conf = model_data.get('confidence')
                    if fv:
                        upside_str = f"{upside:+.1f}%" if upside is not None else "N/A"
                        conf_str = f"{conf:.0f}%" if conf else "N/A"
                        val_rows.append([
                            model_name.upper(),
                            f"${fv:.2f}",
                            upside_str,
                            conf_str
                        ])

            if len(val_rows) > 1:
                val_table = Table(val_rows, colWidths=[1.5*inch, 1.2*inch, 1*inch, 1*inch])
                val_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#374151')),
                    ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(val_table)
                story.append(Spacer(1, 8))

                # Valuation Bar Chart visualization
                current_price = data.get('current_price')
                if current_price and current_price > 0:
                    valid_model_count = len([m for m in valuation_models.values() if isinstance(m, dict) and m.get('fair_value_per_share')])
                    if valid_model_count > 0:
                        bar_chart_height = 35 + 28 * valid_model_count
                        story.append(Paragraph("Fair Value Comparison", self.styles['SectionHeader']))
                        story.append(ValuationBarChart(450, bar_chart_height, current_price, valuation_models))
                        story.append(Spacer(1, 16))

            # Valuation Methodology explanation box
            methodology = data.get('valuation_methodology', {})
            # If no explicit methodology data, extract from valuation models
            if not methodology:
                dcf = valuation_models.get('dcf', {})
                if dcf:
                    methodology = {
                        'wacc': dcf.get('wacc', dcf.get('discount_rate')),
                        'terminal_growth': dcf.get('terminal_growth_rate'),
                        'confidence': dcf.get('confidence'),
                    }
                # Add PE/PS targets if available
                pe_model = valuation_models.get('pe', {})
                if pe_model:
                    methodology['target_pe'] = pe_model.get('target_pe', pe_model.get('pe_ratio'))
                ps_model = valuation_models.get('ps', {})
                if ps_model:
                    methodology['target_ps'] = ps_model.get('target_ps', ps_model.get('ps_ratio'))

            if methodology and any(methodology.values()):
                story.append(ValuationMethodologyBox(400, 60, methodology))
                story.append(Spacer(1, 16))

        # Technical Analysis section
        tech_data = data.get('technical_data', {})
        if tech_data:
            story.append(Paragraph("Technical Analysis", self.styles['SectionHeader']))

            tech_content = []
            signal = tech_data.get('overall_signal', 'NEUTRAL')
            strength = data.get('technical_strength', 'NEUTRAL')
            signal_color = '#22c55e' if signal.lower() == 'bullish' else '#ef4444' if signal.lower() == 'bearish' else '#6b7280'
            tech_content.append(Paragraph(
                f'Signal: <font color="{signal_color}"><b>{signal.upper()}</b></font> | Strength: {strength}',
                self.styles['ReportBody']
            ))

            # Support/Resistance levels
            sr = tech_data.get('support_resistance', {})
            if sr:
                support = sr.get('support_levels', {}).get('support_1')
                resistance = sr.get('resistance_levels', {}).get('resistance_1')
                w52 = sr.get('52_week', {})
                if support or resistance:
                    levels_text = []
                    if support:
                        levels_text.append(f"Support: ${support:.2f}")
                    if resistance:
                        levels_text.append(f"Resistance: ${resistance:.2f}")
                    tech_content.append(Paragraph(' | '.join(levels_text), self.styles['ReportBody']))
                if w52:
                    tech_content.append(Paragraph(
                        f"52-Week: ${w52.get('low', 0):.2f} - ${w52.get('high', 0):.2f}",
                        self.styles['ReportBody']
                    ))

            for p in tech_content:
                story.append(p)
            story.append(Spacer(1, 12))

        # Market Regime section with visual indicator
        market_regime = data.get('market_regime', {})
        if market_regime:
            story.append(Paragraph("Market Environment", self.styles['SectionHeader']))

            regime_name = market_regime.get('regime', 'Normal')
            vix = market_regime.get('vix')
            yield_curve = market_regime.get('yield_curve_slope')

            # Use visual indicator
            story.append(MarketRegimeIndicator(400, 35, regime_name, vix, yield_curve))
            story.append(Spacer(1, 12))

        # Valuation summary (from LLM)
        val_summary = data.get('valuation_summary', '')
        if val_summary:
            story.append(Paragraph("Valuation Summary", self.styles['SectionHeader']))
            story.append(Paragraph(val_summary, self.styles['ReportBody']))
            story.append(Spacer(1, 12))

        # Peer Comparison section with visual chart
        peer_comparison = data.get('peer_comparison', {})
        peers = peer_comparison.get('peers', [])
        peer_metrics = peer_comparison.get('metrics', {})
        peer_summary = peer_comparison.get('summary', '')

        if peers:
            story.append(Paragraph("Peer Comparison", self.styles['SectionHeader']))

            # Show peer summary if available
            if peer_summary:
                story.append(Paragraph(peer_summary, self.styles['ReportBody']))
                story.append(Spacer(1, 8))

            # Calculate target upside from data
            current = data.get('current_price')
            target = data.get('target_price')
            target_upside = ((target - current) / current * 100) if current and target else 0

            # Show upside comparison chart
            chart_height = 20 + 20 * min(len(peers), 4)  # Dynamic height based on peer count
            story.append(PeerComparisonChart(450, chart_height, symbol, target_upside, peers))
            story.append(Spacer(1, 12))

            # Build peer table
            peer_header = ['Peer', 'Match', 'Market Cap', 'P/E', 'Upside']
            peer_rows = [peer_header]
            for peer in peers[:5]:
                sym = peer.get('symbol', 'N/A')
                match_type = peer.get('match_type', 'sector').capitalize()
                mcap = peer.get('market_cap')
                mcap_str = f"${mcap/1e9:.1f}B" if mcap else "N/A"
                val = peer.get('valuation') or {}
                pe = val.get('pe_ratio')
                pe_str = f"{pe:.1f}x" if pe else "N/A"
                upside = val.get('upside_pct')
                upside_str = f"{upside:+.1f}%" if upside else "N/A"
                peer_rows.append([sym, match_type, mcap_str, pe_str, upside_str])

            if len(peer_rows) > 1:
                peer_table = Table(peer_rows, colWidths=[1*inch, 0.9*inch, 1.1*inch, 0.9*inch, 1*inch])
                peer_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#374151')),
                    ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(peer_table)

            # Show peer group medians
            if peer_metrics:
                median_parts = []
                if 'pe_ratio_median' in peer_metrics:
                    median_parts.append(f"Peer P/E Median: {peer_metrics['pe_ratio_median']:.1f}x")
                if 'upside_pct_median' in peer_metrics:
                    median_parts.append(f"Peer Upside Median: {peer_metrics['upside_pct_median']:+.1f}%")
                if median_parts:
                    story.append(Spacer(1, 6))
                    story.append(Paragraph(' | '.join(median_parts), self.styles['ReportBody']))

            story.append(Spacer(1, 12))

        # Time horizon and position size
        horizon = data.get('time_horizon', 'MEDIUM-TERM')
        position = data.get('position_size', 'MODERATE')

        meta_data = [
            ['Time Horizon', horizon],
            ['Position Size', position],
        ]
        meta_table = Table(meta_data, colWidths=[1.5*inch, 2*inch])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#4b5563')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(meta_table)

        # Disclaimer footer
        if self.config.include_disclaimer:
            story.append(Spacer(1, 30))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e5e7eb')))
            story.append(Spacer(1, 8))
            story.append(Paragraph(
                "AI-generated analysis for educational purposes only. Not investment advice. "
                "Consult a licensed financial advisor before making investment decisions.",
                self.styles['Disclaimer']
            ))

        return story


def generate_professional_report(data: Dict[str, Any], output_dir: Path = None) -> str:
    """Convenience function to generate a professional report"""
    generator = ProfessionalReportGenerator(output_dir=output_dir)
    return generator.generate_report(data)
