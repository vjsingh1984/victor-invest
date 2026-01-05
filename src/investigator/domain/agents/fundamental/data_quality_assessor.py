"""
Data Quality Assessor - Extracted from FundamentalAnalysisAgent for SRP.

This module handles data quality assessment:
- Comprehensive data quality scoring (0-100)
- Confidence level calculation based on data quality
- Single quarter quality assessment
- Quality grade mapping (Excellent, Good, Fair, Poor, Very Poor)

Part of Phase 5 refactoring to break up monolithic agent.py.

Author: InvestiGator Team
Date: 2025-01-05
"""

import logging
from typing import Any, Dict, List, Optional

from investigator.domain.services.data_normalizer import DataNormalizer

logger = logging.getLogger(__name__)


class DataQualityAssessor:
    """
    Assesses data quality and maps to confidence levels.

    Extracted from FundamentalAnalysisAgent to follow Single Responsibility Principle.
    All data quality assessment logic is centralized here.

    Quality assessment includes:
    - Completeness scoring (core metrics, market data, ratios)
    - Consistency checking (detecting impossible values)
    - Quality grade mapping (Excellent to Very Poor)
    - Confidence level derivation from data quality
    """

    # Required fields for quarter quality assessment
    QUARTER_REQUIRED_FIELDS = [
        "revenues",
        "net_income",
        "total_assets",
        "total_liabilities",
        "stockholders_equity",
        "operating_cash_flow",
        "capital_expenditures",
    ]

    # Ratio metrics checked for completeness
    RATIO_METRICS = [
        "pe_ratio",
        "price_to_book",
        "current_ratio",
        "debt_to_equity",
        "roe",
        "roa",
        "gross_margin",
        "operating_margin",
    ]

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize data quality assessor.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def assess_data_quality(self, company_data: Dict, ratios: Dict) -> Dict:
        """
        Assess the quality and completeness of financial data.
        Returns a data quality score (0-100) and detailed assessment.

        FEATURE #1: Data Quality Scoring (migrated from old solution)
        Similar to old solution's extraction-level quality scoring.

        Also tracks raw extraction quality vs enhanced quality to quantify
        value of data enrichment (market data integration, ratio calculations).

        ENHANCEMENT: Now uses DataNormalizer for schema harmonization and
        includes debt metrics in completeness scoring.

        Args:
            company_data: Company data dictionary with financials and market_data
            ratios: Calculated financial ratios

        Returns:
            Dictionary with quality assessment including:
                - data_quality_score: Overall score (0-100)
                - quality_grade: Grade string (Excellent, Good, Fair, Poor, Very Poor)
                - completeness_score: Completeness percentage
                - consistency_score: Consistency score (100 minus penalties)
                - core_metrics_populated: "X/Y" string
                - market_metrics_populated: "X/2" string
                - ratio_metrics_populated: "X/Y" string
                - consistency_issues: List of detected issues
                - assessment: Human-readable assessment string
                - extraction_quality: Quality before enrichment
                - quality_improvement: Points gained from enrichment
                - improvement_sources: List of enrichment sources
                - enhancement_summary: Summary of data enrichment impact
        """
        financials = company_data.get("financials") or {}
        market_data = company_data.get("market_data") or {}

        # Step 1: Normalize field names to snake_case for internal consistency
        # Note: DataNormalizer.assess_completeness() internally converts to camelCase
        # for checking against CORE_METRICS, so we normalize to snake_case here first
        normalized_financials = DataNormalizer.normalize_field_names(financials, to_camel_case=False)

        # Step 2: Use DataNormalizer's enhanced completeness assessment (includes debt metrics)
        completeness_assessment = DataNormalizer.assess_completeness(normalized_financials, include_debt_metrics=True)

        core_populated = completeness_assessment["core_metrics_count"]
        debt_populated = completeness_assessment["debt_metrics_count"]

        # Log warnings for missing debt metrics (explicit upstream gap tracking)
        if completeness_assessment["missing_debt"]:
            symbol = company_data.get("symbol", "UNKNOWN")
            missing_debt_str = ", ".join(completeness_assessment["missing_debt"])
            self.logger.warning(
                f"UPSTREAM DATA GAP for {symbol}: Missing debt metrics: {missing_debt_str}. "
                f"Debt-related ratios may be unreliable."
            )

        # Market data metrics (check both camelCase and snake_case for compatibility).
        # NOTE: this dual check keeps legacy prompts (marketCap) and new pipeline fields (market_cap)
        # in sync, which prevents false "1/2 populated" warnings in data quality logs.
        has_price = market_data.get("current_price", 0) != 0 or company_data.get("current_price", 0) != 0
        has_market_cap = (
            market_data.get("market_cap", 0) != 0
            or market_data.get("market_cap", 0) != 0
            or company_data.get("market_cap", 0) != 0
            or company_data.get("market_cap", 0) != 0
        )
        market_populated = (1 if has_price else 0) + (1 if has_market_cap else 0)

        # Calculated ratio metrics (from _calculate_financial_ratios)
        ratio_populated = sum(1 for m in self.RATIO_METRICS if ratios.get(m, 0) != 0)

        # Calculate completeness scores
        # Use DataNormalizer's score for core metrics, then add market/ratio components
        core_completeness = completeness_assessment["score"]  # Already includes debt metrics
        market_completeness = (market_populated / 2) * 100  # 2 metrics: price + market_cap
        ratio_completeness = (ratio_populated / len(self.RATIO_METRICS)) * 100

        # Overall completeness (weighted average)
        # Core+Debt: 50%, Market data: 25%, Ratios: 25%
        completeness_score = core_completeness * 0.50 + market_completeness * 0.25 + ratio_completeness * 0.25

        # Check for data consistency (red flags)
        consistency_issues = self._check_consistency(financials, ratios)

        # Calculate consistency score (100 if no issues, -10 for each issue)
        consistency_score = max(0, 100 - (len(consistency_issues) * 10))

        # ENHANCEMENT: Explicit warnings for zeroed critical ratios due to upstream gaps
        symbol = company_data.get("symbol", "UNKNOWN")
        DataNormalizer.validate_and_warn(ratios, symbol, self.logger)

        # Calculate overall data quality score
        # Completeness: 70%, Consistency: 30%
        data_quality_score = (completeness_score * 0.70) + (consistency_score * 0.30)

        # Determine quality grade
        quality_grade = self._score_to_grade(data_quality_score)

        # FEATURE #3: Enhanced vs Extraction Quality Comparison
        # Calculate "extraction quality" (raw financial data only, before enrichment)
        extraction_completeness = core_completeness  # Only SEC financial data
        extraction_quality = (extraction_completeness * 0.70) + (consistency_score * 0.30)

        # Calculate enhancement delta
        quality_improvement = data_quality_score - extraction_quality
        improvement_sources = []

        if market_populated > 0:
            improvement_sources.append(f"market data (+{market_populated} metrics)")
        if ratio_populated > 0:
            improvement_sources.append(f"calculated ratios (+{ratio_populated} metrics)")

        # Generate enhancement summary
        if quality_improvement > 0:
            enhancement_summary = (
                f"Data enrichment improved quality by {quality_improvement:.1f} points "
                f"({extraction_quality:.1f}% -> {data_quality_score:.1f}%) through: "
                f"{', '.join(improvement_sources)}"
            )
        else:
            enhancement_summary = "No data enrichment applied (extraction-only data)"

        return {
            "data_quality_score": round(data_quality_score, 1),
            "quality_grade": quality_grade,
            "completeness_score": round(completeness_score, 1),
            "consistency_score": round(consistency_score, 1),
            "core_metrics_populated": f"{core_populated}/{completeness_assessment['core_metrics_total']}",
            "market_metrics_populated": f"{market_populated}/2",
            "ratio_metrics_populated": f"{ratio_populated}/{len(self.RATIO_METRICS)}",
            "consistency_issues": consistency_issues,
            "assessment": f"Data quality is {quality_grade.lower()} with {completeness_score:.0f}% completeness",
            # FEATURE #3: Enhanced vs extraction quality tracking
            "extraction_quality": round(extraction_quality, 1),
            "quality_improvement": round(quality_improvement, 1),
            "improvement_sources": improvement_sources,
            "enhancement_summary": enhancement_summary,
        }

    def _check_consistency(self, financials: Dict, ratios: Dict) -> List[str]:
        """
        Check for data consistency issues (red flags).

        Args:
            financials: Financial data dictionary
            ratios: Calculated ratios dictionary

        Returns:
            List of consistency issue descriptions
        """
        consistency_issues = []

        # Check for impossible values (with None-safe comparisons)
        net_income = financials.get("net_income") or 0
        total_revenue = financials.get("revenues") or 0
        if net_income < 0 and total_revenue > 0:
            if abs(net_income) > total_revenue:
                consistency_issues.append("Net loss exceeds revenue (possible data error)")

        current_liabilities = financials.get("current_liabilities") or 0
        total_assets = financials.get("total_assets") or 0
        if current_liabilities > 0 and total_assets > 0 and current_liabilities > total_assets:
            consistency_issues.append("Current liabilities exceed total assets (data warning)")

        current_ratio = ratios.get("current_ratio") or 0
        if current_ratio > 100:  # Impossibly high current ratio
            consistency_issues.append("Unrealistic current ratio (possible unit error)")

        return consistency_issues

    def _score_to_grade(self, score: float) -> str:
        """
        Convert numeric score to quality grade.

        Args:
            score: Quality score (0-100)

        Returns:
            Quality grade string
        """
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 40:
            return "Poor"
        else:
            return "Very Poor"

    def calculate_confidence_level(self, data_quality: Dict) -> Dict:
        """
        Calculate confidence level based on data quality score.

        FEATURE #2: Confidence Level Adjustment (migrated from old solution concept)
        Maps data quality score to confidence level for investment decisions.

        Args:
            data_quality: Data quality assessment from assess_data_quality()

        Returns:
            Dict with:
                - confidence_level: VERY HIGH, HIGH, MODERATE, LOW, VERY LOW
                - confidence_score: Numeric score (0-100)
                - rationale: Explanation string
                - based_on_data_quality: Original data quality score
                - quality_grade: Original quality grade
        """
        data_quality_score = data_quality.get("data_quality_score", 0)
        quality_grade = data_quality.get("quality_grade", "Unknown")
        consistency_issues = data_quality.get("consistency_issues", [])

        # Map data quality score to confidence level
        if data_quality_score >= 90:
            confidence_level = "VERY HIGH"
            confidence_score = 95
            rationale = "Excellent data quality with complete, consistent financial metrics"
        elif data_quality_score >= 75:
            confidence_level = "HIGH"
            confidence_score = 85
            rationale = "Good data quality with minor gaps, analysis is reliable"
        elif data_quality_score >= 60:
            confidence_level = "MODERATE"
            confidence_score = 70
            rationale = "Fair data quality with some gaps, exercise caution in decision-making"
        elif data_quality_score >= 40:
            confidence_level = "LOW"
            confidence_score = 50
            rationale = "Poor data quality with significant gaps, recommendations should be treated with skepticism"
        else:
            confidence_level = "VERY LOW"
            confidence_score = 30
            rationale = "Very poor data quality, analysis may be unreliable, seek additional data sources"

        # Adjust confidence down if there are consistency issues
        if consistency_issues:
            confidence_score -= 10
            rationale += f" (adjusted down due to {len(consistency_issues)} data consistency issue(s))"

        return {
            "confidence_level": confidence_level,
            "confidence_score": confidence_score,
            "rationale": rationale,
            "based_on_data_quality": data_quality_score,
            "quality_grade": quality_grade,
        }

    def assess_quarter_quality(self, financial_data: Dict) -> Dict:
        """
        Assess data quality for a single quarter.

        Args:
            financial_data: Financial metrics for the quarter

        Returns:
            Dictionary with quality metrics:
                - completeness: Percentage of required fields present (0-100)
                - consistency: Consistency score (0-100, penalties for issues)
                - issues: List of detected issues
        """
        # Calculate completeness (% of required fields present and non-zero)
        present_fields = sum(1 for field in self.QUARTER_REQUIRED_FIELDS if financial_data.get(field, 0) != 0)
        completeness = (present_fields / len(self.QUARTER_REQUIRED_FIELDS)) * 100

        # Calculate consistency (basic sanity checks)
        consistency_score = 100.0
        issues = []

        # Check: Assets = Liabilities + Equity (within 5% tolerance)
        assets = financial_data.get("total_assets", 0)
        liabilities = financial_data.get("total_liabilities", 0)
        equity = financial_data.get("stockholders_equity", 0)

        if assets > 0:
            balance = liabilities + equity
            balance_error = abs(assets - balance) / assets
            if balance_error > 0.05:  # 5% tolerance
                consistency_score -= 20
                issues.append(f"Balance sheet mismatch: {balance_error:.1%}")

        # Check: Revenue > 0 and reasonable relative to assets
        revenue = financial_data.get("revenues", 0)
        if revenue <= 0:
            consistency_score -= 30
            issues.append("Zero or negative revenue")

        # Check: OCF should be reasonable relative to net income
        ocf = financial_data.get("operating_cash_flow", 0)
        net_income = financial_data.get("net_income", 0)
        if net_income != 0 and ocf != 0:
            cash_conversion = ocf / net_income
            if cash_conversion < 0.3 or cash_conversion > 5.0:
                consistency_score -= 15
                issues.append(f"Unusual cash conversion: {cash_conversion:.1f}x")

        return {
            "completeness": completeness,
            "consistency": max(0, consistency_score),  # Ensure non-negative
            "issues": issues,
        }


# Singleton instance
_assessor_instance: Optional[DataQualityAssessor] = None


def get_data_quality_assessor(logger: Optional[logging.Logger] = None) -> DataQualityAssessor:
    """
    Get singleton DataQualityAssessor instance.

    Args:
        logger: Optional logger (only used on first call)

    Returns:
        DataQualityAssessor instance
    """
    global _assessor_instance
    if _assessor_instance is None:
        _assessor_instance = DataQualityAssessor(logger)
    return _assessor_instance
