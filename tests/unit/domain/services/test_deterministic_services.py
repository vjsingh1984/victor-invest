"""
Unit tests for deterministic services that replace LLM calls.

Tests cover:
- DeterministicValuationSynthesizer
- DeterministicConflictResolver
- TemplateBasedThesisGenerator
- DeterministicInsightExtractor
- DeterministicCompetitiveAnalyzer
"""

from typing import Any, Dict, List

import pytest

from investigator.domain.services.deterministic_competitive_analyzer import (
    SECTOR_PROFILES,
    CompetitiveContext,
    DeterministicCompetitiveAnalyzer,
    MarketPosition,
    MoatWidth,
    analyze_competitive_position,
)
from investigator.domain.services.deterministic_conflict_resolver import (
    Conflict,
    ConflictSeverity,
    ConflictType,
    DataQualityConflictDetector,
    DeterministicConflictResolver,
    RecommendationConflictDetector,
    TimeHorizonConflictDetector,
    reconcile_conflicts,
)
from investigator.domain.services.deterministic_insight_extractor import (
    DeterministicInsightExtractor,
    FundamentalInsightExtractor,
    MetricThresholds,
    SECInsightExtractor,
    TechnicalInsightExtractor,
    extract_key_insights,
)

# Import all services
from investigator.domain.services.deterministic_valuation_synthesizer import (
    DeterministicValuationSynthesizer,
    ModelContribution,
    RiskBasedMarginOfSafetyCalculator,
    SynthesisContext,
    TemplateBasedExplanationGenerator,
    ThresholdBasedStanceDeterminer,
    ValuationStance,
    synthesize_valuation,
)
from investigator.domain.services.template_thesis_generator import (
    CoreNarrativeGenerator,
    InvestmentStance,
    TemplateBasedThesisGenerator,
    ThesisContext,
    TimeHorizon,
    ValueDriversGenerator,
    generate_investment_thesis,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_model_contributions() -> List[ModelContribution]:
    """Sample model contributions for testing."""
    return [
        ModelContribution(
            model_name="dcf",
            fair_value=150.0,
            weight=0.40,
            is_applicable=True,
            assumptions={"growth_rate": 0.08, "discount_rate": 0.10},
        ),
        ModelContribution(
            model_name="pe", fair_value=145.0, weight=0.30, is_applicable=True, assumptions={"target_pe": 22.5}
        ),
        ModelContribution(
            model_name="ps", fair_value=160.0, weight=0.20, is_applicable=True, assumptions={"target_ps": 5.5}
        ),
        ModelContribution(
            model_name="ggm", fair_value=None, weight=0.0, is_applicable=False, reason="Dividend payout below threshold"
        ),
    ]


@pytest.fixture
def sample_synthesis_context(sample_model_contributions) -> SynthesisContext:
    """Sample synthesis context for testing."""
    return SynthesisContext(
        symbol="AAPL",
        current_price=130.0,
        blended_fair_value=152.0,
        overall_confidence=0.75,
        model_agreement_score=0.72,
        divergence_flag=False,
        data_quality_score=82.0,
        quality_grade="Good",
        sector="Technology",
        industry="Consumer Electronics",
        model_contributions=sample_model_contributions,
        notes=["Strong brand value", "High R&D investment"],
        archetypes=["cash_cow", "dividend_payer"],
    )


@pytest.fixture
def sample_fundamental_analysis() -> Dict[str, Any]:
    """Sample fundamental analysis data."""
    return {
        "valuation": {
            "current_price": 130.0,
            "fair_value": 150.0,
            "pe_ratio": 25.5,
            "ps_ratio": 7.2,
            "dividend_yield": 0.006,
        },
        "ratios": {
            "revenue_growth": 0.12,
            "profit_margin": 0.22,
            "roe": 0.45,
            "debt_to_equity": 1.2,
        },
        "quality_score": 85,
        "data_quality": {
            "data_quality_score": 80,
            "quality_grade": "Good",
        },
        "multi_model_summary": {
            "blended_fair_value": 150.0,
            "model_agreement_score": 0.75,
            "overall_confidence": 0.72,
            "divergence_flag": False,
        },
        "recommendation": "Buy",
        "analysis": {
            "strengths": ["Strong cash flow", "Market leader"],
            "weaknesses": ["Supply chain risks", "Competition"],
        },
    }


@pytest.fixture
def sample_technical_analysis() -> Dict[str, Any]:
    """Sample technical analysis data."""
    return {
        "signals": {
            "trend": "bullish",
            "rsi": 55,
            "ma_signal": "above 200MA",
            "volume_signal": "increasing",
        },
        "levels": {
            "support": 125.0,
            "resistance": 145.0,
            "current_price": 130.0,
        },
        "overall_signal": "buy",
    }


@pytest.fixture
def sample_sec_analysis() -> Dict[str, Any]:
    """Sample SEC analysis data."""
    return {
        "analysis": {
            "overall_rating": 8,
            "revenue_trend": "growing",
            "profit_trend": "improving",
        },
        "data_quality": {
            "data_quality_score": 85,
        },
        "risks": [
            {"description": "Regulatory compliance costs increasing"},
            {"description": "International trade tensions"},
        ],
    }


# ============================================================================
# DeterministicValuationSynthesizer Tests
# ============================================================================


class TestDeterministicValuationSynthesizer:
    """Tests for DeterministicValuationSynthesizer."""

    def test_synthesize_undervalued_stock(self, sample_synthesis_context):
        """Test synthesis for undervalued stock."""
        synthesizer = DeterministicValuationSynthesizer()
        result = synthesizer.synthesize(sample_synthesis_context)

        # Check basic structure
        assert result.fair_value_estimate == 152.0
        assert result.implied_upside_downside > 0  # Should be positive (undervalued)
        assert "Undervalued" in result.valuation_stance
        assert result.margin_of_safety_target > 0

    def test_synthesize_overvalued_stock(self, sample_model_contributions):
        """Test synthesis for overvalued stock."""
        context = SynthesisContext(
            symbol="XYZ",
            current_price=200.0,  # Higher than fair value
            blended_fair_value=150.0,
            overall_confidence=0.70,
            model_agreement_score=0.65,
            divergence_flag=False,
            data_quality_score=75.0,
            quality_grade="Good",
            sector="Technology",
            industry=None,
            model_contributions=sample_model_contributions,
        )

        synthesizer = DeterministicValuationSynthesizer()
        result = synthesizer.synthesize(context)

        assert result.implied_upside_downside < 0  # Negative = overvalued
        assert "Overvalued" in result.valuation_stance

    def test_stance_determiner_thresholds(self):
        """Test stance determination at various upside levels."""
        determiner = ThresholdBasedStanceDeterminer()

        # Significantly undervalued
        assert determiner.determine_stance(0.35, 0.8) == ValuationStance.SIGNIFICANTLY_UNDERVALUED

        # Undervalued
        assert determiner.determine_stance(0.18, 0.8) == ValuationStance.UNDERVALUED

        # Fairly valued
        assert determiner.determine_stance(0.02, 0.8) == ValuationStance.FAIRLY_VALUED

        # Overvalued
        assert determiner.determine_stance(-0.18, 0.8) == ValuationStance.OVERVALUED

    def test_margin_of_safety_calculator(self):
        """Test margin of safety calculation."""
        calculator = RiskBasedMarginOfSafetyCalculator()

        # High quality, high agreement = low margin
        low_margin = calculator.calculate(0.85, 90, 0.80)
        assert low_margin <= 0.15

        # Low agreement, poor quality = high margin
        high_margin = calculator.calculate(0.25, 40, 0.50)
        assert high_margin >= 0.25

    def test_explanation_generator(self, sample_model_contributions):
        """Test explanation generation."""
        generator = TemplateBasedExplanationGenerator()

        influence = generator.generate_model_influence(sample_model_contributions)
        assert "DCF" in influence
        assert "40%" in influence

        confidence = generator.generate_confidence_caution(0.75, 82, False, "Good")
        assert "moderate" in confidence.lower() or "high" in confidence.lower()

    def test_synthesize_valuation_convenience_function(self):
        """Test the drop-in replacement function."""
        result = synthesize_valuation(
            symbol="AAPL",
            current_price=130.0,
            valuation_results={
                "dcf": {"fair_value": 150.0, "weight": 0.5, "applicable": True, "assumptions": {}},
                "pe": {"fair_value": 145.0, "weight": 0.3, "applicable": True, "assumptions": {}},
            },
            multi_model_summary={
                "blended_fair_value": 148.0,
                "overall_confidence": 0.72,
                "model_agreement_score": 0.75,
                "divergence_flag": False,
            },
            data_quality={"data_quality_score": 80, "quality_grade": "Good"},
            company_profile={"sector": "Technology"},
        )

        assert isinstance(result, dict)
        assert "fair_value_estimate" in result
        assert "valuation_stance" in result


# ============================================================================
# DeterministicConflictResolver Tests
# ============================================================================


class TestDeterministicConflictResolver:
    """Tests for DeterministicConflictResolver."""

    def test_detect_recommendation_conflict(self, sample_fundamental_analysis, sample_technical_analysis):
        """Test detection of recommendation conflicts."""
        # Create a conflict scenario
        fundamental = sample_fundamental_analysis.copy()
        fundamental["recommendation"] = "buy"

        technical = sample_technical_analysis.copy()
        technical["signals"]["trend"] = "bearish"
        technical["overall_signal"] = "sell"

        detector = RecommendationConflictDetector()
        conflicts = detector.detect(fundamental, technical, None, None)

        assert len(conflicts) >= 1
        assert any(c.conflict_type == ConflictType.RECOMMENDATION for c in conflicts)

    def test_detect_no_conflict_when_aligned(self, sample_fundamental_analysis, sample_technical_analysis):
        """Test no conflict detected when analyses align."""
        # Both bullish
        detector = RecommendationConflictDetector()
        conflicts = detector.detect(sample_fundamental_analysis, sample_technical_analysis, None, None)

        # Should have no high-severity recommendation conflicts
        rec_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.RECOMMENDATION]
        assert len(rec_conflicts) == 0

    def test_detect_time_horizon_conflict(self, sample_fundamental_analysis, sample_technical_analysis):
        """Test detection of time horizon conflicts."""
        # Fundamentally undervalued but technically bearish
        fundamental = sample_fundamental_analysis.copy()
        fundamental["valuation"]["fair_value"] = 200.0  # Big upside
        fundamental["valuation"]["current_price"] = 130.0

        technical = sample_technical_analysis.copy()
        technical["signals"]["trend"] = "bearish"

        detector = TimeHorizonConflictDetector()
        conflicts = detector.detect(fundamental, technical, None, None)

        assert any(c.conflict_type == ConflictType.TIME_HORIZON for c in conflicts)

    def test_reconcile_conflicts(self, sample_fundamental_analysis, sample_technical_analysis):
        """Test conflict reconciliation."""
        resolver = DeterministicConflictResolver()

        result = resolver.reconcile(
            conflicts=None,
            fundamental=sample_fundamental_analysis,
            technical=sample_technical_analysis,
        )

        assert result.overall_coherence is not None
        assert result.reconciled_recommendation is not None
        assert isinstance(result.resolutions, list)

    def test_reconcile_conflicts_convenience_function(self):
        """Test the drop-in replacement function."""
        result = reconcile_conflicts(
            fundamental={"recommendation": "buy"},
            technical={"signal": "sell"},
        )

        assert isinstance(result, dict)
        assert "overall_coherence" in result
        assert "reconciled_recommendation" in result


# ============================================================================
# TemplateBasedThesisGenerator Tests
# ============================================================================


class TestTemplateBasedThesisGenerator:
    """Tests for TemplateBasedThesisGenerator."""

    def test_generate_bullish_thesis(self):
        """Test thesis generation for bullish scenario."""
        context = ThesisContext(
            symbol="AAPL",
            company_name="Apple Inc.",
            sector="Technology",
            industry="Consumer Electronics",
            overall_score=75,
            confidence=80,
            upside=0.20,  # 20% upside
            current_price=130.0,
            fair_value=156.0,
            positive_factors=["Strong brand", "Cash generation"],
            negative_factors=["Competition", "Supply chain"],
            revenue_growth=0.15,
            profit_margin=0.25,
        )

        generator = TemplateBasedThesisGenerator()
        thesis = generator.generate(context)

        assert "AAPL" in thesis.core_investment_narrative
        assert len(thesis.key_value_drivers) >= 2
        assert len(thesis.competitive_advantages) >= 1
        assert len(thesis.growth_catalysts) >= 2
        assert len(thesis.bear_case_considerations) >= 2

    def test_generate_bearish_thesis(self):
        """Test thesis generation for bearish scenario."""
        context = ThesisContext(
            symbol="XYZ",
            company_name="XYZ Corp",
            sector="Consumer",
            industry=None,
            overall_score=40,
            confidence=60,
            upside=-0.25,  # 25% downside
            current_price=100.0,
            fair_value=75.0,
            negative_factors=["Revenue decline", "Margin pressure"],
        )

        generator = TemplateBasedThesisGenerator()
        thesis = generator.generate(context)

        # Should reflect bearish stance
        assert (
            "downside" in thesis.core_investment_narrative.lower()
            or "caution" in thesis.core_investment_narrative.lower()
        )

    def test_time_horizon_determination(self):
        """Test time horizon determination logic."""
        from investigator.domain.services.template_thesis_generator import TimeHorizonDeterminer

        determiner = TimeHorizonDeterminer()

        # High growth = long term
        growth_context = ThesisContext(
            symbol="X",
            company_name="X",
            sector="Technology",
            industry=None,
            overall_score=70,
            confidence=75,
            upside=0.10,
            current_price=100,
            fair_value=110,
            revenue_growth=0.25,  # High growth
        )
        assert "3-5" in determiner.determine(growth_context)

        # Overvalued = short term
        overvalued_context = ThesisContext(
            symbol="X",
            company_name="X",
            sector="Technology",
            industry=None,
            overall_score=70,
            confidence=75,
            upside=-0.20,  # Overvalued
            current_price=100,
            fair_value=80,
        )
        assert "6-12" in determiner.determine(overvalued_context)

    def test_generate_investment_thesis_convenience_function(self, sample_fundamental_analysis):
        """Test the drop-in replacement function."""
        result = generate_investment_thesis(
            symbol="AAPL",
            key_insights={
                "fundamental": {
                    "positive_factors": ["Strong brand"],
                    "negative_factors": ["Competition"],
                }
            },
            composite_scores={"overall_score": 75, "confidence": 80},
            fundamental_analysis=sample_fundamental_analysis,
            company_profile={"sector": "Technology", "company_name": "Apple Inc."},
        )

        assert isinstance(result, dict)
        assert "core_investment_narrative" in result
        assert "key_value_drivers" in result


# ============================================================================
# DeterministicInsightExtractor Tests
# ============================================================================


class TestDeterministicInsightExtractor:
    """Tests for DeterministicInsightExtractor."""

    def test_extract_fundamental_insights(self, sample_fundamental_analysis):
        """Test fundamental insight extraction."""
        extractor = FundamentalInsightExtractor()
        insight = extractor.extract(sample_fundamental_analysis)

        assert insight is not None
        assert insight.source == "fundamental"
        assert len(insight.positive_factors) >= 1
        assert insight.confidence > 0

    def test_extract_technical_insights(self, sample_technical_analysis):
        """Test technical insight extraction."""
        extractor = TechnicalInsightExtractor()
        insight = extractor.extract(sample_technical_analysis)

        assert insight is not None
        assert insight.source == "technical"
        assert "bullish" in insight.critical_metric.lower() or "trend" in insight.critical_metric.lower()

    def test_extract_sec_insights(self, sample_sec_analysis):
        """Test SEC insight extraction."""
        extractor = SECInsightExtractor()
        insight = extractor.extract(sample_sec_analysis)

        assert insight is not None
        assert insight.source == "sec"
        assert insight.confidence >= 40  # SEC data is authoritative

    def test_extract_all_insights(self, sample_fundamental_analysis, sample_technical_analysis, sample_sec_analysis):
        """Test extracting insights from all sources."""
        extractor = DeterministicInsightExtractor()
        insights = extractor.extract(
            fundamental=sample_fundamental_analysis,
            technical=sample_technical_analysis,
            sec=sample_sec_analysis,
        )

        assert insights.fundamental is not None
        assert insights.technical is not None
        assert insights.sec is not None
        assert "quantitative" in insights.to_dict()

    def test_extract_key_insights_convenience_function(self, sample_fundamental_analysis, sample_technical_analysis):
        """Test the drop-in replacement function."""
        result = extract_key_insights(
            fundamental=sample_fundamental_analysis,
            technical=sample_technical_analysis,
        )

        assert isinstance(result, dict)
        assert "fundamental" in result
        assert "technical" in result


# ============================================================================
# DeterministicCompetitiveAnalyzer Tests
# ============================================================================


class TestDeterministicCompetitiveAnalyzer:
    """Tests for DeterministicCompetitiveAnalyzer."""

    def test_analyze_tech_company(self):
        """Test competitive analysis for tech company."""
        context = CompetitiveContext(
            symbol="AAPL",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=2_500_000_000_000,  # $2.5T
            revenue=380_000_000_000,
            profit_margin=0.25,
            revenue_growth=0.10,
            roe=0.45,
            debt_to_equity=1.5,
            data_quality_score=85,
        )

        analyzer = DeterministicCompetitiveAnalyzer()
        analysis = analyzer.analyze(context)

        # Large cap should be leader
        assert analysis.market_position_and_share.assessment == "Leader"

        # High ROE and margins should indicate wide moat
        assert "Wide" in analysis.competitive_advantages_moat.assessment

        # Tech sector should have high barriers
        assert analysis.barriers_to_entry.score >= 70

        # Should have risks
        assert len(analysis.competitive_risks) >= 2

    def test_analyze_small_company(self):
        """Test competitive analysis for small company."""
        context = CompetitiveContext(
            symbol="SMALL",
            sector="Consumer",
            industry=None,
            market_cap=500_000_000,  # $500M
            revenue=100_000_000,
            profit_margin=0.08,
            revenue_growth=-0.05,  # Declining
            roe=0.08,
            debt_to_equity=0.5,
            data_quality_score=60,
        )

        analyzer = DeterministicCompetitiveAnalyzer()
        analysis = analyzer.analyze(context)

        # Small cap should be niche
        assert analysis.market_position_and_share.assessment == "Niche Player"

        # Low metrics = limited moat
        assert (
            "Limited" in analysis.competitive_advantages_moat.assessment
            or "Narrow" in analysis.competitive_advantages_moat.assessment
        )

    def test_sector_profiles_exist(self):
        """Test that key sector profiles are defined."""
        assert "Technology" in SECTOR_PROFILES
        assert "Healthcare" in SECTOR_PROFILES
        assert "Financials" in SECTOR_PROFILES
        assert "Consumer" in SECTOR_PROFILES
        assert "Industrials" in SECTOR_PROFILES

    def test_strategic_positioning_score_bounds(self):
        """Test that strategic score stays within bounds."""
        context = CompetitiveContext(
            symbol="X",
            sector="Technology",
            industry=None,
            market_cap=50_000_000_000,
            revenue=10_000_000_000,
            profit_margin=0.20,
            roe=0.25,
            revenue_growth=0.15,
            debt_to_equity=0.8,
        )

        analyzer = DeterministicCompetitiveAnalyzer()
        analysis = analyzer.analyze(context)

        assert 0 <= analysis.strategic_positioning_score <= 100

    def test_analyze_competitive_position_convenience_function(self):
        """Test the drop-in replacement function."""
        result = analyze_competitive_position(
            symbol="AAPL",
            company_data={
                "market_data": {"market_cap": 2_500_000_000_000},
                "ratios": {"profit_margin": 0.25, "roe": 0.45},
                "sector": "Technology",
            },
        )

        assert isinstance(result, dict)
        assert "market_position_and_share" in result
        assert "competitive_advantages_moat" in result
        assert "strategic_positioning_score" in result


# ============================================================================
# Integration Tests
# ============================================================================


class TestDeterministicServicesIntegration:
    """Integration tests for deterministic services working together."""

    def test_full_synthesis_pipeline(self, sample_fundamental_analysis, sample_technical_analysis, sample_sec_analysis):
        """Test running through a complete synthesis pipeline."""
        # 1. Extract insights
        insights = extract_key_insights(
            fundamental=sample_fundamental_analysis,
            technical=sample_technical_analysis,
            sec=sample_sec_analysis,
        )

        assert "fundamental" in insights

        # 2. Detect and resolve conflicts
        conflicts = reconcile_conflicts(
            fundamental=sample_fundamental_analysis,
            technical=sample_technical_analysis,
        )

        assert "overall_coherence" in conflicts

        # 3. Generate thesis
        thesis = generate_investment_thesis(
            symbol="AAPL",
            key_insights=insights,
            composite_scores={"overall_score": 75, "confidence": 80},
            fundamental_analysis=sample_fundamental_analysis,
        )

        assert "core_investment_narrative" in thesis

        # 4. Analyze competitive position
        competitive = analyze_competitive_position(
            symbol="AAPL",
            company_data={
                "market_data": {"market_cap": 2_500_000_000_000},
                "ratios": sample_fundamental_analysis.get("ratios", {}),
                "sector": "Technology",
            },
        )

        assert "strategic_positioning_score" in competitive

        # 5. Synthesize valuation
        valuation = synthesize_valuation(
            symbol="AAPL",
            current_price=130.0,
            valuation_results={},
            multi_model_summary={
                "blended_fair_value": 150.0,
                "overall_confidence": 0.75,
                "model_agreement_score": 0.72,
            },
            data_quality={"data_quality_score": 80, "quality_grade": "Good"},
            company_profile={"sector": "Technology"},
        )

        assert "fair_value_estimate" in valuation
        assert "valuation_stance" in valuation

    def test_all_services_return_dict(self, sample_fundamental_analysis, sample_technical_analysis):
        """Ensure all convenience functions return dicts for API compatibility."""
        # All should return dicts
        result1 = extract_key_insights(fundamental=sample_fundamental_analysis)
        assert isinstance(result1, dict)

        result2 = reconcile_conflicts(fundamental=sample_fundamental_analysis)
        assert isinstance(result2, dict)

        result3 = generate_investment_thesis(
            symbol="X",
            key_insights={},
            composite_scores={"overall_score": 50, "confidence": 50},
        )
        assert isinstance(result3, dict)

        result4 = analyze_competitive_position(symbol="X", company_data={"sector": "Technology"})
        assert isinstance(result4, dict)

        result5 = synthesize_valuation(
            symbol="X",
            current_price=100,
            valuation_results={},
            multi_model_summary={"blended_fair_value": 110},
            data_quality={},
            company_profile={},
        )
        assert isinstance(result5, dict)
