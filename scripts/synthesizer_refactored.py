#!/usr/bin/env python3
"""
InvestiGator - Refactored Analysis Synthesis Module
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Refactored version with better separation of concerns and reduced coupling.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models (Separated from business logic)
# ============================================================================


@dataclass
class InvestmentRecommendation:
    """Data class for investment recommendations"""

    symbol: str
    overall_score: float
    fundamental_score: float
    technical_score: float
    income_score: float
    cashflow_score: float
    balance_score: float
    growth_score: float
    value_score: float
    business_quality_score: float
    recommendation: str
    confidence: str
    price_target: Optional[float]
    current_price: Optional[float]
    investment_thesis: str
    time_horizon: str
    position_size: str
    key_catalysts: List[str]
    key_risks: List[str]
    key_insights: List[str]
    entry_strategy: str
    exit_strategy: str
    stop_loss: Optional[float]
    analysis_timestamp: datetime
    data_quality_score: float
    analysis_thinking: Optional[str] = None
    synthesis_details: Optional[str] = None


# ============================================================================
# Interfaces (Dependency Inversion)
# ============================================================================


class DataFetcherInterface(ABC):
    """Interface for data fetching operations"""

    @abstractmethod
    def fetch_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data for a symbol"""
        pass

    @abstractmethod
    def fetch_technical_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch technical data for a symbol"""
        pass


class CacheInterface(ABC):
    """Interface for cache operations"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        pass


class AnalysisInterface(ABC):
    """Interface for analysis operations"""

    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis on data"""
        pass


class ReportGeneratorInterface(ABC):
    """Interface for report generation"""

    @abstractmethod
    def generate(self, recommendation: InvestmentRecommendation) -> Path:
        """Generate report from recommendation"""
        pass


# ============================================================================
# Core Business Logic (Decoupled from dependencies)
# ============================================================================


class InvestmentSynthesisEngine:
    """
    Core synthesis engine - pure business logic without external dependencies
    """

    def __init__(self, fundamental_weight: float = 0.6, technical_weight: float = 0.4):
        """Initialize synthesis engine with configurable weights"""
        self.fundamental_weight = fundamental_weight
        self.technical_weight = technical_weight
        self._validate_weights()

    def _validate_weights(self):
        """Validate that weights sum to 1.0"""
        total = self.fundamental_weight + self.technical_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def calculate_overall_score(self, fundamental_score: float, technical_score: float) -> float:
        """Calculate weighted overall score"""
        return fundamental_score * self.fundamental_weight + technical_score * self.technical_weight

    def determine_recommendation(self, overall_score: float) -> str:
        """Determine recommendation based on score"""
        if overall_score >= 8.5:
            return "STRONG BUY"
        elif overall_score >= 7.0:
            return "BUY"
        elif overall_score >= 5.5:
            return "HOLD"
        elif overall_score >= 4.0:
            return "SELL"
        else:
            return "STRONG SELL"

    def determine_confidence(self, data_quality_score: float) -> str:
        """Determine confidence level based on data quality"""
        if data_quality_score >= 0.9:
            return "HIGH"
        elif data_quality_score >= 0.7:
            return "MEDIUM"
        else:
            return "LOW"

    def determine_position_size(self, overall_score: float, confidence: str) -> str:
        """Determine recommended position size"""
        if confidence == "HIGH":
            if overall_score >= 8.0:
                return "LARGE"
            elif overall_score >= 6.5:
                return "MODERATE"
            else:
                return "SMALL"
        elif confidence == "MEDIUM":
            if overall_score >= 8.5:
                return "MODERATE"
            else:
                return "SMALL"
        else:
            return "AVOID"

    def determine_time_horizon(self, growth_score: float, value_score: float) -> str:
        """Determine investment time horizon"""
        if growth_score >= 8.0:
            return "LONG-TERM"
        elif value_score >= 7.5:
            return "MEDIUM-TERM"
        else:
            return "SHORT-TERM"

    def calculate_price_target(self, current_price: float, growth_score: float, value_score: float) -> float:
        """Calculate price target based on scores"""
        # Simplified calculation - can be made more sophisticated
        upside_potential = ((growth_score + value_score) / 20) * 0.3  # Max 30% upside
        return current_price * (1 + upside_potential)

    def calculate_stop_loss(self, current_price: float, confidence: str) -> float:
        """Calculate stop loss based on confidence"""
        stop_loss_percentages = {
            "HIGH": 0.10,  # 10% stop loss
            "MEDIUM": 0.08,  # 8% stop loss
            "LOW": 0.05,  # 5% stop loss
        }
        percentage = stop_loss_percentages.get(confidence, 0.08)
        return current_price * (1 - percentage)

    def synthesize(self, fundamental_data: Dict[str, Any], technical_data: Dict[str, Any]) -> InvestmentRecommendation:
        """
        Synthesize fundamental and technical data into recommendation
        Pure business logic - no external dependencies
        """
        # Extract scores
        fundamental_score = fundamental_data.get("score", 5.0)
        technical_score = technical_data.get("score", 5.0)

        # Extract detailed scores
        income_score = fundamental_data.get("income_score", 5.0)
        cashflow_score = fundamental_data.get("cashflow_score", 5.0)
        balance_score = fundamental_data.get("balance_score", 5.0)
        growth_score = fundamental_data.get("growth_score", 5.0)
        value_score = fundamental_data.get("value_score", 5.0)
        business_quality_score = fundamental_data.get("business_quality_score", 5.0)

        # Current price
        current_price = technical_data.get("current_price", 100.0)

        # Calculate overall score
        overall_score = self.calculate_overall_score(fundamental_score, technical_score)

        # Data quality
        data_quality_score = fundamental_data.get("data_quality", 0.8)

        # Determine recommendations
        recommendation = self.determine_recommendation(overall_score)
        confidence = self.determine_confidence(data_quality_score)
        position_size = self.determine_position_size(overall_score, confidence)
        time_horizon = self.determine_time_horizon(growth_score, value_score)

        # Calculate targets
        price_target = self.calculate_price_target(current_price, growth_score, value_score)
        stop_loss = self.calculate_stop_loss(current_price, confidence)

        # Generate thesis
        investment_thesis = self._generate_investment_thesis(fundamental_data, technical_data, overall_score)

        # Extract insights
        key_catalysts = fundamental_data.get("catalysts", [])
        key_risks = fundamental_data.get("risks", [])
        key_insights = fundamental_data.get("insights", [])

        # Strategies
        entry_strategy = self._generate_entry_strategy(technical_data, recommendation)
        exit_strategy = self._generate_exit_strategy(time_horizon, price_target)

        return InvestmentRecommendation(
            symbol=fundamental_data.get("symbol", "UNKNOWN"),
            overall_score=overall_score,
            fundamental_score=fundamental_score,
            technical_score=technical_score,
            income_score=income_score,
            cashflow_score=cashflow_score,
            balance_score=balance_score,
            growth_score=growth_score,
            value_score=value_score,
            business_quality_score=business_quality_score,
            recommendation=recommendation,
            confidence=confidence,
            price_target=price_target,
            current_price=current_price,
            investment_thesis=investment_thesis,
            time_horizon=time_horizon,
            position_size=position_size,
            key_catalysts=key_catalysts,
            key_risks=key_risks,
            key_insights=key_insights,
            entry_strategy=entry_strategy,
            exit_strategy=exit_strategy,
            stop_loss=stop_loss,
            analysis_timestamp=datetime.now(),
            data_quality_score=data_quality_score,
        )

    def _generate_investment_thesis(
        self, fundamental_data: Dict[str, Any], technical_data: Dict[str, Any], overall_score: float
    ) -> str:
        """Generate investment thesis"""
        thesis_parts = []

        if overall_score >= 7.0:
            thesis_parts.append("Strong investment opportunity")
        elif overall_score >= 5.5:
            thesis_parts.append("Moderate investment opportunity")
        else:
            thesis_parts.append("Weak investment opportunity")

        if fundamental_data.get("score", 0) >= 7.0:
            thesis_parts.append("with solid fundamentals")

        if technical_data.get("score", 0) >= 7.0:
            thesis_parts.append("and favorable technical setup")

        return " ".join(thesis_parts) + "."

    def _generate_entry_strategy(self, technical_data: Dict[str, Any], recommendation: str) -> str:
        """Generate entry strategy"""
        if "BUY" in recommendation:
            if technical_data.get("rsi", 50) < 30:
                return "Accumulate on oversold conditions"
            else:
                return "Scale in over multiple trading sessions"
        else:
            return "Wait for better entry point"

    def _generate_exit_strategy(self, time_horizon: str, price_target: float) -> str:
        """Generate exit strategy"""
        return f"Hold for {time_horizon.lower().replace('-', ' ')} with target of ${price_target:.2f}"


# ============================================================================
# Service Layer (Orchestrates dependencies)
# ============================================================================


class InvestmentSynthesizer:
    """
    Service layer that orchestrates the synthesis process
    Handles external dependencies through interfaces
    """

    def __init__(
        self,
        synthesis_engine: InvestmentSynthesisEngine,
        data_fetcher: DataFetcherInterface,
        cache: Optional[CacheInterface] = None,
        report_generator: Optional[ReportGeneratorInterface] = None,
    ):
        """
        Initialize with injected dependencies
        """
        self.synthesis_engine = synthesis_engine
        self.data_fetcher = data_fetcher
        self.cache = cache
        self.report_generator = report_generator
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_symbol(
        self, symbol: str, force_refresh: bool = False, generate_report: bool = False
    ) -> InvestmentRecommendation:
        """
        Analyze a symbol and generate recommendation
        """
        self.logger.info(f"Starting analysis for {symbol}")

        # Check cache if not forcing refresh
        if not force_refresh and self.cache:
            cached_result = self._get_cached_result(symbol)
            if cached_result:
                self.logger.info(f"Using cached result for {symbol}")
                return cached_result

        # Fetch data
        fundamental_data = self.data_fetcher.fetch_fundamental_data(symbol)
        technical_data = self.data_fetcher.fetch_technical_data(symbol)

        # Add symbol to data
        fundamental_data["symbol"] = symbol
        technical_data["symbol"] = symbol

        # Synthesize recommendation
        recommendation = self.synthesis_engine.synthesize(fundamental_data, technical_data)

        # Cache result
        if self.cache:
            self._cache_result(symbol, recommendation)

        # Generate report if requested
        if generate_report and self.report_generator:
            report_path = self.report_generator.generate(recommendation)
            self.logger.info(f"Report generated at {report_path}")

        return recommendation

    def analyze_portfolio(self, symbols: List[str], generate_report: bool = True) -> List[InvestmentRecommendation]:
        """
        Analyze multiple symbols
        """
        recommendations = []

        for symbol in symbols:
            try:
                recommendation = self.analyze_symbol(symbol)
                recommendations.append(recommendation)
            except Exception as e:
                self.logger.error(f"Failed to analyze {symbol}: {e}")

        if generate_report and self.report_generator:
            # Generate portfolio report
            self._generate_portfolio_report(recommendations)

        return recommendations

    def _get_cached_result(self, symbol: str) -> Optional[InvestmentRecommendation]:
        """Get cached result if available"""
        cache_key = f"synthesis_{symbol}"
        return self.cache.get(cache_key) if self.cache else None

    def _cache_result(self, symbol: str, recommendation: InvestmentRecommendation):
        """Cache the recommendation"""
        if self.cache:
            cache_key = f"synthesis_{symbol}"
            self.cache.set(cache_key, recommendation)

    def _generate_portfolio_report(self, recommendations: List[InvestmentRecommendation]):
        """Generate portfolio report"""
        # This would be implemented by the report generator
        pass


# ============================================================================
# Factory Functions (Dependency Injection)
# ============================================================================


def create_synthesizer(config: Optional[Dict[str, Any]] = None) -> InvestmentSynthesizer:
    """
    Factory function to create synthesizer with dependencies
    This is where actual implementations are wired together
    """
    from investigator.config import get_config

    # Get configuration
    if config is None:
        config = get_config()

    # Create synthesis engine with configured weights
    synthesis_engine = InvestmentSynthesisEngine(
        fundamental_weight=config.analysis.weights.get("fundamental", 0.6),
        technical_weight=config.analysis.weights.get("technical", 0.4),
    )

    # Create data fetcher (adapter pattern for existing modules)
    from adapters.data_fetcher_adapter import DataFetcherAdapter

    data_fetcher = DataFetcherAdapter(config)

    # Create cache adapter if enabled
    cache = None
    if config.cache_control.use_cache:
        from adapters.cache_adapter import CacheAdapter

        cache = CacheAdapter(config)

    # Create report generator if enabled
    report_generator = None
    if config.pdf_report.enabled:
        from adapters.report_generator_adapter import ReportGeneratorAdapter

        report_generator = ReportGeneratorAdapter(config)

    return InvestmentSynthesizer(
        synthesis_engine=synthesis_engine, data_fetcher=data_fetcher, cache=cache, report_generator=report_generator
    )


# ============================================================================
# Adapters (Bridge to existing code)
# ============================================================================


class DataFetcherAdapter(DataFetcherInterface):
    """Adapter to connect to existing data fetching modules"""

    def __init__(self, config):
        self.config = config
        # Import existing modules
        from sec_fundamental import SECAnalyzer
        from yahoo_technical import TechnicalAnalyzer

        self.sec_analyzer = SECAnalyzer()
        self.tech_analyzer = TechnicalAnalyzer()

    def fetch_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data using existing SECAnalyzer"""
        return self.sec_analyzer.analyze(symbol)

    def fetch_technical_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch technical data using existing TechnicalAnalyzer"""
        return self.tech_analyzer.analyze(symbol)


class CacheAdapter(CacheInterface):
    """Adapter to connect to existing cache system"""

    def __init__(self, config):
        from utils.cache import get_cache_manager, CacheType

        self.cache_manager = get_cache_manager()
        self.cache_type = CacheType.SYNTHESIS

    def get(self, key: str) -> Optional[Any]:
        """Get from cache manager"""
        return self.cache_manager.get(self.cache_type, key)

    def set(self, key: str, value: Any) -> None:
        """Set in cache manager"""
        self.cache_manager.set(self.cache_type, key, value)


class ReportGeneratorAdapter(ReportGeneratorInterface):
    """Adapter to connect to existing report generator"""

    def __init__(self, config):
        from utils.report_generator import PDFReportGenerator, ReportConfig

        self.generator = PDFReportGenerator(
            config.reports_dir / "synthesis",
            ReportConfig(
                title="Investment Analysis Report", subtitle="Comprehensive Stock Analysis", include_charts=True
            ),
        )

    def generate(self, recommendation: InvestmentRecommendation) -> Path:
        """Generate report using existing PDF generator"""
        return self.generator.generate_synthesis_report(recommendation)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Investment Synthesizer")
    parser.add_argument("symbol", help="Stock symbol to analyze")
    parser.add_argument("--report", action="store_true", help="Generate PDF report")
    parser.add_argument("--force", action="store_true", help="Force refresh (bypass cache)")

    args = parser.parse_args()

    # Create synthesizer
    synthesizer = create_synthesizer()

    # Analyze symbol
    recommendation = synthesizer.analyze_symbol(args.symbol, force_refresh=args.force, generate_report=args.report)

    # Print results
    print(f"\nAnalysis for {recommendation.symbol}")
    print(f"Overall Score: {recommendation.overall_score:.2f}")
    print(f"Recommendation: {recommendation.recommendation}")
    print(f"Confidence: {recommendation.confidence}")
    print(f"Price Target: ${recommendation.price_target:.2f}")
    print(f"Investment Thesis: {recommendation.investment_thesis}")
