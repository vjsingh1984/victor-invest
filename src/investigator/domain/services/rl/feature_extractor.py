"""
Valuation Context Extractor

Extracts features for RL state representation from financial data.
This module transforms raw financial metrics into normalized features
suitable for the RL agent to make weight predictions.

Follows Interface Segregation: Only extraction logic, no policy or training concerns.

Usage:
    from investigator.domain.services.rl import ValuationContextExtractor

    extractor = ValuationContextExtractor()

    context = extractor.extract(
        symbol="AAPL",
        financials=financials_dict,
        ratios=ratios_dict,
        market_context=market_context,
        data_quality=data_quality_dict,
    )

    # Convert to feature tensor for RL
    features = extractor.to_tensor(context)
"""

import logging
import math
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from investigator.domain.services.rl.models import (
    ValuationContext,
    GrowthStage,
    CompanySize,
)

logger = logging.getLogger(__name__)


# GICS Sector list for one-hot encoding
GICS_SECTORS = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
    "Unknown",
]

# Market cap thresholds (in billions USD)
MARKET_CAP_THRESHOLDS = {
    "micro_cap": 0.3,  # < $300M
    "small_cap": 2.0,  # $300M - $2B
    "mid_cap": 10.0,  # $2B - $10B
    "large_cap": 200.0,  # $10B - $200B
    # mega_cap: > $200B
}


class ValuationContextExtractor:
    """
    Extracts context features for RL state representation.

    Features extracted:
    - Company classification (sector, industry, growth stage, size)
    - Fundamental metrics (profitability, growth, margins, leverage)
    - Data quality indicators
    - Market context (technical trend, sentiment, volatility)
    - Model applicability flags

    All continuous features are normalized to [0, 1] or [-1, 1] range.
    """

    def __init__(
        self,
        metadata_service: Optional[Any] = None,
        profitability_classifier: Optional[Any] = None,
        threshold_registry: Optional[Any] = None,
    ):
        """
        Initialize extractor with optional service dependencies.

        Args:
            metadata_service: Service for company metadata lookup.
            profitability_classifier: Service for profitability classification.
            threshold_registry: Service for sector-specific thresholds.
        """
        self.metadata_service = metadata_service
        self.profitability_classifier = profitability_classifier
        self.threshold_registry = threshold_registry

    def extract(
        self,
        symbol: str,
        financials: Dict[str, Any],
        ratios: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None,
        data_quality: Optional[Dict[str, Any]] = None,
        insider_data: Optional[Dict[str, Any]] = None,
        current_price: Optional[float] = None,
        analysis_date: Optional[date] = None,
    ) -> ValuationContext:
        """
        Extract context features from financial data.

        Args:
            symbol: Stock ticker symbol.
            financials: Dict with financial statement data.
            ratios: Dict with calculated financial ratios.
            market_context: Optional market context data.
            data_quality: Optional data quality scores.
            insider_data: Optional insider sentiment data from Form 4 filings.
            current_price: Current stock price.
            analysis_date: Date of analysis.

        Returns:
            ValuationContext with all extracted features.
        """
        analysis_date = analysis_date or date.today()
        market_context = market_context or {}
        data_quality = data_quality or {}
        insider_data = insider_data or {}

        # Extract company classification
        sector = self._extract_sector(financials, ratios)
        industry = self._extract_industry(financials, ratios)
        growth_stage = self._classify_growth_stage(financials, ratios)
        company_size = self._classify_company_size(financials, current_price)

        # Extract fundamental metrics
        profitability_score = self._calculate_profitability_score(financials, ratios)
        pe_level = self._normalize_pe(ratios.get("pe_ratio") or ratios.get("trailing_pe"))
        revenue_growth = self._safe_get_float(ratios, "revenue_growth", 0.0)
        fcf_margin = self._safe_get_float(ratios, "fcf_margin", 0.0)
        rule_of_40 = self._calculate_rule_of_40(financials, ratios)
        payout_ratio = self._safe_get_float(ratios, "payout_ratio", 0.0)
        debt_to_equity = self._safe_get_float(ratios, "debt_to_equity", 0.0)
        gross_margin = self._safe_get_float(ratios, "gross_margin", 0.0)
        operating_margin = self._safe_get_float(ratios, "operating_margin", 0.0)

        # Extract data quality
        data_quality_score = self._safe_get_float(data_quality, "overall_score", 50.0)
        quarters_available = self._count_quarters(financials)

        # Extract market context
        technical_trend = self._safe_get_float(market_context, "trend_score", 0.0)
        market_sentiment = self._safe_get_float(market_context, "sentiment_score", 0.0)
        volatility = self._normalize_volatility(market_context.get("volatility"))

        # Extract technical indicators (from market_context.technical_indicators)
        tech_ind = market_context.get("technical_indicators", {})
        rsi_14 = self._safe_get_float(tech_ind, "rsi_14", 50.0)
        macd_histogram = self._normalize_macd_histogram(
            self._safe_get_float(tech_ind, "macd_histogram", 0.0),
            current_price or 100.0
        )
        obv_trend = self._safe_get_float(tech_ind, "obv_trend", 0.0)  # Already -1 to 1
        adx_14 = self._safe_get_float(tech_ind, "adx_14", 25.0)
        stoch_k = self._safe_get_float(tech_ind, "stoch_k", 50.0)
        mfi_14 = self._safe_get_float(tech_ind, "mfi_14", 50.0)

        # Extract entry/exit signals (from market_context.entry_exit_signals)
        entry_exit = market_context.get("entry_exit_signals", {})
        entry_signal_strength = self._safe_get_float(entry_exit, "entry_signal_strength", 0.0)
        exit_signal_strength = self._safe_get_float(entry_exit, "exit_signal_strength", 0.0)
        signal_confluence = self._safe_get_float(entry_exit, "signal_confluence", 0.0)
        days_from_support = self._safe_get_float(entry_exit, "days_from_support", 0.5)
        risk_reward_ratio = self._safe_get_float(entry_exit, "risk_reward_ratio", 2.0)

        # Extract insider sentiment features
        insider_features = self._extract_insider_features(insider_data)

        # Determine model applicability
        model_applicability = self._determine_model_applicability(financials, ratios, growth_stage)

        return ValuationContext(
            symbol=symbol,
            analysis_date=analysis_date,
            sector=sector,
            industry=industry,
            growth_stage=growth_stage,
            company_size=company_size,
            profitability_score=profitability_score,
            pe_level=pe_level,
            revenue_growth=revenue_growth,
            fcf_margin=fcf_margin,
            rule_of_40_score=rule_of_40,
            payout_ratio=min(1.0, payout_ratio),  # Cap at 100%
            debt_to_equity=min(3.0, debt_to_equity),  # Cap at 300%
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            data_quality_score=data_quality_score,
            quarters_available=quarters_available,
            technical_trend=technical_trend,
            market_sentiment=market_sentiment,
            volatility=volatility,
            # Technical indicators
            rsi_14=rsi_14,
            macd_histogram=macd_histogram,
            obv_trend=obv_trend,
            adx_14=adx_14,
            stoch_k=stoch_k,
            mfi_14=mfi_14,
            # Entry/Exit signals
            entry_signal_strength=entry_signal_strength,
            exit_signal_strength=exit_signal_strength,
            signal_confluence=signal_confluence,
            days_from_support=days_from_support,
            risk_reward_ratio=risk_reward_ratio,
            # Insider sentiment features
            insider_sentiment=insider_features["sentiment"],
            insider_buy_ratio=insider_features["buy_ratio"],
            insider_transaction_value=insider_features["transaction_value"],
            insider_cluster_signal=insider_features["cluster_signal"],
            insider_key_exec_activity=insider_features["key_exec_activity"],
            dcf_applicable=model_applicability["dcf"],
            ggm_applicable=model_applicability["ggm"],
            pe_applicable=model_applicability["pe"],
            ps_applicable=model_applicability["ps"],
            pb_applicable=model_applicability["pb"],
            evebitda_applicable=model_applicability["ev_ebitda"],
            fiscal_period=financials.get("fiscal_period"),
            current_price=current_price,
        )

    def to_tensor(
        self,
        context: ValuationContext,
        include_categorical: bool = True,
    ) -> np.ndarray:
        """
        Convert context to normalized feature tensor for RL.

        Args:
            context: ValuationContext to convert.
            include_categorical: If True, include one-hot encoded categoricals.

        Returns:
            Numpy array of normalized features.
        """
        features = []

        # Continuous features (normalized to 0-1 or -1 to 1)
        features.extend(
            [
                context.profitability_score,  # 0-1
                context.pe_level,  # 0-1
                self._normalize_growth(context.revenue_growth),  # 0-1
                self._normalize_margin(context.fcf_margin),  # 0-1
                context.rule_of_40_score / 100.0,  # 0-1
                context.payout_ratio,  # 0-1
                context.debt_to_equity / 3.0,  # 0-1 (capped at 300%)
                context.gross_margin,  # 0-1
                context.operating_margin,  # 0-1
                context.data_quality_score / 100.0,  # 0-1
                min(1.0, context.quarters_available / 8.0),  # 0-1 (cap at 8 quarters)
                (context.technical_trend + 1) / 2,  # 0-1 (from -1 to 1)
                (context.market_sentiment + 1) / 2,  # 0-1 (from -1 to 1)
                context.volatility,  # 0-1
            ]
        )

        # Technical indicators (normalized to 0-1)
        features.extend(
            [
                context.rsi_14 / 100.0,  # 0-1
                (context.macd_histogram + 1) / 2,  # 0-1 (from -1 to 1)
                (context.obv_trend + 1) / 2,  # 0-1 (from -1 to 1)
                context.adx_14 / 100.0,  # 0-1
                context.stoch_k / 100.0,  # 0-1
                context.mfi_14 / 100.0,  # 0-1
            ]
        )

        # Entry/Exit signal features (normalized to 0-1)
        features.extend(
            [
                (context.entry_signal_strength + 1) / 2,  # 0-1 (from -1 to 1)
                (context.exit_signal_strength + 1) / 2,  # 0-1 (from -1 to 1)
                (context.signal_confluence + 1) / 2,  # 0-1 (from -1 to 1)
                context.days_from_support,  # Already 0-1
                min(1.0, context.risk_reward_ratio / 5.0),  # 0-1 (cap at 5:1)
            ]
        )

        # Insider sentiment features (normalized to 0-1)
        features.extend(
            [
                (context.insider_sentiment + 1) / 2,  # 0-1 (from -1 to 1)
                context.insider_buy_ratio,  # Already 0-1
                (context.insider_transaction_value + 1) / 2,  # 0-1 (from -1 to 1)
                (context.insider_cluster_signal + 1) / 2,  # 0-1 (from -1 to 1)
                (context.insider_key_exec_activity + 1) / 2,  # 0-1 (from -1 to 1)
            ]
        )

        # Model applicability flags (binary)
        features.extend(
            [
                1.0 if context.dcf_applicable else 0.0,
                1.0 if context.ggm_applicable else 0.0,
                1.0 if context.pe_applicable else 0.0,
                1.0 if context.ps_applicable else 0.0,
                1.0 if context.pb_applicable else 0.0,
                1.0 if context.evebitda_applicable else 0.0,
            ]
        )

        if include_categorical:
            # One-hot encode sector
            sector_encoding = self._one_hot_sector(context.sector)
            features.extend(sector_encoding)

            # Encode growth stage (ordinal: 0-4)
            stage_order = {
                GrowthStage.PRE_PROFIT: 0,
                GrowthStage.EARLY_GROWTH: 1,
                GrowthStage.HIGH_GROWTH: 2,
                GrowthStage.TRANSITIONING: 3,
                GrowthStage.MATURE: 4,
                GrowthStage.DIVIDEND_PAYING: 5,
            }
            features.append(stage_order.get(context.growth_stage, 2) / 5.0)

            # Encode company size (ordinal: 0-4)
            size_order = {
                CompanySize.MICRO_CAP: 0,
                CompanySize.SMALL_CAP: 1,
                CompanySize.MID_CAP: 2,
                CompanySize.LARGE_CAP: 3,
                CompanySize.MEGA_CAP: 4,
            }
            features.append(size_order.get(context.company_size, 2) / 4.0)

        return np.array(features, dtype=np.float32)

    def get_feature_names(self, include_categorical: bool = True) -> List[str]:
        """Get list of feature names in order they appear in tensor."""
        names = [
            # Fundamental features
            "profitability_score",
            "pe_level",
            "revenue_growth_norm",
            "fcf_margin_norm",
            "rule_of_40_norm",
            "payout_ratio",
            "debt_to_equity_norm",
            "gross_margin",
            "operating_margin",
            "data_quality_norm",
            "quarters_available_norm",
            "technical_trend_norm",
            "market_sentiment_norm",
            "volatility",
            # Technical indicators
            "rsi_14_norm",
            "macd_histogram_norm",
            "obv_trend_norm",
            "adx_14_norm",
            "stoch_k_norm",
            "mfi_14_norm",
            # Entry/Exit signals
            "entry_signal_strength_norm",
            "exit_signal_strength_norm",
            "signal_confluence_norm",
            "days_from_support",
            "risk_reward_ratio_norm",
            # Insider sentiment features
            "insider_sentiment_norm",
            "insider_buy_ratio",
            "insider_transaction_value_norm",
            "insider_cluster_signal_norm",
            "insider_key_exec_activity_norm",
            # Model applicability flags
            "dcf_applicable",
            "ggm_applicable",
            "pe_applicable",
            "ps_applicable",
            "pb_applicable",
            "evebitda_applicable",
        ]

        if include_categorical:
            for sector in GICS_SECTORS:
                names.append(f"sector_{sector.lower().replace(' ', '_')}")
            names.append("growth_stage_ordinal")
            names.append("company_size_ordinal")

        return names

    def _extract_sector(
        self,
        financials: Dict[str, Any],
        ratios: Dict[str, Any],
    ) -> str:
        """Extract sector from financial data."""
        # Check multiple possible field names
        for key in ["sector", "gics_sector", "sec_sector"]:
            if key in financials and financials[key]:
                return str(financials[key])
            if key in ratios and ratios[key]:
                return str(ratios[key])
        return "Unknown"

    def _extract_industry(
        self,
        financials: Dict[str, Any],
        ratios: Dict[str, Any],
    ) -> str:
        """Extract industry from financial data."""
        for key in ["industry", "gics_industry", "sec_industry"]:
            if key in financials and financials[key]:
                return str(financials[key])
            if key in ratios and ratios[key]:
                return str(ratios[key])
        return "Unknown"

    def _classify_growth_stage(
        self,
        financials: Dict[str, Any],
        ratios: Dict[str, Any],
    ) -> GrowthStage:
        """Classify company growth stage based on financial characteristics."""
        # Check profitability
        net_income = self._safe_get_float(financials, "net_income", 0)
        ebitda = self._safe_get_float(financials, "ebitda", 0)
        revenue_growth = self._safe_get_float(ratios, "revenue_growth", 0)
        payout_ratio = self._safe_get_float(ratios, "payout_ratio", 0)

        # Pre-profit: negative net income and negative/low EBITDA
        if net_income < 0 and ebitda <= 0:
            return GrowthStage.PRE_PROFIT

        # Early growth: low profitability, high growth
        if net_income <= 0 and revenue_growth > 0.20:
            return GrowthStage.EARLY_GROWTH

        # High growth: profitable with >20% growth
        if revenue_growth > 0.20:
            return GrowthStage.HIGH_GROWTH

        # Dividend paying: significant payout ratio
        if payout_ratio > 0.30:
            return GrowthStage.DIVIDEND_PAYING

        # Transitioning: 10-20% growth
        if revenue_growth > 0.10:
            return GrowthStage.TRANSITIONING

        # Mature: stable, low growth
        return GrowthStage.MATURE

    def _classify_company_size(
        self,
        financials: Dict[str, Any],
        current_price: Optional[float],
    ) -> CompanySize:
        """Classify company by market cap."""
        market_cap = self._safe_get_float(financials, "market_cap", 0)

        # Try to calculate from shares if not directly available
        if market_cap <= 0 and current_price:
            shares = self._safe_get_float(financials, "shares_outstanding", 0)
            if shares > 0:
                market_cap = shares * current_price

        # Convert to billions
        market_cap_b = market_cap / 1e9

        if market_cap_b < MARKET_CAP_THRESHOLDS["micro_cap"]:
            return CompanySize.MICRO_CAP
        elif market_cap_b < MARKET_CAP_THRESHOLDS["small_cap"]:
            return CompanySize.SMALL_CAP
        elif market_cap_b < MARKET_CAP_THRESHOLDS["mid_cap"]:
            return CompanySize.MID_CAP
        elif market_cap_b < MARKET_CAP_THRESHOLDS["large_cap"]:
            return CompanySize.LARGE_CAP
        else:
            return CompanySize.MEGA_CAP

    def _calculate_profitability_score(
        self,
        financials: Dict[str, Any],
        ratios: Dict[str, Any],
    ) -> float:
        """Calculate profitability score (0-1)."""
        # Use profitability classifier if available
        if self.profitability_classifier:
            try:
                result = self.profitability_classifier.classify(financials, ratios)
                return result.get("score", 0.5)
            except Exception:
                pass

        # Fallback: simple calculation based on margins
        gross_margin = self._safe_get_float(ratios, "gross_margin", 0)
        operating_margin = self._safe_get_float(ratios, "operating_margin", 0)
        net_margin = self._safe_get_float(ratios, "net_margin", 0)
        fcf_margin = self._safe_get_float(ratios, "fcf_margin", 0)

        # Weighted average of margins (normalized)
        score = (
            0.2 * min(1.0, gross_margin)
            + 0.3 * min(1.0, max(0, operating_margin + 0.2) / 0.4)  # -20% to 20% -> 0 to 1
            + 0.2 * min(1.0, max(0, net_margin + 0.1) / 0.3)  # -10% to 20% -> 0 to 1
            + 0.3 * min(1.0, max(0, fcf_margin + 0.1) / 0.4)  # -10% to 30% -> 0 to 1
        )

        return max(0.0, min(1.0, score))

    def _normalize_pe(self, pe: Optional[float]) -> float:
        """Normalize P/E ratio to 0-1 scale."""
        if pe is None or pe <= 0:
            return 0.5  # Unknown/negative earnings

        # Map P/E: 0-50 -> 0-1 (log scale for better distribution)
        if pe <= 5:
            return 0.1
        elif pe >= 100:
            return 1.0
        else:
            # Log scale normalization
            return min(1.0, math.log(pe) / math.log(100))

    def _calculate_rule_of_40(
        self,
        financials: Dict[str, Any],
        ratios: Dict[str, Any],
    ) -> float:
        """Calculate Rule of 40 score."""
        revenue_growth = self._safe_get_float(ratios, "revenue_growth", 0) * 100
        fcf_margin = self._safe_get_float(ratios, "fcf_margin", 0) * 100

        # If no FCF margin, use operating margin
        if fcf_margin == 0:
            fcf_margin = self._safe_get_float(ratios, "operating_margin", 0) * 100

        return revenue_growth + fcf_margin

    def _count_quarters(self, financials: Dict[str, Any]) -> int:
        """Count available quarters of data."""
        # Check for quarterly data availability
        if "quarterly_data" in financials:
            return len(financials["quarterly_data"])

        # Check for quarters_available field
        if "quarters_available" in financials:
            return int(financials["quarters_available"])

        # Default assumption
        return 4

    def _normalize_volatility(self, volatility: Optional[float]) -> float:
        """Normalize volatility to 0-1."""
        if volatility is None:
            return 0.5

        # Typical stock volatility ranges from 15% to 80%
        # Map to 0-1
        return max(0.0, min(1.0, (volatility - 0.10) / 0.70))

    def _normalize_macd_histogram(self, histogram: float, price: float) -> float:
        """Normalize MACD histogram relative to price, to -1 to 1 range."""
        if price <= 0:
            return 0.0

        # MACD histogram is typically 0.1% to 5% of price for significant moves
        # Normalize to -1 to 1 based on percentage of price
        pct_of_price = histogram / price * 100  # As percentage
        # Clamp to -5% to +5% and map to -1 to 1
        normalized = max(-1.0, min(1.0, pct_of_price / 5.0))
        return normalized

    def _normalize_growth(self, growth: float) -> float:
        """Normalize growth rate to 0-1."""
        # Map -50% to +200% -> 0 to 1
        return max(0.0, min(1.0, (growth + 0.5) / 2.5))

    def _normalize_margin(self, margin: float) -> float:
        """Normalize margin to 0-1."""
        # Map -30% to +50% -> 0 to 1
        return max(0.0, min(1.0, (margin + 0.3) / 0.8))

    def _extract_insider_features(
        self,
        insider_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """Extract insider sentiment features from Form 4 data.

        Args:
            insider_data: Dict with insider sentiment data, expected keys:
                - sentiment_score: -1 to +1 overall sentiment
                - buy_count: Number of buy transactions
                - sell_count: Number of sell transactions
                - buy_value: Total value of buy transactions
                - sell_value: Total value of sell transactions
                - cluster_detected: Whether a cluster signal was detected
                - key_insider_activity: List of key insider transactions

        Returns:
            Dict with normalized insider features for RL.
        """
        # Default neutral values
        defaults = {
            "sentiment": 0.0,
            "buy_ratio": 0.5,
            "transaction_value": 0.0,
            "cluster_signal": 0.0,
            "key_exec_activity": 0.0,
        }

        if not insider_data:
            return defaults

        # Sentiment score (-1 to +1)
        sentiment = self._safe_get_float(insider_data, "sentiment_score", 0.0)
        defaults["sentiment"] = max(-1.0, min(1.0, sentiment))

        # Buy ratio (0 to 1)
        buy_count = self._safe_get_float(insider_data, "buy_count", 0)
        sell_count = self._safe_get_float(insider_data, "sell_count", 0)
        total_count = buy_count + sell_count
        if total_count > 0:
            defaults["buy_ratio"] = buy_count / total_count
        else:
            defaults["buy_ratio"] = 0.5  # Neutral when no activity

        # Normalized transaction value (-1 to +1)
        # Net value = buy_value - sell_value, normalized by typical large transaction ($1M)
        buy_value = self._safe_get_float(insider_data, "buy_value", 0)
        sell_value = self._safe_get_float(insider_data, "sell_value", 0)
        net_value = buy_value - sell_value
        # Normalize: $10M+ buying = +1, $10M+ selling = -1
        normalized_value = net_value / 10_000_000  # $10M scale
        defaults["transaction_value"] = max(-1.0, min(1.0, normalized_value))

        # Cluster signal (binary but can be float for soft signal)
        cluster_detected = insider_data.get("cluster_detected", False)
        if cluster_detected:
            # Positive cluster (buying) vs negative cluster (selling)
            if net_value >= 0:
                defaults["cluster_signal"] = 1.0
            else:
                defaults["cluster_signal"] = -1.0
        else:
            defaults["cluster_signal"] = 0.0

        # Key executive activity (-1 to +1)
        # Higher weight for CEO/CFO/Director transactions
        key_insider_activity = insider_data.get("key_insider_activity", [])
        if key_insider_activity:
            key_buy = sum(1 for t in key_insider_activity if t.get("is_buy", False))
            key_sell = sum(1 for t in key_insider_activity if not t.get("is_buy", True))
            key_total = key_buy + key_sell
            if key_total > 0:
                defaults["key_exec_activity"] = (key_buy - key_sell) / key_total
        else:
            # Check for aggregated key insider data
            key_buy_count = self._safe_get_float(insider_data, "key_insider_buy_count", 0)
            key_sell_count = self._safe_get_float(insider_data, "key_insider_sell_count", 0)
            key_total = key_buy_count + key_sell_count
            if key_total > 0:
                defaults["key_exec_activity"] = (key_buy_count - key_sell_count) / key_total

        return defaults

    def _determine_model_applicability(
        self,
        financials: Dict[str, Any],
        ratios: Dict[str, Any],
        growth_stage: GrowthStage,
    ) -> Dict[str, bool]:
        """Determine which valuation models can be applied."""
        applicability = {
            "dcf": True,
            "ggm": False,
            "pe": True,
            "ps": True,
            "pb": True,
            "ev_ebitda": True,
        }

        # DCF requires FCF data (at least 4 quarters)
        fcf = self._safe_get_float(financials, "free_cash_flow", 0)
        if fcf <= 0 or self._count_quarters(financials) < 4:
            applicability["dcf"] = False

        # GGM requires dividend payout and stable growth
        payout_ratio = self._safe_get_float(ratios, "payout_ratio", 0)
        if payout_ratio < 0.40 or growth_stage in [GrowthStage.PRE_PROFIT, GrowthStage.HIGH_GROWTH]:
            applicability["ggm"] = False
        else:
            applicability["ggm"] = True

        # P/E requires positive earnings
        net_income = self._safe_get_float(financials, "net_income", 0)
        if net_income <= 0:
            applicability["pe"] = False

        # EV/EBITDA requires positive EBITDA
        ebitda = self._safe_get_float(financials, "ebitda", 0)
        if ebitda <= 0:
            applicability["ev_ebitda"] = False

        return applicability

    def _one_hot_sector(self, sector: str) -> List[float]:
        """One-hot encode sector."""
        encoding = [0.0] * len(GICS_SECTORS)
        try:
            idx = GICS_SECTORS.index(sector)
            encoding[idx] = 1.0
        except ValueError:
            # Unknown sector
            encoding[-1] = 1.0
        return encoding

    def _safe_get_float(
        self,
        data: Dict[str, Any],
        key: str,
        default: float = 0.0,
    ) -> float:
        """Safely extract float value from dict."""
        value = data.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


# Factory function
def get_feature_extractor() -> ValuationContextExtractor:
    """Get ValuationContextExtractor instance."""
    return ValuationContextExtractor()
