"""
Dynamic Multi-Model Weighting Service

Implements tier-based dynamic weighting for multi-model valuation blending.
Configuration-driven approach with 15 sub-tiers, 5% weight increments.

Author: InvestiGator Team
Date: 2025-11-06
Updated: 2025-11-07 (refactored to use shared services)
Updated: 2025-11-14 (added market context dynamic adjustments)
Updated: 2025-11-27 (integrated bounded multipliers and audit trail)
Updated: 2025-12-29 (added auto manufacturing valuation tier P1-A)
"""

import logging
from typing import Dict, Any, Optional, Tuple

from investigator.domain.models.market_context import MarketContext
from investigator.domain.services.company_metadata_service import CompanyMetadataService
from investigator.domain.services.model_applicability import ModelApplicabilityRules
from investigator.domain.services.weight_normalizer import WeightNormalizer
from investigator.domain.services.weight_bounds import (
    BoundedMultiplierApplicator,
    BoundConfig,
)
from investigator.domain.services.weight_audit_trail import (
    WeightAuditTrail,
    WeightAdjustment,
)
from investigator.domain.services.threshold_registry import (
    ThresholdRegistry,
    get_threshold_registry,
    PELevel,
)
from investigator.domain.services.profitability_classifier import (
    ProfitabilityClassifier,
    get_profitability_classifier,
    ProfitabilityStage,
)


logger = logging.getLogger(__name__)

# Known Dividend Aristocrats (50+ years consecutive dividend increases)
# These companies should NEVER be classified as pre_profit unless confirmed business distress
# Used as data quality safeguard against YTD normalization failures
KNOWN_DIVIDEND_ARISTOCRATS = {
    # Dividend Kings (50+ years) - S&P 500 members
    "JNJ": "Johnson & Johnson",  # 62 years
    "PG": "Procter & Gamble",  # 68 years
    "KO": "Coca-Cola",  # 62 years
    "MMM": "3M Company",  # 66 years
    "CL": "Colgate-Palmolive",  # 62 years
    "ED": "Consolidated Edison",  # 50 years
    "DOV": "Dover Corporation",  # 69 years
    "EMR": "Emerson Electric",  # 67 years
    "GPC": "Genuine Parts",  # 68 years
    "HRL": "Hormel Foods",  # 58 years
    "ITW": "Illinois Tool Works",  # 51 years
    "LOW": "Lowe's Companies",  # 61 years
    "NWN": "Northwest Natural",  # 68 years
    "PPG": "PPG Industries",  # 53 years
    "PH": "Parker-Hannifin",  # 68 years
    "SWK": "Stanley Black & Decker",  # 57 years
    "SCL": "Stepan Company",  # 56 years
    "TGT": "Target",  # 52 years
    "ABT": "Abbott Laboratories",  # 52 years (since 2013 spin-off)
    "ABBV": "AbbVie",  # Spin-off from Abbott, continues streak
    "PEP": "PepsiCo",  # 52 years
    "WMT": "Walmart",  # 51 years
    "XOM": "Exxon Mobil",  # 42 years (still major dividend payer)
    "CVX": "Chevron",  # 37 years (major dividend payer)
    "MCD": "McDonald's",  # 49 years
}


class DynamicModelWeightingService:
    """
    Service for determining dynamic model weights based on company characteristics.

    Flow:
    1. Fetch sector/industry from database
    2. Extract financial metrics (payout, growth, Rule of 40)
    3. Classify into tier (15 sub-tiers)
    4. Get base weights from config
    5. Apply industry overrides
    6. Apply model applicability filters
    7. Apply data quality adjustments
    8. Normalize and round to 5%
    """

    def __init__(self, valuation_config: Dict[str, Any]):
        """
        Initialize with valuation configuration section.

        Args:
            valuation_config: Valuation section from config.yaml with tier_thresholds,
                              tier_base_weights, sector_normalization, market_context_multipliers, etc.
        """
        # Direct dict access - valuation_config is the "valuation" section from config.yaml
        self.tier_thresholds = valuation_config.get("tier_thresholds", {})
        self.tier_base_weights = valuation_config.get("tier_base_weights", {})
        self.industry_specific = valuation_config.get("industry_specific_weights", {})
        self.data_quality_thresholds = valuation_config.get("data_quality_thresholds", {})
        self.market_context_multipliers = valuation_config.get("market_context_multipliers", {})

        # Initialize shared services
        self.metadata_service = CompanyMetadataService(
            sector_normalization=valuation_config.get("sector_normalization", {})
        )
        self.applicability_rules = ModelApplicabilityRules(
            applicability_config=valuation_config.get("model_applicability", {})
        )
        self.weight_normalizer = WeightNormalizer(rounding_increment=5)

        # Initialize bounded multiplier applicator (M4)
        bounds_config = valuation_config.get("multiplier_bounds", {})
        if bounds_config.get("enabled", True):
            self.bounds_applicator = BoundedMultiplierApplicator(
                config=BoundConfig(
                    cumulative_floor=bounds_config.get("cumulative_floor", 0.50),
                    cumulative_ceiling=bounds_config.get("cumulative_ceiling", 1.50),
                    per_model_minimum=bounds_config.get("per_model_floors", {}).get("dcf", 5),
                    warning_threshold=bounds_config.get("warning_threshold", 0.70),
                )
            )
        else:
            self.bounds_applicator = None

        # Audit trail configuration (M4)
        self.audit_config = valuation_config.get("audit_trail", {})
        self.audit_enabled = self.audit_config.get("enabled", True)

        # Sector-aware threshold registry (M5)
        pe_thresholds = valuation_config.get("pe_extremeness_thresholds", {})
        self.threshold_registry = ThresholdRegistry(
            sector_thresholds=pe_thresholds.get("sectors", {}), industry_overrides=pe_thresholds.get("industries", {})
        )

        # Multi-indicator profitability classifier (M5)
        self.profitability_classifier = get_profitability_classifier()

    def determine_weights(
        self,
        symbol: str,
        financials: Dict[str, Any],
        ratios: Dict[str, Any],
        data_quality: Optional[Dict[str, Any]] = None,
        market_context: Optional[MarketContext] = None,
    ) -> Tuple[Dict[str, float], str, Optional[WeightAuditTrail]]:
        """
        Determine dynamic model weights for a company.

        Args:
            symbol: Stock symbol
            financials: Financial metrics (net_income, revenue, dividends, etc.)
            ratios: Financial ratios (payout_ratio, rule_of_40_score, etc.)
            data_quality: Data quality assessment per model
            market_context: Optional market context for dynamic weight adjustments

        Returns:
            Tuple of (weights_dict, tier_classification, audit_trail):
            - weights_dict: Dict mapping model names to weights (sum = 100%, increments = 5%)
                Example: {"dcf": 50, "pe": 30, "ev_ebitda": 15, "ps": 5, "pb": 0, "ggm": 0}
            - tier_classification: String tier name (e.g., "pre_profit_negative_ebitda")
            - audit_trail: WeightAuditTrail with full traceability (None if disabled)
        """
        # Initialize audit trail if enabled
        audit_trail = WeightAuditTrail(symbol=symbol) if self.audit_enabled else None
        step_number = 0

        # 1. Get sector/industry using metadata service
        sector, industry = self.metadata_service.get_sector_industry(symbol)

        # 2. Extract metrics (with None-safety: default to 0 if None or missing)
        net_income = financials.get("net_income") or 0
        revenue = financials.get("revenue") or 0
        payout_ratio = ratios.get("payout_ratio") or 0
        rule_of_40 = ratios.get("rule_of_40_score") or 0
        revenue_growth = ratios.get("revenue_growth_pct") or 0
        fcf_margin = ratios.get("fcf_margin_pct") or 0
        ebitda = financials.get("ebitda") or 0
        market_cap = financials.get("market_cap") or 0
        current_price = financials.get("current_price") or 0
        ttm_eps = ratios.get("ttm_eps") or 0
        operating_income = financials.get("operating_income")  # None if missing (for multi-indicator)
        free_cash_flow = financials.get("free_cash_flow")  # None if missing (for multi-indicator)

        # 3. Classify company characteristics
        company_size = self._classify_company_size(market_cap)
        profitability_stage = self._classify_profitability_stage(
            net_income=net_income,
            revenue=revenue,
            revenue_growth=revenue_growth,
            ebitda=ebitda,
            operating_income=operating_income,
            free_cash_flow=free_cash_flow,
            fcf_margin=fcf_margin if fcf_margin != 0 else None,
        )
        market_pe = self._calculate_market_pe(current_price, ttm_eps)

        # Extract stockholders equity for insurance tier classification
        stockholders_equity = financials.get("stockholders_equity") or 0

        # For insurance companies, fetch stockholders_equity from DB if missing
        # and update financials dict so applicability filters can use it
        if sector == "Financials" and industry and "insur" in industry.lower():
            if stockholders_equity <= 0:
                try:
                    from investigator.domain.services.valuation.insurance_valuation import _fetch_from_database

                    db_equity, db_shares, _ = _fetch_from_database(symbol, None)
                    if db_equity:
                        stockholders_equity = db_equity
                        financials["stockholders_equity"] = stockholders_equity
                        logger.info(
                            f"{symbol} - Insurance: Updated financials with DB equity: ${stockholders_equity/1e9:.2f}B"
                        )
                    if db_shares and not financials.get("shares_outstanding"):
                        financials["shares_outstanding"] = db_shares
                        # Also calculate book_value_per_share for P/B applicability
                        if stockholders_equity and db_shares:
                            financials["book_value_per_share"] = stockholders_equity / db_shares
                            logger.info(
                                f"{symbol} - Insurance: Calculated BV/share: ${financials['book_value_per_share']:.2f}"
                            )
                except Exception as e:
                    logger.warning(f"{symbol} - Insurance: Could not fetch from database: {e}")

        # 4. Classify tier
        tier, sub_tier = self._classify_tier(
            net_income=net_income,
            payout_ratio=payout_ratio,
            rule_of_40=rule_of_40,
            revenue_growth=revenue_growth,
            fcf_margin=fcf_margin,
            sector=sector,
            revenue=revenue,
            ebitda=ebitda,
            industry=industry,
            stockholders_equity=stockholders_equity,
            symbol=symbol,
        )

        # 5. Get base weights for tier
        base_weights = self._get_tier_base_weights(sub_tier)
        logger.info(f"{symbol} - Tier '{sub_tier}' base weights from config: {base_weights}")

        # Audit: Capture base weights
        if audit_trail:
            step_number += 1
            audit_trail.capture(
                step_number=step_number,
                step_name="base_weights",
                weights_before={},
                weights_after=base_weights.copy(),
                adjustments=[],
                metadata={"tier": tier, "sub_tier": sub_tier, "sector": sector},
            )

        # 6. Apply industry overrides (if configured)
        weights_before_industry = base_weights.copy()
        if industry:
            base_weights = self._apply_industry_overrides(base_weights, industry)

            # Audit: Capture industry override
            if audit_trail and weights_before_industry != base_weights:
                step_number += 1
                adjustments = [
                    WeightAdjustment(
                        model=model,
                        source="industry_override",
                        multiplier=(
                            base_weights[model] / weights_before_industry[model]
                            if weights_before_industry.get(model, 0) > 0
                            else 1.0
                        ),
                        reason=f"Industry: {industry}",
                    )
                    for model in base_weights
                    if base_weights.get(model, 0) != weights_before_industry.get(model, 0)
                ]
                audit_trail.capture(
                    step_number=step_number,
                    step_name="industry_override",
                    weights_before=weights_before_industry,
                    weights_after=base_weights.copy(),
                    adjustments=adjustments,
                    metadata={"industry": industry},
                )

        # 7. Apply company-specific adjustments (now with sector-aware thresholds)
        weights_before_company = base_weights.copy()
        base_weights = self._apply_company_specific_adjustments(
            base_weights,
            company_size=company_size,
            profitability_stage=profitability_stage,
            market_pe=market_pe,
            revenue_growth=revenue_growth,
            symbol=symbol,
            sector=sector,
            industry=industry,
        )

        # Audit: Capture company adjustments
        if audit_trail and weights_before_company != base_weights:
            step_number += 1
            adjustments = [
                WeightAdjustment(
                    model=model,
                    source="company_adjustment",
                    multiplier=(
                        base_weights[model] / weights_before_company[model]
                        if weights_before_company.get(model, 0) > 0
                        else 1.0
                    ),
                    reason=f"Size: {company_size}, Stage: {profitability_stage}",
                )
                for model in base_weights
                if base_weights.get(model, 0) != weights_before_company.get(model, 0)
            ]
            audit_trail.capture(
                step_number=step_number,
                step_name="company_adjustment",
                weights_before=weights_before_company,
                weights_after=base_weights.copy(),
                adjustments=adjustments,
                metadata={"company_size": company_size, "profitability_stage": profitability_stage},
            )

        # 7b. Apply market context adjustments with bounds enforcement
        if market_context:
            weights_before_market = base_weights.copy()
            base_weights = self.apply_market_context_adjustments(base_weights, market_context, symbol)

            # Audit: Capture market context adjustments
            if audit_trail and weights_before_market != base_weights:
                step_number += 1
                adjustments = [
                    WeightAdjustment(
                        model=model,
                        source="market_context",
                        multiplier=(
                            base_weights[model] / weights_before_market[model]
                            if weights_before_market.get(model, 0) > 0
                            else 1.0
                        ),
                        reason=f"Trend: {market_context.technical_trend.value}",
                    )
                    for model in base_weights
                    if base_weights.get(model, 0) != weights_before_market.get(model, 0)
                ]
                audit_trail.capture(
                    step_number=step_number,
                    step_name="market_context",
                    weights_before=weights_before_market,
                    weights_after=base_weights.copy(),
                    adjustments=adjustments,
                    metadata={"market_context": str(market_context)},
                )

        # 8. Apply applicability filters using shared service
        weights_before_filter = base_weights.copy()
        weights = self._apply_applicability_filters(base_weights, financials)
        logger.info(f"{symbol} - Weights after applicability filters: {weights}")

        # Audit: Capture applicability filter
        if audit_trail and weights_before_filter != weights:
            step_number += 1
            adjustments = [
                WeightAdjustment(
                    model=model, source="applicability_filter", multiplier=0.0, reason="Model not applicable"
                )
                for model in weights_before_filter
                if weights_before_filter.get(model, 0) > 0 and weights.get(model, 0) == 0
            ]
            audit_trail.capture(
                step_number=step_number,
                step_name="applicability_filter",
                weights_before=weights_before_filter,
                weights_after=weights.copy(),
                adjustments=adjustments,
            )

        # 9. Apply data quality adjustments
        if data_quality:
            weights_before_quality = weights.copy()
            weights = self._apply_data_quality_adjustments(weights, data_quality)

            # Audit: Capture data quality adjustments
            if audit_trail and weights_before_quality != weights:
                step_number += 1
                adjustments = [
                    WeightAdjustment(
                        model=model,
                        source="data_quality",
                        multiplier=(
                            weights[model] / weights_before_quality[model]
                            if weights_before_quality.get(model, 0) > 0
                            else 1.0
                        ),
                        reason=f"Quality: {data_quality.get('model_quality', {}).get(model, 'unknown')}",
                    )
                    for model in weights
                    if weights.get(model, 0) != weights_before_quality.get(model, 0)
                ]
                audit_trail.capture(
                    step_number=step_number,
                    step_name="data_quality",
                    weights_before=weights_before_quality,
                    weights_after=weights.copy(),
                    adjustments=adjustments,
                    metadata={"data_quality": data_quality},
                )

        # 10. Normalize and round using shared service
        weights_before_norm = weights.copy()
        try:
            weights = self.weight_normalizer.normalize(
                weights, model_order=["dcf", "ggm", "pe", "ps", "pb", "ev_ebitda"]
            )
        except ValueError as e:
            logger.warning(f"Failed to normalize weights: {e}, using fallback")
            # Fallback to balanced default if normalization fails
            weights = self._get_tier_base_weights("balanced_default")
            weights = self.weight_normalizer.normalize(
                weights, model_order=["dcf", "ggm", "pe", "ps", "pb", "ev_ebitda"]
            )

        # Audit: Capture normalization
        if audit_trail:
            step_number += 1
            audit_trail.capture(
                step_number=step_number,
                step_name="normalization",
                weights_before=weights_before_norm,
                weights_after=weights.copy(),
                adjustments=[],
            )

        # 11. Log decision with enhanced context
        self._log_weighting_decision(
            symbol,
            tier,
            sub_tier,
            sector,
            industry,
            weights,
            company_size=company_size,
            profitability_stage=profitability_stage,
            market_pe=market_pe,
        )

        # Return weights, tier classification, and audit trail
        # Note: Caller is responsible for calling audit_trail.log_summary() if desired
        # This prevents duplicate logging when the caller also logs the summary
        return weights, sub_tier, audit_trail

    def _classify_tier(
        self,
        net_income: float,
        payout_ratio: float,
        rule_of_40: float,
        revenue_growth: float,
        fcf_margin: float,
        sector: str,
        revenue: float = 0,
        ebitda: float = 0,
        industry: Optional[str] = None,
        stockholders_equity: float = 0,
        symbol: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Classify company into one of 15+ sub-tiers based on decision tree.

        Args:
            net_income: Net income (TTM)
            payout_ratio: Dividend payout ratio (%)
            rule_of_40: Rule of 40 score (%)
            revenue_growth: Revenue growth (%)
            fcf_margin: FCF margin (%)
            sector: Normalized sector name
            revenue: Total revenue (for integrated cyclical check)
            ebitda: EBITDA (for pre-profit sub-tier check)
            industry: Industry name (for insurance/bank detection)
            stockholders_equity: Stockholders equity (for ROE calculation)
            symbol: Stock symbol (for database lookup if needed)

        Returns:
            (tier_name, sub_tier_name) tuple
        """
        # TIER 0: INSURANCE COMPANIES - Special handling (check FIRST)
        # Insurance companies should use P/BV valuation, not DCF
        if sector == "Financials" and industry and "insur" in industry.lower():
            # If stockholders_equity is missing, try to fetch from database
            if stockholders_equity <= 0 and symbol:
                try:
                    from investigator.domain.services.valuation.insurance_valuation import _fetch_from_database

                    db_equity, _, _ = _fetch_from_database(symbol, None)  # Will use config
                    if db_equity:
                        stockholders_equity = db_equity
                        logger.info(
                            f"{symbol} - Insurance tier: Fetched stockholders_equity from database: ${stockholders_equity/1e9:.2f}B"
                        )
                except Exception as e:
                    logger.warning(f"{symbol} - Insurance tier: Could not fetch equity from database: {e}")

            # Calculate ROE if we have the data
            if net_income > 0 and stockholders_equity > 0:
                roe = (net_income / stockholders_equity) * 100
                logger.info(
                    f"{symbol or 'UNKNOWN'} - Insurance tier classification: ROE={roe:.1f}% (NI=${net_income/1e9:.2f}B, Equity=${stockholders_equity/1e9:.2f}B)"
                )
                if roe >= 12:
                    return ("insurance", "insurance_high_quality")
                elif roe >= 8:
                    return ("insurance", "insurance_average")
                else:
                    return ("insurance", "insurance_challenged")
            else:
                # No ROE data, use default insurance tier
                logger.warning(
                    f"{symbol or 'UNKNOWN'} - Insurance tier: Cannot calculate ROE (NI={net_income}, Equity={stockholders_equity}), using default"
                )
                return ("insurance", "insurance_average")

        # TIER 0.5: AUTO MANUFACTURING - Special handling for automotive industry
        # Auto manufacturing has 0.254 avg reward with 199% error rate - worst performing major industry
        # Uses EV/EBITDA as primary (capital-intensive), P/B for tangible assets, P/E cyclically adjusted
        if self._is_auto_manufacturing(industry):
            return self._classify_auto_manufacturing_tier(
                symbol=symbol,
                industry=industry,
                financials_context={
                    "net_income": net_income,
                    "ebitda": ebitda,
                    "revenue": revenue,
                },
            )

        # TIER 0.55: Defense Contractors (P2-B)
        # Defense contractors need backlog-adjusted valuation
        if self._is_defense_contractor(industry, symbol):
            return self._classify_defense_tier(
                symbol,
                industry,
                {
                    "net_income": net_income,
                    "ebitda": ebitda,
                    "revenue": revenue,
                },
            )

        # TIER 0.6: Cloud/SaaS Growth (V2) - Check BEFORE pre-profit
        # SaaS companies have unique valuation drivers even when pre-profit
        # Must check before pre-profit tier to prevent SaaS companies from being
        # misclassified as generic pre-profit
        saas_config = self.tier_thresholds.get("saas_growth", {})
        saas_industries = saas_config.get(
            "industries",
            [
                "Software - Application",
                "Software - Infrastructure",
                "Information Technology Services",
                "Internet Content & Information",
                "Software",
                "Internet Software & Services",
                "EDP Services",
            ],
        )
        if industry and revenue > 0:
            industry_lower = industry.lower()
            is_saas = any(ind.lower() in industry_lower or industry_lower in ind.lower() for ind in saas_industries)
            if is_saas:
                # Calculate Rule of 40 for SaaS classification
                saas_rule_of_40 = revenue_growth + fcf_margin
                hyper_growth_threshold = saas_config.get("rule_of_40_hyper_growth", 60)
                strong_threshold = saas_config.get("rule_of_40_strong", 40)

                if saas_rule_of_40 > hyper_growth_threshold:
                    logger.info(
                        f"{symbol or 'UNKNOWN'} - SaaS hyper growth tier "
                        f"(industry={industry}, Rule of 40={saas_rule_of_40:.1f}%)"
                    )
                    return ("saas_growth", "saas_hyper_growth")
                elif saas_rule_of_40 > strong_threshold:
                    logger.info(
                        f"{symbol or 'UNKNOWN'} - SaaS growth strong tier "
                        f"(industry={industry}, Rule of 40={saas_rule_of_40:.1f}%)"
                    )
                    return ("saas_growth", "saas_growth_strong")
                else:
                    logger.info(
                        f"{symbol or 'UNKNOWN'} - SaaS maturing tier "
                        f"(industry={industry}, Rule of 40={saas_rule_of_40:.1f}%)"
                    )
                    return ("saas_growth", "saas_maturing")

        # TIER 0.65: Semiconductor Cyclical (P0-2)
        # Must check BEFORE high-growth to ensure proper industry classification
        # even for high-growth semis like NVDA
        semiconductor_industries = self.tier_thresholds.get("semiconductor_cyclical", {}).get(
            "industries",
            [
                "Semiconductors",
                "Semiconductor Equipment",
                "Semiconductors & Semiconductor Equipment",
                "Semiconductor Equipment & Materials",
            ],
        )
        if industry:
            industry_lower = industry.lower()
            is_semiconductor = any(
                semi_ind.lower() in industry_lower or industry_lower in semi_ind.lower()
                for semi_ind in semiconductor_industries
            )
            if is_semiconductor:
                logger.info(f"{symbol or 'UNKNOWN'} - Semiconductor cyclical tier (industry={industry})")
                return ("semiconductor_cyclical", "semiconductor_cyclical")

        # TIER 0.7: REITs (P1-C)
        # REITs need FFO-based valuation, not P/E or DCF
        # P/E is misleading due to depreciation; DCF doesn't capture property value
        if self._is_reit(sector, industry, symbol):
            return self._classify_reit_tier(symbol, sector, industry)

        # CRITICAL FIX: Use FCF as fallback when EBITDA is missing
        # Many profitable companies don't report EBITDA (or it's not extracted correctly)
        # Use FCF margin > 0 as proxy for profitability
        profitability_indicator = ebitda if ebitda != 0 else (fcf_margin * revenue / 100 if fcf_margin > 0 else 0)

        # Log tier classification inputs for debugging
        logger.debug(
            f"Tier classification inputs: net_income={net_income:.2f}B, "
            f"ebitda={ebitda:.2f}B, fcf_margin={fcf_margin:.1f}%, "
            f"rule_of_40={rule_of_40:.1f}%, revenue_growth={revenue_growth:.1f}%"
        )

        # DATA QUALITY SAFEGUARD: Known Dividend Aristocrats
        # Prevent misclassification due to YTD normalization failures or data extraction issues
        # Known dividend aristocrats (50+ years of increases) should ALWAYS be classified as dividend_aristocrat
        if symbol and symbol.upper() in KNOWN_DIVIDEND_ARISTOCRATS:
            company_name = KNOWN_DIVIDEND_ARISTOCRATS[symbol.upper()]
            if net_income <= 0:
                # This is almost certainly a data quality issue, not actual pre-profit status
                logger.warning(
                    f"{symbol} - DATA QUALITY SAFEGUARD TRIGGERED: "
                    f"{company_name} appears to have net_income <= 0 ({net_income:.2f}B), "
                    f"but this is a known Dividend Aristocrat with 50+ years of consecutive increases. "
                    f"Overriding classification from pre_profit to dividend_aristocrat_pure. "
                    f"Investigation recommended: Check YTD normalization and quarterly data extraction."
                )
                return ("dividend_aristocrat", "dividend_aristocrat_pure")
            else:
                # Known aristocrat with positive income - ALWAYS classify as dividend aristocrat
                # regardless of calculated payout ratio (payout may be <40% for growth-oriented aristocrats)
                # payout_ratio is in ratio format (0.0-1.0), so 0.60 = 60%
                if payout_ratio >= 0.60:
                    logger.info(
                        f"{symbol} - Known Dividend Aristocrat {company_name}: "
                        f"High payout ratio ({payout_ratio*100:.1f}%) - dividend_aristocrat_pure"
                    )
                    return ("dividend_aristocrat", "dividend_aristocrat_pure")
                else:
                    logger.info(
                        f"{symbol} - Known Dividend Aristocrat {company_name}: "
                        f"Payout ratio {payout_ratio*100:.1f}% - dividend_aristocrat_growth"
                    )
                    return ("dividend_aristocrat", "dividend_aristocrat_growth")

        # TIER 1: Pre-Profit
        if net_income <= 0:
            # Use profitability indicator (EBITDA or FCF-based) instead of just EBITDA
            if profitability_indicator <= 0:
                # True pre-profit: negative net income AND negative cash generation
                return ("pre_profit", "pre_profit_negative_ebitda")
            elif rule_of_40 > 40:
                # Pre-profit but high growth potential
                return ("pre_profit", "pre_profit_high_growth")
            else:
                # Pre-profit but positive EBITDA/FCF
                return ("pre_profit", "pre_profit_positive_ebitda")

        # TIER 2: High Dividend Payers (includes true Aristocrats and regular high-payout companies)
        # True Dividend Aristocrats: 25+ years of consecutive dividend increases (KNOWN_DIVIDEND_ARISTOCRATS)
        # High Dividend Payers: 40%+ payout ratio but <25 years of consecutive increases
        # payout_ratio is in ratio format (0.0-1.0), thresholds are also in ratio format
        dividend_threshold = self.tier_thresholds.get("dividend_aristocrat", {}).get("min_payout_ratio", 0.40)
        if payout_ratio >= dividend_threshold:
            pure_growth_cutoff = self.tier_thresholds.get("dividend_aristocrat", {}).get("sub_tier_growth_cutoff", 5)
            pure_payout_threshold = 0.60  # Pure aristocrat/payer threshold (60%)

            # Check if this is a KNOWN Dividend Aristocrat (25+ years verified)
            # Only these get the "dividend_aristocrat" tier with heavy GGM weighting
            if symbol and symbol.upper() in KNOWN_DIVIDEND_ARISTOCRATS:
                if payout_ratio >= pure_payout_threshold and revenue_growth < pure_growth_cutoff:
                    return ("dividend_aristocrat", "dividend_aristocrat_pure")
                else:
                    return ("dividend_aristocrat", "dividend_aristocrat_growth")
            else:
                # High dividend payer but NOT a verified aristocrat (e.g., ORCL with 11 years)
                # Use more balanced weights with less GGM emphasis
                if payout_ratio >= pure_payout_threshold and revenue_growth < pure_growth_cutoff:
                    logger.info(
                        f"{symbol or 'UNKNOWN'} - High dividend payer (payout={payout_ratio*100:.1f}%) "
                        f"but not in KNOWN_DIVIDEND_ARISTOCRATS list → high_dividend_payer_mature"
                    )
                    return ("high_dividend_payer", "high_dividend_payer_mature")
                else:
                    logger.info(
                        f"{symbol or 'UNKNOWN'} - High dividend payer (payout={payout_ratio*100:.1f}%) "
                        f"with growth ({revenue_growth:.1f}%) → high_dividend_payer_growth"
                    )
                    return ("high_dividend_payer", "high_dividend_payer_growth")

        # TIER 3: High-Growth
        high_growth_r40 = self.tier_thresholds.get("high_growth", {}).get("min_rule_of_40", 40)
        high_growth_revenue = self.tier_thresholds.get("high_growth", {}).get("min_revenue_growth_pct", 15)
        if rule_of_40 > high_growth_r40 or revenue_growth > high_growth_revenue:
            hyper_growth_r40 = self.tier_thresholds.get("high_growth", {}).get("hyper_growth_rule_of_40", 60)
            if rule_of_40 > hyper_growth_r40:
                return ("high_growth", "high_growth_hyper")
            else:
                return ("high_growth", "high_growth_strong")

        # TIER 4: Financial Services
        financial_sectors = self.tier_thresholds.get("financial_services", {}).get("sectors", [])
        if sector in financial_sectors:
            # V1 FIX: Check INDUSTRY for bank keywords (not sector)
            # Banks are in "Financials" sector with industry like "Banks", "Regional Banks", "Major Banks"
            bank_keywords = ["bank", "banking"]
            if industry and any(kw in industry.lower() for kw in bank_keywords):
                logger.info(f"{symbol or 'UNKNOWN'} - Traditional bank tier (industry={industry})")
                return ("financial", "financial_traditional_bank")
            else:
                return ("financial", "financial_asset_manager")

        # NOTE: SaaS tier check moved to TIER 0.6 (before pre-profit check)
        # to ensure SaaS companies get proper classification even when pre-profit
        # NOTE: Semiconductor tier check moved to TIER 0.65 (before pre-profit check)
        # to ensure semiconductors are classified before high-growth check (NVDA fix)

        # TIER 5: Cyclical
        cyclical_sectors = self.tier_thresholds.get("cyclical", {}).get("sectors", [])
        if sector in cyclical_sectors:
            # Large integrated cyclical (e.g., integrated oil majors)
            if revenue > 50e9:
                return ("cyclical", "cyclical_integrated")
            else:
                return ("cyclical", "cyclical_commodity")

        # TIER 6: Growth Hybrid
        hybrid_min_payout = self.tier_thresholds.get("growth_hybrid", {}).get("min_payout_ratio", 20)
        hybrid_max_payout = self.tier_thresholds.get("growth_hybrid", {}).get("max_payout_ratio", 40)
        hybrid_min_r40 = self.tier_thresholds.get("growth_hybrid", {}).get("min_rule_of_40", 25)
        hybrid_min_rev_growth = self.tier_thresholds.get("growth_hybrid", {}).get("min_revenue_growth_pct", 8)

        if (hybrid_min_payout <= payout_ratio < hybrid_max_payout) and (
            rule_of_40 > hybrid_min_r40 or revenue_growth > hybrid_min_rev_growth
        ):
            if sector == "Technology":
                return ("growth_hybrid", "growth_hybrid_tech")
            else:
                return ("growth_hybrid", "growth_hybrid_industrial")

        # TIER 7: Mature FCF Machine
        fcf_min_margin = self.tier_thresholds.get("mature_fcf_machine", {}).get("min_fcf_margin_pct", 20)
        fcf_max_rev_growth = self.tier_thresholds.get("mature_fcf_machine", {}).get("max_revenue_growth_pct", 8)
        fcf_max_payout = self.tier_thresholds.get("mature_fcf_machine", {}).get("max_payout_ratio", 20)

        if fcf_margin > fcf_min_margin and revenue_growth < fcf_max_rev_growth and payout_ratio < fcf_max_payout:
            if sector == "Technology":
                return ("mature_fcf", "mature_fcf_tech")
            else:
                return ("mature_fcf", "mature_fcf_industrial")

        # TIER 8: Balanced Default
        return ("balanced", "balanced_default")

    def _get_tier_base_weights(self, sub_tier: str) -> Dict[str, float]:
        """
        Get base weights for a given sub-tier from config.

        Args:
            sub_tier: Sub-tier name (e.g., "dividend_aristocrat_pure")

        Returns:
            Dict mapping model names to weights
        """
        weights = self.tier_base_weights.get(sub_tier)

        if not weights:
            logger.warning(f"No base weights found for sub_tier '{sub_tier}', using balanced_default")
            weights = self.tier_base_weights.get(
                "balanced_default", {"dcf": 30, "pe": 25, "ev_ebitda": 20, "ps": 15, "pb": 10, "ggm": 0}
            )

        return weights.copy()

    def _apply_industry_overrides(self, base_weights: Dict[str, float], industry: str) -> Dict[str, float]:
        """
        Apply industry-specific weight adjustments from config.

        Args:
            base_weights: Base weights from tier classification
            industry: Industry name

        Returns:
            Adjusted weights
        """
        industry_config = self.industry_specific.get(industry)

        if not industry_config:
            return base_weights  # No override for this industry

        # Check if tier should be overridden
        tier_override = industry_config.get("tier_override")
        if tier_override:
            logger.info(f"Industry {industry} triggers tier override: {tier_override}")
            # Get weights for override tier
            return self._get_tier_base_weights(tier_override)

        # Apply percentage adjustments
        adjustments = industry_config.get("weight_adjustments", {})
        adjusted_weights = base_weights.copy()

        for model, adjustment in adjustments.items():
            if model not in adjusted_weights:
                continue

            if isinstance(adjustment, str) and adjustment.endswith("%"):
                # Percentage adjustment (e.g., "+10%")
                adj_str = adjustment.rstrip("%")
                if adj_str == "100":
                    # Absolute assignment (e.g., "100%")
                    adjusted_weights[model] = 100.0
                else:
                    # Relative adjustment (e.g., "+10%")
                    pct_change = float(adj_str) / 100.0
                    adjusted_weights[model] *= 1.0 + pct_change

        logger.info(f"Applied industry override for {industry}: {industry_config.get('reason', 'No reason provided')}")

        return adjusted_weights

    def apply_market_context_adjustments(
        self, base_weights: Dict[str, float], market_context: MarketContext, symbol: str
    ) -> Dict[str, float]:
        """
        Apply market context multipliers to tier-based weights.

        This method applies cumulative multipliers based on:
        - Technical trend (strong_uptrend to strong_downtrend)
        - Market sentiment (very_bullish to very_bearish)
        - Risk level (very_low to very_high)
        - Quality tier (exceptional to poor)

        Multipliers are applied cumulatively:
            adjusted_weight = base_weight × trend_mult × sentiment_mult × risk_mult × quality_mult

        Args:
            base_weights: Tier-based weights before market context adjustments
            market_context: MarketContext with trend/sentiment/risk/quality factors
            symbol: Stock symbol (for logging)

        Returns:
            Adjusted weights after applying market context multipliers

        Example:
            Base: {"dcf": 50.0, "pe": 30.0, "ps": 15.0, "pb": 5.0}
            Context: strong_downtrend + very_bearish + high_risk
            Result: {"dcf": 37.0, "pe": 27.0, "ps": 11.0, "pb": 14.0, ...}
        """
        if not self.market_context_multipliers:
            logger.warning(f"{symbol} - market_context_multipliers not configured, skipping market context adjustments")
            return base_weights

        adjusted_weights = base_weights.copy()
        models = ["dcf", "ggm", "pe", "ps", "pb", "ev_ebitda"]

        # Track multipliers for logging
        multipliers_applied = {}

        # 1. Apply technical trend multipliers
        trend_config = self.market_context_multipliers.get("technical_trend", {})
        trend_key = market_context.technical_trend.value
        trend_multipliers = trend_config.get(trend_key, {})

        if trend_multipliers:
            multipliers_applied["trend"] = trend_key
            for model in models:
                mult = trend_multipliers.get(model, 1.0)
                adjusted_weights[model] = adjusted_weights.get(model, 0) * mult

        # 2. Apply market sentiment multipliers
        sentiment_config = self.market_context_multipliers.get("market_sentiment", {})
        sentiment_key = market_context.market_sentiment.value
        sentiment_multipliers = sentiment_config.get(sentiment_key, {})

        if sentiment_multipliers:
            multipliers_applied["sentiment"] = sentiment_key
            for model in models:
                mult = sentiment_multipliers.get(model, 1.0)
                adjusted_weights[model] = adjusted_weights.get(model, 0) * mult

        # 3. Apply risk level multipliers
        risk_config = self.market_context_multipliers.get("risk_level", {})
        risk_key = market_context.risk_level.value
        risk_multipliers = risk_config.get(risk_key, {})

        if risk_multipliers:
            multipliers_applied["risk"] = risk_key
            for model in models:
                mult = risk_multipliers.get(model, 1.0)
                adjusted_weights[model] = adjusted_weights.get(model, 0) * mult

        # 4. Apply quality tier multipliers (optional)
        if market_context.quality_tier:
            quality_config = self.market_context_multipliers.get("quality_tier", {})
            quality_key = market_context.quality_tier
            quality_multipliers = quality_config.get(quality_key, {})

            if quality_multipliers:
                multipliers_applied["quality"] = quality_key
                for model in models:
                    mult = quality_multipliers.get(model, 1.0)
                    adjusted_weights[model] = adjusted_weights.get(model, 0) * mult

        # Log market context adjustments
        logger.info(
            f"{symbol} - Market context adjustments applied: {market_context}. " f"Multipliers: {multipliers_applied}"
        )
        logger.debug(
            f"{symbol} - Weight changes: "
            f"DCF: {base_weights.get('dcf', 0):.1f}→{adjusted_weights.get('dcf', 0):.1f}, "
            f"PE: {base_weights.get('pe', 0):.1f}→{adjusted_weights.get('pe', 0):.1f}, "
            f"PS: {base_weights.get('ps', 0):.1f}→{adjusted_weights.get('ps', 0):.1f}, "
            f"PB: {base_weights.get('pb', 0):.1f}→{adjusted_weights.get('pb', 0):.1f}"
        )

        return adjusted_weights

    def _apply_applicability_filters(
        self,
        weights: Dict[str, float],
        financials: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Set model weight to 0% if fundamental assumptions are violated.

        Uses ModelApplicabilityRules shared service for centralized logic.

        Args:
            weights: Base weights
            financials: Financial metrics

        Returns:
            Filtered weights
        """
        filtered = weights.copy()

        # Debug: Log financials keys for troubleshooting
        logger.debug(f"Applicability filter inputs - financials keys: {list(financials.keys())}")

        # Check each model with non-zero weight
        for model, weight in weights.items():
            if weight > 0:
                is_applicable, reason = self.applicability_rules.is_applicable(model, financials)
                if not is_applicable:
                    filtered[model] = 0
                    logger.info(f"{model.upper()} filtered out: {reason}")
                else:
                    logger.debug(f"{model.upper()} passed applicability filter: {reason}")

        return filtered

    def _apply_data_quality_adjustments(
        self,
        weights: Dict[str, float],
        data_quality: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Reduce model weights based on data quality issues.

        Args:
            weights: Base weights
            data_quality: Data quality assessment

        Returns:
            Adjusted weights
        """
        adjusted = weights.copy()

        # Get data quality per model (if available)
        model_quality = data_quality.get("model_quality", {})

        for model, weight in weights.items():
            if weight == 0:
                continue  # Already filtered out

            quality_grade = model_quality.get(model, "good")
            quality_config = self.data_quality_thresholds.get(quality_grade, {})
            adjustment_pct = quality_config.get("weight_adjustment", 0)

            # Apply adjustment
            if adjustment_pct == 0:
                continue  # No change
            elif adjustment_pct == -100:
                adjusted[model] = 0  # Exclude
            else:
                # Relative adjustment (e.g., -10% means multiply by 0.9)
                multiplier = 1.0 + (adjustment_pct / 100.0)
                adjusted[model] = weight * multiplier

        return adjusted

    def _classify_company_size(self, market_cap: float) -> str:
        """
        Classify company by market capitalization.

        Args:
            market_cap: Market capitalization in dollars

        Returns:
            Size classification: "mega_cap", "large_cap", "mid_cap", or "small_cap"
        """
        if market_cap <= 0:
            return "unknown"
        elif market_cap >= 200e9:  # >= $200B
            return "mega_cap"
        elif market_cap >= 10e9:  # >= $10B
            return "large_cap"
        elif market_cap >= 2e9:  # >= $2B
            return "mid_cap"
        else:  # < $2B
            return "small_cap"

    def _classify_profitability_stage(
        self,
        net_income: float,
        revenue: float,
        revenue_growth: float,
        ebitda: float,
        operating_income: Optional[float] = None,
        free_cash_flow: Optional[float] = None,
        fcf_margin: Optional[float] = None,
    ) -> str:
        """
        Classify company's profitability stage using multi-indicator analysis.

        Uses ProfitabilityClassifier (M5) to check multiple indicators:
        net_income, operating_income, ebitda, free_cash_flow in priority order.

        Args:
            net_income: Net income (TTM)
            revenue: Total revenue
            revenue_growth: Revenue growth rate (%)
            ebitda: EBITDA
            operating_income: Operating income (optional)
            free_cash_flow: Free cash flow (optional)
            fcf_margin: FCF margin % (optional)

        Returns:
            Profitability stage: "pre_profit", "early_profitable", "transitioning", or "mature_profitable"
        """
        # Build financials dict for classifier
        financials = {
            "net_income": net_income if net_income != 0 else None,
            "revenue": revenue if revenue != 0 else None,
            "ebitda": ebitda if ebitda != 0 else None,
            "operating_income": operating_income,
            "free_cash_flow": free_cash_flow,
        }

        # Build ratios dict with calculated margins
        ratios = {
            "revenue_growth_pct": revenue_growth,
        }

        # Calculate margins if possible
        if revenue and revenue > 0:
            if net_income:
                ratios["net_margin"] = (net_income / revenue) * 100
            if operating_income:
                ratios["operating_margin"] = (operating_income / revenue) * 100
            if ebitda:
                ratios["ebitda_margin"] = (ebitda / revenue) * 100

        if fcf_margin is not None:
            ratios["fcf_margin"] = fcf_margin

        # Use multi-indicator classifier
        classification = self.profitability_classifier.classify(financials, ratios)

        # Map ProfitabilityStage to expected string values for backward compatibility
        stage_mapping = {
            ProfitabilityStage.PROFITABLE: "mature_profitable",
            ProfitabilityStage.MARGINALLY_PROFITABLE: "early_profitable",
            ProfitabilityStage.TRANSITIONING: "transitioning",
            ProfitabilityStage.PRE_PROFIT: "pre_profit",
            ProfitabilityStage.UNKNOWN: "pre_profit",  # Default to conservative
        }

        stage = stage_mapping.get(classification.stage, "mature_profitable")

        # Log classification details for debugging
        logger.debug(
            f"Profitability classification: {classification.stage.value} → {stage} "
            f"(indicators: {classification.indicators_positive}/{classification.indicators_total}, "
            f"confidence: {classification.confidence:.0%})"
        )

        return stage

    def _calculate_market_pe(self, current_price: float, ttm_eps: float) -> Optional[float]:
        """
        Calculate market's implied P/E ratio.

        Args:
            current_price: Current stock price
            ttm_eps: Trailing twelve month EPS

        Returns:
            Market P/E ratio, or None if calculation not possible
        """
        if current_price <= 0 or ttm_eps <= 0:
            return None

        market_pe = current_price / ttm_eps
        return round(market_pe, 2)

    def _apply_company_specific_adjustments(
        self,
        base_weights: Dict[str, float],
        company_size: str,
        profitability_stage: str,
        market_pe: Optional[float],
        revenue_growth: float,
        symbol: str,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Apply company-specific weight adjustments based on characteristics.

        This is the key method for handling edge cases like DASH (hypergrowth, early profitable).
        Uses sector-aware P/E thresholds via ThresholdRegistry (M5).

        Args:
            base_weights: Base weights from tier classification
            company_size: Company size classification
            profitability_stage: Profitability stage
            market_pe: Market's implied P/E ratio
            revenue_growth: Revenue growth rate (%)
            symbol: Stock symbol (for logging)
            sector: Company sector for threshold lookup
            industry: Company industry for threshold lookup

        Returns:
            Adjusted weights
        """
        adjusted_weights = base_weights.copy()
        adjustments_made = []

        # ADJUSTMENT 1: Extreme Market P/E - Reduce P/E Model Weight
        # Uses sector-aware thresholds from ThresholdRegistry (M5)
        if market_pe is not None:
            # Get sector-aware thresholds (defaults to Technology: 300/150/75 or Financials: 50/30/20)
            pe_thresholds = self.threshold_registry.get_pe_thresholds(sector, industry)
            pe_level = pe_thresholds.classify(market_pe)

            if pe_level == PELevel.EXTREME:
                # Extreme P/E: Reduce P/E weight to 10% max
                if adjusted_weights.get("pe", 0) > 10:
                    old_pe_weight = adjusted_weights["pe"]
                    adjusted_weights["pe"] = 10.0
                    # Redistribute to DCF
                    weight_reduction = old_pe_weight - 10.0
                    adjusted_weights["dcf"] = adjusted_weights.get("dcf", 0) + weight_reduction
                    adjustments_made.append(
                        f"Extreme market P/E ({market_pe:.0f}x > {pe_thresholds.extreme_high}x for {sector or 'default'}) "
                        f"→ Reduced PE weight to 10%, boosted DCF"
                    )

            elif pe_level == PELevel.HIGH:
                # High P/E: Reduce P/E weight by 50%
                if adjusted_weights.get("pe", 0) > 20:
                    old_pe_weight = adjusted_weights["pe"]
                    adjusted_weights["pe"] = old_pe_weight * 0.5
                    # Redistribute to DCF
                    weight_reduction = old_pe_weight * 0.5
                    adjusted_weights["dcf"] = adjusted_weights.get("dcf", 0) + weight_reduction
                    adjustments_made.append(
                        f"High market P/E ({market_pe:.0f}x > {pe_thresholds.high}x for {sector or 'default'}) "
                        f"→ Reduced PE weight by 50%, boosted DCF"
                    )

            elif pe_level == PELevel.MODERATE:
                # Moderate P/E above threshold: Reduce P/E weight by 25%
                if adjusted_weights.get("pe", 0) > 30:
                    old_pe_weight = adjusted_weights["pe"]
                    adjusted_weights["pe"] = old_pe_weight * 0.75
                    # Redistribute to DCF
                    weight_reduction = old_pe_weight * 0.25
                    adjusted_weights["dcf"] = adjusted_weights.get("dcf", 0) + weight_reduction
                    adjustments_made.append(
                        f"Moderate-high market P/E ({market_pe:.0f}x > {pe_thresholds.moderate}x for {sector or 'default'}) "
                        f"→ Reduced PE weight by 25%, boosted DCF"
                    )

        # ADJUSTMENT 2: Early Profitable + High Growth - Favor DCF Over P/E
        # Companies like DASH: recently turned profitable, high growth, TTM earnings unreliable
        if profitability_stage == "early_profitable" and revenue_growth > 30:
            # Target: DCF 70%, P/E 30% max
            if adjusted_weights.get("pe", 0) > 30:
                old_pe_weight = adjusted_weights["pe"]
                adjusted_weights["pe"] = 30.0
                # Redistribute to DCF
                weight_reduction = old_pe_weight - 30.0
                adjusted_weights["dcf"] = adjusted_weights.get("dcf", 0) + weight_reduction
                adjustments_made.append(
                    f"Early profitable + high growth ({revenue_growth:.1f}%) → Capped PE at 30%, boosted DCF"
                )

        # ADJUSTMENT 3: Small-Cap Volatility - Increase DCF Weight
        # Small-caps have less reliable multiples, favor intrinsic valuation
        if company_size == "small_cap":
            # Reduce multiple-based models (PE, PS, PB, EV/EBITDA) by 20%
            multiple_models = ["pe", "ps", "pb", "ev_ebitda"]
            total_reduction = 0
            for model in multiple_models:
                if adjusted_weights.get(model, 0) > 0:
                    old_weight = adjusted_weights[model]
                    adjusted_weights[model] = old_weight * 0.8
                    total_reduction += old_weight * 0.2

            if total_reduction > 0:
                # Redistribute to DCF
                adjusted_weights["dcf"] = adjusted_weights.get("dcf", 0) + total_reduction
                adjustments_made.append(f"Small-cap → Reduced multiple-based models by 20%, boosted DCF")

        # ADJUSTMENT 4: Mega-Cap Stability - More Balanced
        # Mega-caps have reliable data across models, can use more balanced approach
        elif company_size == "mega_cap":
            # Ensure no single model dominates excessively (cap at 50%)
            for model in ["dcf", "pe", "ps", "ev_ebitda"]:
                if adjusted_weights.get(model, 0) > 50:
                    excess = adjusted_weights[model] - 50.0
                    adjusted_weights[model] = 50.0
                    # Distribute excess evenly to other non-zero models
                    other_models = [m for m in adjusted_weights if m != model and adjusted_weights.get(m, 0) > 0]
                    if other_models:
                        per_model_boost = excess / len(other_models)
                        for other_model in other_models:
                            adjusted_weights[other_model] = adjusted_weights.get(other_model, 0) + per_model_boost
                        adjustments_made.append(f"Mega-cap → Capped {model.upper()} at 50% for balanced approach")

        # ADJUSTMENT 5: Pre-Profit High Growth - Favor Price/Sales
        # Pre-profit companies: P/E doesn't work, favor PS ratio and DCF
        if profitability_stage == "pre_profit" and revenue_growth > 30:
            # Boost PS weight if present
            if adjusted_weights.get("ps", 0) > 0:
                old_ps_weight = adjusted_weights.get("ps", 0)
                adjusted_weights["ps"] = old_ps_weight * 1.5
                adjustments_made.append(f"Pre-profit high growth → Boosted PS by 50%")

        # Log adjustments if any were made
        if adjustments_made:
            logger.info(
                f"🔧 {symbol} - Company-specific adjustments applied:\n"
                + "\n".join([f"   • {adj}" for adj in adjustments_made])
            )

        return adjusted_weights

    def _log_weighting_decision(
        self,
        symbol: str,
        tier: str,
        sub_tier: str,
        sector: str,
        industry: Optional[str],
        weights: Dict[str, float],
        company_size: Optional[str] = None,
        profitability_stage: Optional[str] = None,
        market_pe: Optional[float] = None,
    ) -> None:
        """
        Log weighting decision for audit trail with enhanced context.

        Args:
            symbol: Stock symbol
            tier: Primary tier name
            sub_tier: Sub-tier name
            sector: Normalized sector
            industry: Industry name
            weights: Final weights
            company_size: Company size classification (optional)
            profitability_stage: Profitability stage (optional)
            market_pe: Market P/E ratio (optional)
        """
        # Format weights for logging
        non_zero_weights = {k: v for k, v in weights.items() if v > 0}
        weights_str = ", ".join([f"{model.upper()}={weight:.0f}%" for model, weight in non_zero_weights.items()])

        # Build enhanced context string
        context_parts = [f"Tier={sub_tier}", f"Sector={sector}"]

        if industry:
            context_parts.append(f"Industry={industry}")

        if company_size:
            context_parts.append(f"Size={company_size}")

        if profitability_stage:
            context_parts.append(f"Stage={profitability_stage}")

        if market_pe is not None:
            context_parts.append(f"Market_PE={market_pe:.0f}x")

        context_str = " | ".join(context_parts)

        logger.info(f"🎯 {symbol} - Dynamic Weighting: {context_str} | Weights: {weights_str}")

    # =========================================================================
    # AUTO MANUFACTURING TIER METHODS (P1-A)
    # =========================================================================

    def _is_auto_manufacturing(self, industry: Optional[str]) -> bool:
        """
        Check if the industry is an auto manufacturing industry.

        Uses config-driven industry pattern matching.

        Args:
            industry: Industry name from company metadata

        Returns:
            True if industry matches auto manufacturing patterns
        """
        if not industry:
            return False

        # Get auto manufacturing config from tier thresholds
        auto_config = self.tier_thresholds.get("auto_manufacturing", {})
        industry_patterns = auto_config.get(
            "industries",
            [
                "Auto Manufacturing",
                "Automobile Manufacturers",
                "Motor Vehicles",
                "Auto Manufacturers",
                "Automotive",
                "Auto - Manufacturers",
                "Automobiles",
            ],
        )

        industry_lower = industry.lower()

        # Check for pattern matches (case-insensitive)
        for pattern in industry_patterns:
            if pattern.lower() in industry_lower or industry_lower in pattern.lower():
                return True

        # Also check for common automotive keywords
        automotive_keywords = ["auto", "automobile", "motor vehicle", "car manufacturer"]
        for keyword in automotive_keywords:
            if keyword in industry_lower:
                return True

        return False

    def _classify_auto_manufacturing_tier(
        self,
        symbol: Optional[str],
        industry: Optional[str],
        financials_context: Dict[str, Any],
    ) -> Tuple[str, str]:
        """
        Classify auto manufacturing company into appropriate sub-tier.

        Sub-tiers are based on EV transition status:
        - auto_manufacturing_ev_leader: >50% EV revenue (TSLA, RIVN)
        - auto_manufacturing_transitioning: 25-50% EV revenue (F, GM in transition)
        - auto_manufacturing_traditional: <25% EV revenue (traditional ICE manufacturers)

        Args:
            symbol: Stock symbol
            industry: Industry name
            financials_context: Dict with net_income, ebitda, revenue

        Returns:
            Tuple of (tier_name, sub_tier_name)
        """
        # Get EV revenue percentage from company-specific data
        # This would ideally come from financials, but we use symbol mapping for now
        ev_revenue_pct = self._get_ev_revenue_percentage(symbol)

        # Get thresholds from config
        auto_config = self.tier_thresholds.get("auto_manufacturing", {})
        ev_transition = auto_config.get("ev_transition", {})
        ev_leader_threshold = ev_transition.get("ev_leader_threshold", 0.50)
        transitioning_threshold = ev_transition.get("transitioning_threshold", 0.25)

        # Classify based on EV revenue percentage
        if ev_revenue_pct >= ev_leader_threshold:
            sub_tier = "auto_manufacturing_ev_leader"
            logger.info(
                f"{symbol or 'UNKNOWN'} - Auto Manufacturing tier: EV Leader "
                f"(EV revenue {ev_revenue_pct*100:.0f}% >= {ev_leader_threshold*100:.0f}%)"
            )
        elif ev_revenue_pct >= transitioning_threshold:
            sub_tier = "auto_manufacturing_transitioning"
            logger.info(
                f"{symbol or 'UNKNOWN'} - Auto Manufacturing tier: Transitioning "
                f"(EV revenue {ev_revenue_pct*100:.0f}% >= {transitioning_threshold*100:.0f}%)"
            )
        else:
            sub_tier = "auto_manufacturing_traditional"
            logger.info(
                f"{symbol or 'UNKNOWN'} - Auto Manufacturing tier: Traditional ICE "
                f"(EV revenue {ev_revenue_pct*100:.0f}% < {transitioning_threshold*100:.0f}%)"
            )

        return ("auto_manufacturing", sub_tier)

    def _get_ev_revenue_percentage(self, symbol: Optional[str], xbrl_data: Optional[Dict] = None) -> float:
        """
        Get EV revenue percentage for an auto manufacturer.

        Tries XBRL extraction first, then falls back to known company mappings.
        In production, XBRL data would come from SEC filings when available.

        Args:
            symbol: Stock symbol
            xbrl_data: Optional XBRL data dict with EV metrics

        Returns:
            EV revenue percentage as a decimal (0.0 to 1.0)
        """
        if not symbol:
            return 0.0

        # Try XBRL extraction first (preferred source)
        if xbrl_data:
            try:
                from utils.xbrl_tag_aliases import XBRLTagAliasMapper

                mapper = XBRLTagAliasMapper()
                ev_pct = mapper.extract_value_with_fallbacks(xbrl_data, "ev_sales_mix_pct")
                if ev_pct is not None and 0 <= ev_pct <= 1:
                    logger.info(f"{symbol} - EV mix from XBRL: {ev_pct:.1%}")
                    return ev_pct
            except Exception as e:
                logger.debug(f"{symbol} - XBRL EV mix extraction failed: {e}")

        # Fall back to known mapping (static estimates as of 2024)
        # These should be updated periodically or extracted from financial data
        EV_REVENUE_ESTIMATES = {
            # EV Leaders (>50% EV)
            "TSLA": 0.95,  # Tesla - almost 100% EV
            "RIVN": 0.95,  # Rivian - 100% EV
            "LCID": 0.95,  # Lucid - 100% EV
            "NIO": 0.95,  # NIO - 100% EV
            "XPEV": 0.95,  # XPeng - 100% EV
            "LI": 0.95,  # Li Auto - primarily EV/PHEV
            # Transitioning (25-50% EV)
            "GM": 0.15,  # General Motors - growing EV presence
            "F": 0.12,  # Ford - F-150 Lightning, Mustang Mach-E
            "STLA": 0.10,  # Stellantis - various EV launches
            "VWAGY": 0.12,  # Volkswagen - ID series
            "BMWYY": 0.15,  # BMW - i-series
            "MBGAF": 0.10,  # Mercedes-Benz - EQ series
            "HMC": 0.05,  # Honda - partnership with GM
            "TM": 0.05,  # Toyota - lagging in EV
            # Traditional ICE (<10% EV)
            # Most other manufacturers default to this tier
        }

        return EV_REVENUE_ESTIMATES.get(symbol.upper(), 0.05)  # Default to 5% for legacy

    def calculate_ev_transition_premium(
        self,
        symbol: str,
        base_fair_value: float,
    ) -> Tuple[float, float, str]:
        """
        Calculate EV transition premium for auto manufacturers.

        Implements P1-A2: EV Transition Adjustment
        - >50% EV revenue: 20% premium
        - 25-50% EV revenue: 10% premium
        - 10-25% EV revenue: 5% premium
        - <10% EV revenue: No premium

        Args:
            symbol: Stock symbol
            base_fair_value: Base fair value before EV premium

        Returns:
            Tuple of (adjusted_fair_value, premium_pct, tier_name)
        """
        ev_pct = self._get_ev_revenue_percentage(symbol)

        # Get thresholds from config
        auto_config = self.tier_thresholds.get("auto_manufacturing", {})
        ev_transition = auto_config.get("ev_transition", {})

        ev_leader_threshold = ev_transition.get("ev_leader_threshold", 0.50)
        transitioning_threshold = ev_transition.get("transitioning_threshold", 0.25)
        emerging_threshold = ev_transition.get("emerging_threshold", 0.10)

        ev_leader_premium = ev_transition.get("ev_leader_premium", 0.20)
        transitioning_premium = ev_transition.get("transitioning_premium", 0.10)
        emerging_premium = ev_transition.get("emerging_premium", 0.05)

        # Determine premium based on EV revenue percentage
        if ev_pct >= ev_leader_threshold:
            premium = ev_leader_premium
            tier = "EV Leader"
        elif ev_pct >= transitioning_threshold:
            premium = transitioning_premium
            tier = "EV Transitioning"
        elif ev_pct >= emerging_threshold:
            premium = emerging_premium
            tier = "EV Emerging"
        else:
            premium = 0.0
            tier = "Traditional ICE"

        adjusted_value = base_fair_value * (1 + premium)

        logger.info(
            f"{symbol} - EV Transition Premium: {tier} "
            f"(EV revenue {ev_pct*100:.0f}%) -> {premium*100:.0f}% premium, "
            f"Adjusted value: ${adjusted_value:.2f} (was ${base_fair_value:.2f})"
        )

        return adjusted_value, premium, tier

    def detect_capex_burden(
        self,
        symbol: str,
        capex: float,
        depreciation: float,
    ) -> Tuple[bool, float, str]:
        """
        Detect high capital intensity based on capex/depreciation ratio.

        Implements P1-A3: Capex Burden Detection
        - Capex/Depreciation > 1.5x: High capital intensity flag
        - Capex/Depreciation > 2.0x: Warning flag

        Args:
            symbol: Stock symbol
            capex: Capital expenditures (should be positive)
            depreciation: Depreciation expense (should be positive)

        Returns:
            Tuple of (is_high_intensity, ratio, severity)
            severity: "normal", "high", or "warning"
        """
        if depreciation <= 0:
            logger.warning(f"{symbol} - Cannot calculate capex burden: depreciation <= 0")
            return False, 0.0, "unknown"

        # Ensure capex is positive (sometimes reported as negative in statements)
        capex_abs = abs(capex)
        ratio = capex_abs / depreciation

        # Get thresholds from config
        auto_config = self.tier_thresholds.get("auto_manufacturing", {})
        capex_config = auto_config.get("capex_burden", {})
        high_threshold = capex_config.get("high_intensity_threshold", 1.5)
        warning_threshold = capex_config.get("warning_threshold", 2.0)

        if ratio >= warning_threshold:
            severity = "warning"
            is_high = True
            logger.warning(
                f"{symbol} - CAPEX BURDEN WARNING: Capex/Depreciation = {ratio:.2f}x "
                f"(> {warning_threshold}x threshold). Heavy investment phase may pressure FCF."
            )
        elif ratio >= high_threshold:
            severity = "high"
            is_high = True
            logger.info(
                f"{symbol} - High capex intensity: Capex/Depreciation = {ratio:.2f}x "
                f"(> {high_threshold}x threshold). Capital-intensive phase."
            )
        else:
            severity = "normal"
            is_high = False
            logger.debug(f"{symbol} - Normal capex intensity: Capex/Depreciation = {ratio:.2f}x")

        return is_high, ratio, severity

    def get_auto_manufacturing_dcf_parameters(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Get DCF parameters specific to auto manufacturing.

        Returns config-driven parameters with:
        - Lower terminal growth (2% vs 3% default)
        - Lower terminal margins (6-8% vs tech's 25-35%)
        - Longer projection periods for capital cycles

        Args:
            symbol: Stock symbol

        Returns:
            Dict with terminal_growth, terminal_margin_range, projection_years
        """
        auto_config = self.tier_thresholds.get("auto_manufacturing", {})

        terminal_growth = auto_config.get("terminal_growth", 0.02)
        margin_range = auto_config.get("terminal_margin_range", {"min": 0.06, "max": 0.08})

        # Adjust parameters based on EV status
        ev_pct = self._get_ev_revenue_percentage(symbol)

        # EV leaders may have higher terminal growth potential
        if ev_pct >= 0.50:
            terminal_growth = min(terminal_growth + 0.005, 0.03)  # Max 3%
            margin_range = {"min": 0.08, "max": 0.12}  # EV margins potentially higher

        params = {
            "terminal_growth": terminal_growth,
            "terminal_margin_min": margin_range.get("min", 0.06),
            "terminal_margin_max": margin_range.get("max", 0.08),
            "projection_years": 7,  # Longer for capital-intensive industries
            "ev_revenue_pct": ev_pct,
        }

        logger.debug(
            f"{symbol} - Auto Manufacturing DCF params: "
            f"terminal_growth={terminal_growth*100:.1f}%, "
            f"margin_range=[{margin_range['min']*100:.0f}%-{margin_range['max']*100:.0f}%]"
        )

        return params

    # =========================================================================
    # DEFENSE CONTRACTOR TIER METHODS (P2-B)
    # =========================================================================

    def _is_defense_contractor(self, industry: Optional[str], symbol: Optional[str]) -> bool:
        """Check if company is a defense contractor."""
        from investigator.domain.services.valuation.defense_valuation import (
            KNOWN_DEFENSE_CONTRACTORS,
            DEFENSE_INDUSTRIES,
        )

        if symbol and symbol.upper() in KNOWN_DEFENSE_CONTRACTORS:
            return True

        if industry:
            industry_lower = industry.lower()
            for defense_ind in DEFENSE_INDUSTRIES:
                if defense_ind.lower() in industry_lower or industry_lower in defense_ind.lower():
                    return True

        return False

    def _classify_defense_tier(
        self, symbol: Optional[str], industry: Optional[str], financials_context: Dict
    ) -> Tuple[str, str]:
        """Classify defense contractor tier based on backlog visibility."""
        # Use defense_contractor tier with backlog-adjusted weights
        logger.info(f"{symbol or 'UNKNOWN'} - Defense contractor tier (industry={industry})")
        return ("defense_contractor", "defense_contractor")

    # =========================================================================
    # REIT TIER CLASSIFICATION (P1-C)
    # =========================================================================

    def _is_reit(self, sector: Optional[str], industry: Optional[str], symbol: Optional[str]) -> bool:
        """
        Check if company is a REIT (Real Estate Investment Trust).

        REITs are identified by:
        1. Known REIT symbol mapping
        2. Sector = "Real Estate"
        3. Industry contains "REIT" or "Real Estate Investment Trust"
        """
        # Import known REIT mappings
        try:
            from investigator.domain.services.valuation.reit_valuation import KNOWN_REIT_MAPPINGS

            if symbol and symbol.upper() in KNOWN_REIT_MAPPINGS:
                return True
        except ImportError:
            pass

        # Check sector
        if sector and "real estate" in sector.lower():
            return True

        # Check industry
        if industry:
            industry_lower = industry.lower()
            reit_keywords = ["reit", "real estate investment trust", "real estate investment"]
            for keyword in reit_keywords:
                if keyword in industry_lower:
                    return True

        return False

    def _classify_reit_tier(
        self, symbol: Optional[str], sector: Optional[str], industry: Optional[str]
    ) -> Tuple[str, str]:
        """
        Classify REIT into property-type-specific tier.

        Uses FFO-based valuation with property-type-specific multiples:
        - Data centers, cell towers, industrial: Premium (20-26x FFO)
        - Residential, healthcare, net lease: Mid-tier (14-20x FFO)
        - Office, retail: Discount (8-15x FFO)
        """
        try:
            from investigator.domain.services.valuation.reit_valuation import (
                KNOWN_REIT_MAPPINGS,
                REITPropertyType,
                detect_reit_property_type,
            )

            # Try to detect property type
            if symbol and symbol.upper() in KNOWN_REIT_MAPPINGS:
                property_type = KNOWN_REIT_MAPPINGS[symbol.upper()]
                logger.info(
                    f"{symbol or 'UNKNOWN'} - REIT tier (property_type={property_type.value}, "
                    f"detected via known mapping)"
                )
            else:
                # Try to detect from company name/industry
                property_type = detect_reit_property_type(symbol or "", industry)
                logger.info(
                    f"{symbol or 'UNKNOWN'} - REIT tier (property_type={property_type.value}, "
                    f"detected from industry={industry})"
                )

            # Map property type to sub-tier
            premium_types = [
                REITPropertyType.INDUSTRIAL_LOGISTICS,
                REITPropertyType.DATA_CENTERS,
                REITPropertyType.CELL_TOWERS,
                REITPropertyType.SELF_STORAGE,
            ]
            discount_types = [
                REITPropertyType.OFFICE_CLASS_B,
                REITPropertyType.OFFICE_GENERAL,
                REITPropertyType.REGIONAL_MALLS,
            ]

            if property_type in premium_types:
                return ("reit", "reit_premium_growth")
            elif property_type in discount_types:
                return ("reit", "reit_value")
            else:
                return ("reit", "reit_core")

        except ImportError as e:
            logger.warning(f"REIT valuation module not available: {e}")
            return ("reit", "reit_core")
