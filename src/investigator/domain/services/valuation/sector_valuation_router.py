"""
Sector-Aware Valuation Router

Routes to appropriate valuation methods based on company sector and industry:
- Technology: DCF with high growth rates
- Insurance: Price-to-Book (P/BV) and Dividend Discount Model (DDM)
- Banks: Return on Equity (ROE) multiples
- REITs: Funds from Operations (FFO) multiples
- Biotech (Pre-Revenue): Pipeline probability-weighted valuation (P2-A)
- Defense Contractors: Backlog-adjusted EV/EBITDA + P/E valuation (P2-B)
- Default: Standard DCF

Now integrates with IndustryDatasetRegistry for industry-specific metrics extraction
and valuation adjustments.

Author: Claude Code
Date: 2025-11-10
Updated: 2025-12-29 - Added biotech pre-revenue valuation (P2-A)
Updated: 2025-12-29 - Added defense contractor valuation tier (P2-B)
Updated: 2025-12-30 - Integrated IndustryDatasetRegistry for metrics extraction
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValuationResult:
    """Result from sector-specific valuation"""
    method: str  # e.g., "DCF", "P/BV", "DDM", "ROE_Multiple", "FFO_Multiple"
    fair_value: float
    current_price: float
    upside_percent: float
    confidence: str  # "high", "medium", "low"
    details: Dict  # Method-specific details
    warnings: list  # Any warnings or caveats
    # Industry-specific metrics from IndustryDatasetRegistry
    industry_metrics: Optional[Dict[str, Any]] = None
    industry_adjustments: Optional[List[Dict[str, Any]]] = None
    adjusted_fair_value: Optional[float] = None  # Fair value after industry adjustments


class SectorValuationRouter:
    """Routes to appropriate valuation method based on sector/industry"""

    # Sector/industry routing map
    VALUATION_METHODS = {
        # Insurance companies - use P/BV and DDM
        ('Financials', 'Insurance'): 'insurance',

        # Banks - use ROE multiples
        ('Financials', 'Banks'): 'bank',

        # REITs - use FFO multiples
        ('Real Estate', 'REITs'): 'reit',
        ('Real Estate', None): 'reit',  # All Real Estate defaults to REIT

        # Biotech (pre-revenue) - use pipeline probability-weighted valuation
        ('Healthcare', 'Biotechnology'): 'biotech',
        ('Healthcare', 'Biopharmaceuticals'): 'biotech',
        ('Healthcare', 'Biological Products'): 'biotech',

        # Defense Contractors (P2-B) - use backlog-adjusted valuation
        ('Industrials', 'Aerospace & Defense'): 'defense',
        ('Industrials', 'Defense'): 'defense',
        ('Industrials', 'Military/Government Technical'): 'defense',
        ('Industrials', 'Government Services'): 'defense',
        ('Industrials', 'Defense Primes'): 'defense',
        ('Industrials', 'Defense Electronics'): 'defense',

        # Technology - use DCF with high growth
        ('Technology', None): 'dcf_growth',

        # Default for other sectors
        ('default', None): 'dcf_standard',
    }

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def route_valuation(
        self,
        symbol: str,
        sector: Optional[str],
        industry: Optional[str],
        financials: Dict,
        current_price: float,
        database_url: Optional[str] = None,
        xbrl_data: Optional[Dict] = None
    ) -> ValuationResult:
        """
        Route to appropriate valuation method based on sector/industry

        Args:
            symbol: Stock symbol
            sector: Company sector (e.g., "Financials", "Technology")
            industry: Company industry (e.g., "Insurance", "Banks")
            financials: Dictionary of financial metrics
            current_price: Current stock price
            database_url: Optional database connection string
            xbrl_data: Optional raw XBRL data for sector-specific metric extraction (P1-B)

        Returns:
            ValuationResult with fair value and details
        """
        # Flexible matching for insurance companies
        # Matches: "Insurance", "Property-Casualty Insurers", "Life Insurance", etc.
        # Database uses "Finance" sector, but some sources use "Financials"
        is_financial_sector = sector in ('Finance', 'Financials', 'Financial Services')
        if is_financial_sector and industry and 'insur' in industry.lower():
            valuation_type = 'insurance'
        # Flexible matching for banks
        # Matches: "Banks", "Commercial Banks", "Regional Banks", etc.
        elif is_financial_sector and industry and 'bank' in industry.lower():
            valuation_type = 'bank'
        # Flexible matching for defense contractors (P2-B)
        # Matches: "Aerospace & Defense", "Defense", "Military/Government Technical", etc.
        elif self._is_defense_industry(industry, sector, symbol):
            valuation_type = 'defense'
        # Flexible matching for biotech companies (P2-A)
        # Matches: "Biotechnology", "Biopharmaceuticals", "Biological Products", etc.
        elif self._is_biotech_industry(industry, sector, financials):
            valuation_type = 'biotech'
        # Flexible REIT matching - catches more variations
        # Matches: "REIT", "Real Estate Investment Trust", sector "Real Estate", etc.
        elif self._is_reit(sector, industry):
            valuation_type = 'reit'
        # Use exact dictionary matching for other sectors
        else:
            method_key = (sector, industry)
            valuation_type = self.VALUATION_METHODS.get(
                method_key,
                self.VALUATION_METHODS.get((sector, None),
                                          self.VALUATION_METHODS[('default', None)])
            )

        self.logger.info(
            f"{symbol} - Routing to {valuation_type} valuation "
            f"(sector={sector}, industry={industry})"
        )

        # Route to appropriate method
        if valuation_type == 'insurance':
            # P1-B Enhanced: Pass XBRL data for combined ratio extraction
            return self._value_insurance(
                symbol=symbol,
                financials=financials,
                current_price=current_price,
                database_url=database_url,
                xbrl_data=xbrl_data,
                industry=industry
            )
        elif valuation_type == 'bank':
            return self._value_bank(symbol, financials, current_price)
        elif valuation_type == 'reit':
            # Extract company_name from financials if available
            company_name = financials.get('company_name') or financials.get('entityName')
            return self._value_reit(
                symbol=symbol,
                financials=financials,
                current_price=current_price,
                company_name=company_name,
                industry=industry
            )
        elif valuation_type == 'biotech':
            # P2-A: Pre-revenue biotech pipeline valuation
            company_name = financials.get('company_name') or financials.get('entityName')
            pipeline = financials.get('pipeline', [])  # Pipeline data if available
            return self._value_biotech(
                symbol=symbol,
                financials=financials,
                current_price=current_price,
                company_name=company_name,
                industry=industry,
                pipeline=pipeline
            )
        elif valuation_type == 'defense':
            # P2-B: Defense contractor backlog-adjusted valuation
            return self._value_defense_contractor(
                symbol=symbol,
                financials=financials,
                current_price=current_price,
                xbrl_data=xbrl_data,
                industry=industry
            )
        else:
            # For non-special sectors (including dcf_growth, dcf_standard), return None
            # to signal that the caller should use standard DCF valuation
            return None

    def _value_insurance(
        self,
        symbol: str,
        financials: Dict,
        current_price: float,
        database_url: Optional[str] = None,
        xbrl_data: Optional[Dict] = None,
        industry: Optional[str] = None
    ) -> ValuationResult:
        """
        Value insurance company using Price-to-Book (P/BV) method

        Insurance companies are valued primarily on book value because:
        - Balance sheet quality is critical
        - Float management is key asset
        - ROE stability matters more than growth

        P1-B Enhanced: Now extracts actual combined ratio from XBRL data
        instead of using net margin as a proxy.
        """
        from investigator.domain.services.valuation.insurance_valuation import (
            value_insurance_company,
            InsuranceType,
        )

        # Detect insurance type from industry string
        insurance_type = self._detect_insurance_type(industry)

        try:
            result = value_insurance_company(
                symbol=symbol,
                financials=financials,
                current_price=current_price,
                database_url=database_url,
                xbrl_data=xbrl_data,
                insurance_type=insurance_type
            )

            upside = ((result['fair_value'] - current_price) / current_price) * 100

            # Build method string with insurance type if known
            if insurance_type != InsuranceType.UNKNOWN:
                method = f"P/BV (Insurance - {insurance_type.value.replace('_', ' ').title()})"
            else:
                method = "P/BV (Insurance)"

            # Build enhanced details including P1-B combined ratio metrics
            details = {
                'book_value_per_share': result.get('book_value_per_share'),
                'target_pb_ratio': result.get('target_pb_ratio'),
                'current_pb_ratio': result.get('current_pb_ratio'),
                'roe': result.get('roe'),
                'net_margin': result.get('net_margin'),
                # P1-B Enhanced: Actual combined ratio from XBRL
                'combined_ratio': result.get('combined_ratio'),
                'loss_ratio': result.get('loss_ratio'),
                'expense_ratio': result.get('expense_ratio'),
                'underwriting_quality': result.get('underwriting_quality'),
                'underwriting_description': result.get('underwriting_description'),
                'insurance_type': result.get('insurance_type'),
                'target_combined_ratio': result.get('target_combined_ratio'),
            }

            return ValuationResult(
                method=method,
                fair_value=result['fair_value'],
                current_price=current_price,
                upside_percent=upside,
                confidence=result.get('confidence', 'medium'),
                details=details,
                warnings=result.get('warnings', [])
            )
        except Exception as e:
            self.logger.warning(f"{symbol} - Insurance valuation failed: {e}")
            raise

    def _detect_insurance_type(self, industry: Optional[str]) -> 'InsuranceType':
        """
        Detect insurance type from industry string.

        Args:
            industry: Industry classification string

        Returns:
            InsuranceType enum value
        """
        from investigator.domain.services.valuation.insurance_valuation import InsuranceType

        if not industry:
            return InsuranceType.UNKNOWN

        industry_lower = industry.lower()

        # Property & Casualty patterns
        if any(pattern in industry_lower for pattern in [
            'property', 'casualty', 'p&c', 'p/c', 'property-casualty',
            'auto insur', 'home insur', 'workers comp'
        ]):
            return InsuranceType.PROPERTY_CASUALTY

        # Life insurance patterns
        if any(pattern in industry_lower for pattern in [
            'life insur', 'life & health', 'annuity', 'life insurance'
        ]):
            return InsuranceType.LIFE

        # Health insurance patterns
        if any(pattern in industry_lower for pattern in [
            'health insur', 'health care insur', 'managed care',
            'health maintenance', 'hmo', 'ppo'
        ]):
            return InsuranceType.HEALTH

        # Reinsurance patterns
        if any(pattern in industry_lower for pattern in [
            'reinsur', 're-insur', 'reinsurance'
        ]):
            return InsuranceType.REINSURANCE

        # Multi-line patterns
        if any(pattern in industry_lower for pattern in [
            'multi-line', 'multiline', 'diversified insur'
        ]):
            return InsuranceType.MULTI_LINE

        return InsuranceType.UNKNOWN

    def _value_bank(
        self,
        symbol: str,
        financials: Dict,
        current_price: float
    ) -> ValuationResult:
        """
        Value bank using ROE multiples method

        Banks are valued on ROE because:
        - Leverage is inherent to business model
        - ROE drives shareholder returns
        - P/BV ratio correlates with ROE
        """
        warnings = []

        try:
            # Extract key metrics
            stockholders_equity = financials.get('stockholders_equity', 0)
            net_income = financials.get('net_income', 0)
            shares_outstanding = financials.get('shares_outstanding', 0)

            if not all([stockholders_equity, net_income, shares_outstanding]):
                raise ValueError("Missing required metrics for bank valuation")

            # Calculate ROE
            roe = (net_income / stockholders_equity) * 100
            book_value_per_share = stockholders_equity / shares_outstanding

            # Target P/BV based on ROE (industry relationship)
            # High ROE banks (>15%) deserve P/BV of 1.5-2.0x
            # Medium ROE banks (10-15%) deserve P/BV of 1.0-1.5x
            # Low ROE banks (<10%) deserve P/BV of 0.8-1.0x
            if roe >= 15:
                target_pb = 1.75
                confidence = "high"
            elif roe >= 10:
                target_pb = 1.25
                confidence = "medium"
            else:
                target_pb = 0.9
                confidence = "low"
                warnings.append(f"Low ROE ({roe:.1f}%) suggests challenged bank")

            fair_value = book_value_per_share * target_pb
            upside = ((fair_value - current_price) / current_price) * 100

            self.logger.info(
                f"{symbol} - Bank valuation: ROE={roe:.1f}%, "
                f"BV/share=${book_value_per_share:.2f}, "
                f"Target P/BV={target_pb:.2f}x, Fair value=${fair_value:.2f}"
            )

            return ValuationResult(
                method="ROE Multiple (Bank)",
                fair_value=fair_value,
                current_price=current_price,
                upside_percent=upside,
                confidence=confidence,
                details={
                    'roe': roe,
                    'book_value_per_share': book_value_per_share,
                    'target_pb_ratio': target_pb,
                    'current_pb_ratio': current_price / book_value_per_share if book_value_per_share > 0 else 0,
                },
                warnings=warnings
            )

        except Exception as e:
            self.logger.warning(f"{symbol} - Bank valuation failed: {e}")
            raise

    def _value_reit(
        self,
        symbol: str,
        financials: Dict,
        current_price: float,
        company_name: Optional[str] = None,
        industry: Optional[str] = None
    ) -> ValuationResult:
        """
        Value REIT using property-type-specific FFO multiples with rate adjustment.

        REITs are valued on FFO because:
        - FFO better represents cash available for dividends
        - Depreciation is non-cash and often understates true economics
        - Dividend yield is primary investor consideration

        Property Type Matters:
        - Industrial/Data Centers/Cell Towers: Premium multiples (20-26x)
        - Residential: Varies by geography (16-20x)
        - Healthcare: Moderate multiples (14-16x)
        - Office: Discounted multiples (8-14x)
        - Regional Malls: Deep discount (6-10x)

        Rate Sensitivity:
        - Higher 10-year Treasury yields = lower multiples
        - Lower yields = higher multiples
        """
        from investigator.domain.services.valuation.reit_valuation import (
            value_reit,
            REITValuationResult,
        )

        try:
            # Use the new property-type-aware REIT valuation
            reit_result: REITValuationResult = value_reit(
                symbol=symbol,
                financials=financials,
                current_price=current_price,
                company_name=company_name,
                industry=industry,
                use_rate_adjustment=True
            )

            upside = ((reit_result.fair_value - current_price) / current_price) * 100

            # Determine confidence based on property type detection and data quality
            if reit_result.property_type_confidence == "high":
                confidence = "high" if not reit_result.warnings else "medium"
            elif reit_result.property_type_confidence == "medium":
                confidence = "medium"
            else:
                confidence = "low"

            self.logger.info(
                f"{symbol} - REIT valuation: property_type={reit_result.property_type.value}, "
                f"FFO/share=${reit_result.ffo_per_share:.2f}, "
                f"base_multiple={reit_result.base_ffo_multiple:.1f}x, "
                f"adjusted_multiple={reit_result.adjusted_ffo_multiple:.1f}x, "
                f"Fair value=${reit_result.fair_value:.2f}"
            )

            return ValuationResult(
                method=f"FFO Multiple (REIT - {reit_result.property_type.value.replace('_', ' ').title()})",
                fair_value=reit_result.fair_value,
                current_price=current_price,
                upside_percent=upside,
                confidence=confidence,
                details={
                    'ffo_per_share': reit_result.ffo_per_share,
                    'base_ffo_multiple': reit_result.base_ffo_multiple,
                    'adjusted_ffo_multiple': reit_result.adjusted_ffo_multiple,
                    'property_type': reit_result.property_type.value,
                    'property_type_confidence': reit_result.property_type_confidence,
                    'detection_method': reit_result.detection_method,
                    'current_10yr_yield': reit_result.current_10yr_yield,
                    'rate_adjustment': reit_result.rate_adjustment,
                    'current_ffo_yield': (reit_result.ffo_per_share / current_price * 100) if current_price > 0 else 0,
                },
                warnings=reit_result.warnings
            )

        except Exception as e:
            self.logger.warning(f"{symbol} - REIT valuation failed: {e}")
            raise

    def _is_biotech_industry(
        self,
        industry: Optional[str],
        sector: Optional[str],
        financials: Dict
    ) -> bool:
        """
        Determine if company should use biotech pre-revenue valuation.

        Uses biotech valuation if:
        1. Industry matches biotech patterns AND
        2. Company is pre-revenue (< $100M annual revenue)

        Args:
            industry: Industry classification
            sector: Sector classification
            financials: Financial metrics dictionary

        Returns:
            True if biotech pre-revenue valuation should be used
        """
        from investigator.domain.services.valuation.biotech_valuation import (
            is_pre_revenue_biotech,
        )

        if not industry:
            return False

        # Get revenue from financials
        revenue = financials.get('revenue', 0) or financials.get('total_revenue', 0)
        company_name = financials.get('company_name') or financials.get('entityName')

        is_biotech, reason = is_pre_revenue_biotech(
            industry=industry,
            company_name=company_name,
            sector=sector,
            revenue=revenue,
        )

        if is_biotech:
            self.logger.info(f"Biotech detected: {reason}")

        return is_biotech

    def _value_biotech(
        self,
        symbol: str,
        financials: Dict,
        current_price: float,
        company_name: Optional[str] = None,
        industry: Optional[str] = None,
        pipeline: Optional[list] = None
    ) -> ValuationResult:
        """
        Value pre-revenue biotech using pipeline probability-weighted valuation.

        P2-A Implementation:
        - Probability-weighted pipeline value (60% weight)
        - Cash runway analysis (25% weight)
        - Market comparable transactions (15% weight) - not yet implemented

        Pre-revenue biotech companies cannot be valued using:
        - P/S ratio (meaningless at $0 revenue)
        - P/E ratio (no earnings)
        - DCF (too speculative)

        Args:
            symbol: Stock ticker symbol
            financials: Dictionary of financial metrics
            current_price: Current stock price
            company_name: Company name (optional)
            industry: Industry classification (optional)
            pipeline: List of drug candidate dictionaries (optional)

        Returns:
            ValuationResult with fair value and details
        """
        from investigator.domain.services.valuation.biotech_valuation import (
            value_biotech,
            BiotechValuationResult,
        )

        try:
            result: BiotechValuationResult = value_biotech(
                symbol=symbol,
                financials=financials,
                current_price=current_price,
                pipeline=pipeline,
            )

            upside = ((result.fair_value_per_share - current_price) / current_price) * 100

            # Determine method string based on methodology
            if result.methodology == "pipeline_probability_weighted":
                method = "Pipeline PW (Biotech Pre-Revenue)"
            else:
                method = "Cash Value (Biotech Pre-Revenue)"

            # Map confidence
            confidence = result.confidence

            self.logger.info(
                f"{symbol} - Biotech valuation: "
                f"Pipeline=${result.pipeline_value/1e9:.2f}B, "
                f"Cash=${result.cash_value/1e6:.1f}M, "
                f"EV=${result.total_enterprise_value/1e9:.2f}B, "
                f"Fair value=${result.fair_value_per_share:.2f}, "
                f"Runway={result.cash_runway.months:.1f} months"
            )

            return ValuationResult(
                method=method,
                fair_value=result.fair_value_per_share,
                current_price=current_price,
                upside_percent=upside,
                confidence=confidence,
                details={
                    'pipeline_value': result.pipeline_value,
                    'cash_value': result.cash_value,
                    'total_enterprise_value': result.total_enterprise_value,
                    'cash_runway_months': result.cash_runway.months,
                    'cash_runway_risk': result.cash_runway.risk.value,
                    'cash_runway_description': result.cash_runway.risk_description,
                    'dilution_warning': result.cash_runway.dilution_warning,
                    'pipeline_drugs_count': len(result.pipeline_details.drug_values),
                    'probability_weighted_sales': result.pipeline_details.probability_weighted_sales,
                    'methodology': result.methodology,
                    'drug_details': result.pipeline_details.drug_values,
                },
                warnings=result.warnings
            )

        except Exception as e:
            self.logger.warning(f"{symbol} - Biotech valuation failed: {e}")
            raise

    def _is_defense_industry(
        self,
        industry: Optional[str],
        sector: Optional[str],
        symbol: str
    ) -> bool:
        """
        Determine if company should use defense contractor valuation.

        P2-B: Uses defense contractor valuation if:
        1. Industry matches defense patterns OR
        2. Symbol is a known defense contractor

        Args:
            industry: Industry classification
            sector: Sector classification
            symbol: Stock ticker symbol

        Returns:
            True if defense contractor valuation should be used
        """
        from investigator.domain.services.valuation.defense_valuation import (
            classify_defense_contractor,
        )

        classification = classify_defense_contractor(
            symbol=symbol,
            industry=industry,
            sector=sector,
        )

        if classification.is_defense_contractor:
            self.logger.info(
                f"{symbol} - Defense contractor detected via {classification.detection_method}: "
                f"{classification.contractor_type.value}"
            )

        return classification.is_defense_contractor

    def _is_reit(
        self,
        sector: Optional[str],
        industry: Optional[str]
    ) -> bool:
        """
        Determine if company should use REIT valuation.

        Uses REIT valuation if:
        1. Sector is "Real Estate" OR
        2. Industry contains "REIT" or "Real Estate Investment Trust"

        This catches variations like:
        - "REIT - Residential"
        - "Equity Real Estate Investment Trusts (REITs)"
        - "Mortgage REIT"
        - etc.

        Args:
            sector: Sector classification
            industry: Industry classification

        Returns:
            True if REIT valuation should be used
        """
        # Check if sector is Real Estate
        if sector == "Real Estate":
            self.logger.debug(f"REIT detected via sector: {sector}")
            return True

        # Check if industry contains REIT-related keywords
        if industry:
            industry_lower = industry.lower()
            reit_keywords = ['reit', 'real estate investment trust']
            if any(kw in industry_lower for kw in reit_keywords):
                self.logger.debug(f"REIT detected via industry keyword: {industry}")
                return True

        return False

    def _value_defense_contractor(
        self,
        symbol: str,
        financials: Dict,
        current_price: float,
        xbrl_data: Optional[Dict] = None,
        industry: Optional[str] = None,
    ) -> ValuationResult:
        """
        Value defense contractor with backlog-adjusted valuation.

        P2-B Implementation:
        Defense contractors are valued using:
        - EV/EBITDA (35% weight) - primary multiple for stable businesses
        - P/E (30% weight) - important for mature contractors
        - DCF (25% weight) - with conservative growth assumptions
        - Backlog premium (10% weight) - premium/discount based on backlog ratio

        Key adjustments:
        - Backlog premium: +10% for 3x+ backlog ratio, +5% for 2x+, -5% for <1x
        - Contract mix adjustment: -5% for cost-plus heavy, +5% for fixed-price heavy
        - Terminal growth: 2.5% (defense spending grows slowly)
        - Terminal margin: 10% (typical operating margin)

        Args:
            symbol: Stock ticker symbol
            financials: Dictionary of financial metrics
            current_price: Current stock price
            xbrl_data: Optional raw XBRL data for backlog extraction
            industry: Industry classification

        Returns:
            ValuationResult with fair value and details
        """
        from investigator.domain.services.valuation.defense_valuation import (
            value_defense_contractor,
            DefenseValuationResult,
            get_defense_tier_weights,
            classify_defense_contractor,
        )

        try:
            # First, we need a base fair value from standard models
            # For now, use a simple EV/EBITDA approach as the base
            base_fair_value = self._calculate_defense_base_value(
                symbol, financials, current_price
            )

            # Apply defense-specific adjustments
            result: DefenseValuationResult = value_defense_contractor(
                symbol=symbol,
                financials=financials,
                current_price=current_price,
                base_fair_value=base_fair_value,
                xbrl_data=xbrl_data,
                industry=industry,
            )

            upside = ((result.fair_value - current_price) / current_price) * 100

            # Build method string
            contractor_type = result.contractor_type.value.replace('_', ' ').title()
            method = f"EV/EBITDA+Backlog (Defense - {contractor_type})"

            # Build details
            details = {
                'base_fair_value': result.base_fair_value,
                'backlog_premium': result.backlog_premium,
                'contract_mix_adjustment': result.contract_mix_adjustment,
                'total_backlog': result.total_backlog,
                'backlog_ratio': result.backlog_ratio,
                'backlog_value_npv': result.backlog_value,
                'contractor_type': result.contractor_type.value,
                'tier_weights': get_defense_tier_weights(),
            }

            backlog_str = f"backlog_ratio={result.backlog_ratio:.2f}x" if result.backlog_ratio else "backlog_ratio=N/A"
            self.logger.info(
                f"{symbol} - Defense contractor valuation: "
                f"base=${result.base_fair_value:.2f}, "
                f"backlog_premium={result.backlog_premium:.2f}x, "
                f"{backlog_str}, "
                f"adjusted=${result.fair_value:.2f}"
            )

            return ValuationResult(
                method=method,
                fair_value=result.fair_value,
                current_price=current_price,
                upside_percent=upside,
                confidence=result.confidence,
                details=details,
                warnings=result.warnings
            )

        except Exception as e:
            self.logger.warning(f"{symbol} - Defense contractor valuation failed: {e}")
            raise

    def _calculate_defense_base_value(
        self,
        symbol: str,
        financials: Dict,
        current_price: float
    ) -> float:
        """
        Calculate base fair value for defense contractor using EV/EBITDA.

        This provides the base value before backlog adjustments are applied.

        Defense contractors typically trade at 10-14x EV/EBITDA based on:
        - Contract visibility
        - Margin stability
        - Government customer creditworthiness

        Args:
            symbol: Stock ticker symbol
            financials: Dictionary of financial metrics
            current_price: Current stock price

        Returns:
            Base fair value per share
        """
        # Extract key metrics
        ebitda = financials.get('ebitda', 0) or financials.get('operating_income', 0)
        total_debt = financials.get('total_debt', 0) or financials.get('long_term_debt', 0)
        cash = financials.get('cash_and_equivalents', 0) or financials.get('cash', 0)
        shares_outstanding = financials.get('shares_outstanding', 0)

        if not ebitda or not shares_outstanding:
            # Fallback to P/E if EBITDA not available
            net_income = financials.get('net_income', 0)
            if net_income and shares_outstanding:
                eps = net_income / shares_outstanding
                target_pe = 16.0  # Defense contractors typically trade at 14-18x P/E
                return eps * target_pe
            else:
                self.logger.warning(
                    f"{symbol} - Insufficient data for defense base valuation, "
                    "using current price as base"
                )
                return current_price

        # Use 12x EV/EBITDA as base multiple (middle of defense range)
        target_ev_ebitda = 12.0

        # Calculate enterprise value
        enterprise_value = ebitda * target_ev_ebitda

        # Convert to equity value
        equity_value = enterprise_value - total_debt + cash

        # Calculate per share value
        fair_value_per_share = equity_value / shares_outstanding

        self.logger.debug(
            f"{symbol} - Defense base value: EBITDA=${ebitda/1e9:.2f}B, "
            f"EV/EBITDA={target_ev_ebitda}x, EV=${enterprise_value/1e9:.2f}B, "
            f"Fair value=${fair_value_per_share:.2f}"
        )

        return fair_value_per_share

    def enhance_with_industry_metrics(
        self,
        result: ValuationResult,
        symbol: str,
        sector: Optional[str],
        industry: Optional[str],
        financials: Dict,
        xbrl_data: Optional[Dict] = None,
        use_cache: bool = True,
        cache_ttl_days: int = 7,
    ) -> ValuationResult:
        """
        Enhance a valuation result with industry-specific metrics and adjustments.

        This method uses the IndustryDatasetRegistry to:
        1. Extract industry-specific metrics (e.g., inventory days for semiconductors)
        2. Calculate valuation adjustments based on those metrics
        3. Apply adjustments to the fair value

        Now integrates with IndustryMetricsCache for:
        - Cache-first retrieval (avoids re-extracting recently analyzed stocks)
        - Automatic caching of extracted metrics

        Args:
            result: Base ValuationResult to enhance
            symbol: Stock symbol
            sector: Company sector
            industry: Company industry
            financials: Financial metrics dictionary
            xbrl_data: Optional raw XBRL data for detailed extraction
            use_cache: Whether to use cached metrics (default: True)
            cache_ttl_days: Cache time-to-live in days (default: 7)

        Returns:
            Enhanced ValuationResult with industry_metrics, industry_adjustments,
            and adjusted_fair_value fields populated
        """
        try:
            from investigator.domain.services.industry_datasets import (
                extract_industry_metrics,
                get_valuation_adjustments,
                apply_adjustments_to_fair_value,
                get_industry_summary,
            )

            # Try to get cached metrics first
            cached_entry = None
            cache = None
            if use_cache:
                cached_entry, cache = self._get_cached_industry_metrics(symbol)

            if cached_entry:
                # Use cached metrics
                self.logger.info(
                    f"{symbol} - Using cached industry metrics "
                    f"(quality={cached_entry.quality}, coverage={cached_entry.coverage:.0%})"
                )

                result.industry_metrics = {
                    "industry": cached_entry.industry,
                    "quality": cached_entry.quality,
                    "coverage": cached_entry.coverage,
                    "metrics": cached_entry.metrics,
                    "warnings": cached_entry.warnings,
                    "metadata": {**cached_entry.metadata, "from_cache": True},
                }

                if cached_entry.adjustments:
                    result.industry_adjustments = cached_entry.adjustments

                    # Recalculate adjusted fair value with current base value
                    total_factor = 1.0
                    for adj in cached_entry.adjustments:
                        total_factor *= adj.get("factor", 1.0)

                    if total_factor != 1.0:
                        result.adjusted_fair_value = result.fair_value * total_factor
                        if result.current_price > 0:
                            result.upside_percent = (
                                (result.adjusted_fair_value - result.current_price)
                                / result.current_price
                            ) * 100

                return result

            # No cache hit - extract metrics fresh
            metrics = extract_industry_metrics(
                symbol=symbol,
                xbrl_data=xbrl_data,
                financials=financials,
                industry=industry,
                sector=sector,
            )

            if not metrics:
                self.logger.debug(
                    f"{symbol} - No industry-specific metrics available"
                )
                return result

            # Store metrics in result
            result.industry_metrics = {
                "industry": metrics.industry,
                "quality": metrics.quality.value,
                "coverage": metrics.coverage,
                "metrics": metrics.metrics,
                "warnings": metrics.warnings,
                "metadata": metrics.metadata,
            }

            # Get valuation adjustments
            adjustments, _ = get_valuation_adjustments(
                symbol=symbol,
                xbrl_data=xbrl_data,
                financials=financials,
                industry=industry,
                sector=sector,
            )

            adjustments_list = []
            if adjustments:
                # Store adjustments in result
                adjustments_list = [
                    {
                        "type": adj.adjustment_type,
                        "factor": adj.factor,
                        "reason": adj.reason,
                        "confidence": adj.confidence,
                        "affects_models": adj.affects_models,
                    }
                    for adj in adjustments
                ]
                result.industry_adjustments = adjustments_list

                # Apply adjustments to fair value
                adjusted_value, reasons = apply_adjustments_to_fair_value(
                    base_fair_value=result.fair_value,
                    adjustments=adjustments,
                )

                if adjusted_value != result.fair_value:
                    result.adjusted_fair_value = adjusted_value

                    # Update upside based on adjusted value
                    if result.current_price > 0:
                        result.upside_percent = (
                            (adjusted_value - result.current_price)
                            / result.current_price
                        ) * 100

                    self.logger.info(
                        f"{symbol} - Applied {len(adjustments)} industry adjustments: "
                        f"${result.fair_value:.2f} -> ${adjusted_value:.2f}"
                    )

            # Cache the extracted metrics for future use
            if use_cache:
                self._cache_industry_metrics(
                    symbol=symbol,
                    industry=industry or metrics.industry,
                    sector=sector,
                    metrics=metrics,
                    adjustments=adjustments_list,
                    cache=cache,
                    ttl_days=cache_ttl_days,
                )

            return result

        except ImportError:
            self.logger.debug(
                f"{symbol} - IndustryDatasetRegistry not available, skipping enhancement"
            )
            return result
        except Exception as e:
            self.logger.warning(
                f"{symbol} - Failed to enhance with industry metrics: {e}"
            )
            return result

    def _get_cached_industry_metrics(self, symbol: str):
        """
        Try to get cached industry metrics for a symbol.

        Returns:
            Tuple of (cached_entry, cache_instance) or (None, None)
        """
        try:
            from investigator.infrastructure.cache.industry_metrics_cache import (
                IndustryMetricsCache,
            )

            cache = IndustryMetricsCache()
            entry = cache.get(symbol)

            if entry:
                # Check if cache is expired (based on cached_at and ttl)
                from datetime import datetime, timezone

                cached_at = datetime.fromisoformat(entry.cached_at)
                now = datetime.now(timezone.utc)
                age_days = (now - cached_at).days

                # Default TTL is 7 days, but check expires_at if set
                if entry.expires_at:
                    expires_at = datetime.fromisoformat(entry.expires_at)
                    if now > expires_at:
                        self.logger.debug(
                            f"{symbol} - Cache expired (expires_at={entry.expires_at})"
                        )
                        return None, cache
                elif age_days > 7:
                    self.logger.debug(
                        f"{symbol} - Cache too old ({age_days} days)"
                    )
                    return None, cache

                return entry, cache

            return None, cache

        except ImportError:
            self.logger.debug("IndustryMetricsCache not available")
            return None, None
        except Exception as e:
            self.logger.debug(f"Cache lookup failed: {e}")
            return None, None

    def _cache_industry_metrics(
        self,
        symbol: str,
        industry: str,
        sector: Optional[str],
        metrics,
        adjustments: List[Dict],
        cache=None,
        ttl_days: int = 7,
    ):
        """Cache extracted industry metrics for future use."""
        try:
            if cache is None:
                from investigator.infrastructure.cache.industry_metrics_cache import (
                    IndustryMetricsCache,
                )
                cache = IndustryMetricsCache()

            # Get dataset info from metrics
            dataset_name = metrics.metadata.get("dataset_name", "unknown")
            dataset_version = metrics.metadata.get("dataset_version", "1.0.0")

            success = cache.set(
                symbol=symbol,
                industry=industry,
                sector=sector,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                quality=metrics.quality.value,
                coverage=metrics.coverage,
                metrics=metrics.metrics,
                adjustments=adjustments,
                tier_weights=metrics.metadata.get("tier_weights"),
                warnings=metrics.warnings,
                metadata=metrics.metadata,
                ttl_days=ttl_days,
            )

            if success:
                self.logger.info(
                    f"{symbol} - Cached industry metrics "
                    f"(quality={metrics.quality.value}, ttl={ttl_days}d)"
                )

        except Exception as e:
            self.logger.debug(f"Failed to cache industry metrics: {e}")

        # Also update industry benchmarks if we have enough symbols
        self._maybe_update_industry_benchmarks(cache, industry)

    def _maybe_update_industry_benchmarks(self, cache, industry: str, min_symbols: int = 3):
        """
        Update industry-level benchmarks if enough symbols are cached.

        This allows industry benchmarks to be shared across all stocks in an industry,
        providing peer statistics and comparison data.
        """
        try:
            if cache is None:
                return

            # Check if we have enough symbols for this industry
            industry_symbols = cache.get_by_industry(industry)
            if len(industry_symbols) >= min_symbols:
                # Check if benchmarks exist and are recent
                existing = cache.get_industry_benchmarks(industry)
                if existing:
                    from datetime import datetime, timezone
                    cached_at = datetime.fromisoformat(existing.cached_at)
                    age_days = (datetime.now(timezone.utc) - cached_at).days
                    # Only recompute if old or symbol count changed significantly
                    if age_days < 1 and abs(existing.symbol_count - len(industry_symbols)) < 2:
                        return

                # Compute and cache industry benchmarks
                benchmarks = cache.compute_and_cache_industry_benchmarks(industry)
                if benchmarks:
                    self.logger.info(
                        f"Updated industry benchmarks for {industry} "
                        f"({benchmarks.symbol_count} symbols)"
                    )

        except Exception as e:
            self.logger.debug(f"Failed to update industry benchmarks: {e}")

    def get_industry_benchmarks(
        self,
        industry: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get shared industry-level benchmarks (peer statistics, thresholds).

        This returns benchmarks computed from all cached symbols in the industry,
        allowing comparison of a stock against its industry peers.

        Args:
            industry: Industry name

        Returns:
            Dictionary with peer statistics, benchmarks, and cycle indicators
        """
        try:
            from investigator.infrastructure.cache.industry_metrics_cache import (
                IndustryMetricsCache,
            )

            cache = IndustryMetricsCache()
            benchmarks = cache.get_industry_benchmarks(industry)

            if benchmarks:
                return {
                    "industry": benchmarks.industry,
                    "symbol_count": benchmarks.symbol_count,
                    "symbols_included": benchmarks.symbols_included,
                    "peer_statistics": benchmarks.peer_statistics,
                    "benchmarks": benchmarks.benchmarks,
                    "tier_weights": benchmarks.tier_weights,
                    "cycle_indicators": benchmarks.cycle_indicators,
                    "cached_at": benchmarks.cached_at,
                }

            return None

        except Exception as e:
            self.logger.debug(f"Failed to get industry benchmarks: {e}")
            return None

    def get_industry_summary(
        self,
        symbol: str,
        sector: Optional[str],
        industry: Optional[str],
        financials: Dict,
        xbrl_data: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a comprehensive industry summary for a stock.

        This is a convenience method that returns industry analysis
        without requiring a valuation to be performed first.

        Args:
            symbol: Stock symbol
            sector: Company sector
            industry: Company industry
            financials: Financial metrics dictionary
            xbrl_data: Optional raw XBRL data

        Returns:
            Dictionary with industry analysis, or None if no dataset found
        """
        try:
            from investigator.domain.services.industry_datasets import (
                get_industry_summary,
            )

            return get_industry_summary(
                symbol=symbol,
                xbrl_data=xbrl_data,
                financials=financials,
                industry=industry,
                sector=sector,
            )

        except ImportError:
            self.logger.debug(
                f"{symbol} - IndustryDatasetRegistry not available"
            )
            return None
        except Exception as e:
            self.logger.warning(
                f"{symbol} - Failed to get industry summary: {e}"
            )
            return None

