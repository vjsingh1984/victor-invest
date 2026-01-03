"""
Sector-Aware Valuation Router

Routes to appropriate valuation methods based on company sector and industry:
- Technology: DCF with high growth rates
- Insurance: Price-to-Book (P/BV) and Dividend Discount Model (DDM)
- Banks: Return on Equity (ROE) multiples
- REITs: Funds from Operations (FFO) multiples
- Default: Standard DCF

Author: Claude Code
Date: 2025-11-10
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

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


class SectorValuationRouter:
    """Routes to appropriate valuation method based on sector/industry"""

    # Sector/industry routing map
    VALUATION_METHODS = {
        # Insurance companies - use P/BV and DDM
        ("Financials", "Insurance"): "insurance",
        # Banks - use ROE multiples
        ("Financials", "Banks"): "bank",
        # REITs - use FFO multiples
        ("Real Estate", "REITs"): "reit",
        ("Real Estate", None): "reit",  # All Real Estate defaults to REIT
        # Technology - use DCF with high growth
        ("Technology", None): "dcf_growth",
        # Default for other sectors
        ("default", None): "dcf_standard",
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
    ) -> ValuationResult:
        """
        Route to appropriate valuation method based on sector/industry

        Args:
            symbol: Stock symbol
            sector: Company sector (e.g., "Financials", "Technology")
            industry: Company industry (e.g., "Insurance", "Banks")
            financials: Dictionary of financial metrics
            current_price: Current stock price

        Returns:
            ValuationResult with fair value and details
        """
        # Flexible matching for insurance companies
        # Matches: "Insurance", "Property-Casualty Insurers", "Life Insurance", etc.
        if sector == "Financials" and industry and "insur" in industry.lower():
            valuation_type = "insurance"
        # Flexible matching for banks
        # Matches: "Banks", "Commercial Banks", "Regional Banks", etc.
        elif sector == "Financials" and industry and "bank" in industry.lower():
            valuation_type = "bank"
        # Use exact dictionary matching for other sectors
        else:
            method_key = (sector, industry)
            valuation_type = self.VALUATION_METHODS.get(
                method_key, self.VALUATION_METHODS.get((sector, None), self.VALUATION_METHODS[("default", None)])
            )

        self.logger.info(f"{symbol} - Routing to {valuation_type} valuation " f"(sector={sector}, industry={industry})")

        # Route to appropriate method
        if valuation_type == "insurance":
            return self._value_insurance(symbol, financials, current_price, database_url)
        elif valuation_type == "bank":
            return self._value_bank(symbol, financials, current_price)
        elif valuation_type == "reit":
            return self._value_reit(symbol, financials, current_price)
        else:
            # For non-special sectors (including dcf_growth, dcf_standard), return None
            # to signal that the caller should use standard DCF valuation
            return None

    def _value_insurance(
        self, symbol: str, financials: Dict, current_price: float, database_url: Optional[str] = None
    ) -> ValuationResult:
        """
        Value insurance company using Price-to-Book (P/BV) method

        Insurance companies are valued primarily on book value because:
        - Balance sheet quality is critical
        - Float management is key asset
        - ROE stability matters more than growth
        """
        from utils.insurance_valuation import value_insurance_company

        try:
            result = value_insurance_company(symbol, financials, current_price, database_url)

            upside = ((result["fair_value"] - current_price) / current_price) * 100

            return ValuationResult(
                method="P/BV (Insurance)",
                fair_value=result["fair_value"],
                current_price=current_price,
                upside_percent=upside,
                confidence=result.get("confidence", "medium"),
                details={
                    "book_value_per_share": result.get("book_value_per_share"),
                    "target_pb_ratio": result.get("target_pb_ratio"),
                    "current_pb_ratio": result.get("current_pb_ratio"),
                    "roe": result.get("roe"),
                    "combined_ratio": result.get("combined_ratio"),
                },
                warnings=result.get("warnings", []),
            )
        except Exception as e:
            self.logger.warning(f"{symbol} - Insurance valuation failed: {e}")
            raise

    def _value_bank(self, symbol: str, financials: Dict, current_price: float) -> ValuationResult:
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
            stockholders_equity = financials.get("stockholders_equity", 0)
            net_income = financials.get("net_income", 0)
            shares_outstanding = financials.get("shares_outstanding", 0)

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
                    "roe": roe,
                    "book_value_per_share": book_value_per_share,
                    "target_pb_ratio": target_pb,
                    "current_pb_ratio": current_price / book_value_per_share if book_value_per_share > 0 else 0,
                },
                warnings=warnings,
            )

        except Exception as e:
            self.logger.warning(f"{symbol} - Bank valuation failed: {e}")
            raise

    def _value_reit(self, symbol: str, financials: Dict, current_price: float) -> ValuationResult:
        """
        Value REIT using FFO (Funds from Operations) multiples

        REITs are valued on FFO because:
        - FFO better represents cash available for dividends
        - Depreciation is non-cash and often understates true economics
        - Dividend yield is primary investor consideration
        """
        warnings = []

        try:
            # FFO = Net Income + Depreciation - Gains on Property Sales
            net_income = financials.get("net_income", 0)
            depreciation = financials.get("depreciation_amortization", 0)
            shares_outstanding = financials.get("shares_outstanding", 0)

            if not all([net_income, shares_outstanding]):
                raise ValueError("Missing required metrics for REIT valuation")

            # Estimate FFO (simplified - we don't have property sale gains)
            ffo = net_income + depreciation
            ffo_per_share = ffo / shares_outstanding

            # Target FFO multiple based on REIT sector averages (12-18x)
            # Use middle of range: 15x
            target_ffo_multiple = 15.0

            fair_value = ffo_per_share * target_ffo_multiple
            upside = ((fair_value - current_price) / current_price) * 100

            # Check if depreciation data is available
            if not depreciation:
                warnings.append("Depreciation not available, FFO may be understated")
                confidence = "low"
            else:
                confidence = "medium"

            self.logger.info(
                f"{symbol} - REIT valuation: FFO/share=${ffo_per_share:.2f}, "
                f"Target multiple={target_ffo_multiple:.1f}x, Fair value=${fair_value:.2f}"
            )

            return ValuationResult(
                method="FFO Multiple (REIT)",
                fair_value=fair_value,
                current_price=current_price,
                upside_percent=upside,
                confidence=confidence,
                details={
                    "ffo_per_share": ffo_per_share,
                    "ffo_multiple": target_ffo_multiple,
                    "current_ffo_yield": (ffo_per_share / current_price * 100) if current_price > 0 else 0,
                },
                warnings=warnings,
            )

        except Exception as e:
            self.logger.warning(f"{symbol} - REIT valuation failed: {e}")
            raise
