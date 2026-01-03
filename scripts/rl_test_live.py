#!/usr/bin/env python3
"""
RL Policy Live Test Script

Test the trained RL policy on live valuation for a given symbol.
Compares RL-based weights against rule-based weights.

Usage:
    PYTHONPATH=./src:. python scripts/rl_test_live.py AAPL
    PYTHONPATH=./src:. python scripts/rl_test_live.py NVDA TSLA MSFT

Environment:
    PYTHONPATH=./src:. python scripts/rl_test_live.py
"""

import argparse
import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from investigator.config import get_config
from investigator.domain.services.rl.policy.contextual_bandit import ContextualBanditPolicy
from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer
from investigator.domain.services.rl.models import ValuationContext, GrowthStage, CompanySize
from investigator.domain.services.dynamic_model_weighting import DynamicModelWeightingService
from investigator.domain.services.rl.price_history import get_price_history_service
from investigator.infrastructure.database.market_data import get_market_data_fetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path("data/rl_models")
ACTIVE_POLICY_PATH = MODEL_DIR / "active_policy.pkl"
ACTIVE_NORMALIZER_PATH = MODEL_DIR / "active_normalizer.pkl"


def load_active_policy():
    """Load the active deployed policy."""
    if not ACTIVE_POLICY_PATH.exists():
        raise FileNotFoundError(f"No active policy found at {ACTIVE_POLICY_PATH}")
    if not ACTIVE_NORMALIZER_PATH.exists():
        raise FileNotFoundError(f"No active normalizer found at {ACTIVE_NORMALIZER_PATH}")

    normalizer = FeatureNormalizer()
    normalizer.load(str(ACTIVE_NORMALIZER_PATH))

    policy = ContextualBanditPolicy(normalizer=normalizer)
    policy.load(str(ACTIVE_POLICY_PATH))

    return policy, normalizer


def get_stock_context(symbol: str, config) -> dict:
    """Get stock context from market data."""
    fetcher = get_market_data_fetcher(config)
    info = fetcher.get_stock_info(symbol)

    if not info:
        raise ValueError(f"Could not fetch info for {symbol}")

    return info


def infer_growth_stage(info: dict) -> GrowthStage:
    """Infer growth stage from stock info."""
    # Simple heuristics - in production this would be more sophisticated
    sector = (info.get("sector") or "").lower()

    if "technology" in sector or "communication" in sector:
        return GrowthStage.HIGH_GROWTH
    elif "utilities" in sector or "real estate" in sector:
        return GrowthStage.DIVIDEND_PAYING
    elif "financials" in sector:
        return GrowthStage.MATURE
    elif "energy" in sector or "materials" in sector:
        return GrowthStage.TRANSITIONING
    else:
        return GrowthStage.MATURE


def infer_company_size(market_cap: float) -> CompanySize:
    """Infer company size from market cap."""
    if market_cap is None:
        return CompanySize.MID_CAP

    if market_cap >= 200e9:
        return CompanySize.MEGA_CAP
    elif market_cap >= 10e9:
        return CompanySize.LARGE_CAP
    elif market_cap >= 2e9:
        return CompanySize.MID_CAP
    elif market_cap >= 300e6:
        return CompanySize.SMALL_CAP
    else:
        return CompanySize.MICRO_CAP


def build_valuation_context(symbol: str, info: dict) -> ValuationContext:
    """Build ValuationContext from stock info."""
    return ValuationContext(
        symbol=symbol,
        analysis_date=date.today(),
        sector=info.get("sector") or "Unknown",
        industry=info.get("industry") or "Unknown",
        growth_stage=infer_growth_stage(info),
        company_size=infer_company_size(info.get("market_cap")),
        profitability_score=0.7,  # Default - would come from fundamentals
        pe_level=0.5,  # Default
        revenue_growth=0.1,  # Default
    )


def get_rule_based_weights(symbol: str, sector: str, config) -> dict:
    """Get rule-based weights from DynamicModelWeightingService."""
    weighting = DynamicModelWeightingService(config)

    # Build a minimal company profile for weight calculation
    company_profile = {
        "symbol": symbol,
        "sector": sector,
        "industry": "Unknown",
        "current_price": 100,  # Placeholder
        "is_profitable": True,
        "has_positive_fcf": True,
        "revenue_growth_3y": 0.1,
        "pe_ratio": 20,
    }

    return weighting.get_model_weights(company_profile)


def test_symbol(symbol: str, policy, config):
    """Test RL policy on a single symbol."""
    print(f"\n{'='*60}")
    print(f"TESTING: {symbol}")
    print("=" * 60)

    # Get stock info
    try:
        info = get_stock_context(symbol, config)
    except Exception as e:
        print(f"ERROR fetching info: {e}")
        return

    print(f"\nStock Info:")
    print(f"  Sector: {info.get('sector', 'N/A')}")
    print(f"  Industry: {info.get('industry', 'N/A')}")
    print(f"  Current Price: ${info.get('current_price', 'N/A')}")
    print(f"  Market Cap: ${info.get('market_cap', 0)/1e9:.1f}B" if info.get("market_cap") else "  Market Cap: N/A")

    # Build context
    context = build_valuation_context(symbol, info)
    print(f"\nValuation Context:")
    print(f"  Growth Stage: {context.growth_stage.value}")
    print(f"  Company Size: {context.company_size.value}")

    # Get RL-based weights
    print("\nRL Policy Weights:")
    try:
        rl_weights = policy.predict(context)
        for model, weight in sorted(rl_weights.items(), key=lambda x: -x[1]):
            if weight > 0:
                print(f"  {model}: {weight:.0f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        rl_weights = {}

    # Get rule-based weights
    print("\nRule-Based Weights:")
    try:
        rule_weights = get_rule_based_weights(symbol, info.get("sector", "Unknown"), config)
        for model, weight in sorted(rule_weights.items(), key=lambda x: -x[1]):
            if weight > 0:
                print(f"  {model}: {weight:.0f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        rule_weights = {}

    # Compare
    if rl_weights and rule_weights:
        print("\nWeight Differences (RL - Rule):")
        all_models = set(rl_weights.keys()) | set(rule_weights.keys())
        for model in sorted(all_models):
            rl = rl_weights.get(model, 0)
            rule = rule_weights.get(model, 0)
            diff = rl - rule
            if abs(diff) > 1:
                print(f"  {model}: {diff:+.0f}%")


def main():
    parser = argparse.ArgumentParser(description="Test RL policy on live symbols")
    parser.add_argument("symbols", nargs="+", help="Stock symbols to test")
    args = parser.parse_args()

    print("=" * 60)
    print("RL POLICY LIVE TEST")
    print("=" * 60)

    # Load config
    config = get_config()

    # Load policy
    print("\nLoading active RL policy...")
    try:
        policy, normalizer = load_active_policy()
        print(f"  Loaded: {policy.name} v{policy.version}")
        print(f"  Updates: {policy._update_count}")
        print(f"  Ready: {policy.is_ready()}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nHint: Run 'PYTHONPATH=./src:. python scripts/rl_deploy.py' first")
        sys.exit(1)

    # Test each symbol
    for symbol in args.symbols:
        test_symbol(symbol.upper(), policy, config)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
