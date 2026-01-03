#!/usr/bin/env python3
"""
Test Russell 1000 classification integration.

This script tests that the IndustryClassifier correctly loads and uses
the Russell 1000 classifications.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.industry_classifier import IndustryClassifier


def test_russell1000_integration():
    """Test Russell 1000 classification integration."""
    print("=" * 80)
    print("Testing Russell 1000 Classification Integration")
    print("=" * 80)

    # Initialize classifier
    classifier = IndustryClassifier()

    # Check that Russell 1000 overrides were loaded
    print(f"\n📊 Russell 1000 overrides loaded: {len(classifier.russell1000_overrides)} tickers")

    if not classifier.russell1000_overrides:
        print("❌ FAILED: No Russell 1000 overrides loaded!")
        return False

    # Test sample tickers from different sectors
    test_tickers = [
        ("AAPL", "Technology", "Consumer Electronics"),
        ("NVDA", "Technology", "Semiconductors"),
        ("JPM", "Finance", "Banks"),  # Also in SYMBOL_OVERRIDES, should use that first
        ("META", "Technology", "Computer Software: Prepackaged Software"),
        ("GOOGL", "Technology", "Computer Software: Programming, Data Processing"),
        ("WMT", "Consumer Staples", "Food Chains"),
        ("CVX", "Energy", "Integrated oil Companies"),
        ("LLY", "Health Care", "Other Pharmaceuticals"),
    ]

    print("\n🔍 Testing sample classifications:")
    print("-" * 80)

    all_passed = True
    for ticker, expected_sector, expected_industry in test_tickers:
        sector, industry = classifier.classify(ticker)

        if sector and industry:
            status = "✅" if sector == expected_sector else "⚠️"
            print(f"{status} {ticker:6s} → {sector:20s} / {industry}")

            if sector != expected_sector:
                print(f"    Expected sector: {expected_sector}")
                all_passed = False
        else:
            print(f"❌ {ticker:6s} → NO CLASSIFICATION")
            all_passed = False

    # Test a ticker NOT in Russell 1000
    print("\n🔍 Testing non-Russell 1000 ticker:")
    print("-" * 80)
    sector, industry = classifier.classify("FAKE")
    if sector:
        print(f"⚠️  FAKE → {sector} / {industry} (should be None)")
    else:
        print(f"✅ FAKE → No classification (expected)")

    # Summary
    print("\n" + "=" * 80)
    if all_passed and len(classifier.russell1000_overrides) >= 900:
        print("✅ ALL TESTS PASSED")
        print(f"   - {len(classifier.russell1000_overrides)} Russell 1000 tickers loaded")
        print("   - Sample classifications working correctly")
    else:
        print("❌ SOME TESTS FAILED")
        if len(classifier.russell1000_overrides) < 900:
            print(f"   - Only {len(classifier.russell1000_overrides)} tickers loaded (expected 924)")
        if not all_passed:
            print("   - Some classifications did not match expected values")

    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = test_russell1000_integration()
    sys.exit(0 if success else 1)
