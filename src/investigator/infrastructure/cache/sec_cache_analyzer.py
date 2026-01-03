#!/usr/bin/env python3
"""
Analyze cache hit/miss patterns for SEC data types
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from investigator.config import get_config
from investigator.infrastructure.cache.cache_manager import get_cache_manager
from investigator.infrastructure.cache.cache_types import CacheType
from utils.cache_stats import CacheStatsMonitor


def analyze_sec_cache_patterns():
    """Analyze cache patterns for SEC data types"""

    cache_manager = get_cache_manager()
    stats_monitor = CacheStatsMonitor()
    config = get_config()

    print("\n" + "=" * 80)
    print("🔍 SEC CACHE ANALYSIS REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    # CIK mappings for testing
    cik_map = {
        "AAPL": "0000320193",
        "MSFT": "0001564590",
        "GOOGL": "0001652044",
        "AMZN": "0001018724",
        "TSLA": "0001318605",
    }

    # Track results
    results = {
        "submission_data": {"hits": 0, "misses": 0, "errors": 0},
        "company_facts": {"hits": 0, "misses": 0, "errors": 0},
        "quarterly_metrics": {"hits": 0, "misses": 0, "errors": 0},
        "sec_response": {"hits": 0, "misses": 0, "errors": 0},
    }

    # 1. Test SEC Submissions Cache
    print("\n📄 SEC SUBMISSIONS CACHE")
    print("-" * 60)

    for symbol in test_symbols:
        cik = cik_map.get(symbol, "unknown")

        # Test different key formats
        submission_keys = [
            (symbol, "recent_10"),  # Legacy format
            {"symbol": symbol, "cik": cik},  # New format
            {"symbol": symbol},  # Symbol only
        ]

        for key in submission_keys:
            try:
                result = cache_manager.get(CacheType.SUBMISSION_DATA, key)
                if result:
                    results["submission_data"]["hits"] += 1
                    print(f"✅ HIT  - {symbol} with key {key}")
                    # Check data freshness
                    if "metadata" in result:
                        cached_at = result["metadata"].get("cached_at", "unknown")
                        print(f"         Cached at: {cached_at}")
                else:
                    results["submission_data"]["misses"] += 1
                    print(f"❌ MISS - {symbol} with key {key}")
            except Exception as e:
                results["submission_data"]["errors"] += 1
                print(f"💥 ERROR - {symbol}: {str(e)[:50]}")

    # 2. Test Company Facts Cache
    print("\n🏢 COMPANY FACTS CACHE")
    print("-" * 60)

    for symbol in test_symbols:
        cik = cik_map.get(symbol, "unknown")

        # Test different key formats
        facts_keys = [
            {"symbol": symbol},  # Symbol only
            {"symbol": symbol, "cik": cik},  # With CIK
        ]

        for key in facts_keys:
            try:
                result = cache_manager.get(CacheType.COMPANY_FACTS, key)
                if result:
                    results["company_facts"]["hits"] += 1
                    print(f"✅ HIT  - {symbol} with key {key}")
                    if isinstance(result, dict):
                        facts = result.get("companyfacts", result)
                        if "facts" in facts:
                            num_facts = len(facts.get("facts", {}))
                            print(f"         Contains {num_facts} fact categories")
                else:
                    results["company_facts"]["misses"] += 1
                    print(f"❌ MISS - {symbol} with key {key}")
            except Exception as e:
                results["company_facts"]["errors"] += 1
                print(f"💥 ERROR - {symbol}: {str(e)[:50]}")

    # 3. Test Quarterly Metrics Cache
    print("\n📊 QUARTERLY METRICS CACHE")
    print("-" * 60)

    quarters = ["2024Q1", "2024Q2", "2024Q3", "2023Q4"]

    for symbol in test_symbols:
        for quarter in quarters:
            # Test different key formats
            metrics_keys = [
                (symbol, quarter),  # Tuple format
                {"symbol": symbol, "period": quarter, "form_type": "10-Q"},  # Dict with form
                {"symbol": symbol, "fiscal_year": quarter[:4], "fiscal_period": quarter[4:]},  # Split year/period
            ]

            for key in metrics_keys:
                try:
                    result = cache_manager.get(CacheType.QUARTERLY_METRICS, key)
                    if result:
                        results["quarterly_metrics"]["hits"] += 1
                        print(f"✅ HIT  - {symbol} {quarter} with key type {type(key).__name__}")
                        break  # Don't test other formats if we got a hit
                    else:
                        results["quarterly_metrics"]["misses"] += 1
                except Exception as e:
                    results["quarterly_metrics"]["errors"] += 1
                    print(f"💥 ERROR - {symbol} {quarter}: {str(e)[:50]}")

    # 4. Test SEC Response Cache
    print("\n📑 SEC RESPONSE CACHE")
    print("-" * 60)

    for symbol in test_symbols:
        # Test quarterly summaries
        for year in ["2024", "2023"]:
            for period in ["Q1", "Q2", "Q3", "Q4"]:
                key = {
                    "symbol": symbol,
                    "fiscal_year": year,
                    "fiscal_period": period,
                    "form_type": "10-Q",
                    "category": "quarterly_summary",
                }

                try:
                    result = cache_manager.get(CacheType.SEC_RESPONSE, key)
                    if result:
                        results["sec_response"]["hits"] += 1
                        # Don't print each hit to avoid clutter
                    else:
                        results["sec_response"]["misses"] += 1
                except Exception as e:
                    results["sec_response"]["errors"] += 1

    # Print Summary Statistics
    print("\n" + "=" * 80)
    print("📊 CACHE PERFORMANCE SUMMARY")
    print("=" * 80)

    total_hits = total_misses = total_errors = 0

    for cache_type, stats in results.items():
        total_ops = stats["hits"] + stats["misses"] + stats["errors"]
        if total_ops > 0:
            hit_ratio = (
                (stats["hits"] / (stats["hits"] + stats["misses"])) * 100
                if (stats["hits"] + stats["misses"]) > 0
                else 0
            )

            print(f"\n{cache_type.upper()}:")
            print(f"  Total Operations: {total_ops}")
            print(f"  Hits: {stats['hits']} ({stats['hits']/total_ops*100:.1f}%)")
            print(f"  Misses: {stats['misses']} ({stats['misses']/total_ops*100:.1f}%)")
            print(f"  Errors: {stats['errors']} ({stats['errors']/total_ops*100:.1f}%)")
            print(f"  Hit Ratio: {hit_ratio:.1f}%")

            total_hits += stats["hits"]
            total_misses += stats["misses"]
            total_errors += stats["errors"]

    # Overall statistics
    print("\n" + "-" * 60)
    print("OVERALL SEC CACHE STATISTICS:")
    total_all = total_hits + total_misses + total_errors
    if total_all > 0:
        overall_hit_ratio = (total_hits / (total_hits + total_misses)) * 100 if (total_hits + total_misses) > 0 else 0
        print(f"  Total Operations: {total_all}")
        print(f"  Total Hits: {total_hits} ({total_hits/total_all*100:.1f}%)")
        print(f"  Total Misses: {total_misses} ({total_misses/total_all*100:.1f}%)")
        print(f"  Total Errors: {total_errors} ({total_errors/total_all*100:.1f}%)")
        print(f"  Overall Hit Ratio: {overall_hit_ratio:.1f}%")

    # Get cache manager statistics
    print("\n" + "=" * 80)
    print("🔧 CACHE MANAGER STATISTICS")
    print("=" * 80)

    perf_stats = cache_manager.get_performance_stats()

    # Print stats for SEC-related cache types
    sec_types = ["submission_data", "company_facts", "quarterly_metrics", "sec_response"]

    for cache_type in sec_types:
        if cache_type in perf_stats:
            stats = perf_stats[cache_type]
            ops = stats["operations"]
            perf = stats["performance"]

            if ops["total_reads"] > 0:
                print(f"\n{cache_type.upper()} (from cache manager):")
                print(f"  Hit Ratio: {perf['hit_ratio_pct']:.1f}%")
                print(f"  Average Time: {perf['avg_time_ms']:.1f}ms")
                print(f"  Total Time: {perf['total_time_ms']:.1f}ms")

                # Handler breakdown
                if stats["handlers"]:
                    print("  Handlers:")
                    for handler, h_stats in stats["handlers"].items():
                        if h_stats["hits"] + h_stats["misses"] > 0:
                            h_ratio = (h_stats["hits"] / (h_stats["hits"] + h_stats["misses"])) * 100
                            print(f"    {handler}: {h_ratio:.1f}% hit ratio, {h_stats['total_time_ms']:.1f}ms total")

    # Check cache directories
    print("\n" + "=" * 80)
    print("💾 CACHE STORAGE ANALYSIS")
    print("=" * 80)

    cache_dirs = {
        "SEC Cache": Path("data/sec_cache"),
        "Submissions": Path("data/sec_cache/submissions"),
        "Company Facts": Path("data/sec_cache/facts/processed"),
        "LLM Cache": Path("data/llm_cache"),
    }

    for name, path in cache_dirs.items():
        if path.exists():
            # Count files
            files = list(path.rglob("*"))
            data_files = [f for f in files if f.is_file() and not f.name.startswith(".")]
            total_size = sum(f.stat().st_size for f in data_files)

            print(f"\n{name} ({path}):")
            print(f"  Files: {len(data_files)}")
            print(f"  Total Size: {total_size/1024/1024:.1f} MB")

            # Sample some files
            if data_files:
                print(f"  Recent files:")
                recent_files = sorted(data_files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]
                for f in recent_files:
                    size_kb = f.stat().st_size / 1024
                    mod_time = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    print(f"    {f.name:<50} {size_kb:>8.1f} KB  {mod_time}")

    # Configuration check
    print("\n" + "=" * 80)
    print("⚙️ CACHE CONFIGURATION")
    print("=" * 80)

    if hasattr(config, "cache_control"):
        cache_control = config.cache_control
        print(f"\nCache Control Settings:")
        print(f"  Cache Enabled: {cache_control.use_cache}")
        print(f"  Read from Cache: {cache_control.read_from_cache}")
        print(f"  Write to Cache: {cache_control.write_to_cache}")
        print(f"  Force Refresh: {cache_control.force_refresh}")

        print(f"\nTTL Settings:")
        for cache_type in sec_types:
            if cache_type in cache_control.cache_types:
                ttl = cache_control.cache_types[cache_type].ttl_hours
                print(f"  {cache_type}: {ttl} hours ({ttl/24:.1f} days)")

    print("\n✅ SEC Cache Analysis Complete!")

    # Export detailed stats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"logs/sec_cache_analysis_{timestamp}.json"

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "test_results": results,
        "performance_stats": perf_stats,
        "config": {
            "cache_enabled": cache_control.use_cache if hasattr(config, "cache_control") else None,
            "ttl_settings": (
                {ct: cache_control.cache_types[ct].ttl_hours for ct in sec_types if ct in cache_control.cache_types}
                if hasattr(config, "cache_control")
                else {}
            ),
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    print(f"\n📄 Detailed report saved to: {report_file}")


if __name__ == "__main__":
    analyze_sec_cache_patterns()
