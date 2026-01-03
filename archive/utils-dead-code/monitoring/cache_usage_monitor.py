#!/usr/bin/env python3
"""
Cache Usage Monitor for FAANG Company Synthesis
Monitors cache hits, misses, and performance during real synthesis operations
"""

import logging
import time
import sys

# Removed cache facade - using cache manager directly
from utils.cache.cache_manager import get_cache_manager
from utils.cache.cache_types import CacheType
from utils.ascii_art import ASCIIArt


def setup_detailed_logging():
    """Setup detailed logging to capture cache operations"""
    # Set cache loggers to DEBUG level
    cache_loggers = [
        "utils.cache.cache_manager",
        # Removed cache facade - using cache manager directly
        "utils.cache.file_cache_handler",
        "utils.cache.parquet_cache_handler",
        "utils.cache.rdbms_cache_handler",
    ]

    for logger_name in cache_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # Add handler to cache loggers
    for logger_name in cache_loggers:
        logger = logging.getLogger(logger_name)
        if not logger.handlers:
            logger.addHandler(console_handler)


def monitor_cache_stats():
    """Monitor and report cache statistics"""
    # Display cache system banner
    ASCIIArt.print_banner("cache")

    print("=== CACHE MONITORING STARTED ===")

    cache_manager = get_cache_manager()

    print("\n--- Initial Cache State ---")
    stats = cache_manager.get_stats()
    print(f"Total handlers: {stats.get('total_handlers', 0)}")
    print(f"Cache types configured: {len(stats.get('cache_types', {}))}")

    for cache_type_name, cache_info in stats.get("cache_types", {}).items():
        handler_count = cache_info.get("handler_count", 0)
        print(f"  {cache_type_name}: {handler_count} handlers")

    return cache_manager


def test_cache_operations(symbol):
    """Test cache operations for a specific symbol"""
    print(f"\n--- Testing Cache Operations for {symbol} ---")

    cache_manager = monitor_cache_stats()

    operations = []

    print(f"\n1. Testing LLM Response Cache...")
    start_time = time.time()
    try:
        llm_key = {"symbol": symbol, "form_type": "10-K", "period": "2023", "llm_type": "fundamental"}
        llm_result = cache_manager.get(CacheType.LLM_RESPONSE, llm_key)
        end_time = time.time()
        status = "HIT" if llm_result else "MISS"
        operations.append(f"LLM Response: {status} ({end_time - start_time:.3f}s)")
        print(f"   LLM Response: {status}")
    except Exception as e:
        operations.append(f"LLM Response: ERROR - {str(e)[:50]}")
        print(f"   LLM Response: ERROR - {str(e)[:50]}")

    print(f"\n2. Testing Company Facts Cache...")
    start_time = time.time()
    try:
        from utils.ticker_cik_mapper import get_ticker_mapper

        mapper = get_ticker_mapper()
        cik = mapper.resolve_cik(symbol)
        if cik:
            facts_key = {"symbol": symbol, "cik": cik}
            facts_result = cache_manager.get(CacheType.COMPANY_FACTS, facts_key)
            end_time = time.time()
            status = "HIT" if facts_result else "MISS"
            operations.append(f"Company Facts: {status} ({end_time - start_time:.3f}s)")
            print(f"   Company Facts: {status}")
            if facts_result:
                print(f"   Facts data size: ~{len(str(facts_result))} chars")
        else:
            print(f"   Company Facts: ERROR - Could not resolve CIK for {symbol}")
    except Exception as e:
        operations.append(f"Company Facts: ERROR - {str(e)[:50]}")
        print(f"   Company Facts: ERROR - {str(e)[:50]}")

    print(f"\n3. Testing SEC Response Cache...")
    start_time = time.time()
    try:
        sec_key = {"symbol": symbol, "fiscal_year": "2023", "fiscal_period": "Q4"}
        sec_result = cache_manager.get(CacheType.SEC_RESPONSE, sec_key)
        end_time = time.time()
        status = "HIT" if sec_result else "MISS"
        operations.append(f"SEC Response: {status} ({end_time - start_time:.3f}s)")
        print(f"   SEC Response: {status}")
    except Exception as e:
        operations.append(f"SEC Response: ERROR - {str(e)[:50]}")
        print(f"   SEC Response: ERROR - {str(e)[:50]}")

    print(f"\n4. Testing Cache Manager Direct Operations...")
    try:
        # Test direct cache manager operations
        test_key = {"symbol": symbol, "test": "monitor"}
        test_data = {"test": "cache_monitoring", "timestamp": time.time()}

        # Test set operation
        set_success = cache_manager.set(CacheType.LLM_RESPONSE, test_key, test_data)
        print(f"   Direct Set: {'SUCCESS' if set_success else 'FAILED'}")

        # Test get operation
        get_result = cache_manager.get(CacheType.LLM_RESPONSE, test_key)
        print(f"   Direct Get: {'SUCCESS' if get_result else 'FAILED'}")

        # Test delete operation
        delete_success = cache_manager.delete(CacheType.LLM_RESPONSE, test_key)
        print(f"   Direct Delete: {'SUCCESS' if delete_success else 'FAILED'}")

    except Exception as e:
        print(f"   Direct operations: ERROR - {str(e)[:50]}")

    print(f"\n--- Cache Performance Stats ---")
    try:
        perf_stats = cache_manager.get_performance_stats()
        print(f"Performance stats collected: {len(perf_stats)} cache types")
        for cache_type, stats in perf_stats.items():
            if isinstance(stats, dict) and stats:
                print(f"  {cache_type}: {len(stats)} metrics")
    except Exception as e:
        print(f"Performance stats error: {str(e)[:50]}")

    return operations


def run_faang_cache_test():
    """Run cache test for all FAANG companies"""
    setup_detailed_logging()

    faang_symbols = ["AAPL", "AMZN", "NFLX", "GOOGL", "META"]

    print("=== FAANG CACHE MONITORING TEST ===")
    print(f"Testing symbols: {', '.join(faang_symbols)}")

    all_operations = {}

    for symbol in faang_symbols:
        print(f"\n{'='*60}")
        print(f"TESTING {symbol}")
        print(f"{'='*60}")

        operations = test_cache_operations(symbol)
        all_operations[symbol] = operations

        print(f"\nOperations summary for {symbol}:")
        for op in operations:
            print(f"  - {op}")

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    for symbol, operations in all_operations.items():
        print(f"\n{symbol}:")
        hits = len([op for op in operations if "HIT" in op])
        misses = len([op for op in operations if "MISS" in op])
        errors = len([op for op in operations if "ERROR" in op])

        print(f"  Cache HITs: {hits}")
        print(f"  Cache MISSes: {misses}")
        print(f"  Errors: {errors}")

        if misses > hits:
            print(f"  ⚠️  More misses than hits - cache may need population")
        if errors > 0:
            print(f"  ❌ Cache errors detected - needs investigation")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        print(f"Testing cache for {symbol}")
        setup_detailed_logging()
        test_cache_operations(symbol)
    else:
        run_faang_cache_test()
