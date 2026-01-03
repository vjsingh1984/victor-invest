#!/usr/bin/env python3
"""
Comprehensive Cache Inspection Report
Analyzes cache system issues and provides detailed diagnostics
"""

import json
import logging
import os
from pathlib import Path

# Removed cache facade - using cache manager directly
from investigator.infrastructure.cache.cache_manager import get_cache_manager
from investigator.infrastructure.cache.cache_types import CacheType


def inspect_cache_directories():
    """Inspect cache directory structure and contents"""
    print("=== CACHE DIRECTORY INSPECTION ===")

    cache_dirs = {
        "LLM Cache": "data/llm_cache",
        "SEC Cache": "data/sec_cache",
        "Technical Cache": "data/technical_cache",
        "Price Cache": "data/price_cache",
    }

    for name, path in cache_dirs.items():
        print(f"\n{name} ({path}):")
        if os.path.exists(path):
            try:
                # Count files and subdirectories
                total_files = 0
                total_dirs = 0
                total_size = 0

                for root, dirs, files in os.walk(path):
                    total_dirs += len(dirs)
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(file_path)
                            total_files += 1
                        except OSError:
                            pass

                print(f"  Files: {total_files}")
                print(f"  Directories: {total_dirs}")
                print(f"  Total size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")

                # Show recent files
                if total_files > 0:
                    recent_files = []
                    for root, dirs, files in os.walk(path):
                        for file in files[:5]:  # Show first 5 files
                            file_path = os.path.join(root, file)
                            try:
                                size = os.path.getsize(file_path)
                                recent_files.append(f"    {file} ({size:,} bytes)")
                            except OSError:
                                recent_files.append(f"    {file} (size unknown)")

                    print(f"  Sample files:")
                    for rf in recent_files[:3]:
                        print(rf)

            except Exception as e:
                print(f"  Error inspecting directory: {str(e)[:50]}")
        else:
            print(f"  Directory does not exist")


def inspect_cache_handlers():
    """Inspect cache handler configuration"""
    print("\n=== CACHE HANDLER INSPECTION ===")

    try:
        cache_manager = get_cache_manager()
        stats = cache_manager.get_stats()

        print(f"Total handlers configured: {stats.get('total_handlers', 0)}")
        print(f"Cache types: {len(stats.get('cache_types', {}))}")

        for cache_type_name, cache_info in stats.get("cache_types", {}).items():
            print(f"\n{cache_type_name.upper()}:")
            handler_count = cache_info.get("handler_count", 0)
            print(f"  Handlers: {handler_count}")

            handlers = cache_info.get("handlers", [])
            for i, handler in enumerate(handlers):
                handler_type = handler.get("handler_type", "Unknown")
                priority = handler.get("priority", "Unknown")
                print(f"    {i+1}. {handler_type} (priority: {priority})")

        print(f"\nHandler summary:")
        handler_summary = stats.get("handler_summary", {})
        for handler_type, count in handler_summary.items():
            print(f"  {handler_type}: {count}")

    except Exception as e:
        print(f"Error inspecting handlers: {str(e)}")


def test_cache_write_operations():
    """Test cache write operations for each cache type"""
    print("\n=== CACHE WRITE OPERATION TESTS ===")

    try:
        cache_manager = get_cache_manager()

        test_cases = [
            (
                CacheType.LLM_RESPONSE,
                {"symbol": "TEST_WRITE", "llm_type": "test", "form_type": "10-K", "period": "2024"},
                {"prompt": "Test prompt", "response": {"test": True}, "metadata": {"test": "write_test"}},
            ),
            (
                CacheType.COMPANY_FACTS,
                {"symbol": "TEST_WRITE", "cik": "0001234567"},
                {"symbol": "TEST_WRITE", "cik": "0001234567", "companyfacts": {"test": "data"}},
            ),
            (
                CacheType.SEC_RESPONSE,
                {"symbol": "TEST_WRITE", "fiscal_year": "2024", "fiscal_period": "Q1"},
                {"analysis": "test analysis", "metadata": {"test": True}},
            ),
        ]

        results = {}

        for cache_type, key, data in test_cases:
            print(f"\nTesting {cache_type.value}:")
            try:
                # Test write
                write_success = cache_manager.set(cache_type, key, data)
                print(f"  Write: {'SUCCESS' if write_success else 'FAILED'}")

                # Test read
                read_result = cache_manager.get(cache_type, key)
                read_success = read_result is not None
                print(f"  Read: {'SUCCESS' if read_success else 'FAILED'}")

                # Test exists
                exists = cache_manager.exists(cache_type, key)
                print(f"  Exists: {'SUCCESS' if exists else 'FAILED'}")

                # Test delete
                delete_success = cache_manager.delete(cache_type, key)
                print(f"  Delete: {'SUCCESS' if delete_success else 'FAILED'}")

                results[cache_type.value] = {
                    "write": write_success,
                    "read": read_success,
                    "exists": exists,
                    "delete": delete_success,
                }

            except Exception as e:
                print(f"  ERROR: {str(e)[:50]}")
                results[cache_type.value] = {"error": str(e)[:100]}

        # Summary
        print(f"\n=== WRITE TEST SUMMARY ===")
        for cache_type, result in results.items():
            if "error" in result:
                print(f"{cache_type}: ERROR - {result['error']}")
            else:
                operations = ["write", "read", "exists", "delete"]
                success_count = sum(1 for op in operations if result.get(op, False))
                print(f"{cache_type}: {success_count}/4 operations successful")

                failed_ops = [op for op in operations if not result.get(op, False)]
                if failed_ops:
                    print(f"  Failed: {', '.join(failed_ops)}")

    except Exception as e:
        print(f"Error in write tests: {str(e)}")


def identify_cache_issues():
    """Identify specific cache issues and provide solutions"""
    print("\n=== CACHE ISSUE ANALYSIS ===")

    issues = []

    # Check for common issues
    try:
        cache_manager = get_cache_manager()

        # Test LLM response cache
        try:
            llm_key = {"symbol": "TEST", "form_type": "10-K", "period": "2024", "llm_type": "test"}
            cache_manager.get(CacheType.LLM_RESPONSE, llm_key)
            issues.append("✅ LLM response cache working via direct cache manager")
        except Exception as e:
            issues.append(f"❌ LLM response cache error: {str(e)[:50]}")

        # Test company facts cache
        try:
            facts_key = {"symbol": "TEST", "cik": "0001234567"}
            cache_manager.get(CacheType.COMPANY_FACTS, facts_key)
            issues.append("✅ Company facts cache working via direct cache manager")
        except Exception as e:
            issues.append(f"❌ Company facts cache error: {str(e)[:50]}")

        # Test SEC response cache
        try:
            sec_key = {"symbol": "TEST", "fiscal_year": "2024", "fiscal_period": "Q1"}
            cache_manager.get(CacheType.SEC_RESPONSE, sec_key)
            issues.append("✅ SEC response cache working via direct cache manager")
        except Exception as e:
            issues.append(f"❌ SEC response cache error: {str(e)[:50]}")

        # Test technical data cache
        try:
            tech_key = {"symbol": "TEST", "analysis_date": "2024-01-01"}
            cache_manager.get(CacheType.TECHNICAL_DATA, tech_key)
            issues.append("✅ Technical data cache working via direct cache manager")
        except Exception as e:
            issues.append(f"❌ Technical data cache error: {str(e)[:50]}")

    except Exception as e:
        issues.append(f"❌ Major cache manager error: {str(e)[:50]}")

    # Check file patterns
    try:
        from investigator.infrastructure.cache.file_cache_handler import FileCacheStorageHandler

        # This will show if there are pattern issues
        handler = FileCacheStorageHandler(CacheType.LLM_RESPONSE, Path("test"), 10)
        test_key = {"symbol": "TEST"}

        try:
            handler._generate_filename(test_key)
        except Exception as e:
            if "Missing required key" in str(e):
                issues.append(f"❌ File cache pattern issue: {str(e)[:100]}")

    except Exception as e:
        issues.append(f"❌ File handler pattern test failed: {str(e)[:50]}")

    print("Issues found:")
    for issue in issues:
        print(f"  {issue}")

    if not issues:
        print("  ✅ No major issues detected")

    return issues


def generate_cache_recommendations(issues):
    """Generate recommendations based on identified issues"""
    print("\n=== CACHE IMPROVEMENT RECOMMENDATIONS ===")

    recommendations = []

    for issue in issues:
        if "get_sec_response method missing" in issue:
            recommendations.append("🔧 Add get_sec_response method to CacheFacade class")

        if "get_technical_data method missing" in issue:
            recommendations.append("🔧 Add get_technical_data method to CacheFacade class")

        if "Missing required key" in issue:
            recommendations.append("🔧 Fix file cache handler key patterns - ensure all required keys are provided")

        if "sec_companyfacts" in issue or "UndefinedTable" in issue:
            recommendations.append("🔧 Create missing database tables or disable RDBMS cache handlers")

        if "Cache WRITE FAILED" in issue:
            recommendations.append("🔧 Fix cache write operations - check key formats and handler configurations")

    # General recommendations
    recommendations.extend(
        [
            "📊 Monitor cache hit/miss ratios during synthesis operations",
            "🗄️ Populate cache with FAANG data before synthesis to improve performance",
            "⚡ Consider disabling problematic RDBMS handlers if database setup is incomplete",
            "🧪 Run integration tests to verify end-to-end cache workflows",
            "📈 Implement cache warming strategies for frequently accessed data",
        ]
    )

    print("Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")


def main():
    """Run comprehensive cache inspection"""
    print("🔍 COMPREHENSIVE CACHE SYSTEM INSPECTION")
    print("=" * 60)

    # Disable debug logging for cleaner output
    logging.getLogger().setLevel(logging.WARNING)

    inspect_cache_directories()
    inspect_cache_handlers()
    test_cache_write_operations()
    issues = identify_cache_issues()
    generate_cache_recommendations(issues)

    print("\n" + "=" * 60)
    print("📋 INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
