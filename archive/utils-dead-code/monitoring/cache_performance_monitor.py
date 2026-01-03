#!/usr/bin/env python3
"""
Monitor Cache HIT/MISS Operations for FAANG Analysis
Tracks cache performance during real InvestiGator analysis
"""

import logging
import time
import sys
from datetime import datetime
from utils.cache.cache_manager import get_cache_manager
from utils.cache.cache_types import CacheType


class CacheMonitor:
    """Monitors cache HIT/MISS operations with detailed statistics"""

    def __init__(self):
        self.stats = {"hits": 0, "misses": 0, "writes": 0, "errors": 0, "by_type": {}, "operations": []}
        self.start_time = time.time()

    def setup_cache_logging(self):
        """Setup cache manager logging to capture HIT/MISS operations"""
        # Get cache manager logger
        cache_logger = logging.getLogger("utils.cache.cache_manager")
        cache_logger.setLevel(logging.INFO)

        # Remove existing handlers
        for handler in cache_logger.handlers[:]:
            cache_logger.removeHandler(handler)

        # Create custom handler that captures our stats
        handler = CacheStatsHandler(self.stats)
        handler.setLevel(logging.INFO)

        # Add our custom handler
        cache_logger.addHandler(handler)
        cache_logger.propagate = False

        print("🔍 Cache monitoring active - tracking HIT/MISS operations")

    def print_realtime_stats(self):
        """Print real-time cache statistics"""
        elapsed = time.time() - self.start_time
        total_ops = self.stats["hits"] + self.stats["misses"] + self.stats["writes"]

        hit_rate = (
            (self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])) * 100
            if (self.stats["hits"] + self.stats["misses"]) > 0
            else 0
        )

        print(f"\n📊 CACHE PERFORMANCE (Elapsed: {elapsed:.1f}s)")
        print(f"   Total Operations: {total_ops}")
        print(f"   ✅ HITs: {self.stats['hits']}")
        print(f"   ❌ MISSes: {self.stats['misses']}")
        print(f"   📝 WRITEs: {self.stats['writes']}")
        print(f"   💥 ERRORs: {self.stats['errors']}")
        print(f"   📈 Hit Rate: {hit_rate:.1f}%")

        if self.stats["by_type"]:
            print(f"\n   By Cache Type:")
            for cache_type, type_stats in self.stats["by_type"].items():
                type_hit_rate = (
                    (type_stats["hits"] / (type_stats["hits"] + type_stats["misses"])) * 100
                    if (type_stats["hits"] + type_stats["misses"]) > 0
                    else 0
                )
                print(
                    f"     {cache_type}: {type_stats['hits']}H/{type_stats['misses']}M/{type_stats['writes']}W ({type_hit_rate:.1f}%)"
                )

    def print_final_report(self):
        """Print final cache performance report"""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("📋 FINAL CACHE PERFORMANCE REPORT")
        print("=" * 60)

        total_ops = self.stats["hits"] + self.stats["misses"] + self.stats["writes"]
        hit_rate = (
            (self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])) * 100
            if (self.stats["hits"] + self.stats["misses"]) > 0
            else 0
        )

        print(f"Analysis Duration: {elapsed:.1f} seconds")
        print(f"Total Cache Operations: {total_ops}")
        print(f"Cache Hit Rate: {hit_rate:.1f}%")
        print(f"Operations per Second: {total_ops/elapsed:.1f}")

        print(f"\nOperations Breakdown:")
        print(f"  ✅ Cache HITs: {self.stats['hits']}")
        print(f"  ❌ Cache MISSes: {self.stats['misses']}")
        print(f"  📝 Cache WRITEs: {self.stats['writes']}")
        print(f"  💥 Cache ERRORs: {self.stats['errors']}")

        if self.stats["by_type"]:
            print(f"\nBy Cache Type:")
            for cache_type, type_stats in self.stats["by_type"].items():
                type_total = type_stats["hits"] + type_stats["misses"] + type_stats["writes"]
                type_hit_rate = (
                    (type_stats["hits"] / (type_stats["hits"] + type_stats["misses"])) * 100
                    if (type_stats["hits"] + type_stats["misses"]) > 0
                    else 0
                )
                print(f"  {cache_type}:")
                print(f"    Total: {type_total} ops, Hit Rate: {type_hit_rate:.1f}%")
                print(f"    HITs: {type_stats['hits']}, MISSes: {type_stats['misses']}, WRITEs: {type_stats['writes']}")

        if self.stats["operations"]:
            print(f"\nRecent Operations (last 10):")
            for op in self.stats["operations"][-10:]:
                print(f"  {op}")


class CacheStatsHandler(logging.Handler):
    """Custom logging handler to capture cache statistics"""

    def __init__(self, stats_dict):
        super().__init__()
        self.stats = stats_dict

    def emit(self, record):
        message = record.getMessage()

        # Parse cache manager log messages
        if "Cache HIT" in message:
            self.stats["hits"] += 1
            cache_type = self._extract_cache_type(message)
            self._update_type_stats(cache_type, "hits")
            self._log_operation(f"HIT: {cache_type} - {self._extract_key(message)}")

        elif "Cache MISS" in message:
            self.stats["misses"] += 1
            cache_type = self._extract_cache_type(message)
            self._update_type_stats(cache_type, "misses")
            self._log_operation(f"MISS: {cache_type} - {self._extract_key(message)}")

        elif "Cache WRITE SUCCESS" in message:
            self.stats["writes"] += 1
            cache_type = self._extract_cache_type(message)
            self._update_type_stats(cache_type, "writes")
            self._log_operation(f"WRITE: {cache_type} - {self._extract_key(message)}")

        elif "Cache ERROR" in message or "Cache WRITE ERROR" in message:
            self.stats["errors"] += 1
            cache_type = self._extract_cache_type(message)
            self._update_type_stats(cache_type, "errors")
            self._log_operation(f"ERROR: {cache_type} - {self._extract_key(message)}")

    def _extract_cache_type(self, message):
        """Extract cache type from log message"""
        for cache_type in [
            "llm_response",
            "company_facts",
            "sec_response",
            "technical_data",
            "submission_data",
            "quarterly_metrics",
        ]:
            if cache_type in message:
                return cache_type
        return "unknown"

    def _extract_key(self, message):
        """Extract cache key from log message"""
        if "Key:" in message:
            key_part = message.split("Key:")[1].split("|")[0].strip()
            return key_part
        return "unknown"

    def _update_type_stats(self, cache_type, operation):
        """Update per-type statistics"""
        if cache_type not in self.stats["by_type"]:
            self.stats["by_type"][cache_type] = {"hits": 0, "misses": 0, "writes": 0, "errors": 0}

        if operation in self.stats["by_type"][cache_type]:
            self.stats["by_type"][cache_type][operation] += 1

    def _log_operation(self, operation):
        """Log operation for debugging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.stats["operations"].append(f"[{timestamp}] {operation}")

        # Keep only last 50 operations
        if len(self.stats["operations"]) > 50:
            self.stats["operations"] = self.stats["operations"][-50:]


def main():
    """Run cache monitoring"""
    print("🔍 Cache HIT/MISS Monitor for InvestiGator")
    print("=" * 50)

    monitor = CacheMonitor()
    monitor.setup_cache_logging()

    print("✅ Cache monitoring is now active!")
    print("📊 Run InvestiGator analysis in another terminal to see cache operations")
    print("🔄 Press Ctrl+C to stop monitoring and see final report")

    try:
        last_update = time.time()

        while True:
            time.sleep(5)  # Update every 5 seconds

            # Print stats every 30 seconds
            if time.time() - last_update >= 30:
                monitor.print_realtime_stats()
                last_update = time.time()

    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
        monitor.print_final_report()


if __name__ == "__main__":
    main()
