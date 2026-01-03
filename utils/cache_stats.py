#!/usr/bin/env python3
"""
InvestiGator - Cache Statistics and Monitoring Utility
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Utility for monitoring cache performance and generating reports
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from investigator.infrastructure.cache.cache_manager import get_cache_manager
from investigator.infrastructure.cache.cache_types import CacheType


class CacheStatsMonitor:
    """Monitor and report cache performance statistics"""
    
    def __init__(self, config=None):
        self.cache_manager = get_cache_manager()
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'performance_stats': self.cache_manager.get_performance_stats(),
            'handler_stats': self.cache_manager.get_stats(),
            'cache_config_summary': self._get_cache_config_summary()
        }
    
    def print_cache_report(self, detailed: bool = False):
        """Print a formatted cache performance report"""
        stats = self.get_comprehensive_stats()
        performance = stats['performance_stats']
        
        print("\n" + "="*80)
        print("🗄️  CACHE PERFORMANCE REPORT")
        print("="*80)
        print(f"Generated: {stats['timestamp']}")
        print()
        
        # Summary table
        print("📊 CACHE TYPE SUMMARY:")
        print("-" * 80)
        print(f"{'Cache Type':<20} {'Hit Ratio':<12} {'Hits':<8} {'Misses':<8} {'Writes':<8} {'Errors':<8} {'Avg Time':<10}")
        print("-" * 80)
        
        for cache_type, type_stats in performance.items():
            ops = type_stats['operations']
            perf = type_stats['performance']
            
            print(f"{cache_type:<20} {perf['hit_ratio_pct']:>8.1f}% {ops['hits']:>8} {ops['misses']:>8} {ops['writes']:>8} {ops['errors']:>8} {perf['avg_time_ms']:>8.1f}ms")
        
        if detailed:
            print("\n📋 DETAILED HANDLER PERFORMANCE:")
            print("-" * 80)
            
            for cache_type, type_stats in performance.items():
                if type_stats['handlers']:
                    print(f"\n🗂️  {cache_type.upper()}:")
                    for handler_name, handler_stats in type_stats['handlers'].items():
                        print(f"  • {handler_name:<25} Hit Ratio: {handler_stats['hit_ratio_pct']:>6.1f}% | "
                              f"Hits: {handler_stats['hits']:>4} | Misses: {handler_stats['misses']:>4} | "
                              f"Writes: {handler_stats['writes']:>4} | Errors: {handler_stats['errors']:>4}")
        
        print("\n" + "="*80)
    
    def get_cache_efficiency_score(self) -> Dict[str, float]:
        """Calculate cache efficiency scores for each cache type"""
        performance = self.cache_manager.get_performance_stats()
        efficiency_scores = {}
        
        for cache_type, stats in performance.items():
            ops = stats['operations']
            perf = stats['performance']
            
            # Calculate efficiency score (0-100)
            hit_ratio = perf['hit_ratio_pct']
            error_ratio = (ops['errors'] / max(1, ops['hits'] + ops['misses'] + ops['writes'])) * 100
            speed_score = max(0, 100 - (perf['avg_time_ms'] / 10))  # Penalty for slow operations
            
            # Weighted efficiency score
            efficiency = (hit_ratio * 0.6) + (speed_score * 0.3) + ((100 - error_ratio) * 0.1)
            efficiency_scores[cache_type] = round(efficiency, 2)
        
        return efficiency_scores
    
    def get_recent_operations_summary(self, cache_type: Optional[CacheType] = None, limit: int = 10) -> Dict[str, Any]:
        """Get summary of recent cache operations"""
        recent_ops = self.cache_manager.get_recent_operations(cache_type, limit)
        
        summary = {}
        for ct, operations in recent_ops.items():
            if operations:
                operation_types = {}
                total_time = 0
                
                for op in operations:
                    op_type = op['operation']
                    if op_type not in operation_types:
                        operation_types[op_type] = 0
                    operation_types[op_type] += 1
                    total_time += op['time_ms']
                
                summary[ct] = {
                    'total_operations': len(operations),
                    'operation_breakdown': operation_types,
                    'avg_time_ms': round(total_time / len(operations), 2) if operations else 0,
                    'recent_operations': operations[-5:]  # Last 5 operations
                }
        
        return summary
    
    def _get_cache_config_summary(self) -> Dict[str, Any]:
        """Get summary of cache configuration"""
        if not self.config or not hasattr(self.config, 'cache_control'):
            return {'status': 'no_config'}
        
        cache_control = self.config.cache_control
        
        return {
            'use_cache': cache_control.use_cache,
            'read_from_cache': cache_control.read_from_cache,
            'write_to_cache': cache_control.write_to_cache,
            'force_refresh': cache_control.force_refresh,
            'enabled_cache_types': [ct.value for ct in CacheType if cache_control.is_cache_type_enabled(ct.value)]
        }
    
    def export_stats_to_json(self, file_path: str = None) -> str:
        """Export cache statistics to JSON file"""
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"logs/cache_stats_{timestamp}.json"
        
        stats = self.get_comprehensive_stats()
        stats['efficiency_scores'] = self.get_cache_efficiency_score()
        stats['recent_operations'] = self.get_recent_operations_summary()
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        self.logger.info(f"Cache statistics exported to: {file_path}")
        return file_path
    
    def log_performance_warning(self):
        """Log warnings for poor cache performance"""
        efficiency_scores = self.get_cache_efficiency_score()
        performance = self.cache_manager.get_performance_stats()
        
        for cache_type, score in efficiency_scores.items():
            if score < 50:
                self.logger.warning(f"⚠️ Poor cache efficiency for {cache_type}: {score:.1f}%")
                
                type_stats = performance.get(cache_type, {})
                ops = type_stats.get('operations', {})
                perf = type_stats.get('performance', {})
                
                if perf.get('hit_ratio_pct', 0) < 30:
                    self.logger.warning(f"   Low hit ratio: {perf.get('hit_ratio_pct', 0):.1f}%")
                
                if perf.get('avg_time_ms', 0) > 100:
                    self.logger.warning(f"   Slow operations: {perf.get('avg_time_ms', 0):.1f}ms average")
                
                if ops.get('errors', 0) > 0:
                    error_rate = (ops['errors'] / max(1, sum(ops.values()))) * 100
                    self.logger.warning(f"   Error rate: {error_rate:.1f}%")


def print_cache_stats(detailed: bool = False):
    """Convenience function to print cache statistics"""
    monitor = CacheStatsMonitor()
    monitor.print_cache_report(detailed=detailed)


def export_cache_stats(file_path: str = None) -> str:
    """Convenience function to export cache statistics"""
    monitor = CacheStatsMonitor()
    return monitor.export_stats_to_json(file_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache Statistics Monitor")
    parser.add_argument("--detailed", action="store_true", help="Show detailed handler statistics")
    parser.add_argument("--export", type=str, help="Export statistics to JSON file")
    parser.add_argument("--efficiency", action="store_true", help="Show efficiency scores only")
    
    args = parser.parse_args()
    
    monitor = CacheStatsMonitor()
    
    if args.efficiency:
        scores = monitor.get_cache_efficiency_score()
        print("\n🎯 CACHE EFFICIENCY SCORES:")
        print("-" * 40)
        for cache_type, score in scores.items():
            print(f"{cache_type:<20} {score:>6.1f}%")
        print()
    elif args.export:
        file_path = monitor.export_stats_to_json(args.export)
        print(f"Statistics exported to: {file_path}")
    else:
        monitor.print_cache_report(detailed=args.detailed)
        monitor.log_performance_warning()