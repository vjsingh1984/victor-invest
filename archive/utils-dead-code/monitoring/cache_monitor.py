#!/usr/bin/env python3
"""
Unified Cache Monitoring Utilities
Consolidates cache_usage, cache_hits_misses, and inspection functionality
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from utils.cache.cache_manager import CacheManager
from utils.cache.cache_types import CacheType

logger = logging.getLogger(__name__)
console = Console()


class MonitorMode(Enum):
    """Cache monitoring modes"""

    LIVE = "live"  # Real-time monitoring
    SNAPSHOT = "snapshot"  # Single snapshot
    REPORT = "report"  # Detailed report
    HITS_MISSES = "hits_misses"  # Focus on cache performance


@dataclass
class CacheStats:
    """Cache statistics data"""

    timestamp: str
    cache_type: str
    total_entries: int
    total_size_mb: float
    hit_rate: float
    miss_rate: float
    avg_access_time_ms: float
    memory_usage_mb: float
    disk_usage_mb: float


class CacheMonitor:
    """
    Comprehensive cache monitoring system
    Consolidates multiple monitoring scripts into one utility
    """

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize cache monitor"""
        self.cache_manager = cache_manager or CacheManager()
        self.console = Console()
        self.stats_history: List[CacheStats] = []
        self.monitoring_active = False

        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.access_times = []

        # Monitoring configuration
        self.log_dir = Path("logs/cache_monitoring")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_snapshot(self) -> Dict[str, Any]:
        """Get current cache snapshot with all statistics"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "cache_types": {},
            "summary": {
                "total_entries": 0,
                "total_size_mb": 0.0,
                "overall_hit_rate": 0.0,
                "memory_usage_mb": 0.0,
                "disk_usage_mb": 0.0,
            },
        }

        # Get stats for each cache type
        for cache_type in CacheType:
            try:
                stats = self._get_cache_type_stats(cache_type)
                snapshot["cache_types"][cache_type.value] = stats

                # Update summary
                snapshot["summary"]["total_entries"] += stats["entries"]
                snapshot["summary"]["total_size_mb"] += stats["size_mb"]
                snapshot["summary"]["memory_usage_mb"] += stats.get("memory_mb", 0)
                snapshot["summary"]["disk_usage_mb"] += stats.get("disk_mb", 0)

            except Exception as e:
                logger.warning(f"Failed to get stats for {cache_type.value}: {e}")

        # Calculate overall hit rate
        if self.hit_count + self.miss_count > 0:
            snapshot["summary"]["overall_hit_rate"] = self.hit_count / (self.hit_count + self.miss_count)

        return snapshot

    def _get_cache_type_stats(self, cache_type: CacheType) -> Dict[str, Any]:
        """Get statistics for a specific cache type"""
        stats = {
            "entries": 0,
            "size_mb": 0.0,
            "memory_mb": 0.0,
            "disk_mb": 0.0,
            "oldest_entry": None,
            "newest_entry": None,
            "top_symbols": [],
        }

        # Get cache handler for this type
        handlers = self.cache_manager._get_handlers_for_type(cache_type)

        for handler in handlers:
            try:
                # Get handler-specific stats
                handler_stats = handler.get_stats() if hasattr(handler, "get_stats") else {}

                stats["entries"] += handler_stats.get("entries", 0)
                stats["size_mb"] += handler_stats.get("size_mb", 0)

                # Determine storage type
                if "File" in handler.__class__.__name__:
                    stats["disk_mb"] += handler_stats.get("size_mb", 0)
                elif "Memory" in handler.__class__.__name__:
                    stats["memory_mb"] += handler_stats.get("size_mb", 0)

            except Exception as e:
                logger.debug(f"Failed to get stats from {handler}: {e}")

        return stats

    def monitor_live(self, duration_seconds: int = 60, interval: int = 5):
        """
        Monitor cache in real-time with live updates

        Args:
            duration_seconds: How long to monitor
            interval: Update interval in seconds
        """
        self.monitoring_active = True
        start_time = time.time()

        with Live(console=self.console, refresh_per_second=1) as live:
            while time.time() - start_time < duration_seconds and self.monitoring_active:
                snapshot = self.get_cache_snapshot()

                # Create display table
                table = self._create_monitoring_table(snapshot)

                # Create performance panel
                perf_panel = self._create_performance_panel()

                # Update display
                live.update(
                    Panel.fit(
                        table, title=f"🔍 Cache Monitor - {datetime.now().strftime('%H:%M:%S')}", subtitle=perf_panel
                    )
                )

                # Log stats
                self._log_snapshot(snapshot)

                time.sleep(interval)

        self.monitoring_active = False
        self.console.print("[green]Monitoring complete![/green]")

    def _create_monitoring_table(self, snapshot: Dict) -> Table:
        """Create rich table for monitoring display"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Cache Type", style="cyan", width=20)
        table.add_column("Entries", justify="right")
        table.add_column("Size (MB)", justify="right")
        table.add_column("Memory (MB)", justify="right")
        table.add_column("Disk (MB)", justify="right")
        table.add_column("Hit Rate", justify="right")

        for cache_type, stats in snapshot["cache_types"].items():
            hit_rate = f"{stats.get('hit_rate', 0):.1%}" if "hit_rate" in stats else "N/A"

            table.add_row(
                cache_type.replace("_", " ").title(),
                str(stats["entries"]),
                f"{stats['size_mb']:.2f}",
                f"{stats['memory_mb']:.2f}",
                f"{stats['disk_mb']:.2f}",
                hit_rate,
            )

        # Add summary row
        summary = snapshot["summary"]
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{summary['total_entries']}[/bold]",
            f"[bold]{summary['total_size_mb']:.2f}[/bold]",
            f"[bold]{summary['memory_usage_mb']:.2f}[/bold]",
            f"[bold]{summary['disk_usage_mb']:.2f}[/bold]",
            f"[bold]{summary['overall_hit_rate']:.1%}[/bold]",
        )

        return table

    def _create_performance_panel(self) -> str:
        """Create performance metrics panel"""
        total_accesses = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_accesses if total_accesses > 0 else 0
        avg_access_time = sum(self.access_times) / len(self.access_times) if self.access_times else 0

        return (
            f"Hits: {self.hit_count} | Misses: {self.miss_count} | "
            f"Hit Rate: {hit_rate:.1%} | Avg Access: {avg_access_time:.1f}ms"
        )

    def track_cache_access(self, cache_type: CacheType, hit: bool, access_time_ms: float):
        """
        Track cache access for performance monitoring

        Args:
            cache_type: Type of cache accessed
            hit: Whether it was a hit or miss
            access_time_ms: Access time in milliseconds
        """
        if hit:
            self.hit_count += 1
        else:
            self.miss_count += 1

        self.access_times.append(access_time_ms)

        # Keep only recent access times (last 1000)
        if len(self.access_times) > 1000:
            self.access_times = self.access_times[-1000:]

    def generate_cache_report(self, output_path: Optional[Path] = None) -> Dict:
        """
        Generate comprehensive cache analysis report

        Args:
            output_path: Optional path to save report

        Returns:
            Comprehensive cache analysis
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "cache_snapshot": self.get_cache_snapshot(),
            "performance_metrics": self._calculate_performance_metrics(),
            "storage_analysis": self._analyze_storage_distribution(),
            "symbol_analysis": self._analyze_symbol_coverage(),
            "recommendations": self._generate_recommendations(),
        }

        # Display report
        self._display_cache_report(report)

        # Save if path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            self.console.print(f"[green]Report saved to {output_path}[/green]")

        return report

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate detailed performance metrics"""
        total_accesses = self.hit_count + self.miss_count

        metrics = {
            "total_accesses": total_accesses,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total_accesses if total_accesses > 0 else 0,
            "miss_rate": self.miss_count / total_accesses if total_accesses > 0 else 0,
        }

        if self.access_times:
            metrics.update(
                {
                    "avg_access_time_ms": sum(self.access_times) / len(self.access_times),
                    "min_access_time_ms": min(self.access_times),
                    "max_access_time_ms": max(self.access_times),
                    "p95_access_time_ms": pd.Series(self.access_times).quantile(0.95),
                }
            )

        return metrics

    def _analyze_storage_distribution(self) -> Dict:
        """Analyze how cache is distributed across storage layers"""
        distribution = {"by_handler": {}, "by_priority": {}, "total_size_mb": 0}

        # Analyze each cache handler
        for handler in self.cache_manager.handlers:
            handler_name = handler.__class__.__name__
            handler_stats = handler.get_stats() if hasattr(handler, "get_stats") else {}

            distribution["by_handler"][handler_name] = {
                "entries": handler_stats.get("entries", 0),
                "size_mb": handler_stats.get("size_mb", 0),
                "priority": handler.priority,
            }

            distribution["total_size_mb"] += handler_stats.get("size_mb", 0)

        return distribution

    def _analyze_symbol_coverage(self) -> Dict:
        """Analyze which symbols have cached data"""
        coverage = {"total_symbols": 0, "symbols_with_cache": [], "cache_by_symbol": {}}

        # This would need to be implemented based on your cache structure
        # For now, returning placeholder
        return coverage

    def _generate_recommendations(self) -> List[str]:
        """Generate cache optimization recommendations"""
        recommendations = []

        snapshot = self.get_cache_snapshot()
        summary = snapshot["summary"]

        # Check hit rate
        if summary["overall_hit_rate"] < 0.7:
            recommendations.append(
                f"Low cache hit rate ({summary['overall_hit_rate']:.1%}). "
                "Consider pre-warming cache for frequently accessed data."
            )

        # Check memory usage
        if summary["memory_usage_mb"] > 1000:
            recommendations.append(
                f"High memory usage ({summary['memory_usage_mb']:.0f}MB). "
                "Consider moving some data to disk-based cache."
            )

        # Check disk usage
        if summary["disk_usage_mb"] > 5000:
            recommendations.append(
                f"High disk usage ({summary['disk_usage_mb']:.0f}MB). " "Consider implementing cache eviction policy."
            )

        if not recommendations:
            recommendations.append("Cache performance is optimal.")

        return recommendations

    def _display_cache_report(self, report: Dict):
        """Display cache report in console"""
        self.console.print("\n[bold cyan]═══ CACHE ANALYSIS REPORT ═══[/bold cyan]\n")

        # Performance metrics
        perf = report["performance_metrics"]
        self.console.print("[yellow]Performance Metrics:[/yellow]")
        self.console.print(f"  • Total Accesses: {perf['total_accesses']:,}")
        self.console.print(f"  • Hit Rate: {perf['hit_rate']:.1%}")
        self.console.print(f"  • Miss Rate: {perf['miss_rate']:.1%}")

        if "avg_access_time_ms" in perf:
            self.console.print(f"  • Avg Access Time: {perf['avg_access_time_ms']:.1f}ms")
            self.console.print(f"  • P95 Access Time: {perf['p95_access_time_ms']:.1f}ms")

        # Storage distribution
        self.console.print("\n[yellow]Storage Distribution:[/yellow]")
        storage = report["storage_analysis"]
        for handler, stats in storage["by_handler"].items():
            self.console.print(f"  • {handler}: {stats['entries']} entries, {stats['size_mb']:.1f}MB")

        # Recommendations
        self.console.print("\n[yellow]Recommendations:[/yellow]")
        for rec in report["recommendations"]:
            self.console.print(f"  ⚠️ {rec}")

    def _log_snapshot(self, snapshot: Dict):
        """Log snapshot to file for historical analysis"""
        log_file = self.log_dir / f"cache_monitor_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(snapshot, default=str) + "\n")

    def analyze_historical_data(self, days: int = 7) -> pd.DataFrame:
        """
        Analyze historical cache monitoring data

        Args:
            days: Number of days to analyze

        Returns:
            DataFrame with historical analysis
        """
        data = []
        cutoff_date = datetime.now() - timedelta(days=days)

        # Read log files
        for log_file in self.log_dir.glob("cache_monitor_*.jsonl"):
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        timestamp = datetime.fromisoformat(entry["timestamp"])

                        if timestamp >= cutoff_date:
                            data.append(
                                {
                                    "timestamp": timestamp,
                                    "total_entries": entry["summary"]["total_entries"],
                                    "total_size_mb": entry["summary"]["total_size_mb"],
                                    "hit_rate": entry["summary"]["overall_hit_rate"],
                                    "memory_mb": entry["summary"]["memory_usage_mb"],
                                    "disk_mb": entry["summary"]["disk_usage_mb"],
                                }
                            )
            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")

        if data:
            df = pd.DataFrame(data)
            df = df.sort_values("timestamp")

            # Add derived metrics
            df["cache_efficiency"] = df["hit_rate"] * 100
            df["total_storage_mb"] = df["memory_mb"] + df["disk_mb"]

            return df
        else:
            return pd.DataFrame()


def main():
    """Main entry point for cache monitoring"""
    import argparse

    parser = argparse.ArgumentParser(description="Cache Monitoring Utility")
    parser.add_argument("mode", choices=["live", "snapshot", "report", "analyze"], help="Monitoring mode")
    parser.add_argument("--duration", type=int, default=60, help="Duration for live monitoring (seconds)")
    parser.add_argument("--interval", type=int, default=5, help="Update interval for live monitoring")
    parser.add_argument("--output", type=str, help="Output file for reports")
    parser.add_argument("--days", type=int, default=7, help="Days to analyze for historical data")

    args = parser.parse_args()

    # Initialize monitor
    monitor = CacheMonitor()

    if args.mode == "live":
        monitor.monitor_live(args.duration, args.interval)

    elif args.mode == "snapshot":
        snapshot = monitor.get_cache_snapshot()
        console.print(monitor._create_monitoring_table(snapshot))

    elif args.mode == "report":
        output_path = Path(args.output) if args.output else None
        monitor.generate_cache_report(output_path)

    elif args.mode == "analyze":
        df = monitor.analyze_historical_data(args.days)
        if not df.empty:
            console.print(f"\n[cyan]Historical Analysis ({args.days} days):[/cyan]")
            console.print(f"Average Hit Rate: {df['hit_rate'].mean():.1%}")
            console.print(f"Peak Storage: {df['total_storage_mb'].max():.1f}MB")
            console.print(f"Average Entries: {df['total_entries'].mean():.0f}")
        else:
            console.print("[yellow]No historical data found[/yellow]")


if __name__ == "__main__":
    main()
