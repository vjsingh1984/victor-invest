"""
Performance Profiling Tools for InvestiGator
Advanced profiling and optimization utilities
"""

import asyncio
import time
import cProfile
import pstats
import io
import tracemalloc
import psutil
import functools
from typing import Dict, List, Callable, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result of a profiling session"""
    function_name: str
    execution_time: float
    cpu_percent: float
    memory_mb: float
    memory_peak_mb: float
    call_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    call_stack: Optional[str] = None
    hotspots: List[Tuple[str, float]] = field(default_factory=list)


class PerformanceProfiler:
    """
    Comprehensive performance profiler for the InvestiGator system
    """
    
    def __init__(self, output_dir: str = "profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Profile storage
        self.profiles: Dict[str, List[ProfileResult]] = {}
        self.active_profiles: Dict[str, Any] = {}
        
        # System baseline
        self.baseline_cpu = psutil.cpu_percent(interval=1)
        self.baseline_memory = psutil.virtual_memory().percent
        
        # Tracing state
        self.memory_tracing = False
        self.cpu_profiler = None
    
    def profile_function(self, name: Optional[str] = None):
        """
        Decorator to profile a function's performance
        
        Usage:
            @profiler.profile_function()
            async def my_function():
                pass
        """
        def decorator(func):
            function_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._profile_async(function_name, func, args, kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._profile_sync(function_name, func, args, kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def _profile_async(self, name: str, func: Callable, args: tuple, kwargs: dict):
        """Profile async function execution"""
        # Start profiling
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        # Enable memory tracing if not already active
        if not self.memory_tracing:
            tracemalloc.start()
            self.memory_tracing = True
        
        snapshot_before = tracemalloc.take_snapshot()
        
        # CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            # Execute function
            result = await func(*args, **kwargs)
            
            # Stop profiling
            profiler.disable()
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            # Get memory snapshot
            snapshot_after = tracemalloc.take_snapshot()
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            memory_peak = max(
                stat.size_diff / 1024 / 1024 
                for stat in top_stats[:10]
            ) if top_stats else 0
            
            # Get CPU usage during execution
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Extract hotspots
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(10)
            
            hotspots = self._extract_hotspots(ps)
            
            # Create profile result
            profile_result = ProfileResult(
                function_name=name,
                execution_time=execution_time,
                cpu_percent=cpu_percent,
                memory_mb=memory_used,
                memory_peak_mb=memory_peak,
                call_count=1,
                hotspots=hotspots
            )
            
            # Store profile
            if name not in self.profiles:
                self.profiles[name] = []
            self.profiles[name].append(profile_result)
            
            # Log if slow
            if execution_time > 1.0:
                logger.warning(f"Slow function: {name} took {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            profiler.disable()
            logger.error(f"Error profiling {name}: {e}")
            raise
    
    def _profile_sync(self, name: str, func: Callable, args: tuple, kwargs: dict):
        """Profile synchronous function execution"""
        # Similar to async but without await
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
            
            profiler.disable()
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Extract profiling stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(10)
            
            hotspots = self._extract_hotspots(ps)
            
            profile_result = ProfileResult(
                function_name=name,
                execution_time=execution_time,
                cpu_percent=cpu_percent,
                memory_mb=memory_used,
                memory_peak_mb=memory_used,
                call_count=1,
                hotspots=hotspots
            )
            
            if name not in self.profiles:
                self.profiles[name] = []
            self.profiles[name].append(profile_result)
            
            return result
            
        except Exception as e:
            profiler.disable()
            logger.error(f"Error profiling {name}: {e}")
            raise
    
    def start_continuous_profiling(self, interval: int = 60):
        """Start continuous system profiling"""
        async def profile_loop():
            while True:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
        
        asyncio.create_task(profile_loop())
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'per_core': psutil.cpu_percent(interval=1, percpu=True),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            'memory': {
                'percent': psutil.virtual_memory().percent,
                'used_mb': psutil.virtual_memory().used / 1024 / 1024,
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            },
            'disk': {
                'usage_percent': psutil.disk_usage('/').percent,
                'read_mb': psutil.disk_io_counters().read_bytes / 1024 / 1024,
                'write_mb': psutil.disk_io_counters().write_bytes / 1024 / 1024
            },
            'network': {
                'sent_mb': psutil.net_io_counters().bytes_sent / 1024 / 1024,
                'recv_mb': psutil.net_io_counters().bytes_recv / 1024 / 1024
            }
        }
        
        # Save metrics
        metrics_file = self.output_dir / f"system_metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _extract_hotspots(self, stats: pstats.Stats) -> List[Tuple[str, float]]:
        """Extract top hotspots from profiling stats"""
        hotspots = []
        
        # Get top 10 functions by cumulative time
        stats.sort_stats('cumulative')
        for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:10]:
            filename, line, func_name = func
            hotspots.append((f"{filename}:{line}:{func_name}", ct))
        
        return hotspots
    
    def analyze_profile(self, function_name: str) -> Dict:
        """Analyze profiling results for a function"""
        if function_name not in self.profiles:
            return {'error': f'No profile data for {function_name}'}
        
        profiles = self.profiles[function_name]
        
        # Calculate statistics
        exec_times = [p.execution_time for p in profiles]
        cpu_usages = [p.cpu_percent for p in profiles]
        memory_usages = [p.memory_mb for p in profiles]
        
        analysis = {
            'function': function_name,
            'call_count': len(profiles),
            'execution_time': {
                'mean': np.mean(exec_times),
                'median': np.median(exec_times),
                'std': np.std(exec_times),
                'min': np.min(exec_times),
                'max': np.max(exec_times),
                'p95': np.percentile(exec_times, 95),
                'p99': np.percentile(exec_times, 99)
            },
            'cpu_usage': {
                'mean': np.mean(cpu_usages),
                'max': np.max(cpu_usages)
            },
            'memory_usage': {
                'mean': np.mean(memory_usages),
                'max': np.max(memory_usages),
                'peak': max(p.memory_peak_mb for p in profiles)
            },
            'hotspots': self._aggregate_hotspots(profiles),
            'performance_trend': self._calculate_trend(exec_times)
        }
        
        return analysis
    
    def _aggregate_hotspots(self, profiles: List[ProfileResult]) -> List[Tuple[str, float]]:
        """Aggregate hotspots across multiple profile runs"""
        hotspot_totals = {}
        
        for profile in profiles:
            for func, time in profile.hotspots:
                if func not in hotspot_totals:
                    hotspot_totals[func] = 0
                hotspot_totals[func] += time
        
        # Sort by total time
        sorted_hotspots = sorted(
            hotspot_totals.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_hotspots[:10]
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate performance trend"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'degrading'
        else:
            return 'improving'
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive profiling report"""
        report = []
        report.append("=" * 80)
        report.append("InvestiGator Performance Profile Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        report.append("")
        
        # Analyze each profiled function
        for function_name in self.profiles.keys():
            analysis = self.analyze_profile(function_name)
            
            report.append(f"\nFunction: {function_name}")
            report.append("-" * 40)
            report.append(f"Calls: {analysis['call_count']}")
            report.append(f"Avg Time: {analysis['execution_time']['mean']:.3f}s")
            report.append(f"P95 Time: {analysis['execution_time']['p95']:.3f}s")
            report.append(f"Max Time: {analysis['execution_time']['max']:.3f}s")
            report.append(f"Avg CPU: {analysis['cpu_usage']['mean']:.1f}%")
            report.append(f"Avg Memory: {analysis['memory_usage']['mean']:.1f}MB")
            report.append(f"Trend: {analysis['performance_trend']}")
            
            if analysis['hotspots']:
                report.append("\nTop Hotspots:")
                for func, time in analysis['hotspots'][:5]:
                    report.append(f"  - {func}: {time:.3f}s")
        
        # System metrics summary
        report.append("\n" + "=" * 80)
        report.append("System Resource Usage")
        report.append("-" * 40)
        report.append(f"Baseline CPU: {self.baseline_cpu:.1f}%")
        report.append(f"Baseline Memory: {self.baseline_memory:.1f}%")
        
        report_text = "\n".join(report)
        
        # Save report
        if output_file:
            output_path = self.output_dir / output_file
        else:
            output_path = self.output_dir / f"profile_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Profile report saved to {output_path}")
        
        return report_text
    
    def plot_performance(self, function_name: str, save_path: Optional[str] = None):
        """Generate performance visualization"""
        if function_name not in self.profiles:
            logger.warning(f"No profile data for {function_name}")
            return
        
        profiles = self.profiles[function_name]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Performance Profile: {function_name}')
        
        # Execution time over time
        times = [p.timestamp for p in profiles]
        exec_times = [p.execution_time for p in profiles]
        axes[0, 0].plot(times, exec_times, 'b-')
        axes[0, 0].set_title('Execution Time')
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # CPU usage
        cpu_usages = [p.cpu_percent for p in profiles]
        axes[0, 1].plot(times, cpu_usages, 'r-')
        axes[0, 1].set_title('CPU Usage')
        axes[0, 1].set_ylabel('CPU %')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage
        memory_usages = [p.memory_mb for p in profiles]
        axes[1, 0].plot(times, memory_usages, 'g-')
        axes[1, 0].set_title('Memory Usage')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Execution time distribution
        axes[1, 1].hist(exec_times, bins=20, edgecolor='black')
        axes[1, 1].set_title('Execution Time Distribution')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(self.output_dir / save_path)
            logger.info(f"Performance plot saved to {self.output_dir / save_path}")
        else:
            plt.show()
    
    def optimize_suggestions(self, function_name: str) -> List[str]:
        """Generate optimization suggestions based on profiling"""
        if function_name not in self.profiles:
            return []
        
        analysis = self.analyze_profile(function_name)
        suggestions = []
        
        # Check execution time
        if analysis['execution_time']['mean'] > 1.0:
            suggestions.append("Consider async/parallel processing for long-running operations")
        
        if analysis['execution_time']['std'] > analysis['execution_time']['mean'] * 0.5:
            suggestions.append("High variance in execution time - investigate inconsistent performance")
        
        # Check CPU usage
        if analysis['cpu_usage']['mean'] > 80:
            suggestions.append("High CPU usage - consider optimizing algorithms or using caching")
        
        # Check memory usage
        if analysis['memory_usage']['peak'] > 1000:  # 1GB
            suggestions.append("High memory usage - consider streaming or chunking data")
        
        # Check trend
        if analysis['performance_trend'] == 'degrading':
            suggestions.append("Performance degrading over time - possible memory leak or resource exhaustion")
        
        # Analyze hotspots
        if analysis['hotspots']:
            top_hotspot = analysis['hotspots'][0]
            if top_hotspot[1] > analysis['execution_time']['mean'] * 0.5:
                suggestions.append(f"Optimize hotspot: {top_hotspot[0]} (consuming >50% of execution time)")
        
        return suggestions


class MemoryProfiler:
    """
    Specialized memory profiler for tracking memory usage patterns
    """
    
    def __init__(self):
        self.snapshots = []
        self.tracking = False
    
    def start(self):
        """Start memory tracking"""
        if not self.tracking:
            tracemalloc.start()
            self.tracking = True
            self.baseline = tracemalloc.take_snapshot()
    
    def snapshot(self, label: str = ""):
        """Take a memory snapshot"""
        if not self.tracking:
            self.start()
        
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            'label': label,
            'timestamp': datetime.now(),
            'snapshot': snapshot
        })
    
    def compare(self, start_label: str, end_label: str) -> List:
        """Compare two snapshots"""
        start_snap = None
        end_snap = None
        
        for snap in self.snapshots:
            if snap['label'] == start_label:
                start_snap = snap['snapshot']
            if snap['label'] == end_label:
                end_snap = snap['snapshot']
        
        if not start_snap or not end_snap:
            return []
        
        return end_snap.compare_to(start_snap, 'lineno')
    
    def report_leaks(self, threshold_mb: float = 10) -> List[str]:
        """Report potential memory leaks"""
        if len(self.snapshots) < 2:
            return []
        
        leaks = []
        current = self.snapshots[-1]['snapshot']
        baseline = self.snapshots[0]['snapshot']
        
        top_stats = current.compare_to(baseline, 'lineno')
        
        for stat in top_stats[:10]:
            size_diff_mb = stat.size_diff / 1024 / 1024
            if size_diff_mb > threshold_mb:
                leaks.append(f"{stat.traceback}: {size_diff_mb:.1f}MB increase")
        
        return leaks
    
    def stop(self):
        """Stop memory tracking"""
        if self.tracking:
            tracemalloc.stop()
            self.tracking = False


# Global profiler instance
_profiler = None


def get_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance"""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler