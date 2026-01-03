"""
Monitoring and Metrics Collection
Comprehensive monitoring for the InvestiGator system
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import prometheus_client as prom
from prometheus_client import Counter, Histogram, Gauge, Summary
import logging
import psutil
import aiohttp


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class PerformanceSnapshot:
    """System performance snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    active_agents: int
    queue_size: int
    cache_hit_rate: float
    average_latency: float


class MetricsCollector:
    """
    Centralized metrics collection for monitoring
    """
    
    def __init__(self, export_interval: int = 60):
        self.export_interval = export_interval
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Prometheus metrics
        self._init_prometheus_metrics()
        
        # Internal metrics storage
        self.metrics_buffer: List[MetricPoint] = []
        self.max_buffer_size = 10000
        
        # Performance tracking
        self.performance_history: List[PerformanceSnapshot] = []
        self.max_history_size = 1440  # 24 hours at 1-minute intervals
        
        # Agent metrics
        self.agent_metrics = {
            'executions': {},
            'failures': {},
            'durations': {},
            'last_execution': {}
        }
        
        # System metrics
        self.system_metrics = {
            'start_time': datetime.now(),
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Background tasks
        self.tasks = []
        self.running = False
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        from prometheus_client import REGISTRY
        
        # Helper function to get or create metric
        def get_or_create_metric(metric_class, name, *args, **kwargs):
            try:
                return metric_class(name, *args, **kwargs)
            except ValueError:
                # Metric already registered, get the existing one
                return REGISTRY._names_to_collectors[name]
        
        # Counters
        self.analysis_counter = get_or_create_metric(
            Counter,
            'investigator_analyses_total',
            'Total number of analyses performed',
            ['symbol', 'mode', 'status']
        )
        
        self.agent_execution_counter = get_or_create_metric(
            Counter,
            'investigator_agent_executions_total',
            'Total agent executions',
            ['agent_type', 'status']
        )
        
        self.cache_counter = get_or_create_metric(
            Counter,
            'investigator_cache_operations_total',
            'Cache operations',
            ['operation', 'tier']
        )
        
        # Histograms
        self.analysis_duration_histogram = get_or_create_metric(
            Histogram,
            'investigator_analysis_duration_seconds',
            'Analysis duration in seconds',
            ['mode'],
            buckets=(1, 5, 10, 30, 60, 120, 300, 600)
        )
        
        self.agent_duration_histogram = get_or_create_metric(
            Histogram,
            'investigator_agent_duration_seconds',
            'Agent execution duration',
            ['agent_type'],
            buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60)
        )
        
        # Gauges
        self.active_agents_gauge = get_or_create_metric(
            Gauge,
            'investigator_active_agents',
            'Number of active agents'
        )
        
        self.queue_size_gauge = get_or_create_metric(
            Gauge,
            'investigator_queue_size',
            'Analysis queue size'
        )
        
        self.cache_hit_rate_gauge = get_or_create_metric(
            Gauge,
            'investigator_cache_hit_rate',
            'Cache hit rate percentage'
        )
        
        self.system_cpu_gauge = get_or_create_metric(
            Gauge,
            'investigator_system_cpu_percent',
            'System CPU usage percentage'
        )
        
        self.system_memory_gauge = get_or_create_metric(
            Gauge,
            'investigator_system_memory_percent',
            'System memory usage percentage'
        )
        
        # Summaries
        self.llm_latency_summary = get_or_create_metric(
            Summary,
            'investigator_llm_latency_seconds',
            'LLM response latency',
            ['model']
        )
    
    async def start(self):
        """Start metrics collection"""
        self.running = True
        
        # Start collection tasks
        self.tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._export_metrics()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        self.logger.info("Metrics collector started")
    
    async def stop(self):
        """Stop metrics collection"""
        self.running = False
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.logger.info("Metrics collector stopped")
    
    def record_analysis(self, symbol: str, mode: str, status: str, duration: float):
        """Record analysis completion"""
        # Update Prometheus metrics
        self.analysis_counter.labels(symbol=symbol, mode=mode, status=status).inc()
        self.analysis_duration_histogram.labels(mode=mode).observe(duration)
        
        # Update internal metrics
        self.system_metrics['total_analyses'] += 1
        if status == 'success':
            self.system_metrics['successful_analyses'] += 1
        else:
            self.system_metrics['failed_analyses'] += 1
        
        # Add to buffer
        self._add_metric(MetricPoint(
            name='analysis_completed',
            value=1,
            labels={'symbol': symbol, 'mode': mode, 'status': status},
            metric_type=MetricType.COUNTER
        ))
    
    def record_agent_execution(self, agent_type: str, duration: float, status: str = 'success'):
        """Record agent execution"""
        # Update Prometheus metrics
        self.agent_execution_counter.labels(agent_type=agent_type, status=status).inc()
        self.agent_duration_histogram.labels(agent_type=agent_type).observe(duration)
        
        # Update internal metrics
        if agent_type not in self.agent_metrics['executions']:
            self.agent_metrics['executions'][agent_type] = 0
            self.agent_metrics['failures'][agent_type] = 0
            self.agent_metrics['durations'][agent_type] = []
        
        self.agent_metrics['executions'][agent_type] += 1
        if status == 'failure':
            self.agent_metrics['failures'][agent_type] += 1
        
        self.agent_metrics['durations'][agent_type].append(duration)
        self.agent_metrics['last_execution'][agent_type] = datetime.now()
        
        # Keep only recent durations
        if len(self.agent_metrics['durations'][agent_type]) > 100:
            self.agent_metrics['durations'][agent_type] = \
                self.agent_metrics['durations'][agent_type][-100:]
    
    def record_agent_failure(self, agent_type: str):
        """Record agent failure"""
        self.record_agent_execution(agent_type, 0, 'failure')
    
    def record_cache_operation(self, operation: str, tier: str, hit: bool = True):
        """Record cache operation"""
        # Update Prometheus metrics
        self.cache_counter.labels(operation=operation, tier=tier).inc()
        
        # Update internal metrics
        if hit:
            self.system_metrics['cache_hits'] += 1
        else:
            self.system_metrics['cache_misses'] += 1
        
        # Update hit rate gauge
        total = self.system_metrics['cache_hits'] + self.system_metrics['cache_misses']
        if total > 0:
            hit_rate = (self.system_metrics['cache_hits'] / total) * 100
            self.cache_hit_rate_gauge.set(hit_rate)
    
    def record_llm_latency(self, model: str, latency: float):
        """Record LLM response latency"""
        self.llm_latency_summary.labels(model=model).observe(latency)
        
        self._add_metric(MetricPoint(
            name='llm_latency',
            value=latency,
            labels={'model': model},
            metric_type=MetricType.HISTOGRAM
        ))
    
    def update_active_agents(self, count: int):
        """Update active agents count"""
        self.active_agents_gauge.set(count)
    
    def update_queue_size(self, size: int):
        """Update queue size"""
        self.queue_size_gauge.set(size)
    
    def record_orchestrator_stats(self, stats: Dict):
        """Record orchestrator statistics"""
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self._add_metric(MetricPoint(
                    name=f'orchestrator_{key}',
                    value=value,
                    metric_type=MetricType.GAUGE
                ))
    
    def _add_metric(self, metric: MetricPoint):
        """Add metric to buffer"""
        self.metrics_buffer.append(metric)
        
        # Trim buffer if too large
        if len(self.metrics_buffer) > self.max_buffer_size:
            self.metrics_buffer = self.metrics_buffer[-self.max_buffer_size:]
    
    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while self.running:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                net_io = psutil.net_io_counters()
                
                # Update Prometheus gauges
                self.system_cpu_gauge.set(cpu_percent)
                self.system_memory_gauge.set(memory.percent)
                
                # Create performance snapshot
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    disk_usage=disk.percent,
                    network_io={
                        'bytes_sent': net_io.bytes_sent,
                        'bytes_recv': net_io.bytes_recv
                    },
                    active_agents=self.active_agents_gauge._value.get(),
                    queue_size=self.queue_size_gauge._value.get(),
                    cache_hit_rate=self.cache_hit_rate_gauge._value.get(),
                    average_latency=self._calculate_average_latency()
                )
                
                self.performance_history.append(snapshot)
                
                # Trim history
                if len(self.performance_history) > self.max_history_size:
                    self.performance_history = self.performance_history[-self.max_history_size:]
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)
    
    async def _export_metrics(self):
        """Export metrics to external systems"""
        while self.running:
            try:
                await asyncio.sleep(self.export_interval)
                
                # Export to file (for backup)
                await self._export_to_file()
                
                # Export to external monitoring system if configured
                # await self._export_to_prometheus_pushgateway()
                
            except Exception as e:
                self.logger.error(f"Error exporting metrics: {e}")
    
    async def _export_to_file(self):
        """Export metrics to JSON file"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': self.system_metrics,
            'agent_metrics': self._serialize_agent_metrics(),
            'recent_metrics': [asdict(m) for m in self.metrics_buffer[-100:]],
            'performance_snapshot': asdict(self.performance_history[-1]) if self.performance_history else None
        }
        
        filename = Path("metrics") / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            filename.parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to export metrics to file: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old metrics data"""
        while self.running:
            await asyncio.sleep(3600)  # Run hourly
            
            # Clean up old metrics files
            import os
            import glob
            
            try:
                # Keep only last 7 days of metrics files
                cutoff_date = datetime.now() - timedelta(days=7)
                pattern = "metrics/metrics_*.json"
                
                for filepath in glob.glob(pattern):
                    file_date_str = os.path.basename(filepath)[8:16]  # Extract date
                    try:
                        file_date = datetime.strptime(file_date_str, '%Y%m%d')
                        if file_date < cutoff_date:
                            os.remove(filepath)
                            self.logger.info(f"Removed old metrics file: {filepath}")
                    except ValueError:
                        continue
                        
            except Exception as e:
                self.logger.error(f"Error cleaning up old metrics: {e}")
    
    def _calculate_average_latency(self) -> float:
        """Calculate average latency across all agents"""
        total_duration = 0
        total_count = 0
        
        for agent_type, durations in self.agent_metrics['durations'].items():
            if durations:
                total_duration += sum(durations)
                total_count += len(durations)
        
        return total_duration / total_count if total_count > 0 else 0
    
    def _serialize_agent_metrics(self) -> Dict:
        """Serialize agent metrics for export"""
        serialized = {}
        
        for agent_type in self.agent_metrics['executions'].keys():
            durations = self.agent_metrics['durations'].get(agent_type, [])
            
            serialized[agent_type] = {
                'executions': self.agent_metrics['executions'].get(agent_type, 0),
                'failures': self.agent_metrics['failures'].get(agent_type, 0),
                'average_duration': sum(durations) / len(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'last_execution': self.agent_metrics['last_execution'].get(agent_type, '').isoformat() if agent_type in self.agent_metrics['last_execution'] else None
            }
        
        return serialized
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        uptime = (datetime.now() - self.system_metrics['start_time']).total_seconds()
        
        return {
            'uptime_seconds': uptime,
            'total_analyses': self.system_metrics['total_analyses'],
            'success_rate': (self.system_metrics['successful_analyses'] / 
                           self.system_metrics['total_analyses'] * 100) 
                           if self.system_metrics['total_analyses'] > 0 else 0,
            'cache_hit_rate': (self.system_metrics['cache_hits'] / 
                             (self.system_metrics['cache_hits'] + self.system_metrics['cache_misses']) * 100)
                             if (self.system_metrics['cache_hits'] + self.system_metrics['cache_misses']) > 0 else 0,
            'agent_stats': self._serialize_agent_metrics(),
            'current_performance': asdict(self.performance_history[-1]) if self.performance_history else None
        }
    
    def get_agent_performance(self, agent_type: str) -> Dict:
        """Get performance metrics for specific agent"""
        if agent_type not in self.agent_metrics['executions']:
            return {'status': 'no_data'}
        
        durations = self.agent_metrics['durations'].get(agent_type, [])
        executions = self.agent_metrics['executions'].get(agent_type, 0)
        failures = self.agent_metrics['failures'].get(agent_type, 0)
        
        return {
            'total_executions': executions,
            'total_failures': failures,
            'success_rate': ((executions - failures) / executions * 100) if executions > 0 else 0,
            'average_duration': sum(durations) / len(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'last_execution': self.agent_metrics['last_execution'].get(agent_type, '').isoformat() if agent_type in self.agent_metrics['last_execution'] else None,
            'recent_durations': durations[-10:] if durations else []
        }
    
    def get_system_health(self) -> Dict:
        """Get system health status"""
        if not self.performance_history:
            return {'status': 'unknown', 'message': 'No performance data available'}
        
        latest = self.performance_history[-1]
        
        # Define health thresholds
        health_status = 'healthy'
        issues = []
        
        if latest.cpu_percent > 80:
            health_status = 'warning'
            issues.append(f"High CPU usage: {latest.cpu_percent:.1f}%")
        
        if latest.memory_percent > 85:
            health_status = 'warning'
            issues.append(f"High memory usage: {latest.memory_percent:.1f}%")
        
        if latest.disk_usage > 90:
            health_status = 'critical'
            issues.append(f"Critical disk usage: {latest.disk_usage:.1f}%")
        
        if latest.average_latency > 30:
            health_status = 'warning'
            issues.append(f"High average latency: {latest.average_latency:.1f}s")
        
        if latest.cache_hit_rate < 50:
            health_status = 'warning'
            issues.append(f"Low cache hit rate: {latest.cache_hit_rate:.1f}%")
        
        return {
            'status': health_status,
            'issues': issues,
            'metrics': {
                'cpu_percent': latest.cpu_percent,
                'memory_percent': latest.memory_percent,
                'disk_usage': latest.disk_usage,
                'active_agents': latest.active_agents,
                'queue_size': latest.queue_size,
                'cache_hit_rate': latest.cache_hit_rate,
                'average_latency': latest.average_latency
            }
        }


class AlertManager:
    """
    Manages alerts based on metrics thresholds
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Alert thresholds
        self.thresholds = {
            'cpu_critical': 90,
            'cpu_warning': 75,
            'memory_critical': 90,
            'memory_warning': 80,
            'disk_critical': 95,
            'disk_warning': 85,
            'latency_critical': 60,
            'latency_warning': 30,
            'error_rate_critical': 20,
            'error_rate_warning': 10
        }
        
        # Alert history
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_history: List[Dict] = []
        
        # Alert channels
        self.alert_channels = []
    
    def add_alert_channel(self, channel: Callable):
        """Add alert notification channel"""
        self.alert_channels.append(channel)
    
    async def check_alerts(self):
        """Check metrics against thresholds and raise alerts"""
        health = self.metrics.get_system_health()
        stats = self.metrics.get_stats()
        
        # Check CPU
        if health['metrics']['cpu_percent'] > self.thresholds['cpu_critical']:
            await self._raise_alert('cpu_critical', f"CPU usage critical: {health['metrics']['cpu_percent']:.1f}%", 'critical')
        elif health['metrics']['cpu_percent'] > self.thresholds['cpu_warning']:
            await self._raise_alert('cpu_warning', f"CPU usage high: {health['metrics']['cpu_percent']:.1f}%", 'warning')
        else:
            await self._clear_alert('cpu_critical')
            await self._clear_alert('cpu_warning')
        
        # Check Memory
        if health['metrics']['memory_percent'] > self.thresholds['memory_critical']:
            await self._raise_alert('memory_critical', f"Memory usage critical: {health['metrics']['memory_percent']:.1f}%", 'critical')
        elif health['metrics']['memory_percent'] > self.thresholds['memory_warning']:
            await self._raise_alert('memory_warning', f"Memory usage high: {health['metrics']['memory_percent']:.1f}%", 'warning')
        else:
            await self._clear_alert('memory_critical')
            await self._clear_alert('memory_warning')
        
        # Check error rate
        error_rate = 100 - stats.get('success_rate', 100)
        if error_rate > self.thresholds['error_rate_critical']:
            await self._raise_alert('error_rate_critical', f"Error rate critical: {error_rate:.1f}%", 'critical')
        elif error_rate > self.thresholds['error_rate_warning']:
            await self._raise_alert('error_rate_warning', f"Error rate high: {error_rate:.1f}%", 'warning')
        else:
            await self._clear_alert('error_rate_critical')
            await self._clear_alert('error_rate_warning')
    
    async def _raise_alert(self, alert_id: str, message: str, severity: str):
        """Raise an alert"""
        if alert_id not in self.active_alerts:
            alert = {
                'id': alert_id,
                'message': message,
                'severity': severity,
                'timestamp': datetime.now(),
                'notified': False
            }
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Notify channels
            for channel in self.alert_channels:
                try:
                    await channel(alert)
                    alert['notified'] = True
                except Exception as e:
                    self.logger.error(f"Failed to send alert to channel: {e}")
            
            self.logger.warning(f"Alert raised: {message}")
    
    async def _clear_alert(self, alert_id: str):
        """Clear an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert['cleared_at'] = datetime.now()
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert cleared: {alert_id}")
