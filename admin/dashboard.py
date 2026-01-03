"""
Admin Dashboard for InvestiGator
Web-based administration interface for system management
"""

from fastapi import FastAPI, WebSocket, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import json

from agents.orchestrator import AgentOrchestrator, AnalysisMode
from utils.cache.cache_manager import CacheManager
from investigator.infrastructure.monitoring import MetricsCollector
from utils.ollama_client import OllamaClient
from utils.profiler import get_profiler
from models.database import Analysis, AgentResult, SystemMetric
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_


# Security
security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify admin credentials"""
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, os.environ.get("ADMIN_PASSWORD", ""))

    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return credentials.username


class AdminDashboard:
    """
    Admin dashboard for InvestiGator system management
    """

    def __init__(
        self, orchestrator: AgentOrchestrator, cache_manager: CacheManager, metrics_collector: MetricsCollector
    ):
        self.orchestrator = orchestrator
        self.cache = cache_manager
        self.metrics = metrics_collector
        self.profiler = get_profiler()

        # WebSocket connections for real-time updates
        self.websocket_connections: List[WebSocket] = []

        # Dashboard stats cache
        self.stats_cache = {}
        self.stats_cache_time = None

        # Start background tasks
        asyncio.create_task(self._update_stats_loop())

    async def _update_stats_loop(self):
        """Background task to update dashboard stats"""
        while True:
            await self._update_stats()
            await asyncio.sleep(10)  # Update every 10 seconds

    async def _update_stats(self):
        """Update dashboard statistics"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "system": await self._get_system_stats(),
            "analyses": await self._get_analysis_stats(),
            "agents": await self._get_agent_stats(),
            "cache": await self._get_cache_stats(),
            "queue": await self._get_queue_stats(),
        }

        self.stats_cache = stats
        self.stats_cache_time = datetime.now()

        # Broadcast to connected clients
        await self._broadcast_stats(stats)

    async def _get_system_stats(self) -> Dict:
        """Get system resource statistics"""
        health = self.metrics.get_system_health()

        return {
            "status": health["status"],
            "cpu_percent": health["metrics"].get("cpu_percent", 0),
            "memory_percent": health["metrics"].get("memory_percent", 0),
            "disk_usage": health["metrics"].get("disk_usage", 0),
            "uptime_hours": (datetime.now() - self.metrics.system_metrics["start_time"]).total_seconds() / 3600,
        }

    async def _get_analysis_stats(self) -> Dict:
        """Get analysis statistics"""
        stats = self.metrics.get_stats()

        # Get recent analyses from orchestrator
        recent = []
        for task_id, task in list(self.orchestrator.completed_tasks.items())[-10:]:
            recent.append(
                {
                    "task_id": task_id,
                    "symbol": task.symbol,
                    "mode": task.mode.value,
                    "status": task.status,
                    "duration": task.results.get("duration", 0) if task.results else 0,
                    "completed_at": task.results.get("completed_at", "") if task.results else "",
                }
            )

        return {
            "total": stats["total_analyses"],
            "successful": stats.get("successful_analyses", 0),
            "failed": stats.get("total_analyses", 0) - stats.get("successful_analyses", 0),
            "success_rate": stats.get("success_rate", 0),
            "active": len(self.orchestrator.active_tasks),
            "queued": self.orchestrator.task_queue.qsize(),
            "recent": recent,
        }

    async def _get_agent_stats(self) -> Dict:
        """Get agent performance statistics"""
        agent_stats = {}

        for agent_type, metrics in self.metrics.agent_metrics["executions"].items():
            perf = self.metrics.get_agent_performance(agent_type)
            agent_stats[agent_type] = {
                "executions": perf["total_executions"],
                "failures": perf["total_failures"],
                "success_rate": perf["success_rate"],
                "avg_duration": perf["average_duration"],
                "last_execution": perf["last_execution"],
            }

        return agent_stats

    async def _get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        cache_stats = await self.cache.get_stats()
        perf_stats = self.cache.get_performance_stats()

        # Aggregate stats
        total_hits = sum(s["operations"]["hits"] for s in perf_stats.values())
        total_misses = sum(s["operations"]["misses"] for s in perf_stats.values())

        return {
            "hit_rate": (total_hits / (total_hits + total_misses) * 100) if (total_hits + total_misses) > 0 else 0,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "l1_memory_mb": cache_stats.get("l1_memory_mb", 0),
            "l2_file_mb": cache_stats.get("l2_file_mb", 0),
            "l3_entries": cache_stats.get("l3_entries", 0),
        }

    async def _get_queue_stats(self) -> Dict:
        """Get task queue statistics"""
        return {
            "size": self.orchestrator.task_queue.qsize(),
            "workers": len(self.orchestrator.workers),
            "max_concurrent": self.orchestrator.max_concurrent_analyses,
        }

    async def _broadcast_stats(self, stats: Dict):
        """Broadcast stats to all connected WebSocket clients"""
        if not self.websocket_connections:
            return

        message = json.dumps({"type": "stats_update", "data": stats})

        # Send to all connected clients
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except:
                disconnected.append(websocket)

        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.remove(ws)

    async def get_dashboard_data(self) -> Dict:
        """Get complete dashboard data"""
        if not self.stats_cache or (datetime.now() - self.stats_cache_time).seconds > 30:
            await self._update_stats()

        return self.stats_cache

    async def clear_cache(self, cache_type: Optional[str] = None) -> Dict:
        """Clear cache (specific type or all)"""
        if cache_type:
            from utils.cache.cache_types import CacheType

            success = self.cache.clear_cache_type(CacheType(cache_type))
        else:
            success = self.cache.clear_all_caches()

        return {"success": success, "message": f"Cache {'cleared' if success else 'clear failed'}"}

    async def manage_agents(self, action: str, agent_type: Optional[str] = None) -> Dict:
        """Manage agent operations"""
        if action == "list":
            return {"agents": list(self.orchestrator.agents.keys()), "status": "success"}

        elif action == "restart" and agent_type:
            # Restart specific agent
            if agent_type in self.orchestrator.agents:
                old_agent = self.orchestrator.agents[agent_type]
                # Recreate agent
                self.orchestrator.agents[agent_type] = self.orchestrator._initialize_agents()[agent_type]
                return {"status": "success", "message": f"Agent {agent_type} restarted"}

        return {"status": "error", "message": "Invalid action or agent type"}

    async def manage_ollama(self, action: str, model: Optional[str] = None) -> Dict:
        """Manage Ollama models"""
        ollama_client = OllamaClient()

        async with ollama_client:
            if action == "list":
                models = await ollama_client.list_models()
                return {"models": [m["name"] for m in models], "count": len(models)}

            elif action == "pull" and model:
                # Pull new model
                success = False
                async for status in ollama_client.pull_model(model):
                    if status.get("status") == "success":
                        success = True
                        break

                return {"success": success, "message": f"Model {model} {'pulled' if success else 'pull failed'}"}

            elif action == "unload" and model:
                success = await ollama_client.unload_model(model)
                return {"success": success, "message": f"Model {model} {'unloaded' if success else 'unload failed'}"}

        return {"status": "error", "message": "Invalid action"}

    async def get_performance_profile(self, function_name: Optional[str] = None) -> Dict:
        """Get performance profiling data"""
        if function_name:
            return self.profiler.analyze_profile(function_name)
        else:
            # Get all profiles
            profiles = {}
            for func in self.profiler.profiles.keys():
                profiles[func] = self.profiler.analyze_profile(func)
            return profiles

    async def run_diagnostics(self) -> Dict:
        """Run system diagnostics"""
        diagnostics = {"timestamp": datetime.now().isoformat(), "checks": {}}

        # Check Ollama
        ollama_client = OllamaClient()
        async with ollama_client:
            ollama_healthy = await ollama_client.health_check()
            diagnostics["checks"]["ollama"] = {
                "status": "healthy" if ollama_healthy else "unhealthy",
                "models": len(await ollama_client.list_models()) if ollama_healthy else 0,
            }

        # Check cache
        cache_healthy = await self.cache.ping()
        diagnostics["checks"]["cache"] = {"status": "healthy" if cache_healthy else "unhealthy"}

        # Check database
        try:
            # Simple query to test connection
            from utils.db import get_db_session

            async with get_db_session() as session:
                result = await session.execute(select(func.count(Analysis.id)))
                count = result.scalar()
                diagnostics["checks"]["database"] = {"status": "healthy", "analyses_count": count}
        except Exception as e:
            diagnostics["checks"]["database"] = {"status": "unhealthy", "error": str(e)}

        # Check agents
        agent_status = {}
        for agent_name, agent in self.orchestrator.agents.items():
            agent_status[agent_name] = {
                "status": agent.status.value if hasattr(agent, "status") else "unknown",
                "health": "healthy",  # Would implement actual health check
            }
        diagnostics["checks"]["agents"] = agent_status

        # Overall health
        all_healthy = all(
            check.get("status") == "healthy"
            for check in diagnostics["checks"].values()
            if isinstance(check, dict) and "status" in check
        )

        diagnostics["overall_status"] = "healthy" if all_healthy else "degraded"

        return diagnostics

    async def get_logs(self, component: str, lines: int = 100) -> List[str]:
        """Get recent logs for a component"""
        log_file = f"logs/{component}.log"

        try:
            with open(log_file, "r") as f:
                all_lines = f.readlines()
                return all_lines[-lines:]
        except FileNotFoundError:
            return [f"Log file not found: {log_file}"]
        except Exception as e:
            return [f"Error reading logs: {str(e)}"]

    async def schedule_analysis(self, symbol: str, mode: str, schedule_time: Optional[datetime] = None) -> Dict:
        """Schedule an analysis for later execution"""
        # For immediate execution
        if not schedule_time or schedule_time <= datetime.now():
            task_id = await self.orchestrator.analyze(symbol, AnalysisMode[mode.upper()])
            return {"status": "submitted", "task_id": task_id, "scheduled": False}

        # For scheduled execution (would need a scheduler implementation)
        return {"status": "scheduled", "symbol": symbol, "mode": mode, "schedule_time": schedule_time.isoformat()}


def create_admin_app(
    orchestrator: AgentOrchestrator, cache_manager: CacheManager, metrics_collector: MetricsCollector
) -> FastAPI:
    """Create FastAPI admin application"""

    app = FastAPI(title="InvestiGator Admin Dashboard")
    dashboard = AdminDashboard(orchestrator, cache_manager, metrics_collector)

    # HTML template for dashboard
    @app.get("/", response_class=HTMLResponse)
    async def root():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>InvestiGator Admin Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
                .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px; }
                .stat-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .stat-value { font-size: 2em; font-weight: bold; color: #3498db; }
                .stat-label { color: #7f8c8d; margin-top: 5px; }
                .status-healthy { color: #27ae60; }
                .status-warning { color: #f39c12; }
                .status-critical { color: #e74c3c; }
                .charts { margin-top: 30px; }
                .controls { margin-top: 20px; padding: 20px; background: white; border-radius: 5px; }
                button { padding: 10px 20px; margin: 5px; border: none; border-radius: 3px; cursor: pointer; }
                .btn-primary { background: #3498db; color: white; }
                .btn-danger { background: #e74c3c; color: white; }
                .btn-success { background: #27ae60; color: white; }
                .log-viewer { background: #2c3e50; color: #1abc9c; padding: 10px; border-radius: 3px; height: 300px; overflow-y: scroll; font-family: monospace; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 InvestiGator Admin Dashboard</h1>
                <p>Agentic AI Investment Analysis System</p>
            </div>
            
            <div class="stats-grid" id="stats-grid">
                <!-- Stats will be populated here -->
            </div>
            
            <div class="controls">
                <h2>System Controls</h2>
                <button class="btn-primary" onclick="runDiagnostics()">Run Diagnostics</button>
                <button class="btn-danger" onclick="clearCache()">Clear Cache</button>
                <button class="btn-success" onclick="refreshStats()">Refresh Stats</button>
                
                <h3>Schedule Analysis</h3>
                <input type="text" id="symbol" placeholder="Symbol" />
                <select id="mode">
                    <option value="quick">Quick</option>
                    <option value="standard">Standard</option>
                    <option value="comprehensive">Comprehensive</option>
                </select>
                <button class="btn-primary" onclick="scheduleAnalysis()">Analyze</button>
            </div>
            
            <div class="charts" id="charts">
                <!-- Charts will be rendered here -->
            </div>
            
            <div class="controls">
                <h3>System Logs</h3>
                <div class="log-viewer" id="logs">
                    Loading logs...
                </div>
            </div>
            
            <script>
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'stats_update') {
                        updateStats(data.data);
                    }
                };
                
                function updateStats(stats) {
                    const grid = document.getElementById('stats-grid');
                    
                    grid.innerHTML = `
                        <div class="stat-card">
                            <div class="stat-value ${getStatusClass(stats.system.status)}">${stats.system.status.toUpperCase()}</div>
                            <div class="stat-label">System Status</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.system.cpu_percent.toFixed(1)}%</div>
                            <div class="stat-label">CPU Usage</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.system.memory_percent.toFixed(1)}%</div>
                            <div class="stat-label">Memory Usage</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.analyses.active}</div>
                            <div class="stat-label">Active Analyses</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.analyses.success_rate.toFixed(1)}%</div>
                            <div class="stat-label">Success Rate</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.cache.hit_rate.toFixed(1)}%</div>
                            <div class="stat-label">Cache Hit Rate</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.queue.size}</div>
                            <div class="stat-label">Queue Size</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.system.uptime_hours.toFixed(1)}h</div>
                            <div class="stat-label">Uptime</div>
                        </div>
                    `;
                }
                
                function getStatusClass(status) {
                    if (status === 'healthy') return 'status-healthy';
                    if (status === 'warning') return 'status-warning';
                    return 'status-critical';
                }
                
                async function runDiagnostics() {
                    const response = await fetch('/api/diagnostics');
                    const data = await response.json();
                    alert('Diagnostics: ' + data.overall_status);
                }
                
                async function clearCache() {
                    if (confirm('Clear all cache?')) {
                        const response = await fetch('/api/cache/clear', {method: 'POST'});
                        const data = await response.json();
                        alert(data.message);
                    }
                }
                
                async function scheduleAnalysis() {
                    const symbol = document.getElementById('symbol').value;
                    const mode = document.getElementById('mode').value;
                    
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({symbol, mode})
                    });
                    
                    const data = await response.json();
                    alert(`Analysis ${data.status}: ${data.task_id || ''}`);
                }
                
                async function refreshStats() {
                    const response = await fetch('/api/stats');
                    const data = await response.json();
                    updateStats(data);
                }
                
                async function loadLogs() {
                    const response = await fetch('/api/logs/system');
                    const logs = await response.json();
                    document.getElementById('logs').innerText = logs.join('');
                }
                
                // Initial load
                refreshStats();
                loadLogs();
                setInterval(loadLogs, 5000);  // Refresh logs every 5 seconds
            </script>
        </body>
        </html>
        """

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""
        await websocket.accept()
        dashboard.websocket_connections.append(websocket)

        try:
            while True:
                await websocket.receive_text()
        except:
            dashboard.websocket_connections.remove(websocket)

    @app.get("/api/stats")
    async def get_stats(username: str = Depends(verify_credentials)):
        """Get dashboard statistics"""
        return await dashboard.get_dashboard_data()

    @app.get("/api/diagnostics")
    async def run_diagnostics(username: str = Depends(verify_credentials)):
        """Run system diagnostics"""
        return await dashboard.run_diagnostics()

    @app.post("/api/cache/clear")
    async def clear_cache(cache_type: Optional[str] = None, username: str = Depends(verify_credentials)):
        """Clear cache"""
        return await dashboard.clear_cache(cache_type)

    @app.post("/api/analyze")
    async def schedule_analysis(symbol: str, mode: str, username: str = Depends(verify_credentials)):
        """Schedule an analysis"""
        return await dashboard.schedule_analysis(symbol, mode)

    @app.get("/api/logs/{component}")
    async def get_logs(component: str, lines: int = 100, username: str = Depends(verify_credentials)):
        """Get component logs"""
        return await dashboard.get_logs(component, lines)

    @app.get("/api/agents")
    async def manage_agents(
        action: str = "list", agent_type: Optional[str] = None, username: str = Depends(verify_credentials)
    ):
        """Manage agents"""
        return await dashboard.manage_agents(action, agent_type)

    @app.get("/api/ollama")
    async def manage_ollama(
        action: str = "list", model: Optional[str] = None, username: str = Depends(verify_credentials)
    ):
        """Manage Ollama models"""
        return await dashboard.manage_ollama(action, model)

    @app.get("/api/profile")
    async def get_profile(function: Optional[str] = None, username: str = Depends(verify_credentials)):
        """Get performance profile"""
        return await dashboard.get_performance_profile(function)

    return app
