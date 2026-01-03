"""
Agent Orchestrator
Coordinates and manages the execution of multiple specialized agents

DEPRECATED: This orchestrator is maintained for backwards compatibility only.
New code should use Victor StateGraph workflows instead:

    from victor_invest.workflows import build_graph_for_mode, AnalysisMode

    workflow = build_graph_for_mode(AnalysisMode.STANDARD)
    result = await workflow.invoke({"symbol": "AAPL", "mode": AnalysisMode.STANDARD})

The Victor workflows provide:
- Declarative state graph definition
- Built-in checkpointing and retry
- Better error handling
- Multi-provider LLM support

See victor_invest/workflows/graphs.py for the new architecture.
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from investigator.config import get_config
from investigator.domain.agents.base import AgentResult, AgentTask, AnalysisType
from investigator.domain.agents.base import TaskStatus as AgentStatus
from investigator.domain.agents.fundamental import FundamentalAnalysisAgent
from investigator.domain.agents.market_context import ETFMarketContextAgent
from investigator.domain.agents.sec import SECAnalysisAgent
from investigator.domain.agents.synthesis import SynthesisAgent
from investigator.domain.agents.symbol_update import SymbolUpdateAgent
from investigator.domain.agents.technical import TechnicalAnalysisAgent
from investigator.infrastructure.cache.cache_manager import CacheManager
from investigator.infrastructure.database.market_data import get_market_data_fetcher
from investigator.infrastructure.llm.ollama import OllamaClient
from investigator.infrastructure.llm.pool import create_resource_aware_pool
from investigator.infrastructure.events import EventBus
from investigator.infrastructure.monitoring import MetricsCollector


class AnalysisMode(Enum):
    """Analysis execution modes"""

    QUICK = "quick"  # Technical only
    STANDARD = "standard"  # Technical + Fundamental
    COMPREHENSIVE = "comprehensive"  # All agents
    CUSTOM = "custom"  # User-defined agent selection


class Priority(Enum):
    """Task priority levels"""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class OrchestrationTask:
    """Task for orchestration"""

    id: str
    symbol: str
    mode: AnalysisMode
    agents: List[str]
    priority: Priority = Priority.NORMAL
    deadline: Optional[datetime] = None
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    results: Dict = field(default_factory=dict)

    def __lt__(self, other):
        """Compare tasks by priority for queue ordering"""
        if isinstance(other, OrchestrationTask):
            return self.priority.value < other.priority.value
        return NotImplemented


class AgentOrchestrator:
    """
    Master orchestrator that coordinates multiple agents for comprehensive analysis

    Control-flow sketch:

        ┌─────────────┐   enqueue()    ┌─────────────────────┐   spawn tasks   ┌─────────────┐
        │ CLI / API   │ ──────────────▶│ AgentOrchestrator   │ ───────────────▶│ Agent Worker│
        └─────┬───────┘                │  • priority queue   │                 │  coroutine  │
              │                        │  • dep graph (DAG)  │◀────────────────└────┬────────┘
              │                        │  • cache & metrics  │    results/emit       │
              │                        └──────────┬──────────┘                      │
              │                                   │                                 │
              │                                   ▼                                 │
              │                          ┌────────────────┐                         │
              └──────────────────────────│  Event Bus     │◀────────────────────────┘
                                         └────────────────┘

    Workers respect agent dependencies (SEC/Technical/Fundamental → Synthesis) and fan-in results.
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        metrics_collector: MetricsCollector,
        max_concurrent_analyses: int = 5,
        max_concurrent_agents: int = 10,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.cache_manager = cache_manager
        self.metrics = metrics_collector
        self.event_bus = EventBus()
        self._logger = logging.getLogger(__name__)

        # Store config for pool creation
        self._config = get_config()
        self.ollama_pool = None
        self.ollama_client = None  # Will be set to pool in start()
        self.symbol_classification_cache: Dict[str, bool] = {}
        try:
            self.market_data_fetcher = get_market_data_fetcher(self._config)
        except Exception as e:
            self.logger.warning("ETF detection disabled (market data fetcher init failed): %s", e, exc_info=True)
            self.market_data_fetcher = None

        self.max_concurrent_analyses = max_concurrent_analyses
        self.max_concurrent_agents = max_concurrent_agents

        # Agents will be initialized in start() after pool is created
        self.agents = {}

        # Task management
        self.task_queue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, OrchestrationTask] = {}
        self.completed_tasks: Dict[str, OrchestrationTask] = {}
        self.completed_analyses: Dict[str, Dict] = {}  # For storing analysis results

        # Execution control (already set in __init__ parameters)
        self.agent_semaphore = asyncio.Semaphore(self.max_concurrent_agents)

        # Dependency graph for agent coordination
        self.dependency_graph = self._build_dependency_graph()

        # Performance tracking
        self.performance_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_duration": 0,
            "cache_hits": 0,
        }

        # Start background workers
        self.workers = []
        self._background_tasks: List[asyncio.Task] = []
        self.running = False

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all available agents"""
        agents = {
            "sec": SECAnalysisAgent("sec_agent_1", self.ollama_client, self.event_bus, self.cache_manager),
            "technical": TechnicalAnalysisAgent("tech_agent_1", self.ollama_client, self.event_bus, self.cache_manager),
            "fundamental": FundamentalAnalysisAgent(
                "fund_agent_1", self.ollama_client, self.event_bus, self.cache_manager
            ),
            "symbol_update": SymbolUpdateAgent(
                "symbol_update_agent_1", self.ollama_client, self.event_bus, self.cache_manager
            ),
            "market_context": ETFMarketContextAgent(
                "market_context_agent_1", self.ollama_client, self.event_bus, self.cache_manager
            ),
            "synthesis": SynthesisAgent("synth_agent_1", self.ollama_client, self.event_bus, self.cache_manager),
        }

        return agents

    def _build_dependency_graph(self) -> nx.DiGraph:
        """
        Build dependency graph for agent coordination

        Dependency structure:
        - Level 0 (no dependencies): sec  # Fetches raw SEC CompanyFacts data ONCE
        - Level 1 (depends on sec): fundamental, technical, market_context
        - Level 2 (depends on fundamental): symbol_update  # Updates symbol table with metrics
        - Level 3 (depends on all): synthesis

        This ensures SEC data is fetched once and cached, then reused by all agents.
        Eliminates redundant SEC API calls.
        """
        G = nx.DiGraph()

        # Add all agent nodes
        G.add_nodes_from(["sec", "technical", "fundamental", "symbol_update", "market_context", "synthesis"])

        # Add dependency edges
        # Level 1 agents depend on SEC data being fetched and cached first
        G.add_edge("sec", "fundamental")
        G.add_edge("sec", "technical")
        G.add_edge("sec", "market_context")

        # Level 2: symbol_update depends on SEC and fundamental data
        G.add_edge("sec", "symbol_update")
        G.add_edge("fundamental", "symbol_update")

        # Synthesis depends on all data-gathering agents
        G.add_edge("sec", "synthesis")
        G.add_edge("technical", "synthesis")
        G.add_edge("fundamental", "synthesis")
        G.add_edge("market_context", "synthesis")
        G.add_edge("symbol_update", "synthesis")  # Synthesis waits for symbol update

        return G

    async def start(self):
        """
        Start the orchestrator and worker tasks

        Resolves Technical Debt Issue 1.2 (HIGH) - Pool resource leak
        """
        self.running = True

        try:
            # Create resource-aware Ollama pool
            self.ollama_pool = create_resource_aware_pool(self._config)
            await self.ollama_pool.__aenter__()
            await self.ollama_pool.initialize_servers()

            # Validate pool initialization
            pool_status = await self.ollama_pool.get_pool_status()

            if pool_status.get("available_servers", 0) == 0:
                raise RuntimeError(
                    "No Ollama servers available. "
                    f"Total servers: {pool_status.get('total_servers', 0)}, "
                    f"Available: {pool_status.get('available_servers', 0)}"
                )

            self.ollama_client = self.ollama_pool  # Agents use this

            # Log pool status
            self.logger.info(
                f"Ollama pool initialized: {pool_status['available_servers']}/{pool_status['total_servers']} servers available, "
                f"{pool_status['total_capacity_gb']}GB total capacity"
            )

            # Initialize agents now that pool is validated
            self.agents = self._initialize_agents()

            # Start EventBus processor (Fix: EventBus pipeline was dead)
            await self.event_bus.start()
            self.logger.info("Event bus started - monitoring and metrics active")

        except Exception as e:
            # Clean up pool if initialization fails
            self.logger.error(f"Failed to initialize orchestrator: {e}")

            # Stop event bus if it was started
            if self.event_bus and self.event_bus.running:
                try:
                    await self.event_bus.stop()
                except Exception as bus_cleanup_error:
                    self.logger.error(f"Error stopping event bus: {bus_cleanup_error}")

            if self.ollama_pool:
                try:
                    await self.ollama_pool.__aexit__(None, None, None)
                except Exception as cleanup_error:
                    self.logger.error(f"Error cleaning up pool: {cleanup_error}")
            raise

        # Start worker tasks
        for i in range(self.max_concurrent_analyses):
            worker = asyncio.create_task(self._worker(f"worker_{i}"))
            self.workers.append(worker)

        # Start event handler
        event_handler = asyncio.create_task(self._handle_events(), name="orchestrator-event-handler")
        self._background_tasks.append(event_handler)

        # Start metrics reporter
        metrics_reporter = asyncio.create_task(self._report_metrics(), name="orchestrator-metrics-reporter")
        self._background_tasks.append(metrics_reporter)

        # Phase 4.1: Start cache cleanup service (TTL enforcement)
        if self.cache_manager:
            try:
                await self.cache_manager.start_cleanup_service(interval_seconds=3600)
                self.logger.info("Cache cleanup service started (1 hour interval)")
            except Exception as e:
                self.logger.warning(f"Failed to start cache cleanup service: {e}")

        self.logger.info(f"Orchestrator started with {len(self.workers)} workers")

    async def stop(self):
        """Stop the orchestrator gracefully"""
        self.running = False

        # Stop EventBus processor (Fix: Ensure events are flushed before shutdown)
        if self.event_bus and self.event_bus.running:
            try:
                await self.event_bus.stop()
                self.logger.info("Event bus stopped - all events flushed")
            except Exception as e:
                self.logger.warning(f"Error stopping event bus: {e}")

        # Phase 4.1: Stop cache cleanup service
        if self.cache_manager:
            try:
                await self.cache_manager.stop_cleanup_service()
            except Exception as e:
                self.logger.warning(f"Error stopping cleanup service: {e}")

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Cancel background tasks (event handler, metrics reporter)
        for task in self._background_tasks:
            task.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Cleanup Ollama pool
        if self.ollama_pool:
            await self.ollama_pool.__aexit__(None, None, None)
            self.logger.info("Ollama pool closed")

        self.logger.info("Orchestrator stopped")

    async def analyze(
        self,
        symbol: str,
        mode: AnalysisMode = AnalysisMode.COMPREHENSIVE,
        priority: Priority = Priority.NORMAL,
        **kwargs,
    ) -> str:
        """
        Submit a symbol for analysis

        Args:
            symbol: Stock symbol to analyze
            mode: Analysis mode (quick/standard/comprehensive/custom)
            priority: Task priority
            **kwargs: Additional parameters for agents

        Returns:
            Task ID for tracking
        """
        kwargs = dict(kwargs)

        is_etf = self._is_etf(symbol)
        agents = self._get_agents_for_mode(symbol, mode, kwargs.get("custom_agents", []), is_etf)
        kwargs["is_etf"] = is_etf

        if is_etf:
            self.logger.info(
                "Symbol %s classified as ETF; limiting analysis to %s",
                symbol,
                agents,
            )

        # Create orchestration task
        task_id = f"{symbol}_{datetime.now().timestamp()}"

        task = OrchestrationTask(
            id=task_id,
            symbol=symbol,
            mode=mode,
            agents=agents,
            priority=priority,
            deadline=kwargs.get("deadline"),
            metadata=kwargs,
        )

        # Add to queue with priority
        await self.task_queue.put((priority.value, task))

        self.logger.info(f"Analysis task {task_id} queued for {symbol} with {mode.value} mode")

        return task_id

    async def analyze_batch(
        self, symbols: List[str], mode: AnalysisMode = AnalysisMode.STANDARD, priority: Priority = Priority.NORMAL
    ) -> List[str]:
        """Submit multiple symbols for analysis"""
        task_ids = []

        for symbol in symbols:
            task_id = await self.analyze(symbol, mode, priority)
            task_ids.append(task_id)

        return task_ids

    async def analyze_peer_group(
        self, target: str, peers: List[str], mode: AnalysisMode = AnalysisMode.COMPREHENSIVE
    ) -> str:
        """Analyze a target company and its peers"""
        # Create tasks for all companies
        all_symbols = [target] + peers
        task_ids = await self.analyze_batch(all_symbols, mode, Priority.HIGH)

        # Create peer comparison task
        comparison_task_id = f"peer_comparison_{target}_{datetime.now().timestamp()}"

        comparison_task = OrchestrationTask(
            id=comparison_task_id,
            symbol=target,
            mode=AnalysisMode.CUSTOM,
            agents=["synthesis"],
            priority=Priority.HIGH,
            dependencies=set(task_ids),
            metadata={
                "analysis_type": "peer_comparison",
                "target": target,
                "peers": peers,
                "component_tasks": task_ids,
            },
        )

        await self.task_queue.put((Priority.HIGH.value, comparison_task))

        return comparison_task_id

    async def get_status(self, task_id: str) -> Dict:
        """Get status of an analysis task"""
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                "status": "completed",
                "symbol": task.symbol,
                "mode": task.mode.value,
                "results": task.results,
                "duration": (datetime.now() - task.created_at).total_seconds(),
            }
        elif task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "status": "processing",
                "symbol": task.symbol,
                "mode": task.mode.value,
                "agents_completed": len([a for a in task.results if "result" in task.results.get(a, {})]),
                "total_agents": len(task.agents),
            }
        else:
            # Check if in queue
            queue_items = []
            while not self.task_queue.empty():
                item = await self.task_queue.get()
                queue_items.append(item)
                if item[1].id == task_id:
                    # Put items back
                    for queue_item in queue_items:
                        await self.task_queue.put(queue_item)
                    return {
                        "status": "queued",
                        "symbol": item[1].symbol,
                        "mode": item[1].mode.value,
                        "position": len(queue_items),
                    }

            # Put items back
            for queue_item in queue_items:
                await self.task_queue.put(queue_item)

            return {"status": "not_found", "task_id": task_id}

    async def get_results(self, task_id: str, wait: bool = False, timeout: int = 300) -> Optional[Dict]:
        """
        Get results of an analysis task

        Args:
            task_id: Task ID to retrieve
            wait: Whether to wait for completion
            timeout: Maximum wait time in seconds

        Returns:
            Analysis results or None if not ready
        """
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].results

        if not wait:
            return None

        # Wait for completion
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].results

            await asyncio.sleep(1)

        return None

    async def _worker(self, worker_id: str):
        """Worker task that processes analysis tasks from queue"""
        self.logger.info(f"Worker {worker_id} started")

        while self.running:
            try:
                # Get task from queue with timeout
                priority, task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                # Check dependencies
                # Resolves Technical Debt Issue 1.3 (HIGH) - Task dependency race condition
                if task.dependencies:
                    missing_deps = [d for d in task.dependencies if d not in self.completed_tasks]

                    if missing_deps:
                        # Track dependency wait count
                        if not hasattr(task, "dependency_wait_count"):
                            task.dependency_wait_count = 0

                        task.dependency_wait_count += 1

                        # Max 100 retries = 50 seconds of waiting
                        if task.dependency_wait_count > 100:
                            self.logger.error(
                                f"Task {task.id} dependencies never completed: {missing_deps}. "
                                f"Waited {task.dependency_wait_count * 0.5}s. Marking as failed."
                            )
                            task.status = "failed"
                            task.results = {"error": f"Dependencies timeout: {missing_deps}"}
                            self.completed_tasks[task.id] = task
                            if task.id in self.active_tasks:
                                del self.active_tasks[task.id]
                            continue

                        # Re-queue task
                        await self.task_queue.put((priority, task))
                        await asyncio.sleep(0.5)
                        continue

                # Process task
                self.active_tasks[task.id] = task

                try:
                    results = await self._process_task(task)
                    task.results = results
                    task.status = "completed"

                    # Move to completed
                    self.completed_tasks[task.id] = task
                    del self.active_tasks[task.id]

                    # Update stats
                    self.performance_stats["successful_analyses"] += 1

                    # Emit completion event
                    await self.event_bus.emit(
                        "analysis_completed", {"task_id": task.id, "symbol": task.symbol, "mode": task.mode.value}
                    )

                except Exception as e:
                    self.logger.error(f"Task {task.id} failed: {e}")
                    task.status = "failed"
                    task.results = {"error": str(e)}

                    # Move to completed (with error)
                    self.completed_tasks[task.id] = task
                    del self.active_tasks[task.id]

                    # Update stats
                    self.performance_stats["failed_analyses"] += 1

                finally:
                    self.performance_stats["total_analyses"] += 1

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} encountered unexpected error: {e}", exc_info=True)

    async def _process_task(self, task: OrchestrationTask) -> Dict:
        """Process a single orchestration task"""
        start_time = datetime.now()
        is_etf = bool(task.metadata.get("is_etf", False))
        if is_etf:
            self.logger.info(
                "Task %s (%s): ETF detected; SEC and Fundamental agents were omitted from the plan.",
                task.id,
                task.symbol,
            )
        results = {
            "task_id": task.id,
            "symbol": task.symbol,
            "mode": task.mode.value,
            "started_at": start_time.isoformat(),
        }

        # Special handling for peer comparison
        if task.metadata.get("analysis_type") == "peer_comparison":
            return await self._process_peer_comparison(task)

        # Get execution order based on dependencies
        execution_order = self._get_execution_order(task.agents)

        # Execute agents in parallel where possible
        agent_results: Dict[str, Any] = {}
        execution_trace: List[Dict[str, Any]] = []

        for step_index, level in enumerate(execution_order, start=1):
            step_name = f"step_{step_index}"
            agents_in_step = list(level)
            agents_display = ", ".join(agents_in_step) if agents_in_step else "none"
            self.logger.info(
                "Task %s (%s): starting %s -> agents=[%s]",
                task.id,
                task.symbol,
                step_name,
                agents_display,
            )

            level_tasks = []
            executed_agents = []

            for agent_name in agents_in_step:
                if agent_name not in self.agents:
                    self.logger.error(
                        "Task %s (%s): %s -> %s skipped (agent not registered)",
                        task.id,
                        task.symbol,
                        step_name,
                        agent_name,
                    )
                    execution_trace.append(
                        {
                            "step": step_index,
                            "step_name": step_name,
                            "agent": agent_name,
                            "status": "skipped",
                            "error": "agent_not_registered",
                        }
                    )
                    agent_results[agent_name] = {"status": "error", "error": "Agent not registered"}
                    continue

                agent_task = AgentTask(
                    task_id=f"{task.id}_{agent_name}",
                    symbol=task.symbol,
                    analysis_type=(
                        AnalysisType.SEC_FUNDAMENTAL
                        if agent_name == "sec"
                        else (
                            AnalysisType.TECHNICAL_ANALYSIS
                            if agent_name == "technical"
                            else (
                                AnalysisType.FUNDAMENTAL_ANALYSIS
                                if agent_name == "fundamental"
                                else (
                                    AnalysisType.MARKET_CONTEXT
                                    if agent_name == "market_context"
                                    else AnalysisType.INVESTMENT_SYNTHESIS
                                )
                            )
                        )
                    ),
                    context={"symbol": task.symbol, **task.metadata},
                )

                agent_task.context.update(
                    {
                        "orchestrator_step": step_index,
                        "orchestrator_step_name": step_name,
                        "orchestrator_agents_in_step": agents_in_step,
                    }
                )

                if agent_name == "synthesis":
                    agent_task.context["analyses"] = agent_results
                elif agent_name == "symbol_update":
                    # Pass previous agent results with _analysis suffix for consistency
                    if "fundamental" in agent_results:
                        agent_task.context["fundamental_analysis"] = agent_results["fundamental"]
                    if "sec" in agent_results:
                        agent_task.context["sec_analysis"] = agent_results["sec"]

                level_tasks.append(self._run_agent_with_semaphore(self.agents[agent_name], agent_task))
                executed_agents.append(agent_name)

            if not level_tasks:
                continue

            level_results = await asyncio.gather(*level_tasks, return_exceptions=True)

            for agent_name, agent_outcome in zip(executed_agents, level_results):
                trace_entry: Dict[str, Any] = {
                    "step": step_index,
                    "step_name": step_name,
                    "agent": agent_name,
                }

                if isinstance(agent_outcome, Exception):
                    error_message = str(agent_outcome)
                    trace_entry.update({"status": "error", "error": error_message})
                    agent_results[agent_name] = {"status": "error", "error": error_message}
                    self.logger.error(
                        "Task %s (%s): %s -> %s failed: %s",
                        task.id,
                        task.symbol,
                        step_name,
                        agent_name,
                        error_message,
                    )
                else:
                    agent_results[agent_name] = agent_outcome.result_data
                    status_value = agent_outcome.status.value if hasattr(agent_outcome, "status") else "completed"
                    trace_entry["status"] = status_value

                    duration = getattr(agent_outcome, "processing_time", None)
                    trace_entry["processing_time"] = duration

                    if getattr(agent_outcome, "error", None):
                        trace_entry["error"] = agent_outcome.error
                    if getattr(agent_outcome, "cache_hit", None):
                        trace_entry["cache_hit"] = agent_outcome.cache_hit
                    if getattr(agent_outcome, "cached", None):
                        trace_entry["cached"] = agent_outcome.cached

                    if hasattr(agent_outcome, "metadata") and agent_outcome.metadata:
                        trace_entry["metadata"] = agent_outcome.metadata

                    duration_display = f"{duration:.2f}s" if isinstance(duration, (int, float)) else "n/a"
                    log_fn = self.logger.info
                    if hasattr(agent_outcome, "is_successful") and not agent_outcome.is_successful():
                        log_fn = self.logger.warning
                    log_fn(
                        "Task %s (%s): %s -> %s completed (%s) duration=%s",
                        task.id,
                        task.symbol,
                        step_name,
                        agent_name,
                        status_value,
                        duration_display,
                    )

                execution_trace.append(trace_entry)

            self.logger.info(
                "Task %s (%s): finished %s",
                task.id,
                task.symbol,
                step_name,
            )

        # Compile final results
        results["agents"] = agent_results
        results["execution_trace"] = execution_trace
        results["completed_at"] = datetime.now().isoformat()
        results["duration"] = (datetime.now() - start_time).total_seconds()
        results["is_etf"] = is_etf

        # Cache comprehensive results
        from investigator.infrastructure.cache.cache_types import CacheType

        cache_key = {
            "symbol": task.symbol,
            "mode": task.mode.value,
            "timestamp": datetime.now().isoformat()[:10],  # Date only for daily cache
            "llm_type": f"orchestrator_{task.mode.value}",
        }
        try:
            # FIX Issue #3: Use async cache to prevent event loop blocking
            await self.cache_manager.set_async(CacheType.LLM_RESPONSE, cache_key, results)
        except Exception as e:
            self.logger.warning(f"Failed to cache comprehensive results: {e}")

        return results

    async def _process_peer_comparison(self, task: OrchestrationTask) -> Dict:
        """Process peer comparison task"""
        target = task.metadata["target"]
        peers = task.metadata["peers"]
        component_task_ids = task.metadata["component_tasks"]

        # Gather all component results
        analyses = {}
        for task_id in component_task_ids:
            if task_id in self.completed_tasks:
                comp_task = self.completed_tasks[task_id]
                symbol = comp_task.symbol
                analyses[symbol] = comp_task.results.get("agents", {})

        # Run synthesis agent for peer comparison
        synthesis_agent = self.agents["synthesis"]
        comparison_result = await synthesis_agent.generate_peer_synthesis(target, peers, analyses)

        return {
            "task_id": task.id,
            "type": "peer_comparison",
            "target": target,
            "peers": peers,
            "comparison": comparison_result,
            "individual_analyses": analyses,
        }

    async def _run_agent_with_semaphore(self, agent, task: AgentTask) -> AgentResult:
        """Run agent with concurrency control"""
        async with self.agent_semaphore:
            return await agent.run(task)

    def _is_etf(self, symbol: str) -> bool:
        """
        Check if symbol is an ETF using database symbol table.

        Uses caching to avoid repeated database queries.
        Falls back to treating unknown symbols as stocks.
        """
        symbol_key = symbol.upper()
        if symbol_key in self.symbol_classification_cache:
            return self.symbol_classification_cache[symbol_key]

        # Use database-based ETF detection (more reliable than API)
        from investigator.infrastructure.database.db import is_etf as db_is_etf

        is_etf_flag = db_is_etf(symbol_key)

        self.symbol_classification_cache[symbol_key] = is_etf_flag
        return is_etf_flag

    def _get_agents_for_mode(
        self,
        symbol: str = "",
        mode: AnalysisMode = AnalysisMode.STANDARD,
        custom_agents: Optional[List[str]] = None,
        is_etf: bool = False,
    ) -> List[str]:
        """
        Get list of agents to run based on analysis mode

        IMPORTANT: SEC Agent must run in STANDARD and COMPREHENSIVE modes because
        it's the single source of truth for SEC CompanyFacts data. All other agents
        (fundamental, technical) depend on SEC Agent's cached data.
        """
        if mode == AnalysisMode.QUICK:
            agents = ["technical", "market_context"]
        elif mode == AnalysisMode.STANDARD:
            agents = ["sec", "technical", "fundamental", "symbol_update", "market_context", "synthesis"]
        elif mode == AnalysisMode.COMPREHENSIVE:
            agents = ["sec", "technical", "fundamental", "symbol_update", "market_context", "synthesis"]
        elif mode == AnalysisMode.CUSTOM:
            agents = custom_agents or ["technical"]
        else:
            agents = ["sec", "technical", "fundamental", "symbol_update", "market_context", "synthesis"]

        # Deduplicate while preserving order
        agents = list(dict.fromkeys(agents))

        if is_etf:
            # ETFs get: technical + market_context + synthesis (no SEC/fundamental/symbol_update)
            filtered = [a for a in agents if a not in {"sec", "fundamental", "symbol_update"}]
            if "technical" not in filtered:
                filtered.insert(0, "technical")
            if "market_context" not in filtered:
                filtered.append("market_context")
            if "synthesis" not in filtered:
                filtered.append("synthesis")
            filtered = list(dict.fromkeys(filtered))
            if filtered != agents:
                self.logger.info(
                    "Symbol %s identified as ETF; limiting agents to %s (skipping SEC/Fundamental/SymbolUpdate)",
                    symbol,
                    filtered,
                )
            agents = filtered

        return agents

    def _get_execution_order(self, agents: List[str]) -> List[List[str]]:
        """
        Determine execution order using proper topological sort

        Uses the dependency graph to dynamically compute execution levels.
        Agents in same level can run in parallel.

        Resolves Technical Debt Issue 1.1 (CRITICAL) - Brittle hardcoded ordering

        Args:
            agents: List of agent names to execute

        Returns:
            List of levels, each level is list of agents that can run in parallel

        Raises:
            ValueError: If circular dependency detected
        """
        if not agents:
            return []

        # Create subgraph with only requested agents
        subgraph = self.dependency_graph.subgraph(agents)

        # Topological sort using Kahn's algorithm
        levels = []
        processed = set()
        in_degree = {node: subgraph.in_degree(node) for node in subgraph.nodes()}

        while len(processed) < len(agents):
            # Find all nodes with no unprocessed dependencies
            level = []
            for node in subgraph.nodes():
                if node not in processed:
                    # Check if all dependencies are processed
                    deps = set(subgraph.predecessors(node))
                    if deps.issubset(processed):
                        level.append(node)

            if not level:
                # No nodes ready - circular dependency detected
                unprocessed = [a for a in agents if a not in processed]
                raise ValueError(
                    f"Circular dependency detected in agents: {unprocessed}. " f"Cannot determine execution order."
                )

            processed.update(level)
            levels.append(level)

        self.logger.info(f"✅ Execution order computed: {len(levels)} levels → {levels}")

        return levels

    async def _handle_events(self):
        """Handle events from agents"""
        # Subscribe to agent events
        self.event_bus.subscribe("agent_*", self._process_event_sync)
        self.event_bus.subscribe("analysis_*", self._process_event_sync)

        while self.running:
            try:
                # Process events from the queue
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                self.logger.error(f"Event handler encountered unexpected error: {e}", exc_info=True)
                await asyncio.sleep(1)

    def _process_event_sync(self, event):
        """Sync wrapper for event processing

        Args:
            event: Event object from EventBus with .type, .data, .source attributes
        """
        event_dict = {"type": event.type, "data": event.data, "source": event.source}
        asyncio.create_task(self._process_event(event_dict))

    async def _process_event(self, event: Dict):
        """Process events from agents"""
        event_type = event.get("type")

        if event_type == "agent_completed":
            # Update metrics
            agent_id = event.get("agent_id")
            duration = event.get("duration")
            self.metrics.record_agent_execution(agent_id, duration)

        elif event_type == "agent_failed":
            # Log failure
            agent_id = event.get("agent_id")
            error = event.get("error")
            self.logger.error(f"Agent {agent_id} failed: {error}")
            self.metrics.record_agent_failure(agent_id)

        elif event_type == "analysis_completed":
            # Log completion
            task_id = event.get("task_id")
            self.logger.info(f"Analysis {task_id} completed")

    async def _report_metrics(self):
        """Periodically report metrics"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Report every minute

                # Calculate average duration
                if self.performance_stats["total_analyses"] > 0:
                    success_rate = (
                        self.performance_stats["successful_analyses"] / self.performance_stats["total_analyses"]
                    ) * 100
                else:
                    success_rate = 0

                self.logger.info(
                    f"Orchestrator Stats - Total: {self.performance_stats['total_analyses']}, "
                    f"Success Rate: {success_rate:.1f}%, "
                    f"Active Tasks: {len(self.active_tasks)}, "
                    f"Queue Size: {self.task_queue.qsize()}"
                )

                # Send to metrics collector
                self.metrics.record_orchestrator_stats(self.performance_stats)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Metrics reporter encountered unexpected error: %s", e, exc_info=True)
                await asyncio.sleep(1)

    async def optimize_agent_allocation(self):
        """Dynamically optimize agent allocation based on workload"""
        # Monitor queue size and adjust workers
        queue_size = self.task_queue.qsize()
        active_count = len(self.active_tasks)

        if queue_size > 10 and len(self.workers) < 10:
            # Add more workers
            for i in range(2):
                worker_id = f"dynamic_worker_{len(self.workers)}"
                worker = asyncio.create_task(self._worker(worker_id))
                self.workers.append(worker)
                self.logger.info(f"Added dynamic worker {worker_id}")

        elif queue_size == 0 and active_count == 0 and len(self.workers) > self.max_concurrent_analyses:
            # Remove excess workers
            excess = len(self.workers) - self.max_concurrent_analyses
            for _ in range(min(excess, 2)):
                if self.workers:
                    worker = self.workers.pop()
                    worker.cancel()
                    self.logger.info("Removed excess worker")

    @property
    def logger(self):
        """Get logger instance"""
        return self._logger
