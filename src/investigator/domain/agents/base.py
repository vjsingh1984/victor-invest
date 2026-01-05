"""
Base Agent Framework for InvestiGator
Defines core agent interfaces and base functionality
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# Import domain models
from investigator.domain.models.analysis import (
    AgentCapability,
    AgentMetrics,
    AgentResult,
    AgentTask,
    AnalysisType,
    Priority,
    TaskStatus,
)

# Import CacheType for cache operations
from investigator.infrastructure.cache.cache_types import CacheType

# ========================================================================================
# Helper Functions
# ========================================================================================


def get_cache_type_for_analysis(analysis_type: AnalysisType) -> CacheType:
    """Map AnalysisType to appropriate CacheType"""
    mapping = {
        AnalysisType.SEC_FUNDAMENTAL: CacheType.SEC_RESPONSE,
        AnalysisType.TECHNICAL_ANALYSIS: CacheType.TECHNICAL_DATA,
        AnalysisType.FUNDAMENTAL_ANALYSIS: CacheType.LLM_RESPONSE,
        AnalysisType.INVESTMENT_SYNTHESIS: CacheType.LLM_RESPONSE,
        AnalysisType.PEER_GROUP: CacheType.LLM_RESPONSE,
        AnalysisType.ESG_ANALYSIS: CacheType.LLM_RESPONSE,
        AnalysisType.MARKET_DATA: CacheType.TECHNICAL_DATA,
        AnalysisType.RISK_ASSESSMENT: CacheType.LLM_RESPONSE,
        AnalysisType.SENTIMENT_ANALYSIS: CacheType.LLM_RESPONSE,
        AnalysisType.OPTIONS_ANALYSIS: CacheType.LLM_RESPONSE,
        AnalysisType.PORTFOLIO_OPTIMIZATION: CacheType.LLM_RESPONSE,
    }
    return mapping.get(analysis_type, CacheType.LLM_RESPONSE)


# ========================================================================================
# Base Agent Classes
# ========================================================================================

# ========================================================================================
# Base Agent Class
# ========================================================================================


class InvestmentAgent(ABC):
    """
    Base class for all investment analysis agents.

    Lifecycle reminder:

        create task ──▶ hydrate context ──▶ process() ──▶ emit AgentResult
              ▲                  │                │
              │                  └──── cache lookups/upserts
              └───── metrics/event bus capture happens outside
    """

    def __init__(self, agent_id: str, ollama_client, event_bus, cache_manager=None):
        self.agent_id = agent_id
        self.ollama = ollama_client
        self.event_bus = event_bus
        self.cache = cache_manager
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.capabilities = self.register_capabilities()
        self.metrics = AgentMetrics(agent_id=agent_id)
        self._shutdown = False
        self.processing_tasks: Set[str] = set()

    @abstractmethod
    def register_capabilities(self) -> List[AgentCapability]:
        """Register what this agent can do"""
        pass

    @abstractmethod
    async def process(self, task: AgentTask) -> AgentResult:
        """Process an analysis task"""
        pass

    async def can_handle_task(self, task: AgentTask) -> bool:
        """Check if agent can handle a specific task"""
        for capability in self.capabilities:
            if capability.analysis_type == task.analysis_type:
                # Check if we have required data in context
                for req_key, req_type in capability.min_data_required.items():
                    if req_key not in task.context:
                        return False
                    if not isinstance(task.context[req_key], req_type):
                        return False
                return True
        return False

    async def pre_process(self, task: AgentTask) -> bool:
        """Pre-processing hook for validation and setup"""
        # Check if task is already being processed
        if task.task_id in self.processing_tasks:
            self.logger.warning(f"Task {task.task_id} already being processed")
            return False

        # Add to processing set
        self.processing_tasks.add(task.task_id)

        # Validate task timeout
        if task.timeout and task.timeout < 10:
            self.logger.warning(f"Task timeout too short: {task.timeout}s")
            task.timeout = 30

        return True

    @staticmethod
    def parse_llm_response(response: Any, default: Any = None) -> Dict:
        """
        Consolidated LLM response parser (shared across all agents)

        Resolves Technical Debt Issue 4.1 (HIGH) - Duplicate parsing in 3 agents

        Args:
            response: LLM response (could be str, dict, or None)
            default: Default value if parsing fails (None or {})

        Returns:
            Parsed dict or default value
        """
        if default is None:
            default = {}

        # Already a dict - return as-is
        if isinstance(response, dict):
            return response

        # None or empty - return default
        if not response:
            return default

        # String - try JSON parsing
        if isinstance(response, str):
            import json

            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Not valid JSON, return as-is in dict
                return {"response": response}

        # Unexpected type - log warning and return default
        import logging

        logging.getLogger(__name__).warning(f"Unexpected LLM response type: {type(response)}")
        return default

    async def post_process(self, task: AgentTask, result: AgentResult) -> AgentResult:
        """Post-processing hook for cleanup, normalization, and metrics"""
        # Remove from processing set
        self.processing_tasks.discard(task.task_id)

        # CRITICAL: Normalize data to snake_case before caching
        # This ensures consistent key naming across all agents
        if result.is_successful() and result.result_data:
            try:
                from investigator.domain.services.data_normalizer import DataNormalizer

                # Convert to snake_case and apply judicious rounding
                result.result_data = DataNormalizer.normalize_and_round(
                    result.result_data, to_camel_case=False  # Use snake_case for internal Python code
                )

                self.logger.debug(
                    f"Normalized {len(result.result_data)} keys to snake_case " f"for task {task.task_id}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to normalize result data for {task.task_id}: {e}. " f"Continuing with unnormalized data."
                )

        # Update metrics
        self.metrics.update(result)

        # Log performance
        if result.is_successful():
            self.logger.info(
                f"Task {task.task_id} completed in {result.processing_time:.2f}s " f"(cache_hit: {result.cache_hit})"
            )
        else:
            self.logger.error(f"Task {task.task_id} failed: {result.error}")

        return result

    async def execute_with_timeout(self, task: AgentTask) -> AgentResult:
        """Execute task with timeout handling"""
        timeout = task.timeout or 900  # Default 15 minutes (matches CLI timeout)

        try:
            result = await asyncio.wait_for(self.process(task), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self.logger.error(f"Task {task.task_id} timed out after {timeout}s")
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                result_data={},
                processing_time=timeout,
                error=f"Task timed out after {timeout} seconds",
            )

    async def execute_with_retry(self, task: AgentTask) -> AgentResult:
        """Execute task with retry logic"""
        while task.retry_count <= task.max_retries:
            try:
                result = await self.execute_with_timeout(task)

                if result.is_successful() or task.retry_count >= task.max_retries:
                    return result

                # Exponential backoff
                wait_time = 2**task.retry_count
                self.logger.info(f"Retrying task {task.task_id} in {wait_time}s")
                await asyncio.sleep(wait_time)

                task.retry_count += 1

            except Exception as e:
                self.logger.error(f"Task {task.task_id} exception: {e}")
                if task.retry_count >= task.max_retries:
                    return AgentResult(
                        task_id=task.task_id,
                        agent_id=self.agent_id,
                        status=TaskStatus.FAILED,
                        result_data={},
                        processing_time=0,
                        error=str(e),
                    )
                task.retry_count += 1

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status=TaskStatus.FAILED,
            result_data={},
            processing_time=0,
            error="Max retries exceeded",
        )

    async def run(self, task: AgentTask) -> AgentResult:
        """Main entry point for task execution"""
        start_time = datetime.now()

        # Pre-process
        if not await self.pre_process(task):
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                result_data={},
                processing_time=0,
                error="Pre-processing failed",
            )

        # Check cache if available
        if self.cache:
            # Use structured cache key for CacheManager
            # CRITICAL: Include fiscal_period for period-specific data (Phase 2 fix)
            cache_key = {
                "symbol": task.symbol,
                "analysis_type": task.analysis_type.value,
                "context_hash": task.get_cache_key()[:8],  # Use first 8 chars of hash
            }

            # Add fiscal_period if available (ensures different cache per period)
            if task.fiscal_period:
                cache_key["fiscal_period"] = task.fiscal_period

            cache_type = get_cache_type_for_analysis(task.analysis_type)
            if cache_type == CacheType.LLM_RESPONSE:
                cache_key["llm_type"] = task.analysis_type.value
            # FIX Issue #3: Use async cache to prevent event loop blocking
            cached_result = await self.cache.get_async(cache_type, cache_key)
            if cached_result:
                self.logger.info(f"Cache hit for task {task.task_id} (period: {task.fiscal_period or 'latest'})")

                # HYBRID CACHING FIX (Phase 1):
                # Cache expensive LLM responses (30-60s) but recalculate deterministic metrics (~100ms)
                # This ensures CompanyProfile and derived metrics are always fresh
                if hasattr(self, "recalculate_derived_metrics"):
                    try:
                        self.logger.debug(f"{task.symbol} - Recalculating derived metrics from cached data")
                        enriched_data = await self.recalculate_derived_metrics(task, cached_result)
                        if enriched_data:
                            cached_result = enriched_data
                            self.logger.debug(f"{task.symbol} - Derived metrics recalculated successfully")
                    except Exception as e:
                        self.logger.warning(
                            f"{task.symbol} - Failed to recalculate derived metrics: {e}", exc_info=True
                        )
                        # Continue with cached data if recalculation fails

                result = AgentResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status=TaskStatus.COMPLETED,
                    result_data=cached_result,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    cached=True,
                    cache_hit=True,
                )
                return await self.post_process(task, result)

        # Execute with retry
        result = await self.execute_with_retry(task)
        result.processing_time = (datetime.now() - start_time).total_seconds()

        # Cache successful results
        if self.cache and result.is_successful():
            # Use structured cache key for CacheManager
            # CRITICAL: Include fiscal_period for period-specific data (Phase 2 fix)
            cache_key = {
                "symbol": task.symbol,
                "analysis_type": task.analysis_type.value,
                "context_hash": task.get_cache_key()[:8],  # Use first 8 chars of hash
            }

            # Add fiscal_period if available (ensures different cache per period)
            if task.fiscal_period:
                cache_key["fiscal_period"] = task.fiscal_period

            try:
                # Note: CacheManager.set is synchronous
                cache_type = get_cache_type_for_analysis(task.analysis_type)
                if cache_type == CacheType.LLM_RESPONSE:
                    cache_key["llm_type"] = task.analysis_type.value
                # FIX Issue #3: Use async cache to prevent event loop blocking
                await self.cache.set_async(cache_type, cache_key, result.result_data)
                result.cached = True
                self.logger.debug(f"Cached result for {task.symbol} (period: {task.fiscal_period or 'latest'})")
            except Exception as e:
                self.logger.warning(f"Failed to cache result: {e}")

        # Post-process
        return await self.post_process(task, result)

    async def _cache_llm_response(
        self,
        response: Any,
        model: str,
        symbol: str,
        llm_type: str,
        prompt: str = "",
        temperature: float = 0.3,
        top_p: float = 0.9,
        format: str = "json",
        period: str = None,
        **extra_params,
    ) -> None:
        """
        Cache LLM response separately for audit/debugging purposes.

        FIX Issue #3: Made async to support non-blocking cache operations.

        This method stores ONLY the LLM response with proper metadata structure
        in the LLM_RESPONSE cache type. This is separate from agent analysis caching.

        Args:
            response: Raw LLM response from Ollama client
            model: Model name used (e.g., "qwen3:30b")
            symbol: Stock symbol
            llm_type: Type of LLM analysis (e.g., "market_context", "risk_analysis")
            prompt: Original prompt sent to LLM
            temperature: Temperature parameter
            top_p: Top-p sampling parameter
            format: Response format ("json" or "text")
            period: Optional fiscal period (e.g., "2024-Q3") for period-specific caching
            **extra_params: Additional model parameters
        """
        if not self.cache:
            return

        try:
            # Build the wrapped LLM response with proper structure
            wrapped_response = self._wrap_llm_response(
                response=response,
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                format=format,
                **extra_params,
            )

            # Create cache key for LLM response
            # CRITICAL FIX #9: Include period in cache key for period-specific responses
            llm_cache_key = {
                "symbol": symbol,
                "llm_type": llm_type,
                "model": model,
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:8] if prompt else "no_prompt",
            }

            # Add period to cache key if provided (ensures different cache per fiscal period)
            if period:
                llm_cache_key["period"] = period

            # Store in LLM_RESPONSE cache
            from investigator.infrastructure.cache.cache_types import CacheType

            # FIX Issue #3: Use async cache to prevent event loop blocking
            await self.cache.set_async(CacheType.LLM_RESPONSE, llm_cache_key, wrapped_response)
            self.logger.info(f"💾 Cached LLM response: {llm_type} for {symbol} (model: {model})")

        except Exception as e:
            self.logger.warning(f"Failed to cache LLM response: {e}")

    def _wrap_llm_response(
        self,
        response: Any,
        model: str,
        prompt: str = "",
        temperature: float = 0.3,
        top_p: float = 0.9,
        format: str = "json",
        **extra_params,
    ) -> Dict[str, Any]:
        """
        Wrap LLM response with proper metadata structure for caching.

        This ensures all LLM responses are cached with consistent structure:
        - response: The actual LLM response text/data (parsed JSON if format="json")
        - model_info: Model configuration (model, temperature, top_p, etc.)
        - metadata: Cache metadata (timestamp, cache key, etc.)

        NOTE: This method only STRUCTURES the data. To actually cache it, use _cache_llm_response()
        which handles both structuring AND caching to the LLM_RESPONSE cache type.

        Args:
            response: Raw LLM response from Ollama client
            model: Model name used (e.g., "qwen3:30b")
            prompt: Original prompt sent to LLM (optional)
            temperature: Temperature parameter (default: 0.3)
            top_p: Top-p sampling parameter (default: 0.9)
            format: Response format ("json" or "text")
            **extra_params: Additional model parameters to include

        Returns:
            Dict with properly structured cache data (NOT cached, just structured)
        """
        # Extract response text from Ollama result
        if isinstance(response, dict):
            response_text = response.get("response", "")
            # Also check for _raw_thinking field (reasoning models)
            raw_thinking = response.get("_raw_thinking", "")
        else:
            response_text = str(response)
            raw_thinking = ""

        # Parse JSON if format is json using the LLM response processor
        parsed_response = response_text
        if format == "json" and response_text:
            from investigator.application.processors import LLMResponseProcessor

            processor = LLMResponseProcessor()

            # Try to extract JSON from markdown code blocks or raw text
            extracted_json = processor.extract_json_from_text(response_text)
            if extracted_json:
                parsed_response = extracted_json
                self.logger.info(f"✅ Successfully extracted JSON from LLM response ({len(str(extracted_json))} chars)")
            else:
                # JSON extraction failed - log as DEBUG since retry logic will handle it
                self.logger.debug(
                    f"⚠️  Failed to extract JSON from response (length: {len(response_text)}). Raw response: {response_text[:500]}..."
                )
                self.logger.debug(f"Full raw response that failed JSON extraction: {response_text}")
                # If JSON extraction fails, return an empty dict to trigger retry logic
                parsed_response = {}

        # Round numeric values in parsed response before caching to reduce token usage and storage
        if isinstance(parsed_response, dict):
            from investigator.domain.services.data_normalizer import DataNormalizer

            parsed_response = DataNormalizer.round_financial_data(parsed_response)
            self.logger.debug("Applied numeric precision rounding to LLM response")

        # Build model_info with all parameters
        model_info = {"model": model, "temperature": temperature, "top_p": top_p, "format": format}

        # Add any extra parameters (like max_tokens, num_ctx, etc.)
        model_info.update(extra_params)

        # Build complete cache structure
        wrapped = {
            "response": parsed_response,
            "prompt": prompt,  # Add full prompt for cache auditing
            "model_info": model_info,
            "metadata": {
                "cached_at": datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "prompt_length": len(prompt) if prompt else 0,
                "prompt_preview": prompt[:200] if prompt else "",  # Save prompt preview for debugging
                "cache_type": "llm_response",
                "raw_thinking": raw_thinking[:500] if raw_thinking else "",  # Save thinking preview
            },
        }

        return wrapped

    async def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        return {
            "agent_id": self.agent_id,
            "status": "healthy" if not self._shutdown else "shutting_down",
            "capabilities": [cap.analysis_type.value for cap in self.capabilities],
            "metrics": {
                "total_tasks": self.metrics.total_tasks,
                "success_rate": (
                    self.metrics.successful_tasks / self.metrics.total_tasks if self.metrics.total_tasks > 0 else 0
                ),
                "avg_processing_time": self.metrics.avg_processing_time,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "error_rate": self.metrics.error_rate,
            },
            "active_tasks": len(self.processing_tasks),
            "timestamp": datetime.now().isoformat(),
        }

    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info(f"Shutting down agent {self.agent_id}")
        self._shutdown = True

        # Wait for active tasks to complete (max 30 seconds)
        wait_time = 0
        while self.processing_tasks and wait_time < 30:
            await asyncio.sleep(1)
            wait_time += 1

        if self.processing_tasks:
            self.logger.warning(f"Agent {self.agent_id} shutting down with {len(self.processing_tasks)} active tasks")


# ========================================================================================
# Agent Utilities
# ========================================================================================


class AgentPool:
    """Manages a pool of agents for load balancing"""

    def __init__(self):
        self.agents: Dict[str, InvestmentAgent] = {}
        self.agent_loads: Dict[str, int] = {}

    def register(self, agent: InvestmentAgent):
        """Register an agent in the pool"""
        self.agents[agent.agent_id] = agent
        self.agent_loads[agent.agent_id] = 0

    async def get_best_agent_for_task(self, task: AgentTask) -> Optional[InvestmentAgent]:
        """Get the best available agent for a task"""
        capable_agents = []

        for agent_id, agent in self.agents.items():
            if await agent.can_handle_task(task):
                capable_agents.append((agent_id, self.agent_loads.get(agent_id, 0)))

        if not capable_agents:
            return None

        # Sort by load and return least loaded agent
        capable_agents.sort(key=lambda x: x[1])
        return self.agents[capable_agents[0][0]]

    def update_load(self, agent_id: str, delta: int):
        """Update agent load"""
        if agent_id in self.agent_loads:
            self.agent_loads[agent_id] += delta
            self.agent_loads[agent_id] = max(0, self.agent_loads[agent_id])

    async def health_check_all(self) -> Dict[str, Any]:
        """Health check all agents"""
        results = {}
        for agent_id, agent in self.agents.items():
            results[agent_id] = await agent.health_check()
        return results

    async def shutdown_all(self):
        """Shutdown all agents"""
        shutdown_tasks = [agent.shutdown() for agent in self.agents.values()]
        await asyncio.gather(*shutdown_tasks)
