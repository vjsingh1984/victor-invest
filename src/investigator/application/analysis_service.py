"""
Analysis Service

High-level service for stock analysis coordination.
Provides simplified interface to the orchestrator.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from investigator.application.orchestrator import AgentOrchestrator, AnalysisMode, Priority
from investigator.infrastructure.cache.cache_manager import CacheManager
from investigator.infrastructure.monitoring import MetricsCollector


class AnalysisService:
    """
    High-level service for coordinating stock analysis

    Provides a simplified interface for:
    - Single stock analysis
    - Batch analysis
    - Peer comparison
    - Result retrieval
    """

    def __init__(
        self, cache_manager: CacheManager, metrics_collector: MetricsCollector, max_concurrent_analyses: int = 5
    ):
        """
        Initialize the analysis service

        Args:
            cache_manager: Cache manager instance
            metrics_collector: Metrics collector for tracking
            max_concurrent_analyses: Maximum concurrent analyses
        """
        self.orchestrator = AgentOrchestrator(
            cache_manager=cache_manager,
            metrics_collector=metrics_collector,
            max_concurrent_analyses=max_concurrent_analyses,
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self._started = False

    async def start(self):
        """Start the service"""
        if not self._started:
            await self.orchestrator.start()
            self._started = True
            self.logger.info("Analysis service started")

    async def stop(self):
        """Stop the service"""
        if self._started:
            await self.orchestrator.stop()
            self._started = False
            self.logger.info("Analysis service stopped")

    async def analyze_stock(
        self, symbol: str, mode: str = "standard", priority: str = "normal", **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze a single stock

        Args:
            symbol: Stock ticker symbol
            mode: Analysis mode (quick, standard, comprehensive)
            priority: Task priority (low, normal, high, critical)
            **kwargs: Additional parameters

        Returns:
            Analysis results dictionary

        Example:
            >>> service = AnalysisService(cache_manager, metrics)
            >>> await service.start()
            >>> results = await service.analyze_stock("AAPL", mode="standard")
            >>> print(results['agents']['fundamental']['metrics'])
        """
        if not self._started:
            await self.start()

        # Convert string modes to enums
        analysis_mode = self._parse_mode(mode)
        priority_level = self._parse_priority(priority)

        # Submit analysis task
        task_id = await self.orchestrator.analyze(symbol=symbol, mode=analysis_mode, priority=priority_level, **kwargs)

        # Wait for completion
        results = await self.orchestrator.get_results(task_id, wait=True, timeout=600)

        if results is None:
            raise TimeoutError(f"Analysis for {symbol} timed out after 600 seconds")

        return results

    async def batch_analyze(
        self, symbols: List[str], mode: str = "standard", priority: str = "normal"
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple stocks in batch

        Args:
            symbols: List of stock ticker symbols
            mode: Analysis mode
            priority: Task priority

        Returns:
            List of analysis results

        Example:
            >>> results = await service.batch_analyze(["AAPL", "MSFT", "GOOGL"])
            >>> for result in results:
            ...     print(f"{result['symbol']}: {result['status']}")
        """
        if not self._started:
            await self.start()

        analysis_mode = self._parse_mode(mode)
        priority_level = self._parse_priority(priority)

        # Submit all tasks
        task_ids = await self.orchestrator.analyze_batch(symbols=symbols, mode=analysis_mode, priority=priority_level)

        # Collect results
        results = []
        for task_id in task_ids:
            result = await self.orchestrator.get_results(task_id, wait=True, timeout=600)
            if result:
                results.append(result)
            else:
                self.logger.warning(f"Task {task_id} timed out or failed")

        return results

    async def peer_comparison(self, target: str, peers: List[str], mode: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze a target company and its peers for comparison

        Args:
            target: Target company ticker
            peers: List of peer company tickers
            mode: Analysis mode (typically comprehensive)

        Returns:
            Peer comparison results

        Example:
            >>> results = await service.peer_comparison(
            ...     target="AAPL",
            ...     peers=["MSFT", "GOOGL", "META"]
            ... )
            >>> print(results['comparison']['relative_valuation'])
        """
        if not self._started:
            await self.start()

        analysis_mode = self._parse_mode(mode)

        # Submit peer comparison task
        task_id = await self.orchestrator.analyze_peer_group(target=target, peers=peers, mode=analysis_mode)

        # Wait for completion
        results = await self.orchestrator.get_results(task_id, wait=True, timeout=900)

        if results is None:
            raise TimeoutError(f"Peer comparison for {target} timed out")

        return results

    async def get_status(self, task_id: str) -> Dict:
        """
        Get status of an analysis task

        Args:
            task_id: Task ID to check

        Returns:
            Status dictionary
        """
        return await self.orchestrator.get_status(task_id)

    async def get_results(self, task_id: str, wait: bool = False, timeout: int = 300) -> Optional[Dict]:
        """
        Get results of an analysis task

        Args:
            task_id: Task ID to retrieve
            wait: Whether to wait for completion
            timeout: Maximum wait time in seconds

        Returns:
            Results or None if not ready
        """
        return await self.orchestrator.get_results(task_id, wait=wait, timeout=timeout)

    def _parse_mode(self, mode: str) -> AnalysisMode:
        """Parse string mode to AnalysisMode enum"""
        mode_map = {
            "quick": AnalysisMode.QUICK,
            "standard": AnalysisMode.STANDARD,
            "comprehensive": AnalysisMode.COMPREHENSIVE,
            "custom": AnalysisMode.CUSTOM,
        }
        return mode_map.get(mode.lower(), AnalysisMode.STANDARD)

    def _parse_priority(self, priority: str) -> Priority:
        """Parse string priority to Priority enum"""
        priority_map = {
            "critical": Priority.CRITICAL,
            "high": Priority.HIGH,
            "normal": Priority.NORMAL,
            "low": Priority.LOW,
        }
        return priority_map.get(priority.lower(), Priority.NORMAL)

    async def __aenter__(self):
        """Context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.stop()
        return False
