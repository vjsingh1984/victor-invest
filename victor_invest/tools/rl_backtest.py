# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RL Backtest Tool for Victor Invest.

Provides RL backtesting functionality integrated with shared services:
- Uses shared market data services (SharesService, PriceService, TechnicalAnalysisService)
- Uses shared valuation config services (ValuationConfigService, SectorMultiplesService)
- Records predictions to valuation_outcomes table with JSONB multi-period data
- Consistent with batch_analysis_runner and victor_invest workflows

Multi-period data stored in per_model_rewards JSONB:
{
    "multi_period": {
        "entry_date": "2025-01-02",
        "prices": {"1m": 270.37, "3m": 271.86, "6m": 280.50, "12m": 290.00, ...},
        "exit_dates": {"1m": "2025-02-01", "3m": "2025-04-02", ...},
        "long_rewards": {"1m": 0.577, "3m": 0.214, ...},
        "short_rewards": {"1m": -0.577, "3m": -0.214, ...}
    }
}
"""

import logging
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from typing import Any, Dict, List, Optional

from victor_invest.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Holding periods in days for multi-period reward calculation
HOLDING_PERIODS = {
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "12m": 365,
    "18m": 540,
    "24m": 730,
    "36m": 1095,
}


class RLBacktestTool(BaseTool):
    """Tool for RL backtesting using shared services.

    This tool provides consistent RL backtest functionality that aligns with
    the victor_invest workflow architecture and uses shared market data and
    valuation services.

    Features:
    - Historical valuation simulation at past dates
    - Multi-period reward calculation (1m, 3m, 6m, 12m, 18m, 24m, 36m)
    - Entry/exit date tracking in JSONB format
    - Dual position recording (LONG and SHORT)
    - Consistent with batch_analysis_runner output

    Actions:
        run_backtest: Run backtest for a symbol at specific lookback periods
        calculate_rewards: Calculate multi-period rewards for a prediction
        record_prediction: Record prediction to database
        get_historical_data: Get historical price and shares data

    Example:
        tool = RLBacktestTool()
        result = await tool.execute(
            action="run_backtest",
            symbol="AAPL",
            lookback_months=[12, 24, 36]
        )
    """

    name = "rl_backtest"
    description = """Run RL backtesting for valuation model training including:
    - Historical valuation simulation using only data available at past dates
    - Multi-period reward calculation (1m, 3m, 6m, 12m, 18m, 24m, 36m)
    - Entry/exit date tracking for position management
    - Dual position recording (LONG and SHORT) for balanced RL training
    - Unified context feature extraction via DataSourceManager"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize the RL backtest tool."""
        super().__init__(config)
        self._shares_service = None
        self._price_service = None
        self._technical_service = None
        self._metadata_service = None
        self._valuation_config = None
        self._sector_multiples = None
        self._outcome_tracker = None
        self._reward_calculator = None
        self._data_source_manager = None
        self._db = None

    async def initialize(self) -> None:
        """Initialize shared services."""
        try:
            # Shared market data services
            from investigator.domain.services.market_data import (
                SharesService,
                PriceService,
                SymbolMetadataService,
                get_technical_analysis_service,
            )

            # Shared valuation config services
            from investigator.domain.services.valuation_shared import (
                ValuationConfigService,
                SectorMultiplesService,
            )

            # RL infrastructure
            from investigator.domain.services.rl.outcome_tracker import OutcomeTracker
            from investigator.domain.services.rl.reward_calculator import get_reward_calculator

            # Data source manager for consolidated data access
            from investigator.domain.services.data_sources.manager import DataSourceManager

            # Database
            from investigator.infrastructure.database.db import get_db_manager

            self._db = get_db_manager()
            # Services use default connection URLs when not specified
            self._shares_service = SharesService()
            self._price_service = PriceService()
            self._metadata_service = SymbolMetadataService()
            self._technical_service = get_technical_analysis_service()
            self._valuation_config = ValuationConfigService()
            self._sector_multiples = SectorMultiplesService()
            self._outcome_tracker = OutcomeTracker()
            self._reward_calculator = get_reward_calculator()
            self._data_source_manager = DataSourceManager()

            self._initialized = True
            logger.info("RLBacktestTool initialized with shared services and DataSourceManager")
        except ImportError as e:
            logger.error(f"Could not import required services: {e}")
            raise

    async def execute(
        self,
        _exec_ctx: Dict[str, Any],
        action: str = "run_backtest",
        symbol: str = "",
        lookback_months: Optional[List[int]] = None,
        analysis_date: Optional[date] = None,
        current_price: float = 0.0,
        fair_value: float = 0.0,
        fair_values: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        tier_classification: str = "",
        context_features: Optional[Dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute RL backtest action.

        Args:
            action: Action to perform:
                - "run_backtest": Run full backtest for symbol at lookback periods
                - "calculate_rewards": Calculate multi-period rewards
                - "record_prediction": Record prediction to database
                - "get_historical_data": Get historical price/shares data
                - "get_context_features": Get RL context features via DataSourceManager
            symbol: Stock symbol
            lookback_months: List of months to look back (e.g., [12, 24, 36])
            analysis_date: Date of analysis (for historical simulation)
            current_price: Price at analysis date
            fair_value: Blended fair value
            fair_values: Dict of model fair values
            weights: Dict of model weights
            tier_classification: Valuation tier classification
            context_features: RL context features dict

        Returns:
            ToolResult with backtest data
        """
        await self.ensure_initialized()

        try:
            if action == "run_backtest":
                return await self._run_backtest(
                    symbol=symbol,
                    lookback_months=lookback_months or [12],
                )
            elif action == "calculate_rewards":
                return await self._calculate_rewards(
                    symbol=symbol,
                    analysis_date=analysis_date or date.today(),
                    current_price=current_price,
                )
            elif action == "record_prediction":
                return await self._record_prediction(
                    symbol=symbol,
                    analysis_date=analysis_date or date.today(),
                    current_price=current_price,
                    fair_value=fair_value,
                    fair_values=fair_values or {},
                    weights=weights or {},
                    tier_classification=tier_classification,
                    context_features=context_features or {},
                )
            elif action == "get_historical_data":
                return await self._get_historical_data(
                    symbol=symbol,
                    analysis_date=analysis_date or date.today(),
                )
            elif action == "get_context_features":
                return await self._get_context_features(
                    symbol=symbol,
                    analysis_date=analysis_date or date.today(),
                )
            else:
                return ToolResult.error_result(
                    f"Unknown action: {action}. Valid actions: run_backtest, "
                    "calculate_rewards, record_prediction, get_historical_data, "
                    "get_context_features"
                )
        except Exception as e:
            logger.error(f"Error in RLBacktestTool: {e}")
            return ToolResult.error_result(str(e))

    async def _run_backtest(
        self,
        symbol: str,
        lookback_months: List[int],
    ) -> ToolResult:
        """Run backtest for a symbol at multiple lookback periods."""
        results = {
            "symbol": symbol,
            "predictions": [],
            "errors": [],
        }

        today = date.today()
        metadata = await self._get_metadata(symbol)

        for months_back in lookback_months:
            try:
                analysis_date = today - relativedelta(months=months_back)

                # Get historical price
                price = self._price_service.get_price(symbol, analysis_date)
                if not price or price <= 0:
                    results["errors"].append(f"{months_back}m: No price data")
                    continue

                # Get multi-period prices and calculate rewards
                multi_period_data = await self._get_multi_period_data(
                    symbol, analysis_date, price, metadata.get("beta", 1.0)
                )

                results["predictions"].append({
                    "lookback_months": months_back,
                    "analysis_date": analysis_date.isoformat(),
                    "price_at_prediction": price,
                    "multi_period": multi_period_data,
                })

            except Exception as e:
                results["errors"].append(f"{months_back}m: {str(e)}")

        return ToolResult.success_result(
            data=results,
            metadata={
                "tool": "rl_backtest",
                "action": "run_backtest",
                "lookback_periods": lookback_months,
            }
        )

    async def _calculate_rewards(
        self,
        symbol: str,
        analysis_date: date,
        current_price: float,
    ) -> ToolResult:
        """Calculate multi-period rewards for a prediction."""
        metadata = await self._get_metadata(symbol)
        beta = metadata.get("beta", 1.0)

        multi_period_data = await self._get_multi_period_data(
            symbol, analysis_date, current_price, beta
        )

        return ToolResult.success_result(
            data={
                "symbol": symbol,
                "analysis_date": analysis_date.isoformat(),
                "current_price": current_price,
                "beta": beta,
                "multi_period": multi_period_data,
            },
            metadata={
                "tool": "rl_backtest",
                "action": "calculate_rewards",
            }
        )

    async def _record_prediction(
        self,
        symbol: str,
        analysis_date: date,
        current_price: float,
        fair_value: float,
        fair_values: Dict[str, float],
        weights: Dict[str, float],
        tier_classification: str,
        context_features: Dict,
    ) -> ToolResult:
        """Record prediction to database.

        If context_features is empty, uses DataSourceManager to fetch
        consolidated data and extract RL features automatically.
        """
        if not self._outcome_tracker:
            return ToolResult.error_result("Outcome tracker not available")

        try:
            # Calculate multi-period data
            metadata = await self._get_metadata(symbol)
            beta = metadata.get("beta", 1.0)
            multi_period_data = await self._get_multi_period_data(
                symbol, analysis_date, current_price, beta
            )

            # If context_features not provided, use DataSourceManager
            if not context_features and self._data_source_manager:
                try:
                    consolidated = self._data_source_manager.get_data(
                        symbol=symbol,
                        as_of_date=analysis_date
                    )
                    context_features = consolidated.get_rl_features()
                    logger.debug(f"Auto-fetched {len(context_features)} RL features for {symbol}")
                except Exception as e:
                    logger.warning(f"Could not fetch RL features via DataSourceManager: {e}")
                    context_features = {}

            record_ids = []
            for position_type in ["LONG", "SHORT"]:
                record_id = self._outcome_tracker.record_prediction(
                    symbol=symbol,
                    analysis_date=analysis_date,
                    blended_fair_value=fair_value,
                    current_price=current_price,
                    fair_values=fair_values,
                    weights=weights,
                    tier_classification=tier_classification,
                    context_features=context_features,
                    per_model_rewards={"multi_period": multi_period_data},
                    position_type=position_type,
                )
                if record_id:
                    record_ids.append(record_id)

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "analysis_date": analysis_date.isoformat(),
                    "record_ids": record_ids,
                    "position_types": ["LONG", "SHORT"],
                    "context_features_count": len(context_features),
                },
                metadata={
                    "tool": "rl_backtest",
                    "action": "record_prediction",
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"Failed to record prediction: {e}")

    async def _get_historical_data(
        self,
        symbol: str,
        analysis_date: date,
    ) -> ToolResult:
        """Get historical price and shares data."""
        price = self._price_service.get_price(symbol, analysis_date)
        shares = self._shares_service.get_sec_shares(symbol, analysis_date)
        metadata = await self._get_metadata(symbol)

        return ToolResult.success_result(
            data={
                "symbol": symbol,
                "analysis_date": analysis_date.isoformat(),
                "price": price,
                "shares_outstanding": shares,
                "metadata": metadata,
            },
            metadata={
                "tool": "rl_backtest",
                "action": "get_historical_data",
            }
        )

    async def _get_context_features(
        self,
        symbol: str,
        analysis_date: date,
    ) -> ToolResult:
        """Get RL context features via DataSourceManager.

        Uses the unified DataSourceManager to fetch consolidated data
        and extract normalized features for RL model training.

        Features include:
        - Price returns (1d, 5d, 1m)
        - Technical indicators (RSI, SMA crossovers)
        - Macro indicators (VIX, GDPNow)
        - Sentiment (insider buys/sells, short interest)
        """
        if not self._data_source_manager:
            return ToolResult.error_result("DataSourceManager not initialized")

        try:
            consolidated = self._data_source_manager.get_data(
                symbol=symbol,
                as_of_date=analysis_date
            )

            features = consolidated.get_rl_features()

            return ToolResult.success_result(
                data={
                    "symbol": symbol,
                    "analysis_date": analysis_date.isoformat(),
                    "features": features,
                    "feature_count": len(features),
                    "sources_succeeded": consolidated.sources_succeeded,
                    "sources_failed": consolidated.sources_failed,
                    "data_quality": consolidated.overall_quality.name,
                },
                metadata={
                    "tool": "rl_backtest",
                    "action": "get_context_features",
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"Failed to get context features: {e}")

    async def _get_multi_period_data(
        self,
        symbol: str,
        analysis_date: date,
        current_price: float,
        beta: float,
    ) -> Dict[str, Any]:
        """Get multi-period prices, exit dates, and rewards."""
        prices = {}
        exit_dates = {}
        long_rewards = {}
        short_rewards = {}

        for period, days in HOLDING_PERIODS.items():
            target_date = analysis_date + timedelta(days=days)
            future_price = self._price_service.get_price(symbol, target_date)

            if future_price and future_price > 0:
                prices[period] = round(future_price, 2)
                exit_dates[period] = target_date.isoformat()

                # Calculate rewards using shared calculator
                # RewardCalculator.calculate() derives LONG/SHORT from predicted_fv vs price:
                # - predicted_fv > price_at_prediction => LONG
                # - predicted_fv < price_at_prediction => SHORT
                # We simulate this by setting fake fair values to force the desired direction
                if current_price > 0:
                    # For LONG: set predicted_fv higher than entry price
                    long_result = self._reward_calculator.calculate(
                        predicted_fv=current_price * 1.10,  # 10% above = LONG signal
                        price_at_prediction=current_price,
                        actual_price=future_price,
                        days=days,
                        beta=beta,
                    )
                    # For SHORT: set predicted_fv lower than entry price
                    short_result = self._reward_calculator.calculate(
                        predicted_fv=current_price * 0.90,  # 10% below = SHORT signal
                        price_at_prediction=current_price,
                        actual_price=future_price,
                        days=days,
                        beta=beta,
                    )
                    long_rewards[period] = round(long_result.reward, 4)
                    short_rewards[period] = round(short_result.reward, 4)
            else:
                prices[period] = None
                exit_dates[period] = None
                long_rewards[period] = None
                short_rewards[period] = None

        return {
            "entry_date": analysis_date.isoformat(),
            "prices": prices,
            "exit_dates": exit_dates,
            "long_rewards": long_rewards,
            "short_rewards": short_rewards,
        }

    async def _get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get symbol metadata from shared service."""
        if self._metadata_service:
            metadata = self._metadata_service.get_metadata(symbol)
            if metadata:
                # Convert SymbolMetadata dataclass to dict
                return {
                    "symbol": metadata.symbol,
                    "sector": metadata.sector,
                    "industry": metadata.industry,
                    "market_cap": metadata.market_cap,
                    "shares_outstanding": metadata.shares_outstanding,
                    "beta": metadata.beta,
                    "is_sp500": metadata.is_sp500,
                    "is_russell1000": metadata.is_russell1000,
                    "cik": metadata.cik,
                    "size_category": metadata.size_category,
                }
        return {}

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["run_backtest", "calculate_rewards", "record_prediction", "get_historical_data", "get_context_features"],
                    "description": "Action to perform",
                    "default": "run_backtest",
                },
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol (e.g., AAPL)",
                },
                "lookback_months": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of months to look back for backtesting",
                    "default": [12],
                },
                "analysis_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Date of analysis (ISO format)",
                },
                "current_price": {
                    "type": "number",
                    "description": "Price at analysis date",
                },
                "fair_value": {
                    "type": "number",
                    "description": "Blended fair value",
                },
                "fair_values": {
                    "type": "object",
                    "description": "Dict of model fair values",
                },
                "weights": {
                    "type": "object",
                    "description": "Dict of model weights",
                },
                "tier_classification": {
                    "type": "string",
                    "description": "Valuation tier classification",
                },
            },
            "required": ["symbol"],
        }
