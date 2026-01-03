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

"""Investment domain compute handlers for victor-invest.

Provides LLM-free handlers for investment workflows, integrating with
existing shared services:
- valuation_compute: Multi-model valuation via ParallelValuationOrchestrator
- rl_weight_decision: RL-based weights via RLModelWeightingService
- sec_data_extract: SEC data via FinancialDataService
- sector_valuation: Sector routing via SectorValuationRouter
- price_data_fetch: Prices via PriceService
- technical_analysis: Indicators via TechnicalAnalysisService
- metadata_fetch: Symbol metadata via SymbolMetadataService

Usage:
    from investigator.domain.handlers import register_handlers
    register_handlers()

    # In YAML workflow:
    - id: run_valuation
      type: compute
      handler: valuation_compute
      inputs:
        symbol: $ctx.symbol
        models: [dcf, pe, ev_ebitda]
      output: valuation_results
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import ComputeNode
    from victor.workflows.executor import NodeResult, NodeStatus, WorkflowContext

logger = logging.getLogger(__name__)


# Lazy service imports to avoid circular dependencies
def _get_price_service():
    from investigator.domain.services.market_data import PriceService
    return PriceService()


def _get_shares_service():
    from investigator.domain.services.market_data import SharesService
    return SharesService()


def _get_metadata_service():
    from investigator.domain.services.market_data import SymbolMetadataService
    return SymbolMetadataService()


def _get_technical_service():
    from investigator.domain.services.market_data import get_technical_analysis_service
    return get_technical_analysis_service()


def _get_financial_data_service():
    from investigator.domain.services.valuation_shared import FinancialDataService
    return FinancialDataService()


def _get_sector_multiples_service():
    from investigator.domain.services.valuation_shared import (
        ValuationConfigService, SectorMultiplesService
    )
    config = ValuationConfigService()
    return SectorMultiplesService(config)


class HandlerBase:
    """Base class for investment handlers with common input extraction."""

    def _get_input(
        self, node: "ComputeNode", context: "WorkflowContext", key: str, default: Any = None
    ) -> Any:
        """Get input value from node mapping or context."""
        value = node.input_mapping.get(key)
        if isinstance(value, str) and value.startswith("$ctx."):
            return context.get(value[5:]) or default
        return value if value is not None else default


@dataclass
class ValuationComputeHandler(HandlerBase):
    """Execute multi-model valuation without LLM.

    Uses ParallelValuationOrchestrator to run DCF, PE, EV/EBITDA, etc.
    in parallel and aggregate results with dynamic weighting.

    Example YAML:
        - id: run_valuation
          type: compute
          handler: valuation_compute
          inputs:
            symbol: $ctx.symbol
            financials: $ctx.financial_data
            ratios: $ctx.ratios
            as_of_date: $ctx.analysis_date
          output: valuation_results
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        # Extract inputs
        symbol = self._get_input(node, context, "symbol")
        financials = self._get_input(node, context, "financials", {})
        ratios = self._get_input(node, context, "ratios", {})
        as_of_date = self._get_input(node, context, "as_of_date")

        if not symbol:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error="No 'symbol' input provided",
                duration_seconds=time.time() - start_time,
            )

        try:
            from investigator.domain.services.parallel_valuation_orchestrator import (
                ParallelValuationOrchestrator,
            )
            from investigator.domain.services.dynamic_model_weighting import (
                DynamicModelWeightingService,
            )

            # Get metadata for sector-specific routing
            metadata_service = _get_metadata_service()
            metadata = metadata_service.get_metadata(symbol)
            sector = metadata.sector if metadata else "Unknown"
            industry = metadata.industry if metadata else "Unknown"

            # Enrich financials with metadata
            financials["sector"] = sector
            financials["industry"] = industry
            financials["market_cap"] = ratios.get("market_cap", 0)

            # Get dynamic weights for model blending
            import yaml
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            valuation_config = config.get("valuation", {})
            weighting_service = DynamicModelWeightingService(valuation_config)
            weights, tier, weight_audit = weighting_service.determine_weights(
                symbol=symbol,
                financials=financials,
                ratios=ratios,
            )

            # Run valuation orchestrator
            orchestrator = ParallelValuationOrchestrator()
            results = await orchestrator.execute_valuation(
                frameworks=list(weights.keys()),
                rule_of_40_score=ratios.get("rule_of_40_score", 0),
                revenue_growth_pct=ratios.get("revenue_growth_pct", 0),
                fcf_margin_pct=ratios.get("fcf_margin", 0) * 100,
                financials=financials,
            )

            output = {
                "symbol": symbol,
                "sector": sector,
                "industry": industry,
                "tier": tier,
                "model_weights": weights,
                "results": results.model_results if results else {},
                "blended_fair_value": results.blended_fair_value if results else None,
                "confidence": results.confidence if results else 0,
                "weight_audit": weight_audit,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Valuation failed for {symbol}: {e}", exc_info=True)
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class RLWeightDecisionHandler(HandlerBase):
    """RL-based model weight decision without LLM.

    Uses RLModelWeightingService with trained dual policy (Technical + Fundamental)
    to determine optimal model weights based on company context, industry, and
    market conditions. Includes industry-level granularity for weight selection.

    Example YAML:
        - id: determine_weights
          type: compute
          handler: rl_weight_decision
          inputs:
            symbol: $ctx.symbol
            financials: $ctx.financial_data
            ratios: $ctx.ratios
            use_dual_policy: true
            technical_policy_path: data/rl_models/active_technical_policy.pkl
            fundamental_policy_path: data/rl_models/active_fundamental_policy.pkl
          output: model_weights
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        symbol = self._get_input(node, context, "symbol")
        financials = self._get_input(node, context, "financials", {})
        ratios = self._get_input(node, context, "ratios", {})

        # Dual policy paths (preferred)
        use_dual_policy = self._get_input(node, context, "use_dual_policy", True)
        technical_policy_path = self._get_input(
            node, context, "technical_policy_path",
            "data/rl_models/active_technical_policy.pkl"
        )
        fundamental_policy_path = self._get_input(
            node, context, "fundamental_policy_path",
            "data/rl_models/active_fundamental_policy.pkl"
        )

        # Legacy single policy path (fallback)
        policy_path = self._get_input(
            node, context, "policy_path", "data/rl_models/active_policy.pkl"
        )

        if not symbol:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error="No 'symbol' input provided",
                duration_seconds=time.time() - start_time,
            )

        try:
            from investigator.domain.services.rl.rl_model_weighting import (
                RLModelWeightingService,
            )
            from investigator.domain.services.dynamic_model_weighting import (
                DynamicModelWeightingService,
            )

            # Load config for fallback service
            import yaml
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            valuation_config = config.get("valuation", {})
            fallback_service = DynamicModelWeightingService(valuation_config)

            # Create RL weighting service with dual policy support
            rl_service = RLModelWeightingService(
                rl_enabled=True,
                fallback_service=fallback_service,
                policy_path=policy_path,
                normalizer_path=policy_path.replace("policy.pkl", "normalizer.pkl"),
                use_dual_policy=use_dual_policy,
                technical_policy_path=technical_policy_path,
                fundamental_policy_path=fundamental_policy_path,
            )

            # Determine weights using RL policy (dual or single)
            weights, tier, weight_audit = rl_service.determine_weights(
                symbol=symbol,
                financials=financials,
                ratios=ratios,
            )

            # Check which policy was used
            dual_active = rl_service.is_dual_policy_active()
            single_active = hasattr(rl_service, 'policy') and rl_service.policy and rl_service.policy.is_ready()

            # Extract additional metadata from audit trail
            position = None
            position_confidence = None
            holding_period = None
            industry_category = None
            if weight_audit and hasattr(weight_audit, 'metadata') and weight_audit.metadata:
                position = weight_audit.metadata.get("position")
                position_confidence = weight_audit.metadata.get("position_confidence")
                holding_period = weight_audit.metadata.get("holding_period")
                industry_category = weight_audit.metadata.get("industry_category")

            output = {
                "symbol": symbol,
                "weights": weights,
                "tier": tier,
                "policy_used": (
                    "dual_rl" if dual_active else
                    "single_rl" if single_active else
                    "fallback_rule_based"
                ),
                "rl_active": dual_active or single_active,
                "dual_policy_active": dual_active,
                "position_signal": position,  # -1=short, 0=skip, 1=long
                "position_confidence": position_confidence,
                "holding_period": holding_period,
                "industry_category": industry_category,
                "weight_audit": weight_audit,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"RL weight decision failed: {e}", exc_info=True)
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class SECDataExtractHandler(HandlerBase):
    """Extract financial data from SEC filings without LLM.

    Uses FinancialDataService to fetch structured quarterly/annual
    data from SEC companyfacts with point-in-time accuracy.

    Example YAML:
        - id: fetch_sec_data
          type: compute
          handler: sec_data_extract
          inputs:
            symbol: $ctx.symbol
            num_quarters: 12
            as_of_date: $ctx.analysis_date
          output: financial_data
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        symbol = self._get_input(node, context, "symbol")
        num_quarters = self._get_input(node, context, "num_quarters", 12)
        as_of_date = self._get_input(node, context, "as_of_date")

        if not symbol:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error="No 'symbol' input provided",
                duration_seconds=time.time() - start_time,
            )

        try:
            # Use FinancialDataService for structured data retrieval
            financial_service = _get_financial_data_service()

            # Convert as_of_date if string
            if isinstance(as_of_date, str):
                as_of_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            elif as_of_date is None:
                as_of_date = date.today()

            # Get quarterly metrics (structured format)
            quarterly_data = financial_service.get_quarterly_metrics(
                symbol=symbol,
                as_of_date=as_of_date,
                num_quarters=num_quarters,
            )

            # Get TTM (trailing twelve months) summary
            ttm_data = financial_service.get_ttm_metrics(
                symbol=symbol,
                as_of_date=as_of_date,
            )

            # Calculate key ratios from TTM data
            ratios = {}
            if ttm_data:
                revenue = ttm_data.get("total_revenue", 0) or 0
                net_income = ttm_data.get("net_income", 0) or 0
                fcf = ttm_data.get("free_cash_flow", 0) or 0
                equity = ttm_data.get("stockholders_equity", 0) or 1
                shares = ttm_data.get("shares_outstanding", 0) or 1

                ratios = {
                    "net_margin": net_income / revenue if revenue > 0 else 0,
                    "fcf_margin": fcf / revenue if revenue > 0 else 0,
                    "roe": net_income / equity if equity > 0 else 0,
                    "ttm_eps": net_income / shares if shares > 0 else 0,
                }

            output = {
                "symbol": symbol,
                "as_of_date": str(as_of_date),
                "quarterly_data": quarterly_data,
                "ttm_data": ttm_data,
                "ratios": ratios,
                "quarters_available": len(quarterly_data) if quarterly_data else 0,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"SEC data extraction failed for {symbol}: {e}", exc_info=True)
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class SectorValuationHandler(HandlerBase):
    """Sector-specific valuation routing without LLM.

    Uses SectorValuationRouter to dispatch to appropriate sector
    valuation (banks, REITs, biotech, semiconductors, insurance, defense).

    Example YAML:
        - id: sector_valuation
          type: compute
          handler: sector_valuation
          inputs:
            symbol: $ctx.symbol
            financials: $ctx.financial_data
            current_price: $ctx.current_price
          output: sector_valuation_result
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        symbol = self._get_input(node, context, "symbol")
        financials = self._get_input(node, context, "financials", {})
        current_price = self._get_input(node, context, "current_price", 0)

        if not symbol:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error="No 'symbol' input provided",
                duration_seconds=time.time() - start_time,
            )

        try:
            # Get metadata for sector routing
            metadata_service = _get_metadata_service()
            metadata = metadata_service.get_metadata(symbol)
            sector = metadata.sector if metadata else "Unknown"
            industry = metadata.industry if metadata else "Unknown"
            shares = financials.get("shares_outstanding", 1) or 1

            result = {"sector": sector, "industry": industry, "model_used": None, "fair_value": None}

            # Route to sector-specific valuation
            if sector == "Financials" and "bank" in industry.lower():
                from investigator.domain.services.valuation.bank_valuation import value_bank
                bank_result = value_bank(symbol, financials, current_price, shares)
                if bank_result and bank_result.fair_value:
                    result["model_used"] = "bank_pb"
                    result["fair_value"] = bank_result.fair_value

            elif sector == "Real Estate":
                from investigator.domain.services.valuation.reit_valuation import value_reit
                reit_result = value_reit(symbol, financials, current_price, shares)
                if reit_result and reit_result.fair_value:
                    result["model_used"] = "reit_ffo"
                    result["fair_value"] = reit_result.fair_value

            elif "insurance" in industry.lower():
                from investigator.domain.services.valuation.insurance_valuation import value_insurance_company
                ins_result = value_insurance_company(symbol, financials, current_price, shares)
                if ins_result:
                    result["model_used"] = "insurance_combined_ratio"
                    result["fair_value"] = ins_result.get("fair_value")

            elif "semiconductor" in industry.lower():
                from investigator.domain.services.valuation.semiconductor_valuation import value_semiconductor
                semi_result = value_semiconductor(symbol, financials, current_price, shares)
                if semi_result and semi_result.fair_value:
                    result["model_used"] = "semiconductor_cycle"
                    result["fair_value"] = semi_result.fair_value

            output = {
                "symbol": symbol,
                "sector": sector,
                "industry": industry,
                "valuation_model": result.get("model_used"),
                "fair_value": result.get("fair_value"),
                "is_sector_specific": result.get("model_used") is not None,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Sector valuation failed for {symbol}: {e}", exc_info=True)
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class PriceDataFetchHandler(HandlerBase):
    """Fetch market price data without LLM.

    Uses PriceService for current/historical prices and SharesService
    for split-adjusted share counts.

    Example YAML:
        - id: fetch_prices
          type: compute
          handler: price_data_fetch
          inputs:
            symbol: $ctx.symbol
            target_date: $ctx.analysis_date
            lookback_days: 365
          output: price_data
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        symbol = self._get_input(node, context, "symbol")
        target_date = self._get_input(node, context, "target_date")
        lookback_days = self._get_input(node, context, "lookback_days", 365)

        if not symbol:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error="No 'symbol' input provided",
                duration_seconds=time.time() - start_time,
            )

        try:
            price_service = _get_price_service()
            shares_service = _get_shares_service()
            metadata_service = _get_metadata_service()

            # Convert target_date
            if isinstance(target_date, str):
                target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
            elif target_date is None:
                target_date = date.today()

            # Get current price
            current_price = price_service.get_price(symbol, target_date)

            # Get price history
            from datetime import timedelta
            start_date = target_date - timedelta(days=lookback_days)
            price_history = price_service.get_price_history(symbol, start_date, target_date)

            # Get volatility
            volatility = price_service.get_volatility(symbol, days=30, end_date=target_date)

            # Get metadata for beta
            metadata = metadata_service.get_metadata(symbol)
            beta = metadata.beta if metadata else 1.0

            # Get shares (split-adjusted)
            shares = shares_service.get_shares(symbol, target_date)

            output = {
                "symbol": symbol,
                "target_date": str(target_date),
                "current_price": current_price,
                "shares_outstanding": shares,
                "market_cap": current_price * shares if current_price and shares else None,
                "beta": beta,
                "volatility": volatility,
                "price_history_days": len(price_history) if price_history else 0,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Price data fetch failed for {symbol}: {e}", exc_info=True)
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class TechnicalAnalysisHandler(HandlerBase):
    """Compute technical indicators without LLM.

    Uses TechnicalAnalysisService for RSI, MACD, OBV, ADX, etc.
    and entry/exit signal generation.

    Example YAML:
        - id: technical_analysis
          type: compute
          handler: technical_analysis
          inputs:
            symbol: $ctx.symbol
            analysis_date: $ctx.analysis_date
            fair_value: $ctx.blended_fair_value
          output: technical_features
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        symbol = self._get_input(node, context, "symbol")
        analysis_date = self._get_input(node, context, "analysis_date")
        fair_value = self._get_input(node, context, "fair_value")
        lookback_days = self._get_input(node, context, "lookback_days", 365)

        if not symbol:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error="No 'symbol' input provided",
                duration_seconds=time.time() - start_time,
            )

        try:
            technical_service = _get_technical_service()

            # Convert analysis_date
            if isinstance(analysis_date, str):
                analysis_date = datetime.strptime(analysis_date, "%Y-%m-%d").date()
            elif analysis_date is None:
                analysis_date = date.today()

            # Get technical features
            features = technical_service.get_technical_features(
                symbol=symbol,
                analysis_date=analysis_date,
                lookback_days=lookback_days,
                fair_value=fair_value,
            )

            # Get entry/exit signals
            signals = technical_service.get_entry_exit_signals(
                symbol=symbol,
                analysis_date=analysis_date,
                fair_value=fair_value,
                lookback_days=lookback_days,
            )

            output = {
                "symbol": symbol,
                "analysis_date": str(analysis_date),
                # Technical indicators
                "rsi_14": features.rsi_14 if features else None,
                "macd_histogram": features.macd_histogram if features else None,
                "adx_14": features.adx_14 if features else None,
                "obv_trend": features.obv_trend if features else None,
                "stoch_k": features.stoch_k if features else None,
                "mfi_14": features.mfi_14 if features else None,
                "volatility": features.volatility if features else None,
                # Entry/exit signals
                "entry_signal_strength": features.entry_signal_strength if features else 0,
                "exit_signal_strength": features.exit_signal_strength if features else 0,
                "signal_confluence": features.signal_confluence if features else 0,
                "risk_reward_ratio": features.risk_reward_ratio if features else 1.0,
                # Trend context
                "price_vs_sma_20": features.price_vs_sma_20 if features else 0,
                "price_vs_sma_50": features.price_vs_sma_50 if features else 0,
                "price_vs_sma_200": features.price_vs_sma_200 if features else 0,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}", exc_info=True)
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class MetadataFetchHandler(HandlerBase):
    """Fetch symbol metadata without LLM.

    Uses SymbolMetadataService for sector, industry, market cap, beta.

    Example YAML:
        - id: fetch_metadata
          type: compute
          handler: metadata_fetch
          inputs:
            symbol: $ctx.symbol
          output: symbol_metadata
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        symbol = self._get_input(node, context, "symbol")

        if not symbol:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error="No 'symbol' input provided",
                duration_seconds=time.time() - start_time,
            )

        try:
            metadata_service = _get_metadata_service()
            metadata = metadata_service.get_metadata(symbol)

            output = {
                "symbol": symbol,
                "sector": metadata.sector if metadata else "Unknown",
                "industry": metadata.industry if metadata else "Unknown",
                "market_cap": metadata.market_cap if metadata else None,
                "shares_outstanding": metadata.shares_outstanding if metadata else None,
                "beta": metadata.beta if metadata else 1.0,
                "is_sp500": metadata.is_sp500 if metadata else False,
                "is_russell1000": metadata.is_russell1000 if metadata else False,
                "cik": metadata.cik if metadata else None,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Metadata fetch failed for {symbol}: {e}", exc_info=True)
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class OutcomeTrackingHandler(HandlerBase):
    """Record prediction for RL training without LLM.

    Uses OutcomeTracker to record predictions for future reward calculation.

    Example YAML:
        - id: track_prediction
          type: compute
          handler: outcome_tracking
          inputs:
            symbol: $ctx.symbol
            blended_fair_value: $ctx.blended_fair_value
            current_price: $ctx.current_price
            model_weights: $ctx.model_weights
            tier: $ctx.tier
          output: tracking_result
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        symbol = self._get_input(node, context, "symbol")
        blended_fv = self._get_input(node, context, "blended_fair_value")
        current_price = self._get_input(node, context, "current_price")
        model_weights = self._get_input(node, context, "model_weights", {})
        model_fair_values = self._get_input(node, context, "model_fair_values", {})
        tier = self._get_input(node, context, "tier", "unknown")

        if not symbol or not blended_fv or not current_price:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error="Missing required inputs (symbol, blended_fair_value, current_price)",
                duration_seconds=time.time() - start_time,
            )

        try:
            from investigator.domain.services.rl.outcome_tracker import OutcomeTracker

            tracker = OutcomeTracker()
            record_id = tracker.record_prediction(
                symbol=symbol,
                analysis_date=date.today(),
                blended_fair_value=blended_fv,
                current_price=current_price,
                model_fair_values=model_fair_values,
                model_weights=model_weights,
                tier_classification=tier,
            )

            output = {
                "symbol": symbol,
                "record_id": record_id,
                "tracked": record_id is not None,
                "predicted_upside_pct": ((blended_fv / current_price) - 1) * 100 if current_price > 0 else 0,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Outcome tracking failed for {symbol}: {e}", exc_info=True)
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class BlendedValuationHandler(HandlerBase):
    """Compute blended fair value from multiple models without LLM.

    Applies weights to valuation results and computes blended value.

    Example YAML:
        - id: blend_valuations
          type: compute
          handler: blended_valuation
          inputs:
            valuation_results: $ctx.valuation_results
            weights: $ctx.model_weights
          output: blended_fair_value
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        valuation_results = self._get_input(node, context, "valuation_results", {})
        weights = self._get_input(node, context, "weights", {})

        try:
            # Compute weighted average
            total_weight = 0.0
            weighted_sum = 0.0
            model_contributions = {}

            for model, result in valuation_results.items():
                fair_value = result.get("fair_value") if isinstance(result, dict) else None
                weight = weights.get(model, 0.0)

                if fair_value is not None and weight > 0:
                    contribution = fair_value * weight
                    weighted_sum += contribution
                    total_weight += weight
                    model_contributions[model] = {
                        "fair_value": fair_value,
                        "weight": weight,
                        "contribution": contribution,
                    }

            blended_value = weighted_sum / total_weight if total_weight > 0 else None

            output = {
                "blended_fair_value": blended_value,
                "total_weight": total_weight,
                "model_contributions": model_contributions,
                "models_used": list(model_contributions.keys()),
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Blended valuation failed: {e}", exc_info=True)
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


# Handler instances - all investment domain handlers
HANDLERS = {
    # Data fetching handlers
    "metadata_fetch": MetadataFetchHandler(),
    "price_data_fetch": PriceDataFetchHandler(),
    "sec_data_extract": SECDataExtractHandler(),
    # Valuation handlers
    "valuation_compute": ValuationComputeHandler(),
    "sector_valuation": SectorValuationHandler(),
    "blended_valuation": BlendedValuationHandler(),
    # RL/ML handlers
    "rl_weight_decision": RLWeightDecisionHandler(),
    "outcome_tracking": OutcomeTrackingHandler(),
    # Technical analysis handlers
    "technical_analysis": TechnicalAnalysisHandler(),
}


def register_handlers() -> None:
    """Register investment domain handlers with the workflow executor.

    Call this function during application startup to make handlers
    available for use in YAML workflows.

    Example:
        from investigator.domain.handlers import register_handlers
        register_handlers()
    """
    try:
        from victor.workflows.executor import register_compute_handler

        for name, handler in HANDLERS.items():
            register_compute_handler(name, handler)
            logger.debug(f"Registered investment handler: {name}")

        logger.info(f"Registered {len(HANDLERS)} investment domain handlers")
    except ImportError:
        logger.warning("Victor workflows not available, handlers not registered")


def get_handler(name: str) -> Any:
    """Get a handler by name."""
    return HANDLERS.get(name)


def list_handlers() -> List[str]:
    """List all available handler names."""
    return list(HANDLERS.keys())


__all__ = [
    # Base class
    "HandlerBase",
    # Handler classes
    "MetadataFetchHandler",
    "PriceDataFetchHandler",
    "SECDataExtractHandler",
    "ValuationComputeHandler",
    "SectorValuationHandler",
    "BlendedValuationHandler",
    "RLWeightDecisionHandler",
    "OutcomeTrackingHandler",
    "TechnicalAnalysisHandler",
    # Registry
    "HANDLERS",
    "register_handlers",
    "get_handler",
    "list_handlers",
]
