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

"""Unit tests for investment domain handlers.

Tests all compute handlers in investigator.domain.handlers:
- MetadataFetchHandler
- PriceDataFetchHandler
- SECDataExtractHandler
- ValuationComputeHandler
- SectorValuationHandler
- BlendedValuationHandler
- RLWeightDecisionHandler
- OutcomeTrackingHandler
- TechnicalAnalysisHandler
"""

import pytest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any, Dict, Optional


# Mock victor framework imports for testing without victor-core
@dataclass
class MockNodeResult:
    node_id: str
    status: str
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    tool_calls_used: int = 0


class MockNodeStatus:
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"


@dataclass
class MockComputeNode:
    id: str
    input_mapping: Dict[str, Any]
    output_key: Optional[str] = None


class MockWorkflowContext:
    def __init__(self, data: Dict[str, Any] = None):
        self._data = data or {}

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value


@pytest.fixture
def mock_victor_imports():
    """Mock victor framework imports."""
    with patch.dict('sys.modules', {
        'victor.workflows.executor': MagicMock(
            NodeResult=MockNodeResult,
            NodeStatus=MockNodeStatus,
        ),
    }):
        yield


@pytest.fixture
def mock_metadata_service():
    """Mock SymbolMetadataService."""
    @dataclass
    class MockMetadata:
        symbol: str = "AAPL"
        sector: str = "Technology"
        industry: str = "Consumer Electronics"
        market_cap: float = 3000000000000
        shares_outstanding: int = 15000000000
        beta: float = 1.2
        is_sp500: bool = True
        is_russell1000: bool = True
        cik: str = "0000320193"

    service = MagicMock()
    service.get_metadata.return_value = MockMetadata()
    return service


@pytest.fixture
def mock_price_service():
    """Mock PriceService."""
    service = MagicMock()
    service.get_price.return_value = 185.50
    service.get_price_history.return_value = [
        {"date": "2024-01-01", "close": 180.0},
        {"date": "2024-01-02", "close": 182.0},
    ]
    service.get_volatility.return_value = 0.25
    return service


@pytest.fixture
def mock_shares_service():
    """Mock SharesService."""
    service = MagicMock()
    service.get_shares.return_value = 15000000000
    return service


@pytest.fixture
def mock_financial_service():
    """Mock FinancialDataService."""
    service = MagicMock()
    service.get_quarterly_metrics.return_value = [
        {"period": "Q4 2023", "revenue": 120000000000},
        {"period": "Q3 2023", "revenue": 115000000000},
    ]
    service.get_ttm_metrics.return_value = {
        "total_revenue": 400000000000,
        "net_income": 100000000000,
        "free_cash_flow": 90000000000,
        "stockholders_equity": 60000000000,
        "shares_outstanding": 15000000000,
    }
    return service


class TestHandlerBase:
    """Test HandlerBase utility methods."""

    def test_get_input_direct_value(self):
        """Test _get_input with direct value."""
        from investigator.domain.handlers import HandlerBase

        handler = HandlerBase()
        node = MockComputeNode(id="test", input_mapping={"symbol": "AAPL"})
        context = MockWorkflowContext()

        result = handler._get_input(node, context, "symbol")
        assert result == "AAPL"

    def test_get_input_context_reference(self):
        """Test _get_input with $ctx reference."""
        from investigator.domain.handlers import HandlerBase

        handler = HandlerBase()
        node = MockComputeNode(id="test", input_mapping={"symbol": "$ctx.target_symbol"})
        context = MockWorkflowContext({"target_symbol": "GOOGL"})

        result = handler._get_input(node, context, "symbol")
        assert result == "GOOGL"

    def test_get_input_default_value(self):
        """Test _get_input with missing key returns default."""
        from investigator.domain.handlers import HandlerBase

        handler = HandlerBase()
        node = MockComputeNode(id="test", input_mapping={})
        context = MockWorkflowContext()

        result = handler._get_input(node, context, "missing_key", "default_value")
        assert result == "default_value"


class TestMetadataFetchHandler:
    """Test MetadataFetchHandler."""

    @pytest.mark.asyncio
    async def test_fetch_metadata_success(self, mock_metadata_service):
        """Test successful metadata fetch."""
        from investigator.domain.handlers import MetadataFetchHandler

        # Patch both the service and the victor imports
        with patch('investigator.domain.handlers._get_metadata_service', return_value=mock_metadata_service), \
             patch.dict('sys.modules', {'victor.workflows.executor': MagicMock(
                 NodeResult=MockNodeResult,
                 NodeStatus=MockNodeStatus,
             )}):
            handler = MetadataFetchHandler()
            node = MockComputeNode(
                id="fetch_metadata",
                input_mapping={"symbol": "AAPL"},
                output_key="symbol_metadata",
            )
            context = MockWorkflowContext()

            result = await handler(node, context, None)

            assert result.node_id == "fetch_metadata"
            assert result.output["symbol"] == "AAPL"
            assert result.output["sector"] == "Technology"

    @pytest.mark.asyncio
    async def test_fetch_metadata_missing_symbol(self):
        """Test metadata fetch fails with missing symbol."""
        from investigator.domain.handlers import MetadataFetchHandler

        with patch.dict('sys.modules', {'victor.workflows.executor': MagicMock(
             NodeResult=MockNodeResult,
             NodeStatus=MockNodeStatus,
        )}):
            handler = MetadataFetchHandler()
            node = MockComputeNode(
                id="fetch_metadata",
                input_mapping={},  # No symbol
                output_key="symbol_metadata",
            )
            context = MockWorkflowContext()

            result = await handler(node, context, None)

            assert result.status == MockNodeStatus.FAILED
            assert "symbol" in result.error.lower()


class TestPriceDataFetchHandler:
    """Test PriceDataFetchHandler."""

    @pytest.mark.asyncio
    async def test_fetch_price_success(self, mock_price_service, mock_shares_service, mock_metadata_service):
        """Test successful price fetch."""
        from investigator.domain.handlers import PriceDataFetchHandler

        with patch('investigator.domain.handlers._get_price_service', return_value=mock_price_service), \
             patch('investigator.domain.handlers._get_shares_service', return_value=mock_shares_service), \
             patch('investigator.domain.handlers._get_metadata_service', return_value=mock_metadata_service), \
             patch.dict('sys.modules', {'victor.workflows.executor': MagicMock(
                 NodeResult=MockNodeResult,
                 NodeStatus=MockNodeStatus,
             )}):

            handler = PriceDataFetchHandler()
            node = MockComputeNode(
                id="fetch_prices",
                input_mapping={
                    "symbol": "AAPL",
                    "target_date": "2024-01-15",
                    "lookback_days": 90,
                },
                output_key="price_data",
            )
            context = MockWorkflowContext()

            result = await handler(node, context, None)

            assert result.node_id == "fetch_prices"
            assert result.status == MockNodeStatus.COMPLETED
            assert result.output["symbol"] == "AAPL"
            assert result.output["current_price"] == 185.50


class TestSECDataExtractHandler:
    """Test SECDataExtractHandler."""

    @pytest.mark.asyncio
    async def test_extract_sec_data_success(self, mock_financial_service):
        """Test successful SEC data extraction."""
        from investigator.domain.handlers import SECDataExtractHandler

        with patch('investigator.domain.handlers._get_financial_data_service', return_value=mock_financial_service), \
             patch.dict('sys.modules', {'victor.workflows.executor': MagicMock(
                 NodeResult=MockNodeResult,
                 NodeStatus=MockNodeStatus,
             )}):
            handler = SECDataExtractHandler()
            node = MockComputeNode(
                id="fetch_sec_data",
                input_mapping={
                    "symbol": "AAPL",
                    "num_quarters": 8,
                },
                output_key="financial_data",
            )
            context = MockWorkflowContext()

            result = await handler(node, context, None)

            assert result.node_id == "fetch_sec_data"
            assert result.status == MockNodeStatus.COMPLETED
            assert result.output["symbol"] == "AAPL"
            assert "quarterly_data" in result.output
            assert "ttm_data" in result.output


class TestBlendedValuationHandler:
    """Test BlendedValuationHandler."""

    @pytest.mark.asyncio
    async def test_blend_valuations_success(self):
        """Test successful valuation blending."""
        from investigator.domain.handlers import BlendedValuationHandler

        with patch.dict('sys.modules', {'victor.workflows.executor': MagicMock(
             NodeResult=MockNodeResult,
             NodeStatus=MockNodeStatus,
        )}):
            handler = BlendedValuationHandler()
            node = MockComputeNode(
                id="blend_valuations",
                input_mapping={
                    "valuation_results": {
                        "dcf": {"fair_value": 200.0},
                        "pe": {"fair_value": 180.0},
                        "ev_ebitda": {"fair_value": 190.0},
                    },
                    "weights": {
                        "dcf": 0.4,
                        "pe": 0.35,
                        "ev_ebitda": 0.25,
                    },
                },
                output_key="blended_fair_value",
            )
            context = MockWorkflowContext()

            result = await handler(node, context, None)

            assert result.node_id == "blend_valuations"
            assert result.status == MockNodeStatus.COMPLETED

            # Calculate expected: (200*0.4 + 180*0.35 + 190*0.25) / 1.0 = 190.5
            expected = (200 * 0.4 + 180 * 0.35 + 190 * 0.25) / 1.0
            assert abs(result.output["blended_fair_value"] - expected) < 0.01

    @pytest.mark.asyncio
    async def test_blend_valuations_partial_weights(self):
        """Test blending with partial weights (some models missing)."""
        from investigator.domain.handlers import BlendedValuationHandler

        with patch.dict('sys.modules', {'victor.workflows.executor': MagicMock(
             NodeResult=MockNodeResult,
             NodeStatus=MockNodeStatus,
        )}):
            handler = BlendedValuationHandler()
            node = MockComputeNode(
                id="blend_valuations",
                input_mapping={
                    "valuation_results": {
                        "dcf": {"fair_value": 200.0},
                        "pe": {"fair_value": None},  # Failed model
                        "ev_ebitda": {"fair_value": 190.0},
                    },
                    "weights": {
                        "dcf": 0.4,
                        "pe": 0.35,
                        "ev_ebitda": 0.25,
                    },
                },
                output_key="blended_fair_value",
            )
            context = MockWorkflowContext()

            result = await handler(node, context, None)

            assert result.status == MockNodeStatus.COMPLETED
            # Only DCF and EV/EBITDA contribute
            expected = (200 * 0.4 + 190 * 0.25) / (0.4 + 0.25)
            assert abs(result.output["blended_fair_value"] - expected) < 0.01
            assert len(result.output["models_used"]) == 2


class TestHandlerRegistry:
    """Test handler registration and lookup."""

    def test_handlers_registered(self):
        """Test all expected handlers are in HANDLERS dict."""
        from investigator.domain.handlers import HANDLERS

        expected_handlers = [
            "metadata_fetch",
            "price_data_fetch",
            "sec_data_extract",
            "valuation_compute",
            "sector_valuation",
            "blended_valuation",
            "rl_weight_decision",
            "outcome_tracking",
            "technical_analysis",
        ]

        for handler_name in expected_handlers:
            assert handler_name in HANDLERS, f"Handler '{handler_name}' not registered"

    def test_get_handler(self):
        """Test get_handler function."""
        from investigator.domain.handlers import get_handler, BlendedValuationHandler

        handler = get_handler("blended_valuation")
        assert handler is not None
        assert isinstance(handler, BlendedValuationHandler)

    def test_list_handlers(self):
        """Test list_handlers function."""
        from investigator.domain.handlers import list_handlers

        handlers = list_handlers()
        assert len(handlers) >= 9
        assert "valuation_compute" in handlers
        assert "rl_weight_decision" in handlers


class TestContextIntegration:
    """Test handler context integration."""

    @pytest.mark.asyncio
    async def test_output_stored_in_context(self):
        """Test handler output is stored in context."""
        from investigator.domain.handlers import BlendedValuationHandler

        with patch.dict('sys.modules', {'victor.workflows.executor': MagicMock(
             NodeResult=MockNodeResult,
             NodeStatus=MockNodeStatus,
        )}):
            handler = BlendedValuationHandler()
            node = MockComputeNode(
                id="blend",
                input_mapping={
                    "valuation_results": {"dcf": {"fair_value": 100.0}},
                    "weights": {"dcf": 1.0},
                },
                output_key="my_output",
            )
            context = MockWorkflowContext()

            await handler(node, context, None)

            # Check output was stored
            stored = context.get("my_output")
            assert stored is not None
            assert stored["blended_fair_value"] == 100.0

    @pytest.mark.asyncio
    async def test_nested_context_reference(self):
        """Test handlers can read nested context values."""
        from investigator.domain.handlers import BlendedValuationHandler

        with patch.dict('sys.modules', {'victor.workflows.executor': MagicMock(
             NodeResult=MockNodeResult,
             NodeStatus=MockNodeStatus,
        )}):
            handler = BlendedValuationHandler()
            node = MockComputeNode(
                id="blend",
                input_mapping={
                    "valuation_results": "$ctx.valuation_results",
                    "weights": "$ctx.model_weights",
                },
                output_key="result",
            )
            context = MockWorkflowContext({
                "valuation_results": {"dcf": {"fair_value": 150.0}},
                "model_weights": {"dcf": 1.0},
            })

            result = await handler(node, context, None)

            assert result.status == MockNodeStatus.COMPLETED
            assert result.output["blended_fair_value"] == 150.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
