import asyncio

import pytest
from victor.tools.base import BaseTool as VictorBaseTool
from victor.tools.registry import ToolRegistry

from victor_invest.tools import register_investment_tools
from victor_invest.tools.base import BaseTool as LocalBaseTool
from victor_invest.tools.base import ToolResult as LocalToolResult


def test_register_investment_tools_registers_core_tools():
    registry = ToolRegistry()
    stats = register_investment_tools(registry)

    assert isinstance(stats, dict)
    assert stats.get("registered", 0) > 0

    for tool_name in [
        "sec_filing",
        "valuation",
        "technical_indicators",
        "market_data",
        "cache",
    ]:
        assert registry.get(tool_name) is not None


def test_local_base_tool_is_victor_base_tool_contract():
    class DummyLocalTool(LocalBaseTool):
        name = "dummy_local_contract_tool"
        description = "Dummy local tool"

        def get_schema(self):
            return {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            }

        async def execute(self, _exec_ctx, **kwargs):
            return LocalToolResult(success=True, output={"ok": True})

    tool = DummyLocalTool()
    assert isinstance(tool, VictorBaseTool)
    assert tool.parameters["required"] == ["symbol"]


def test_register_investment_tools_registers_native_tools_for_registry_execute(monkeypatch):
    class DummyLocalTool(LocalBaseTool):
        name = "dummy_local_tool"
        description = "Dummy local investment tool for adapter contract test"

        def get_schema(self):
            return {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                },
                "required": ["action"],
            }

        async def execute(self, _exec_ctx, action="", **kwargs):
            return LocalToolResult(
                success=True,
                output={"ok": True, "action": action},
                metadata={"source": "dummy"},
            )

    monkeypatch.setattr("victor_invest.tools.TOOL_CLASSES", [DummyLocalTool])

    registry = ToolRegistry()
    stats = register_investment_tools(registry, strict=True)

    assert stats["registered"] == 1
    assert not stats["errors"]
    registered_tool = registry.get("dummy_local_tool")
    assert registered_tool is not None
    assert isinstance(registered_tool, VictorBaseTool)
    assert isinstance(registered_tool, LocalBaseTool)

    result = asyncio.run(registry.execute("dummy_local_tool", {}, action="ping"))
    assert result.success is True
    assert result.output["ok"] is True
    assert result.output["action"] == "ping"


def test_register_investment_tools_rejects_non_victor_tools():
    class DummyNonVictorTool:
        name = "dummy_non_victor_tool"
        description = "Non-native tool used for registry contract validation"

        def __init__(self, config=None):
            self.config = config

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("victor_invest.tools.TOOL_CLASSES", [DummyNonVictorTool])
    try:
        stats = register_investment_tools(ToolRegistry(), strict=False)
        assert stats["registered"] == 0
        assert len(stats["errors"]) == 1
        assert "Expected Victor BaseTool instance" in stats["errors"][0]
    finally:
        monkeypatch.undo()


def test_register_investment_tools_strict_mode_raises_for_non_victor_tools():
    class DummyNonVictorTool:
        name = "dummy_non_victor_tool"
        description = "Non-native tool used for registry strict-mode validation"

        def __init__(self, config=None):
            self.config = config

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("victor_invest.tools.TOOL_CLASSES", [DummyNonVictorTool])
    try:
        with pytest.raises(TypeError, match="Expected Victor BaseTool instance"):
            register_investment_tools(ToolRegistry(), strict=True)
    finally:
        monkeypatch.undo()
