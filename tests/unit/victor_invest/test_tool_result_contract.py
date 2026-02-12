from victor.tools.base import ToolResult as VictorToolResult

from victor_invest.tools.base import ToolResult


def test_local_tool_result_is_victor_tool_result():
    result = ToolResult.create_success(
        output={"ok": True, "symbol": "AAPL"},
        metadata={"source": "test"},
    )

    assert isinstance(result, VictorToolResult)
    assert result.success is True
    assert result.output["ok"] is True
    assert result.output["symbol"] == "AAPL"
    assert not hasattr(result, "warnings")
    assert result.metadata["source"] == "test"

    as_dict = result.to_dict()
    assert as_dict["success"] is True
    assert as_dict["output"]["symbol"] == "AAPL"
    assert "data" not in as_dict
    assert "warnings" not in as_dict


def test_local_tool_result_error_factory():
    result = ToolResult.create_failure(
        error="Symbol not found",
        metadata={"symbol": "UNKNOWN"},
    )

    assert isinstance(result, VictorToolResult)
    assert result.success is False
    assert result.error == "Symbol not found"
    assert not hasattr(result, "warnings")
    assert result.metadata["symbol"] == "UNKNOWN"


def test_local_tool_result_uses_victor_native_factories_only():
    assert hasattr(ToolResult, "create_success")
    assert hasattr(ToolResult, "create_failure")
    assert not hasattr(ToolResult, "success_result")
    assert not hasattr(ToolResult, "error_result")


def test_local_tool_result_factory_signatures_are_strict_victor_style():
    import inspect

    success_sig = inspect.signature(ToolResult.create_success)
    failure_sig = inspect.signature(ToolResult.create_failure)

    assert "data" not in success_sig.parameters
    assert "warnings" not in success_sig.parameters
    assert "data" not in failure_sig.parameters
    assert "warnings" not in failure_sig.parameters
