import inspect

from victor_invest.tools import TOOL_CLASSES


def test_all_tool_execute_methods_accept_optional_exec_context():
    for tool_cls in TOOL_CLASSES:
        signature = inspect.signature(tool_cls.execute)
        exec_ctx = signature.parameters.get("_exec_ctx")
        assert exec_ctx is not None, f"{tool_cls.__name__}.execute must include _exec_ctx parameter"
        assert exec_ctx.default is None, f"{tool_cls.__name__}.execute _exec_ctx must default to None"
