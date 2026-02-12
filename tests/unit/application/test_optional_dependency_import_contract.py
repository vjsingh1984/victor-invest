import builtins
import importlib
import sys
from types import ModuleType
from typing import Iterable


def _reimport_with_blocked_prefixes(module_name: str, blocked_prefixes: Iterable[str]) -> ModuleType:
    blocked = tuple(blocked_prefixes)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in blocked):
            raise ModuleNotFoundError(f"blocked optional dependency: {name}")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        for loaded in list(sys.modules):
            if loaded == module_name or loaded.startswith(f"{module_name}."):
                sys.modules.pop(loaded, None)
        return importlib.import_module(module_name)
    finally:
        builtins.__import__ = real_import


def test_synthesizer_import_isolated_from_optional_peer_and_reporting_deps():
    module = _reimport_with_blocked_prefixes(
        "investigator.application.synthesizer",
        blocked_prefixes=(
            "patterns.analysis.peer_comparison",
            "investigator.infrastructure.reporting",
            "reportlab",
            "yfinance",
        ),
    )
    assert hasattr(module, "InvestmentSynthesizer")


def test_report_generator_importable_when_reportlab_missing():
    module = _reimport_with_blocked_prefixes(
        "investigator.infrastructure.reporting.report_generator",
        blocked_prefixes=("reportlab",),
    )
    assert module.REPORTLAB_AVAILABLE is False
    assert module.ReportConfig().margin > 0


def test_reporting_package_importable_when_reportlab_missing():
    module = _reimport_with_blocked_prefixes(
        "investigator.infrastructure.reporting",
        blocked_prefixes=("reportlab",),
    )
    assert module is not None
    # Accessing only non-reportlab-backed export should remain safe.
    assert "ReportPayloadBuilder" in module.__all__
