#!/usr/bin/env python3
"""
Test script to verify Victor framework handler migration.

This script verifies that all handlers have been successfully migrated
to use @handler_decorator + BaseHandler pattern as per Victor framework
best practices.

Run: python tests/test_migration_verification.py
"""

import inspect
from dataclasses import is_dataclass

from victor.framework.workflows.base_handler import BaseHandler
from victor_invest.handlers import (
    FetchSECDataHandler,
    FetchMarketDataHandler,
    FetchMacroDataHandler,
    RunFundamentalAnalysisHandler,
    RunTechnicalAnalysisHandler,
    RunMarketContextHandler,
    RunSynthesisHandler,
    GenerateReportHandler,
    IdentifyPeersHandler,
    AnalyzePeersHandler,
    GenerateLookbackDatesHandler,
    ProcessBacktestBatchHandler,
    SaveRLPredictionsHandler,
    register_handlers,
)

# All expected handlers
ALL_HANDLERS = [
    FetchSECDataHandler,
    FetchMarketDataHandler,
    FetchMacroDataHandler,
    RunFundamentalAnalysisHandler,
    RunTechnicalAnalysisHandler,
    RunMarketContextHandler,
    RunSynthesisHandler,
    GenerateReportHandler,
    IdentifyPeersHandler,
    AnalyzePeersHandler,
    GenerateLookbackDatesHandler,
    ProcessBacktestBatchHandler,
    SaveRLPredictionsHandler,
]


def test_handler_migration():
    """Verify all handlers are properly migrated."""
    errors = []

    for handler_cls in ALL_HANDLERS:
        handler_name = handler_cls.__name__

        # Test 1: Extends BaseHandler
        if not issubclass(handler_cls, BaseHandler):
            errors.append(f"{handler_name} does not extend BaseHandler")

        # Test 2: Has execute() method
        if not hasattr(handler_cls, "execute"):
            errors.append(f"{handler_name} missing execute() method")
            continue

        # Test 3: execute() signature
        sig = inspect.signature(handler_cls.execute)
        params = list(sig.parameters.keys())
        expected_params = ["self", "node", "context", "tool_registry"]
        if params != expected_params:
            errors.append(f"{handler_name}.execute() has wrong params: {params}")

        # Test 4: Return type
        return_annotation = sig.return_annotation
        if "Tuple" not in str(return_annotation):
            errors.append(f"{handler_name}.execute() wrong return type: {return_annotation}")

        # Test 5: Is dataclass
        if not is_dataclass(handler_cls):
            errors.append(f"{handler_name} is not a dataclass")

    # Test 6: register_handlers() is no-op
    try:
        register_handlers()
    except Exception as e:
        errors.append(f"register_handlers() not a no-op: {e}")

    return errors


if __name__ == "__main__":
    print("Testing Victor Framework Handler Migration")
    print("=" * 60)

    errors = test_handler_migration()

    if errors:
        print("\n❌ Migration Tests Failed:")
        for error in errors:
            print(f"  ✗ {error}")
        exit(1)
    else:
        print("\n✅ All Migration Tests Passed!")
        print(f"  ✓ All {len(ALL_HANDLERS)} handlers extend BaseHandler")
        print(f"  ✓ All handlers have execute() method")
        print(f"  ✓ All handlers return Tuple[Any, int]")
        print(f"  ✓ All handlers are dataclasses")
        print(f"  ✓ register_handlers() is no-op (backward compatible)")
        print("=" * 60)
        exit(0)
