"""
Pytest configuration for metric_extraction tests.

Uses importlib to directly load modules without triggering the parent
package's __init__.py which has circular import issues.
"""

import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

# Get the source directory
# Path from tests/unit/infrastructure/sec/metric_extraction/conftest.py
# to src/ is up 5 levels (conftest -> metric_extraction -> sec -> infrastructure -> unit -> tests -> root)
ROOT_DIR = Path(__file__).resolve().parents[5]  # InvestiGator root
SRC_DIR = ROOT_DIR / "src"
METRIC_EXTRACTION_DIR = SRC_DIR / "investigator" / "infrastructure" / "sec" / "metric_extraction"


def _load_module_directly(module_name: str, file_path: Path):
    """Load a module directly from file without triggering parent package init."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# First, mock the canonical_mapper that orchestrator needs
canonical_mapper_mock = MagicMock()
canonical_mapper_mock.get_canonical_mapper = MagicMock(return_value=MagicMock())
sys.modules['investigator.infrastructure.sec.canonical_mapper'] = canonical_mapper_mock

# Create stub parent packages to allow submodule imports
for pkg_path in [
    'investigator',
    'investigator.infrastructure',
    'investigator.infrastructure.sec',
    'investigator.infrastructure.sec.metric_extraction',
]:
    if pkg_path not in sys.modules:
        stub = type(sys)('stub_' + pkg_path)
        stub.__path__ = []
        sys.modules[pkg_path] = stub

# Now load the actual metric_extraction modules directly
result_module = _load_module_directly(
    'investigator.infrastructure.sec.metric_extraction.result',
    METRIC_EXTRACTION_DIR / 'result.py'
)
strategies_module = _load_module_directly(
    'investigator.infrastructure.sec.metric_extraction.strategies',
    METRIC_EXTRACTION_DIR / 'strategies.py'
)
orchestrator_module = _load_module_directly(
    'investigator.infrastructure.sec.metric_extraction.orchestrator',
    METRIC_EXTRACTION_DIR / 'orchestrator.py'
)

# Make them accessible as submodules
sys.modules['investigator.infrastructure.sec.metric_extraction'].result = result_module
sys.modules['investigator.infrastructure.sec.metric_extraction'].strategies = strategies_module
sys.modules['investigator.infrastructure.sec.metric_extraction'].orchestrator = orchestrator_module
