"""
InvestiGator CLI Module

This module re-exports the CLI from the root cli_orchestrator.py.
The actual CLI implementation is in cli_orchestrator.py at the project root.
"""

import sys
from pathlib import Path

# Ensure root directory is in path for cli_orchestrator import
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Re-export CLI from root cli_orchestrator.py
from cli_orchestrator import cli

__all__ = ["cli"]
