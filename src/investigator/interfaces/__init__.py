"""
Interfaces Layer

External interfaces (CLI, API, etc.) for InvestiGator.
"""

from investigator.interfaces.cli import cli, create_cli

__all__ = ["cli", "create_cli"]
