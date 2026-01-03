"""
InvestiGator CLI Entry Point

Enables running InvestiGator as a module:
    python -m investigator [command] [options]
"""

import sys
from pathlib import Path

# Add root to path for cli_orchestrator import
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import and run CLI from root cli_orchestrator.py
from cli_orchestrator import cli

if __name__ == "__main__":
    cli()
