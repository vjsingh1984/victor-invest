#!/usr/bin/env python3
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

"""Generate Crontab Entries for Scheduled Jobs.

Reads the scheduler configuration and generates crontab entries
that can be installed on the system.

Usage:
    python scripts/scheduled/generate_crontab.py
    python scripts/scheduled/generate_crontab.py --output /tmp/victor_crontab
    python scripts/scheduled/generate_crontab.py --install
    python scripts/scheduled/generate_crontab.py --show
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# Script mapping
JOB_SCRIPTS = {
    # Core data collectors
    "collect_treasury_data": "scripts/scheduled/collect_treasury_data.py",
    "refresh_macro_indicators": "scripts/scheduled/refresh_macro_indicators.py",
    "collect_insider_transactions": "scripts/scheduled/collect_insider_transactions.py",
    "collect_13f_filings": "scripts/scheduled/collect_13f_filings.py",
    "collect_short_interest": "scripts/scheduled/collect_short_interest.py",
    "update_market_regime": "scripts/scheduled/update_market_regime.py",
    "calculate_credit_risk": "scripts/scheduled/calculate_credit_risk.py",

    # ML/RL training data collectors
    "collect_analyst_estimates": "scripts/scheduled/collect_analyst_estimates.py",
    "collect_news_sentiment": "scripts/scheduled/collect_news_sentiment.py",
    "collect_fama_french_factors": "scripts/scheduled/collect_fama_french_factors.py",
    "collect_dividends": "scripts/scheduled/collect_dividends.py",
    "calculate_earnings_quality": "scripts/scheduled/calculate_earnings_quality.py",
}


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load scheduler configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_crontab(
    config: Dict[str, Any],
    python_path: str,
    project_root: Path,
    log_dir: Path,
) -> str:
    """Generate crontab entries from configuration."""
    lines = []

    # Header
    lines.append("# Victor Invest - Scheduled Data Collection Jobs")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append("#")
    lines.append("# Install with: crontab <filename>")
    lines.append("# View with: crontab -l")
    lines.append("")

    # Environment variables
    lines.append("# Environment")
    lines.append(f"SHELL=/bin/bash")
    lines.append(f"PATH=/usr/local/bin:/usr/bin:/bin")
    lines.append(f"PYTHONPATH={project_root}/src:{project_root}")
    lines.append("")

    # Jobs
    lines.append("# Jobs")
    lines.append("# minute hour day month weekday command")
    lines.append("")

    jobs = config.get("jobs", {})
    for job_name, job_config in jobs.items():
        if not job_config.get("enabled", True):
            lines.append(f"# DISABLED: {job_name}")
            continue

        if job_name not in JOB_SCRIPTS:
            lines.append(f"# UNKNOWN: {job_name}")
            continue

        schedule = job_config.get("schedule", {})
        cron_expr = schedule.get("cron", "")
        if not cron_expr:
            lines.append(f"# NO SCHEDULE: {job_name}")
            continue

        # Build command
        script_path = project_root / JOB_SCRIPTS[job_name]
        log_file = log_dir / f"{job_name}.log"

        # Add description as comment
        description = job_config.get("description", "")
        if description:
            lines.append(f"# {job_name}: {description}")

        # Cron entry with logging
        command = (
            f"cd {project_root} && "
            f"{python_path} {script_path} "
            f">> {log_file} 2>&1"
        )

        lines.append(f"{cron_expr} {command}")
        lines.append("")

    # Cleanup job (daily at 3AM)
    lines.append("# Log cleanup (keep last 7 days)")
    cleanup_cmd = f"find {log_dir} -name '*.log' -mtime +7 -delete"
    lines.append(f"0 3 * * * {cleanup_cmd}")
    lines.append("")

    return "\n".join(lines)


def get_python_path() -> str:
    """Get the path to the Python interpreter."""
    # Try to find the virtual environment python
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)

    # Fall back to current python
    return sys.executable


def install_crontab(crontab_content: str) -> bool:
    """Install crontab entries."""
    try:
        # Write to temp file
        temp_file = Path("/tmp/victor_crontab")
        temp_file.write_text(crontab_content)

        # Get existing crontab (without our entries)
        result = subprocess.run(
            ["crontab", "-l"],
            capture_output=True,
            text=True,
        )

        existing = result.stdout if result.returncode == 0 else ""

        # Remove old Victor Invest entries
        filtered_lines = []
        skip_block = False
        for line in existing.split("\n"):
            if "Victor Invest" in line:
                skip_block = True
            elif skip_block and line.strip() == "":
                skip_block = False
                continue
            elif not skip_block:
                filtered_lines.append(line)

        # Combine existing (filtered) with new
        combined = "\n".join(filtered_lines) + "\n\n" + crontab_content

        # Write combined crontab
        combined_file = Path("/tmp/victor_combined_crontab")
        combined_file.write_text(combined)

        # Install
        result = subprocess.run(
            ["crontab", str(combined_file)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error installing crontab: {result.stderr}")
            return False

        print("Crontab installed successfully!")
        return True

    except Exception as e:
        print(f"Error installing crontab: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate crontab entries for scheduled jobs"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "scheduler.yaml"),
        help="Path to scheduler configuration"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show generated crontab"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install crontab entries"
    )
    parser.add_argument(
        "--python",
        type=str,
        help="Path to Python interpreter"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Determine paths
    python_path = args.python or get_python_path()
    log_dir = PROJECT_ROOT / "logs"

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate crontab
    crontab_content = generate_crontab(
        config=config,
        python_path=python_path,
        project_root=PROJECT_ROOT,
        log_dir=log_dir,
    )

    # Handle output
    if args.show or (not args.output and not args.install):
        print(crontab_content)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(crontab_content)
        print(f"Crontab written to: {output_path}")

    if args.install:
        success = install_crontab(crontab_content)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
