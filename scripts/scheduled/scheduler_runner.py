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

"""Main Scheduler Runner for Data Collection Jobs.

This script provides a simple Python-based scheduler that can be used
as an alternative to cron. It reads the scheduler configuration and
runs jobs at their scheduled times.

For production, it's recommended to use the generated crontab instead.

Usage:
    python scripts/scheduled/scheduler_runner.py
    python scripts/scheduled/scheduler_runner.py --config config/scheduler.yaml
    python scripts/scheduled/scheduler_runner.py --run-now collect_treasury_data
    python scripts/scheduled/scheduler_runner.py --run-all
    python scripts/scheduled/scheduler_runner.py --list-jobs
"""

import argparse
import importlib
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.scheduled.base import setup_logging


# Job module mapping
JOB_MODULES = {
    "collect_treasury_data": "scripts.scheduled.collect_treasury_data",
    "refresh_macro_indicators": "scripts.scheduled.refresh_macro_indicators",
    "collect_insider_transactions": "scripts.scheduled.collect_insider_transactions",
    "collect_13f_filings": "scripts.scheduled.collect_13f_filings",
    "collect_short_interest": "scripts.scheduled.collect_short_interest",
    "update_market_regime": "scripts.scheduled.update_market_regime",
    "calculate_credit_risk": "scripts.scheduled.calculate_credit_risk",
}


class SchedulerRunner:
    """Simple Python scheduler for data collection jobs."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or PROJECT_ROOT / "config" / "scheduler.yaml"
        self.logger = setup_logging("scheduler_runner")
        self.config = self._load_config()
        self.running = True

    def _load_config(self) -> Dict[str, Any]:
        """Load scheduler configuration from YAML."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {"jobs": {}}

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all configured jobs."""
        jobs = []
        for job_name, job_config in self.config.get("jobs", {}).items():
            jobs.append({
                "name": job_name,
                "description": job_config.get("description", ""),
                "schedule": job_config.get("schedule", {}).get("cron", ""),
                "enabled": job_config.get("enabled", True),
            })
        return jobs

    def run_job(self, job_name: str) -> int:
        """Run a specific job by name."""
        if job_name not in JOB_MODULES:
            self.logger.error(f"Unknown job: {job_name}")
            return 1

        self.logger.info(f"Running job: {job_name}")

        try:
            # Import the job module
            module_path = JOB_MODULES[job_name]
            module = importlib.import_module(module_path)

            # Run the main function
            if hasattr(module, "main"):
                # Capture exit code without actually exiting
                original_exit = sys.exit
                exit_code = 0

                def mock_exit(code=0):
                    nonlocal exit_code
                    exit_code = code

                sys.exit = mock_exit
                try:
                    module.main()
                finally:
                    sys.exit = original_exit

                return exit_code
            else:
                self.logger.error(f"No main function in {module_path}")
                return 1

        except Exception as e:
            self.logger.exception(f"Job {job_name} failed: {e}")
            return 1

    def run_all(self) -> Dict[str, int]:
        """Run all enabled jobs in order."""
        results = {}
        jobs = self.config.get("jobs", {})

        # Sort by dependencies (simple topological sort)
        ordered_jobs = self._order_by_dependencies(jobs)

        for job_name in ordered_jobs:
            job_config = jobs.get(job_name, {})
            if not job_config.get("enabled", True):
                self.logger.info(f"Skipping disabled job: {job_name}")
                continue

            results[job_name] = self.run_job(job_name)

        return results

    def _order_by_dependencies(self, jobs: Dict[str, Any]) -> List[str]:
        """Order jobs by their dependencies."""
        # Simple implementation - just returns jobs in dependency order
        ordered = []
        remaining = set(jobs.keys())

        while remaining:
            for job_name in list(remaining):
                job_config = jobs.get(job_name, {})
                dependencies = job_config.get("dependencies", [])

                # Check if all dependencies are satisfied
                if all(dep in ordered for dep in dependencies):
                    ordered.append(job_name)
                    remaining.remove(job_name)
                    break
            else:
                # No job could be added - circular dependency or missing dep
                # Just add remaining jobs
                ordered.extend(remaining)
                break

        return ordered

    def _should_run_now(self, cron_expr: str) -> bool:
        """Check if a cron expression matches the current time."""
        # Simple cron parser for minute/hour/day/month/weekday
        try:
            parts = cron_expr.split()
            if len(parts) != 5:
                return False

            minute, hour, day, month, weekday = parts
            now = datetime.now()

            # Check each field
            if not self._matches_field(minute, now.minute):
                return False
            if not self._matches_field(hour, now.hour):
                return False
            if not self._matches_field(day, now.day):
                return False
            if not self._matches_field(month, now.month):
                return False
            if not self._matches_field(weekday, now.weekday()):
                return False

            return True

        except Exception:
            return False

    def _matches_field(self, field: str, value: int) -> bool:
        """Check if a cron field matches a value."""
        if field == "*":
            return True

        # Handle ranges (e.g., "1-5")
        if "-" in field:
            start, end = map(int, field.split("-"))
            return start <= value <= end

        # Handle lists (e.g., "1,15")
        if "," in field:
            return value in [int(x) for x in field.split(",")]

        # Handle step values (e.g., "*/5")
        if "/" in field:
            base, step = field.split("/")
            if base == "*":
                return value % int(step) == 0
            return False

        # Simple value match
        return int(field) == value

    def run_scheduler(self, interval: int = 60) -> None:
        """Run the scheduler loop."""
        self.logger.info("Starting scheduler...")

        # Set up signal handlers for graceful shutdown
        def handle_signal(signum, frame):
            self.logger.info("Received shutdown signal")
            self.running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while self.running:
            now = datetime.now()

            # Check each job
            for job_name, job_config in self.config.get("jobs", {}).items():
                if not job_config.get("enabled", True):
                    continue

                cron_expr = job_config.get("schedule", {}).get("cron", "")
                if not cron_expr:
                    continue

                if self._should_run_now(cron_expr):
                    self.logger.info(f"Triggering scheduled job: {job_name}")
                    self.run_job(job_name)

            # Sleep until next minute
            sleep_time = 60 - datetime.now().second
            time.sleep(sleep_time)

        self.logger.info("Scheduler stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Run scheduled data collection jobs"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to scheduler configuration file"
    )
    parser.add_argument(
        "--run-now",
        type=str,
        metavar="JOB_NAME",
        help="Run a specific job immediately"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all enabled jobs immediately"
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="List all configured jobs"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a daemon (continuous scheduler)"
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    runner = SchedulerRunner(config_path)

    if args.list_jobs:
        jobs = runner.list_jobs()
        print("\nConfigured Jobs:")
        print("-" * 60)
        for job in jobs:
            status = "enabled" if job["enabled"] else "disabled"
            print(f"  {job['name']:<30} [{status}]")
            print(f"    Schedule: {job['schedule']}")
            if job["description"]:
                print(f"    {job['description']}")
            print()
        sys.exit(0)

    if args.run_now:
        exit_code = runner.run_job(args.run_now)
        sys.exit(exit_code)

    if args.run_all:
        results = runner.run_all()
        failed = sum(1 for code in results.values() if code != 0)
        print(f"\nCompleted {len(results)} jobs, {failed} failed")
        sys.exit(1 if failed > 0 else 0)

    if args.daemon:
        runner.run_scheduler()
        sys.exit(0)

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
