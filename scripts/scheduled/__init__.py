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

"""Scheduled Data Collection Scripts.

This package contains scripts for automated data collection that can be
run via cron, systemd timers, or the built-in scheduler.

Scripts:
- collect_treasury_data.py: Treasury yield curve (daily 6PM ET)
- refresh_macro_indicators.py: FRED macro data (daily 9AM ET)
- collect_insider_transactions.py: SEC Form 4 (every 4 hours)
- collect_13f_filings.py: SEC Form 13F (daily 7AM ET)
- collect_short_interest.py: FINRA short interest (bi-monthly)
- update_market_regime.py: Market regime classification (daily 6:30PM ET)
- calculate_credit_risk.py: Credit risk scores (weekly Sunday 8PM ET)

Configuration:
- config/scheduler.yaml: Schedule and job configuration

Usage:
    # Run individual script
    python scripts/scheduled/collect_treasury_data.py

    # Run with scheduler
    python scripts/scheduled/scheduler_runner.py

    # Generate crontab
    python scripts/scheduled/generate_crontab.py
"""
