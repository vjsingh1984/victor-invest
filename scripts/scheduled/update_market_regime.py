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

"""Update Market Regime Classification.

Schedule: Daily at 6:30PM ET (after market close and treasury data collection)
Dependencies: collect_treasury_data, refresh_macro_indicators

Updates:
- Current market regime classification
- Credit cycle phase
- Volatility regime
- Recession probability
- Risk-off signals

Usage:
    python scripts/scheduled/update_market_regime.py
    python scripts/scheduled/update_market_regime.py --lookback 60
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.scheduled.base import (
    BaseCollector,
    CollectionMetrics,
    get_database_connection,
)


class MarketRegimeUpdater(BaseCollector):
    """Updater for market regime classification."""

    def __init__(self, lookback_days: int = 30):
        super().__init__("update_market_regime")
        self.lookback_days = lookback_days

    def collect(self) -> CollectionMetrics:
        """Update market regime classification."""
        try:
            # Import the market regime analyzer
            from investigator.domain.services.market_regime_analyzer import (
                get_market_regime_analyzer,
            )

            analyzer = get_market_regime_analyzer()

            self.logger.info(
                f"Updating market regime classification (lookback: {self.lookback_days} days)"
            )

            # Get comprehensive regime analysis
            regime_data = analyzer.get_comprehensive_regime()

            if not regime_data:
                self.logger.warning("No regime data returned from analyzer")
                self.metrics.warnings.append("Empty regime data")
                return self.metrics

            self.metrics.records_processed = 1

            # Store in database
            conn = get_database_connection()
            cursor = conn.cursor()

            # Store current regime snapshot
            cursor.execute("""
                INSERT INTO market_regime_history
                    (snapshot_date, regime, credit_cycle_phase,
                     volatility_regime, recession_probability,
                     yield_curve_inverted, vix_level, credit_spread,
                     risk_off_signal, recommendations)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (snapshot_date) DO UPDATE SET
                    regime = EXCLUDED.regime,
                    credit_cycle_phase = EXCLUDED.credit_cycle_phase,
                    volatility_regime = EXCLUDED.volatility_regime,
                    recession_probability = EXCLUDED.recession_probability,
                    yield_curve_inverted = EXCLUDED.yield_curve_inverted,
                    vix_level = EXCLUDED.vix_level,
                    credit_spread = EXCLUDED.credit_spread,
                    risk_off_signal = EXCLUDED.risk_off_signal,
                    recommendations = EXCLUDED.recommendations,
                    updated_at = NOW()
            """, (
                datetime.now().date(),
                str(regime_data.regime),
                str(regime_data.credit_cycle_phase.value if hasattr(regime_data.credit_cycle_phase, 'value') else regime_data.credit_cycle_phase),
                str(regime_data.volatility_regime),
                float(regime_data.recession_probability or 0),
                bool(regime_data.yield_curve_inverted),
                float(regime_data.vix_level or 0),
                float(regime_data.credit_spread or 0),
                bool(regime_data.risk_off_signal),
                str(regime_data.recommendations or ""),
            ))

            self.metrics.records_inserted += 1

            # Update current regime status table
            cursor.execute("""
                INSERT INTO current_market_regime
                    (id, regime, credit_cycle_phase, volatility_regime,
                     recession_probability, yield_curve_inverted,
                     risk_off_signal, last_updated)
                VALUES (1, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    regime = EXCLUDED.regime,
                    credit_cycle_phase = EXCLUDED.credit_cycle_phase,
                    volatility_regime = EXCLUDED.volatility_regime,
                    recession_probability = EXCLUDED.recession_probability,
                    yield_curve_inverted = EXCLUDED.yield_curve_inverted,
                    risk_off_signal = EXCLUDED.risk_off_signal,
                    last_updated = NOW()
            """, (
                str(regime_data.regime),
                str(regime_data.credit_cycle_phase.value if hasattr(regime_data.credit_cycle_phase, 'value') else regime_data.credit_cycle_phase),
                str(regime_data.volatility_regime),
                float(regime_data.recession_probability or 0),
                bool(regime_data.yield_curve_inverted),
                bool(regime_data.risk_off_signal),
            ))

            self.metrics.records_updated += 1

            # Store regime transition if changed
            self._check_regime_transition(cursor, regime_data)

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info(
                f"Market regime updated: {regime_data.regime}"
            )

        except ImportError as e:
            self.logger.error(f"Market regime analyzer not available: {e}")
            self.metrics.errors.append(f"Import error: {e}")

        except Exception as e:
            self.logger.exception(f"Market regime update failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics

    def _check_regime_transition(self, cursor, current_regime) -> None:
        """Check and record regime transitions."""
        try:
            # Get previous regime
            cursor.execute("""
                SELECT regime, credit_cycle_phase, volatility_regime
                FROM market_regime_history
                WHERE snapshot_date < CURRENT_DATE
                ORDER BY snapshot_date DESC
                LIMIT 1
            """)

            prev = cursor.fetchone()
            if not prev:
                return

            prev_regime, prev_credit, prev_vol = prev
            curr_regime = current_regime.regime
            curr_credit = current_regime.credit_cycle_phase
            curr_vol = current_regime.volatility_regime

            # Record any transitions
            transitions = []
            if prev_regime != curr_regime:
                transitions.append(("regime", prev_regime, curr_regime))
            if prev_credit != curr_credit:
                transitions.append(("credit_cycle", prev_credit, curr_credit))
            if prev_vol != curr_vol:
                transitions.append(("volatility", prev_vol, curr_vol))

            for transition_type, from_state, to_state in transitions:
                cursor.execute("""
                    INSERT INTO regime_transitions
                        (transition_date, transition_type, from_state, to_state)
                    VALUES (CURRENT_DATE, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (transition_type, from_state, to_state))

                self.logger.info(
                    f"Regime transition: {transition_type} {from_state} -> {to_state}"
                )

        except Exception as e:
            self.logger.debug(f"Could not check regime transition: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Update market regime classification"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Days to look back for analysis (default: 30)"
    )
    args = parser.parse_args()

    updater = MarketRegimeUpdater(lookback_days=args.lookback)
    exit_code = updater.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
