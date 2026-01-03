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

"""Calculate Credit Risk Scores for All Companies.

Schedule: Weekly on Sunday at 8PM ET
Dependencies: None (uses existing fundamental data)

Calculates:
- Altman Z-Score (bankruptcy prediction)
- Beneish M-Score (earnings manipulation detection)
- Piotroski F-Score (financial strength)
- Composite distress tier

Usage:
    python scripts/scheduled/calculate_credit_risk.py
    python scripts/scheduled/calculate_credit_risk.py --symbols AAPL,MSFT
    python scripts/scheduled/calculate_credit_risk.py --force-refresh
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.scheduled.base import (
    BaseCollector,
    CollectionMetrics,
    get_database_connection,
    get_russell1000_symbols,
)


# Distress tier thresholds
DISTRESS_TIERS = {
    "HEALTHY": {"z_min": 2.99, "m_max": -1.78, "f_min": 7},
    "WATCH": {"z_min": 1.81, "m_max": -1.50, "f_min": 5},
    "CONCERN": {"z_min": 1.23, "m_max": -1.20, "f_min": 3},
    "DISTRESSED": {"z_min": 0.5, "m_max": -0.80, "f_min": 2},
    "SEVERE_DISTRESS": {"z_min": float("-inf"), "m_max": float("inf"), "f_min": 0},
}


class CreditRiskCalculator(BaseCollector):
    """Calculator for credit risk scores."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        force_refresh: bool = False,
    ):
        super().__init__("calculate_credit_risk")
        self.symbols = symbols
        self.force_refresh = force_refresh

    def collect(self) -> CollectionMetrics:
        """Calculate credit risk scores for all companies."""
        import asyncio

        try:
            # Import credit risk service
            from investigator.domain.services.credit_risk import (
                get_credit_risk_service,
            )

            service = get_credit_risk_service()

            # Get symbols to process
            if self.symbols:
                symbols = self.symbols
            else:
                symbols = get_russell1000_symbols()

            self.logger.info(
                f"Calculating credit risk scores for {len(symbols)} symbols"
            )

            conn = get_database_connection()
            cursor = conn.cursor()

            for symbol in symbols:
                try:
                    # Check if we need to refresh
                    if not self.force_refresh:
                        cursor.execute("""
                            SELECT updated_at FROM credit_risk_scores
                            WHERE symbol = %s
                              AND updated_at > CURRENT_DATE - INTERVAL '7 days'
                        """, (symbol,))
                        if cursor.fetchone():
                            continue

                    self.metrics.records_processed += 1

                    # Calculate all credit risk scores using the service
                    try:
                        assessment = asyncio.run(
                            service.calculate_from_symbol(symbol)
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate scores for {symbol}: {e}")
                        self.metrics.records_failed += 1
                        continue

                    # Extract scores from assessment
                    z_score = assessment.altman.score if assessment.altman else None
                    z_interpretation = (
                        assessment.altman.zone.value if assessment.altman and assessment.altman.zone else None
                    )

                    m_score = assessment.beneish.score if assessment.beneish else None
                    m_interpretation = (
                        assessment.beneish.risk_level.value if assessment.beneish and assessment.beneish.risk_level else None
                    )

                    f_score = assessment.piotroski.score if assessment.piotroski else None
                    f_interpretation = (
                        assessment.piotroski.strength.value if assessment.piotroski and assessment.piotroski.strength else None
                    )

                    # Get distress tier from composite assessment
                    distress_tier = (
                        assessment.composite.distress_tier.name
                        if assessment.composite and assessment.composite.distress_tier
                        else self._calculate_distress_tier(z_score, m_score, f_score)
                    )

                    # Store results
                    cursor.execute("""
                        INSERT INTO credit_risk_scores
                            (symbol, calculation_date,
                             altman_z_score, altman_z_interpretation,
                             beneish_m_score, beneish_m_interpretation,
                             piotroski_f_score, piotroski_f_interpretation,
                             distress_tier, updated_at)
                        VALUES (%s, CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (symbol, calculation_date) DO UPDATE SET
                            altman_z_score = EXCLUDED.altman_z_score,
                            altman_z_interpretation = EXCLUDED.altman_z_interpretation,
                            beneish_m_score = EXCLUDED.beneish_m_score,
                            beneish_m_interpretation = EXCLUDED.beneish_m_interpretation,
                            piotroski_f_score = EXCLUDED.piotroski_f_score,
                            piotroski_f_interpretation = EXCLUDED.piotroski_f_interpretation,
                            distress_tier = EXCLUDED.distress_tier,
                            updated_at = NOW()
                    """, (
                        symbol,
                        z_score,
                        z_interpretation,
                        m_score,
                        m_interpretation,
                        f_score,
                        f_interpretation,
                        distress_tier,
                    ))

                    self.metrics.records_inserted += 1

                    # Store in current scores table for quick lookup
                    cursor.execute("""
                        INSERT INTO current_credit_risk
                            (symbol, altman_z_score, beneish_m_score,
                             piotroski_f_score, distress_tier, last_updated)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (symbol) DO UPDATE SET
                            altman_z_score = EXCLUDED.altman_z_score,
                            beneish_m_score = EXCLUDED.beneish_m_score,
                            piotroski_f_score = EXCLUDED.piotroski_f_score,
                            distress_tier = EXCLUDED.distress_tier,
                            last_updated = NOW()
                    """, (symbol, z_score, m_score, f_score, distress_tier))

                    self.metrics.records_updated += 1

                except Exception as e:
                    self.logger.warning(f"Failed to calculate for {symbol}: {e}")
                    self.metrics.records_failed += 1
                    self.metrics.warnings.append(f"{symbol}: {e}")

            # Generate summary statistics
            self._generate_summary_stats(cursor)

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info(
                f"Calculated credit risk for {self.metrics.records_inserted} symbols"
            )

        except ImportError as e:
            self.logger.error(f"Credit risk analyzers not available: {e}")
            self.metrics.errors.append(f"Import error: {e}")

        except Exception as e:
            self.logger.exception(f"Credit risk calculation failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics

    def _calculate_distress_tier(
        self,
        z_score: Optional[float],
        m_score: Optional[float],
        f_score: Optional[int],
    ) -> str:
        """Calculate composite distress tier from individual scores."""
        # Default to WATCH if scores unavailable
        if z_score is None and m_score is None and f_score is None:
            return "UNKNOWN"

        # Count flags for each tier
        tier_scores = {
            "HEALTHY": 0,
            "WATCH": 0,
            "CONCERN": 0,
            "DISTRESSED": 0,
            "SEVERE_DISTRESS": 0,
        }

        # Z-Score evaluation (higher is better)
        if z_score is not None:
            if z_score >= 2.99:
                tier_scores["HEALTHY"] += 1
            elif z_score >= 1.81:
                tier_scores["WATCH"] += 1
            elif z_score >= 1.23:
                tier_scores["CONCERN"] += 1
            elif z_score >= 0.5:
                tier_scores["DISTRESSED"] += 1
            else:
                tier_scores["SEVERE_DISTRESS"] += 1

        # M-Score evaluation (lower/more negative is better)
        if m_score is not None:
            if m_score < -2.22:
                tier_scores["HEALTHY"] += 1
            elif m_score < -1.78:
                tier_scores["WATCH"] += 1
            elif m_score < -1.50:
                tier_scores["CONCERN"] += 1
            elif m_score < -0.80:
                tier_scores["DISTRESSED"] += 1
            else:
                tier_scores["SEVERE_DISTRESS"] += 1

        # F-Score evaluation (higher is better, max 9)
        if f_score is not None:
            if f_score >= 7:
                tier_scores["HEALTHY"] += 1
            elif f_score >= 5:
                tier_scores["WATCH"] += 1
            elif f_score >= 3:
                tier_scores["CONCERN"] += 1
            elif f_score >= 2:
                tier_scores["DISTRESSED"] += 1
            else:
                tier_scores["SEVERE_DISTRESS"] += 1

        # Use worst tier that has any score
        # (Conservative approach - if any model signals distress, flag it)
        for tier in ["SEVERE_DISTRESS", "DISTRESSED", "CONCERN", "WATCH", "HEALTHY"]:
            if tier_scores[tier] > 0:
                # But also consider if majority of models agree on better tier
                better_score = sum(
                    tier_scores[t] for t in list(DISTRESS_TIERS.keys())[:list(DISTRESS_TIERS.keys()).index(tier)]
                )
                if better_score >= 2:
                    continue
                return tier

        return "UNKNOWN"

    def _generate_summary_stats(self, cursor) -> None:
        """Generate and store summary statistics."""
        try:
            cursor.execute("""
                INSERT INTO credit_risk_summary
                    (summary_date, total_analyzed,
                     healthy_count, watch_count, concern_count,
                     distressed_count, severe_distress_count,
                     avg_z_score, avg_m_score, avg_f_score)
                SELECT
                    CURRENT_DATE,
                    COUNT(*),
                    COUNT(*) FILTER (WHERE distress_tier = 'HEALTHY'),
                    COUNT(*) FILTER (WHERE distress_tier = 'WATCH'),
                    COUNT(*) FILTER (WHERE distress_tier = 'CONCERN'),
                    COUNT(*) FILTER (WHERE distress_tier = 'DISTRESSED'),
                    COUNT(*) FILTER (WHERE distress_tier = 'SEVERE_DISTRESS'),
                    AVG(altman_z_score),
                    AVG(beneish_m_score),
                    AVG(piotroski_f_score)
                FROM current_credit_risk
                ON CONFLICT (summary_date) DO UPDATE SET
                    total_analyzed = EXCLUDED.total_analyzed,
                    healthy_count = EXCLUDED.healthy_count,
                    watch_count = EXCLUDED.watch_count,
                    concern_count = EXCLUDED.concern_count,
                    distressed_count = EXCLUDED.distressed_count,
                    severe_distress_count = EXCLUDED.severe_distress_count,
                    avg_z_score = EXCLUDED.avg_z_score,
                    avg_m_score = EXCLUDED.avg_m_score,
                    avg_f_score = EXCLUDED.avg_f_score,
                    updated_at = NOW()
            """)
        except Exception as e:
            self.logger.debug(f"Could not generate summary stats: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate credit risk scores for companies"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (default: Russell 1000)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force recalculation even if recent scores exist"
    )
    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    calculator = CreditRiskCalculator(
        symbols=symbols,
        force_refresh=args.force_refresh,
    )
    exit_code = calculator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
