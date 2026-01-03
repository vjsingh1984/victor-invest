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

"""Collect Fama-French Factor Data.

Schedule: Daily at 6AM ET
Source: Kenneth French Data Library (free, updated monthly)

Collects:
- 5-factor model (Mkt-RF, SMB, HML, RMW, CMA)
- Momentum factor (UMD)
- Risk-free rate (RF)

The Fama-French factors are essential for:
- Understanding factor exposures of individual stocks
- Market regime detection
- Risk decomposition

Usage:
    python scripts/scheduled/collect_fama_french_factors.py
    python scripts/scheduled/collect_fama_french_factors.py --years 5
"""

import argparse
import io
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.scheduled.base import (
    BaseCollector,
    CollectionMetrics,
    compute_record_hash,
    get_database_connection,
    get_last_date,
    retry_with_backoff,
)

# Kenneth French Data Library URLs
FF5_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"


class FamaFrenchCollector(BaseCollector):
    """Collector for Fama-French factor data."""

    def __init__(self, lookback_years: int = 5):
        super().__init__("collect_fama_french_factors")
        self.lookback_years = lookback_years

    def _download_and_parse_ff5(self) -> Optional[pd.DataFrame]:
        """Download and parse 5-factor daily data."""
        try:
            response = requests.get(FF5_DAILY_URL, timeout=60)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find the CSV file (case-insensitive)
                csv_name = [n for n in z.namelist() if n.lower().endswith('.csv')][0]
                with z.open(csv_name) as f:
                    # Read raw content
                    content = f.read().decode('utf-8')

                    # Find the start of the data (skip description rows)
                    lines = content.split('\n')
                    start_idx = 0
                    for i, line in enumerate(lines):
                        # Header line starts with comma and contains Mkt-RF
                        if ',Mkt-RF' in line or line.strip().startswith('Mkt-RF'):
                            start_idx = i
                            break

                    if start_idx == 0:
                        self.logger.error("Could not find header row in FF5 data")
                        return None

                    # Parse with pandas
                    df = pd.read_csv(
                        io.StringIO('\n'.join(lines[start_idx:])),
                        index_col=0,
                        parse_dates=False,
                    )

                    # Clean column names
                    df.columns = [c.strip() for c in df.columns]

                    # Convert index to date (index is YYYYMMDD format)
                    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
                    df = df.dropna(how='all')  # Drop completely empty rows
                    df = df[df.index.notna()]  # Drop rows with invalid dates

                    # Convert percentages to decimals
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce') / 100

                    self.logger.info(f"Parsed {len(df)} rows of FF5 factor data")
                    return df

        except Exception as e:
            self.logger.error(f"Failed to download FF5 data: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def _download_and_parse_momentum(self) -> Optional[pd.DataFrame]:
        """Download and parse momentum factor daily data."""
        try:
            response = requests.get(MOM_DAILY_URL, timeout=60)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find the CSV file (case-insensitive)
                csv_name = [n for n in z.namelist() if n.lower().endswith('.csv')][0]
                with z.open(csv_name) as f:
                    content = f.read().decode('utf-8')
                    lines = content.split('\n')

                    # Find header line containing "Mom"
                    start_idx = 0
                    for i, line in enumerate(lines):
                        if ',Mom' in line or line.strip() == 'Mom':
                            start_idx = i
                            break

                    if start_idx == 0:
                        self.logger.error("Could not find header row in momentum data")
                        return None

                    df = pd.read_csv(
                        io.StringIO('\n'.join(lines[start_idx:])),
                        index_col=0,
                        parse_dates=False,
                    )

                    df.columns = ['UMD']  # Rename to standard name
                    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
                    df = df[df.index.notna()]  # Drop rows with invalid dates
                    df['UMD'] = pd.to_numeric(df['UMD'], errors='coerce') / 100

                    # Filter out missing data markers (-99.99 or -999)
                    df = df[df['UMD'] > -0.9]

                    self.logger.info(f"Parsed {len(df)} rows of momentum factor data")
                    return df

        except Exception as e:
            self.logger.error(f"Failed to download momentum data: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def collect(self) -> CollectionMetrics:
        """Collect Fama-French factors with incremental fetching."""
        try:
            conn = get_database_connection()
            cursor = conn.cursor()

            # Get last date in database
            last_date = get_last_date(cursor, "fama_french_factors", "date")

            self.logger.info("Downloading Fama-French 5-factor data...")
            ff5_df = self._download_and_parse_ff5()

            self.logger.info("Downloading momentum factor data...")
            mom_df = self._download_and_parse_momentum()

            if ff5_df is None:
                self.metrics.errors.append("Failed to download FF5 data")
                cursor.close()
                conn.close()
                return self.metrics

            # Merge with momentum if available
            if mom_df is not None:
                factors_df = ff5_df.join(mom_df, how='left')
            else:
                factors_df = ff5_df
                factors_df['UMD'] = None

            # Filter to lookback period and new dates
            cutoff_date = datetime.now().date() - timedelta(days=365 * self.lookback_years)
            factors_df = factors_df[factors_df.index >= pd.Timestamp(cutoff_date)]

            if last_date:
                factors_df = factors_df[factors_df.index > pd.Timestamp(last_date)]
                self.logger.info(f"Incremental fetch: {len(factors_df)} new days after {last_date}")
            else:
                self.logger.info(f"Full fetch: {len(factors_df)} days")

            self.metrics.records_processed = len(factors_df)

            # Helper to convert numpy types to Python native types
            def to_python(val):
                if val is None or (hasattr(val, 'item') and pd.isna(val)):
                    return None
                if pd.isna(val):
                    return None
                if hasattr(val, 'item'):  # numpy scalar
                    return float(val.item())
                return float(val) if val is not None else None

            # Insert/update records
            for date_idx, row in factors_df.iterrows():
                try:
                    record_date = date_idx.date()
                    mkt_rf = to_python(row.get('Mkt-RF'))
                    smb = to_python(row.get('SMB'))
                    hml = to_python(row.get('HML'))
                    rmw = to_python(row.get('RMW'))
                    cma = to_python(row.get('CMA'))
                    rf = to_python(row.get('RF'))
                    umd = to_python(row.get('UMD'))

                    record_hash = compute_record_hash({
                        "date": str(record_date),
                        "mkt_rf": str(mkt_rf),
                        "smb": str(smb),
                        "hml": str(hml),
                        "rmw": str(rmw),
                        "cma": str(cma),
                        "rf": str(rf),
                        "umd": str(umd),
                    })

                    # Check existing
                    cursor.execute(
                        "SELECT source_hash FROM fama_french_factors WHERE date = %s",
                        (record_date,),
                    )
                    existing = cursor.fetchone()

                    if existing:
                        if existing[0] == record_hash:
                            self.metrics.records_skipped += 1
                            continue
                        # Update
                        cursor.execute("""
                            UPDATE fama_french_factors SET
                                mkt_rf = %s, smb = %s, hml = %s, rmw = %s, cma = %s,
                                rf = %s, umd = %s, source_hash = %s,
                                source_fetch_timestamp = NOW(), updated_at = NOW()
                            WHERE date = %s
                        """, (
                            mkt_rf,
                            smb,
                            hml,
                            rmw,
                            cma,
                            rf,
                            umd,
                            record_hash,
                            record_date,
                        ))
                        self.metrics.records_updated += 1
                    else:
                        # Insert
                        cursor.execute("""
                            INSERT INTO fama_french_factors
                                (date, mkt_rf, smb, hml, rmw, cma, rf, umd,
                                 source_hash, source_fetch_timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        """, (
                            record_date,
                            mkt_rf,
                            smb,
                            hml,
                            rmw,
                            cma,
                            rf,
                            umd,
                            record_hash,
                        ))
                        self.metrics.records_inserted += 1

                    # Track high watermark
                    if self.metrics.high_watermark_date is None or record_date > self.metrics.high_watermark_date:
                        self.metrics.high_watermark_date = record_date

                except Exception as e:
                    self.logger.warning(f"Failed to process {date_idx}: {e}")
                    self.metrics.records_failed += 1

            # Update watermark
            if self.metrics.high_watermark_date:
                cursor.execute("""
                    INSERT INTO ff_factors_watermarks (id, last_date)
                    VALUES (1, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        last_date = GREATEST(ff_factors_watermarks.last_date, EXCLUDED.last_date),
                        last_fetch_timestamp = NOW()
                """, (self.metrics.high_watermark_date,))

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info(
                f"Fama-French factors: inserted={self.metrics.records_inserted}, "
                f"updated={self.metrics.records_updated}, skipped={self.metrics.records_skipped}"
            )

        except Exception as e:
            self.logger.exception(f"Fama-French collection failed: {e}")
            self.metrics.errors.append(str(e))

        return self.metrics


def main():
    parser = argparse.ArgumentParser(
        description="Collect Fama-French factor data"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years to fetch (default: 5)"
    )
    args = parser.parse_args()

    collector = FamaFrenchCollector(lookback_years=args.years)
    exit_code = collector.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
