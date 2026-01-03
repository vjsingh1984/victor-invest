#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine, text
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sys, time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class BetaResult:
    """Data class to store beta calculation results"""
    beta: float
    r_squared: float
    observations: int
    start_date: datetime
    end_date: datetime

class DatabaseConnection:
    """Handles database connection and queries"""

    def __init__(self, user: str, password: str, database: str, host: str, port: int):
        """Initialize database connection"""
        self.engine = create_engine(
            f"postgresql://{user}:{password}@{host}:{port}/{database}"
        )

    def execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame

        Args:
            query: SQL query string
            params: Dictionary of parameters to bind to the query
        """
        try:
            return pd.read_sql(query, con=self.engine, params=params)
        except Exception as e:
            logging.error(f"Database query error: {str(e)}")
            raise

    def execute_update(self, statement: str) -> None:
        """Execute SQL update statement"""
        try:
            with self.engine.connect() as connection:
                connection.execute(text(statement))
                connection.commit()
        except Exception as e:
            logging.error(f"Database update error: {str(e)}")
            raise

class StockDataFetcher:
    """Handles retrieval and preprocessing of stock data"""

    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection

    def get_ticker_list(self, start_date: datetime) -> pd.DataFrame:
        """Get list of tickers to process, ordered by market cap"""
        query = """
        SELECT X.ticker, X.mktcap 
        FROM (
            SELECT S.ticker, AVG(D.close * D.volume) as mktcap
            FROM tickerdata D 
            JOIN Symbol S ON (D.ticker = S.ticker)
            WHERE D.date > CAST(%(dtstr)s AS DATE)
            AND S.isListed = True
            AND S.skiptametric = FALSE
            GROUP BY S.ticker
        ) X
        ORDER BY X.mktcap DESC
        """
        return self.db.execute_query(query, {
            'dtstr': start_date.strftime("%Y-%m-%d")
        })

    def get_weekly_prices(self, ticker: str, start_date: datetime) -> pd.DataFrame:
        """
        Fetch weekly (Wednesday) closing prices for a ticker
        """
        query = """
        SELECT DISTINCT Y.close, Y.wkdt
        FROM (
            SELECT FIRST_VALUE(X.close) OVER (
                PARTITION BY DATE_TRUNC('WEEK', date)
                ORDER BY CASE WHEN EXTRACT(DOW FROM date) = 3 THEN 0 ELSE 1 END,
                         date DESC
            ) as close,
            DATE_TRUNC('WEEK', date) as wkdt
            FROM tickerdata X
            WHERE date > DATE_TRUNC('WEEK', CAST(%(fromdt)s AS DATE))
            AND ticker = CAST(%(ticker)s AS VARCHAR)
            ORDER BY date
        ) Y
        ORDER BY Y.wkdt
        """
        # OVERWRITING ABOVE QUERY FOR WEDNESDAY BASED CALC
        query = """
        WITH date_ranges AS (
            SELECT 
                date,
                close,
                date_trunc('week', date) as week_start,
                EXTRACT(DOW FROM date) as day_of_week
            FROM tickerdata
            WHERE ticker = CAST(%(ticker)s AS VARCHAR)
            AND date > CAST(%(fromdt)s AS DATE)
        ),
        weekly_prices AS (
            SELECT 
                week_start,
                COALESCE(
                    -- First try to get Wednesday's price
                    MAX(CASE WHEN day_of_week = 3 THEN close END),
                    -- If no Wednesday, try Tuesday
                    MAX(CASE WHEN day_of_week = 2 THEN close END),
                    -- If no Tuesday, try Thursday
                    MAX(CASE WHEN day_of_week = 4 THEN close END),
                    -- If none of above, get any day's close
                    MAX(close)
                ) as close,
                COALESCE(
                    -- First try to get Wednesday's date
                    MAX(CASE WHEN day_of_week = 3 THEN date END),
                    -- If no Wednesday, try Tuesday
                    MAX(CASE WHEN day_of_week = 2 THEN date END),
                    -- If no Tuesday, try Thursday
                    MAX(CASE WHEN day_of_week = 4 THEN date END),
                    -- If none of above, get any day
                    MAX(date)
                ) as actual_date
            FROM date_ranges
            GROUP BY week_start
            HAVING MAX(date) IS NOT NULL
        )
        SELECT 
            close,
            actual_date as wkdt
        FROM weekly_prices
        ORDER BY actual_date;
        """

        df = self.db.execute_query(query, {
            'fromdt': start_date.strftime("%Y-%m-%d"),
            'ticker': str(ticker)
        })
        # Convert to datetime with UTC timezone, then convert to timezone-naive
        df['wkdt'] = pd.to_datetime(df['wkdt'], utc=True).dt.tz_convert(None)
        return df.set_index('wkdt')

class BetaCalculator:
    """Handles beta calculations and data validation"""

    MINIMUM_OBSERVATIONS = {
        1: 4,    # 1 month: minimum 4 weekly observations
        3: 13,   # 3 months
        6: 26,   # 6 months
        12: 52,  # 1 year
        24: 104, # 2 years
        36: 156, # 3 years
        60: 260  # 5 years
    }

    def __init__(self, data_fetcher: StockDataFetcher):
        self.data_fetcher = data_fetcher

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate weekly returns from price data"""
        returns = prices[['close']].pct_change()
        return returns.dropna()

    def validate_beta(self, beta: float, r_squared: float, ticker: str, period: int) -> bool:
        """
        Validate beta calculation results.

        Validation criteria:
        - Beta must be within reasonable bounds (|beta| <= 40)
        - R² must be >= 5% for statistical significance (was 2.5%, too permissive)
        - Very low betas (< 0.10) with low R² are likely noise
        """
        if abs(beta) > 40:
            logging.warning(f"Unusual beta value ({beta:.2f}) for {ticker} {period}-month calculation")
            return False

        # CRITICAL FIX: Increased R² threshold from 2.5% to 5%
        # 2.5% R² means only 2.5% of variance explained - statistically weak
        R2_THRESHOLD = 0.05  # 5% minimum
        if r_squared < R2_THRESHOLD:
            logging.warning(f"Low R-squared ({r_squared:.4f} < {R2_THRESHOLD}) for {ticker} {period}-month beta")
            return False

        # Additional check: very low beta with marginal R² is likely noise
        if abs(beta) < 0.10 and r_squared < 0.10:
            logging.warning(f"Statistically weak beta ({beta:.4f}) with low R² ({r_squared:.4f}) for {ticker}")
            return False

        return True

    def calculate_single_beta(self,
                            stock_returns: pd.DataFrame,
                            market_returns: pd.DataFrame,
                            ticker: str,
                            period: int) -> Optional[BetaResult]:
        """Calculate beta for a single period"""
        min_obs = self.MINIMUM_OBSERVATIONS[period]

        if len(stock_returns) < min_obs:
            logging.warning(
                f"Insufficient observations for {ticker} {period}-month beta. "
                f"Need {min_obs}, got {len(stock_returns)}"
            )
            return None

        try:
            X = market_returns['spyclose'].values.reshape(-1, 1)  # Specify column name
            y = stock_returns['close'].values.reshape(-1, 1)   # Specify column name

            model = LinearRegression()
            model.fit(X, y)

            beta = model.coef_[0][0]
            r_squared = model.score(X, y)

            if not self.validate_beta(beta, r_squared, ticker, period):
                return None

            return BetaResult(
                beta=beta,
                r_squared=r_squared,
                observations=len(stock_returns),
                start_date=stock_returns.index[0],
                end_date=stock_returns.index[-1]
            )

        except Exception as e:
            logging.error(f"Error calculating {period}-month beta for {ticker}: {str(e)}")
            return None

class BetaProcessor:
    """Main class for processing beta calculations"""

    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize BetaProcessor

        Args:
            db_connection: DatabaseConnection instance for database operations
        """
        self.data_fetcher = StockDataFetcher(db_connection)
        self.calculator = BetaCalculator(self.data_fetcher)
        self.db = db_connection
        self.beta_updates = []


    def save_beta(self, ticker: str, period: int, beta_result: BetaResult) -> None:
        """Save beta calculation results to database"""
        column = f'b_{period}_month'
        update_sql = """
        UPDATE symbol
        SET {column} = CAST({beta} AS DECIMAL(7,4)),
            lastupdts = CURRENT_TIMESTAMP
        WHERE ticker = '{ticker}'
        """.format(column=column, beta=beta_result.beta, ticker=ticker)

        self.db.execute_update(update_sql)

    def save_betas_batch(self) -> None:
        """Save all accumulated beta and R² calculations in a single SQL update"""
        if not self.beta_updates:
            logging.info("No beta updates to save")
            return

        try:
            column_cases = {}
            # Handle both beta and R² columns
            for update in self.beta_updates:
                ticker, period, beta, r_squared = update
                
                # Beta column cases
                beta_column = f'b_{period}_month'
                if beta_column not in column_cases:
                    column_cases[beta_column] = []
                column_cases[beta_column].append(
                    f"WHEN ticker = '{ticker}' THEN CAST({beta} AS DECIMAL(7,4))"
                )
                
                # R² column cases
                r2_column = f'r2_{period}_month'
                if r2_column not in column_cases:
                    column_cases[r2_column] = []
                column_cases[r2_column].append(
                    f"WHEN ticker = '{ticker}' THEN CAST({r_squared} AS DECIMAL(7,4))"
                )

            set_clauses = []
            for column, cases in column_cases.items():
                set_clauses.append(f"{column} = CASE {' '.join(cases)} ELSE {column} END")

            tickers = {update[0] for update in self.beta_updates}
            ticker_list = "', '".join(tickers)

            update_sql = f"""
            UPDATE symbol 
            SET {', '.join(set_clauses)},
                lastupdts = CURRENT_TIMESTAMP 
            WHERE ticker IN ('{ticker_list}')
            """

            self.db.execute_update(update_sql)
            logging.info(f"Successfully updated betas and R² values for {len(tickers)} tickers")
            self.beta_updates = []

        except Exception as e:
            logging.error(f"Error in batch beta update: {str(e)}")
            raise

    def add_beta_update(self, ticker: str, period: int, beta_result: BetaResult) -> None:
        """Add a beta and R² update to the batch"""
        self.beta_updates.append((ticker, period, beta_result.beta, beta_result.r_squared))


    def process_all_periods(self, periods: List[int], end_date: datetime) -> None:
        """
        Process beta calculations for all periods and tickers
        """
        # Convert end_date to timezone-naive
        if end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)
            
        max_period = max(periods)
        start_date = end_date - relativedelta(months=max_period)
        update_count = 0
        batch_size = len(periods)
        
        try:
            spy_data = self.data_fetcher.get_weekly_prices('SPY', start_date)
            if spy_data.empty:
                logging.error("No SPY data available")
                return
                
            spy_returns = self.calculator.calculate_returns(spy_data)
            spy_returns.rename(columns={'close': 'spyclose'}, inplace=True)
            
            tickers_df = self.data_fetcher.get_ticker_list(start_date)
            total_tickers = len(tickers_df)
            
            for idx, row in tickers_df.iterrows():
                ticker = str(row['ticker'])
                logging.info(f"Processing [{idx+1}/{total_tickers}] {ticker}")
                
                #time.sleep(5)
                try:
                    stock_data = self.data_fetcher.get_weekly_prices(ticker, start_date)
                    if stock_data.empty:
                        logging.warning(f"No data available for ticker {ticker}")
                        continue
                        
                    stock_returns = self.calculator.calculate_returns(stock_data)
                    
                    for period in periods:
                        period_start = end_date - relativedelta(months=period)
                        # Ensure period_start is timezone-naive
                        if period_start.tzinfo is not None:
                            period_start = period_start.replace(tzinfo=None)
                            
                        period_returns = stock_returns[stock_returns.index >= period_start]
                        period_spy_returns = spy_returns[spy_returns.index >= period_start]
                        
                        aligned_returns = pd.concat([period_returns, period_spy_returns], axis=1, join='inner')
                        if aligned_returns.empty:
                            logging.warning(f"No aligned data for {ticker} in {period}-month period")
                            continue
                            
                        beta_result = self.calculator.calculate_single_beta(
                            aligned_returns[['close']], 
                            aligned_returns[['spyclose']], 
                            ticker, 
                            period
                        )

                        if beta_result:
                            self.add_beta_update(ticker, period, beta_result)
                            update_count += 1
                            logging.info(
                                f"Calculated {period}-month beta={beta_result.beta:.2f} "
                                f"r²={beta_result.r_squared:.2f} for {ticker}"
                            )
                
                except Exception as e:
                    logging.error(f"Error processing {ticker}: {str(e)}")
                    continue

                # Process batch updates when reaching batch size or at the end
                if update_count >= batch_size or idx == len(tickers_df) - 1:
                    try:
                        self.save_betas_batch()
                        update_count = 0
                    except Exception as e:
                        logging.error(f"Error saving batch updates: {str(e)}")
                        raise
                        #continue
        
        except Exception as e:
            logging.error(f"Fatal error in beta processing: {str(e)}")
            raise

        # Final batch save if any updates remain
        try:
            if self.beta_updates:
                self.save_betas_batch()
        except Exception as e:
            logging.error(f"Error in final batch save: {str(e)}")
            raise

def main():
    """Main function to run beta calculations"""
    import os

    # Create logs directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"beta_calculator.{datetime.now():%Y-%m-%d}.log")

    # Configure logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    try:
        # Initialize database connection
        db_connection = DatabaseConnection(
            user="stockuser",
            password=os.environ.get("STOCK_DB_PASSWORD", ""),
            database="stock",
            host="${DB_HOST:-localhost}",
            port=5432
        )

        # Initialize processor with the database connection
        processor = BetaProcessor(db_connection)

        # Set calculation periods and end date
        periods = [1, 3, 6, 12, 24, 36, 60]

        #Wednesday to wednesday logic
         # Get current time
        current_date = datetime.now()

        # Calculate the most recent Wednesday
        days_since_wednesday = (current_date.weekday() - 2) % 7
        last_wednesday = current_date - timedelta(days=days_since_wednesday)

        # Set to end of trading day
        end_date = last_wednesday.replace(
            hour=16,
            minute=0,
            second=0,
            microsecond=0
        )

        # Process betas
        processor.process_all_periods(periods, end_date)

    except Exception as e:
        logging.critical(f"Application failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
