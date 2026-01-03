#!/usr/bin/env python3

# load libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import logging
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import sys
import time
from sqlalchemy import text


def getConnectionEngine(user, password, database, host, port):
    return create_engine(
        "postgresql://{user}:{password}@{host}:{port}/{database}".format(
            user=user, password=password, database=database, host=host, port=port
        )
    )


def getTickerClosePriceDataframe(connEngine, ticker, dt):
    sqlQuery = """
                SELECT DISTINCT Y.close, Y.wkdt 
                FROM (
                    SELECT FIRST_VALUE(X.close) OVER ( PARTITION BY DATE_TRUNC('WEEK', {dtfield} ) ORDER BY {dtfield} DESC) as close,
                           DATE_TRUNC('WEEK', {dtfield}) as wkdt
                    FROM {table} X
                    WHERE {dtfield} > DATE_TRUNC( 'WEEK', DATE'{fromdt}' )
                    AND ticker = '{ticker}'
                    ORDER BY {dtfield} 
                    ) Y ORDER BY Y.wkdt """.format(
        dtfield="date", table="tickerdata", fromdt=dt.strftime("%Y-%m-%d"), ticker=ticker
    )
    # logging.info("Executing Query: " + sqlQuery);
    try:
        return pd.read_sql(sqlQuery, con=connEngine)
    except Exception as e:
        logging.warning(e)
        raise e


def saveBetaForTicker(connEngine, ticker, column, beta):
    cursor = connEngine.connect()
    sqlStmt = "UPDATE symbol SET {column} = CAST({beta} AS DECIMAL(7,4)), lastupdts = CURRENT_TIMESTAMP WHERE ticker = '{ticker}'".format(
        ticker=ticker, beta=beta, column=column
    )
    logging.info("Executing Update Query: " + sqlStmt)
    try:
        cursor.execute(text(sqlStmt))
        cursor.commit()
    except Exception as e:
        logging.warning(e)
        raise e
    cursor.close()


def getTickerList(connEngine, dt, column):
    sql = """ SELECT X.ticker, X.mktcap 
              FROM ( SELECT S.ticker , AVG(D.close*D.volume) as mktcap
                     FROM tickerdata D JOIN Symbol S ON (D.ticker = S.ticker)
                     WHERE D.date > '{dtstr}'
                     AND S.isListed = True
                     AND S.skiptametric = FALSE
                     Group By S.ticker
              ) X
              Order By 2 DESC """.format(
        dtstr=dt.strftime("%Y-%m-%d"), column=column
    )
    try:
        logging.info("Executing ticker list sql: {sql}".format(sql=sql))
        return pd.read_sql(sql, con=connEngine)
    except Exception as e:
        logging.warning(e)
        raise e


def main():
    connEngine = getConnectionEngine(
        user="stockuser", password=os.environ.get("STOCK_DB_PASSWORD", ""), database="stock", host="${DB_HOST:-localhost}", port=5432
    )
    todate = datetime.now() + relativedelta(hours=9)
    logging.info("todate:{todate}".format(todate=todate))

    # monthsList = [1,3,6,12]
    monthsList = [1, 3, 6, 12, 24, 36, 60]
    for iMonth in reversed(monthsList):
        calcBetaMonth(connEngine, todate, iMonth)


def clean_and_validate_data(y: np.ndarray) -> np.ndarray:
    """
    Clean and validate a single array by removing infinities,
    NaN values, and extremely large numbers.

    Args:
        y: numpy array to clean

    Returns:
        Cleaned numpy array
    """
    # Convert to numpy array if it isn't already
    y = np.array(y)

    # Create mask for valid values
    valid_mask = (
        ~np.isinf(y) & ~np.isnan(y) & (np.abs(y) < 1e100)  # Remove infinity  # Remove NaN  # Remove too large values
    )

    # Apply mask
    y_clean = y[valid_mask]

    # Verify we still have enough data
    if len(y_clean) < 2:
        raise ValueError("Insufficient valid data points after cleaning")

    return y_clean


def calcBetaMonth(connEngine, todate, months):

    dt = todate - relativedelta(months=months)
    logging.info("Process From dt:{dt}".format(dt=dt))
    column = "b_{months}_month".format(months=months)
    r2_column = "r2_{months}_month".format(months=months)  # Also save R²

    # symbols = [stock, market]
    # start date for historical prices
    spydata = getTickerClosePriceDataframe(connEngine, "SPY", dt)
    spy_price_change = spydata[["close"]].pct_change()
    spydf = spy_price_change.drop(spy_price_change.index[0])
    # Add date index for proper alignment
    spydf["wkdt"] = spydata["wkdt"].iloc[1:].values
    spydf = spydf.set_index("wkdt")
    spydf.columns = ["spy_return"]

    tickerDF = getTickerList(connEngine, dt, column)
    tickerLength = len(tickerDF.index)
    counter = 0

    # Minimum observations required for statistical significance
    MIN_OBSERVATIONS = {1: 4, 3: 10, 6: 20, 12: 40, 24: 80, 36: 120, 60: 200}
    min_obs = MIN_OBSERVATIONS.get(months, 20)

    # Beta validation thresholds
    BETA_FLOOR = 0.10  # Minimum meaningful beta
    BETA_CAP = 5.0  # Maximum reasonable beta
    R2_THRESHOLD = 0.05  # Minimum R² (5%) for statistical significance

    for idx, row in tickerDF.iterrows():
        counter += 1
        ticker = row["ticker"]
        logging.info(
            "Process [{counter}/{tickerLength}]\t[{ticker}]\t from [{dtstr}]".format(
                ticker=ticker, dtstr=dt.strftime("%Y-%m-%d"), counter=counter, tickerLength=tickerLength
            )
        )
        # Convert historical stock prices to weekly percent change
        tickerdata = getTickerClosePriceDataframe(connEngine, ticker, dt)
        tickerdata_price_change = tickerdata[["close"]].pct_change()
        # Deletes row one containing the NaN
        tickerdf = tickerdata_price_change.drop(tickerdata_price_change.index[0])
        tickerdf["wkdt"] = tickerdata["wkdt"].iloc[1:].values
        tickerdf = tickerdf.set_index("wkdt")
        tickerdf.columns = ["stock_return"]

        try:
            # CRITICAL FIX: Align data by date using inner join
            aligned = pd.concat([tickerdf, spydf], axis=1, join="inner").dropna()

            if len(aligned) < min_obs:
                logging.info(
                    "Skipped [{counter}/{tickerLength}]\t[{ticker}]\t insufficient data ({obs} < {min_obs})".format(
                        ticker=ticker, counter=counter, tickerLength=tickerLength, obs=len(aligned), min_obs=min_obs
                    )
                )
                continue

            # Clean data
            x = aligned["spy_return"].values.reshape(-1, 1)
            y = aligned["stock_return"].values.reshape(-1, 1)

            # CRITICAL FIX: Use cleaned data for regression (not overwritten)
            model = LinearRegression().fit(x, y)
            beta = model.coef_[0][0]
            r_squared = model.score(x, y)

            # Validation checks
            skip_reason = None
            if abs(beta) > 40:
                skip_reason = f"extreme beta ({beta:.2f})"
            elif r_squared < R2_THRESHOLD:
                skip_reason = f"low R² ({r_squared:.4f} < {R2_THRESHOLD})"

            if skip_reason:
                logging.warning(f"Skip [{ticker}]: {skip_reason}")
                continue

            logging.info(
                "Completed [{counter}/{tickerLength}]\t[{ticker}]\t beta={beta:.4f}\t R²={r2:.4f}\t obs={obs}".format(
                    ticker=ticker, counter=counter, tickerLength=tickerLength, beta=beta, r2=r_squared, obs=len(aligned)
                )
            )

            saveBetaForTicker(connEngine, ticker, column, beta)
            # Also save R² for downstream quality checks
            saveBetaForTicker(connEngine, ticker, r2_column, r_squared)

        except ValueError as e:
            logging.warning(f"Ticker: {ticker} encountered exception: {str(e)}")


if __name__ == "__main__":
    currdate = datetime.now()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                "/var/log/saveBetaForStock.{currdatestr}.log".format(currdatestr=currdate.strftime("%Y-%m-%d"))
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )

    main()
