#!/usr/bin/env python3
"""
Update Symbol Index Membership Flags

Updates symbol table with proper index membership flags (sp500, nasdaq100, dow30)
based on current index constituent data.

Usage:
    python3 scripts/update_symbol_index_flags.py

Author: InvestiGator Team
Date: 2025-12-28
"""

import logging
import sys
from typing import List, Set

from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Database connection
ENGINE = create_engine("postgresql://stockuser:${STOCK_DB_PASSWORD}@${STOCK_DB_HOST}:5432/stock")

# S&P 500 constituents (fetched 2025-12-28 from slickcharts.com)
SP500_SYMBOLS = (
    """
NVDA,AAPL,MSFT,AMZN,GOOGL,GOOG,META,AVGO,TSLA,BRK.B,LLY,JPM,WMT,V,ORCL,MA,XOM,JNJ,PLTR,BAC,
ABBV,NFLX,COST,AMD,HD,PG,GE,MU,CSCO,UNH,KO,CVX,WFC,MS,IBM,CAT,GS,MRK,AXP,PM,CRM,RTX,APP,TMUS,
LRCX,MCD,TMO,ABT,C,AMAT,ISRG,DIS,LIN,PEP,INTU,QCOM,SCHW,GEV,AMGN,BKNG,T,TJX,INTC,VZ,BA,UBER,
BLK,APH,KLAC,NEE,ACN,ANET,DHR,TXN,SPGI,NOW,COF,GILD,ADBE,PFE,BSX,UNP,LOW,ADI,SYK,PGR,PANW,
WELL,DE,HON,ETN,MDT,CB,CRWD,BX,PLD,VRTX,KKR,NEM,COP,CEG,PH,LMT,BMY,HCA,CMCSA,HOOD,ADP,MCK,
CVS,DASH,CME,SBUX,MO,SO,ICE,MCO,GD,MMC,SNPS,DUK,NKE,WM,TT,CDNS,CRH,APO,MMM,DELL,USB,UPS,HWM,
MAR,PNC,ABNB,AMT,REGN,NOC,BK,SHW,RCL,ORLY,ELV,GM,CTAS,GLW,AON,EMR,FCX,MNST,ECL,EQIX,JCI,CI,
TDG,ITW,WMB,CMI,WBD,MDLZ,FDX,TEL,HLT,CSX,AJG,COR,RSG,NSC,TRV,TFC,PWR,CL,COIN,ADSK,MSI,STX,
WDC,CVNA,AEP,SPG,FTNT,KMI,PCAR,ROST,WDAY,SRE,AFL,AZO,NDAQ,SLB,EOG,PYPL,NXPI,BDX,ZTS,LHX,APD,
IDXX,VST,ALL,DLR,F,MET,URI,O,PSX,EA,D,EW,VLO,CMG,CAH,MPC,CBRE,GWW,ROP,DDOG,AME,FAST,TTWO,
AIG,AMP,AXON,DAL,OKE,PSA,CTVA,MPWR,CARR,TGT,ROK,LVS,BKR,XEL,MSCI,EXC,DHI,YUM,FANG,FICO,ETR,
CTSH,PAYX,CCL,XYZ,PEG,KR,PRU,GRMN,TRGP,OXY,A,MLM,VMC,EL,HIG,IQV,EBAY,CCI,KDP,GEHC,NUE,CPRT,
WAB,VTR,HSY,ARES,STT,UAL,SNDK,FISV,ED,RMD,SYY,KEYS,EXPE,MCHP,FIS,ACGL,PCG,WEC,OTIS,FIX,LYV,
XYL,EQT,KMB,ODFL,KVUE,HPE,RJF,IR,WTW,FITB,MTB,TER,HUM,SYF,NRG,VRSK,DG,VICI,IBKR,ROL,MTD,
FSLR,KHC,CSGP,EME,HBAN,ADM,EXR,BRO,DOV,ATO,EFX,TSCO,AEE,ULTA,TPR,WRB,CHTR,CBOE,DTE,BR,NTRS,
DXCM,EXE,BIIB,PPL,AVB,FE,LEN,CINF,CFG,STLD,AWK,VLTO,ES,JBL,OMC,GIS,STE,CNP,DLTR,LULU,RF,TDY,
STZ,IRM,HUBB,EQR,LDOS,HAL,PPG,PHM,KEY,WAT,EIX,TROW,VRSN,WSM,DVN,ON,L,DRI,NTAP,RL,CPAY,HPQ,
LUV,CMS,IP,LH,PTC,TSN,SBAC,CHD,EXPD,PODD,SW,NVR,CNC,TYL,TPL,NI,WST,INCY,PFG,CTRA,DGX,CHRW,
AMCR,TRMB,GPN,JBHT,PKG,TTD,MKC,SNA,SMCI,IT,CDW,ZBH,FTV,ALB,Q,GPC,LII,PNR,DD,IFF,BG,GDDY,TKO,
GEN,WY,ESS,INVH,LNT,EVRG,APTV,HOLX,DOW,COO,MAA,J,TXT,FOXA,FOX,FFIV,DECK,PSKY,ERIE,BBY,DPZ,
UHS,VTRS,EG,BALL,AVY,SOLV,LYB,ALLE,KIM,HII,NDSN,IEX,JKHY,MAS,HRL,WYNN,REG,AKAM,HST,BEN,ZBRA,
MRNA,BF.B,CF,UDR,AIZ,CLX,IVZ,EPAM,SWK,CPT,HAS,BLDR,ALGN,GL,DOC,DAY,BXP,RVTY,FDS,SJM,PNW,
NCLH,MGM,CRL,AES,BAX,NWSA,SWKS,AOS,TECH,TAP,HSIC,FRT,PAYC,POOL,APA,MOH,ARE,CPB,GNRC,CAG,DVA,
MOS,MTCH,LW,NWS
""".replace(
        "\n", ""
    )
    .replace(" ", "")
    .split(",")
)

# NASDAQ 100 constituents (fetched 2025-12-28 from slickcharts.com)
NASDAQ100_SYMBOLS = (
    """
NVDA,AAPL,MSFT,AMZN,GOOGL,GOOG,META,AVGO,TSLA,PLTR,ASML,NFLX,COST,AMD,MU,CSCO,AZN,APP,TMUS,
LRCX,SHOP,AMAT,ISRG,LIN,PEP,INTU,QCOM,AMGN,BKNG,INTC,KLAC,PDD,TXN,GILD,ADBE,ADI,PANW,HON,
CRWD,VRTX,ARM,CEG,CMCSA,ADP,MELI,DASH,SBUX,SNPS,CDNS,MAR,ABNB,REGN,ORLY,CTAS,MNST,MRVL,WBD,
MDLZ,CSX,ADSK,AEP,FTNT,TRI,PCAR,ROST,WDAY,PYPL,NXPI,IDXX,EA,ROP,DDOG,FAST,TTWO,AXON,MSTR,
BKR,XEL,EXC,TEAM,FANG,CTSH,CCEP,PAYX,KDP,GEHC,CPRT,ZS,MCHP,ODFL,VRSK,KHC,CSGP,CHTR,DXCM,
BIIB,LULU,ON,GFS,TTD,CDW
""".replace(
        "\n", ""
    )
    .replace(" ", "")
    .split(",")
)

# Dow 30 constituents (fetched 2025-12-28)
DOW30_SYMBOLS = (
    """
AAPL,AMGN,AMZN,AXP,BA,CAT,CRM,CSCO,CVX,DIS,GS,HD,HON,IBM,JNJ,JPM,KO,MCD,MMM,MRK,MSFT,NKE,
NVDA,PG,SHW,TRV,UNH,V,VZ,WMT
""".replace(
        "\n", ""
    )
    .replace(" ", "")
    .split(",")
)


def parse_symbols(symbols_list: List[str]) -> Set[str]:
    """Parse and clean symbol list."""
    return {s.strip().upper() for s in symbols_list if s.strip()}


def update_index_flag(flag_name: str, symbols: Set[str], batch_size: int = 100) -> dict:
    """
    Update a specific index flag for given symbols.

    Args:
        flag_name: Column name (sp500, nasdaq100, dow30)
        symbols: Set of ticker symbols to mark as True
        batch_size: Number of symbols to update per batch

    Returns:
        dict with update statistics
    """
    symbols_list = sorted(list(symbols))
    total = len(symbols_list)

    logger.info(f"Updating {flag_name} for {total} symbols in batches of {batch_size}")

    # First, reset all to FALSE
    with ENGINE.begin() as conn:
        result = conn.execute(text(f"UPDATE symbol SET {flag_name} = FALSE WHERE {flag_name} = TRUE"))
        reset_count = result.rowcount
        logger.info(f"  Reset {reset_count} existing {flag_name} flags to FALSE")

    # Then set TRUE for matching symbols in batches
    updated_count = 0
    not_found = []

    for i in range(0, total, batch_size):
        batch = symbols_list[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        with ENGINE.begin() as conn:
            # Update matching symbols
            result = conn.execute(
                text(
                    f"""
                    UPDATE symbol
                    SET {flag_name} = TRUE
                    WHERE ticker = ANY(:symbols)
                """
                ),
                {"symbols": batch},
            )
            batch_updated = result.rowcount
            updated_count += batch_updated

            # Find symbols not in database
            check_result = conn.execute(
                text(
                    """
                    SELECT unnest(:symbols) AS sym
                    EXCEPT
                    SELECT ticker FROM symbol WHERE ticker = ANY(:symbols)
                """
                ),
                {"symbols": batch},
            )
            batch_not_found = [row[0] for row in check_result.fetchall()]
            not_found.extend(batch_not_found)

        logger.info(f"  Batch {batch_num}/{total_batches}: " f"updated {batch_updated}/{len(batch)} symbols")

    return {
        "flag": flag_name,
        "total_symbols": total,
        "updated": updated_count,
        "not_found": not_found,
        "reset": reset_count,
    }


def verify_updates() -> None:
    """Verify the updates by counting flags."""
    with ENGINE.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT
                SUM(CASE WHEN sp500 = TRUE THEN 1 ELSE 0 END) as sp500_count,
                SUM(CASE WHEN nasdaq100 = TRUE THEN 1 ELSE 0 END) as nasdaq100_count,
                SUM(CASE WHEN dow30 = TRUE THEN 1 ELSE 0 END) as dow30_count,
                SUM(CASE WHEN russell1000 = TRUE THEN 1 ELSE 0 END) as russell1000_count
            FROM symbol
        """
            )
        )
        row = result.fetchone()

        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION - Current Index Flag Counts")
        logger.info("=" * 60)
        logger.info(f"  S&P 500:      {row[0]:,}")
        logger.info(f"  NASDAQ 100:   {row[1]:,}")
        logger.info(f"  Dow 30:       {row[2]:,}")
        logger.info(f"  Russell 1000: {row[3]:,}")
        logger.info("=" * 60)


def infer_russell1000() -> None:
    """
    Infer Russell 1000 membership from market cap.

    Russell 1000 = top ~1000 US stocks by market cap.
    We can approximate by taking top 1000 by mktcap.
    """
    logger.info("\nInferring Russell 1000 from market cap rankings...")

    with ENGINE.begin() as conn:
        # Reset existing
        conn.execute(text("UPDATE symbol SET russell1000 = FALSE WHERE russell1000 = TRUE"))

        # Set top 1000 by market cap as Russell 1000
        result = conn.execute(
            text(
                """
            WITH top_1000 AS (
                SELECT ticker
                FROM symbol
                WHERE islisted = TRUE
                  AND mktcap IS NOT NULL
                  AND mktcap > 0
                ORDER BY mktcap DESC
                LIMIT 1000
            )
            UPDATE symbol
            SET russell1000 = TRUE
            WHERE ticker IN (SELECT ticker FROM top_1000)
        """
            )
        )

        logger.info(f"  Set {result.rowcount} symbols as Russell 1000 (by market cap)")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("SYMBOL INDEX FLAGS UPDATE")
    logger.info("=" * 60)

    # Parse symbol lists
    sp500 = parse_symbols(SP500_SYMBOLS)
    nasdaq100 = parse_symbols(NASDAQ100_SYMBOLS)
    dow30 = parse_symbols(DOW30_SYMBOLS)

    logger.info(f"S&P 500:    {len(sp500)} symbols")
    logger.info(f"NASDAQ 100: {len(nasdaq100)} symbols")
    logger.info(f"Dow 30:     {len(dow30)} symbols")
    logger.info("=" * 60)

    # Update each index flag
    results = []

    results.append(update_index_flag("sp500", sp500, batch_size=100))
    results.append(update_index_flag("nasdaq100", nasdaq100, batch_size=100))
    results.append(update_index_flag("dow30", dow30, batch_size=100))

    # Infer Russell 1000 from market cap
    infer_russell1000()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("UPDATE SUMMARY")
    logger.info("=" * 60)

    for r in results:
        logger.info(f"\n{r['flag'].upper()}:")
        logger.info(f"  Total in index:     {r['total_symbols']}")
        logger.info(f"  Updated in DB:      {r['updated']}")
        logger.info(f"  Not found in DB:    {len(r['not_found'])}")
        if r["not_found"]:
            # Show first 10 not found
            sample = r["not_found"][:10]
            logger.info(f"  Missing samples:    {', '.join(sample)}")
            if len(r["not_found"]) > 10:
                logger.info(f"                      ... and {len(r['not_found']) - 10} more")

    # Verify
    verify_updates()

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
