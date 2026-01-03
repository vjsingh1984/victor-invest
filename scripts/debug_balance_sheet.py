#!/usr/bin/env python3
"""Debug balance sheet identity issues for AMT and NEE"""

from dao.sec_bulk_dao import SECBulkDAO
from sqlalchemy import text

dao = SECBulkDAO()

# Get AMT equity values (latest only)
amt_cik = dao.get_cik("AMT")
with dao.engine.connect() as conn:
    query = text(
        """
        SELECT DISTINCT ON (n.tag) n.tag, n.value
        FROM sec_num_data n
        JOIN sec_sub_data s ON n.adsh = s.adsh AND n.quarter_id = s.quarter_id
        WHERE s.cik = :cik
          AND s.fy = 2024
          AND s.fp = 'FY'
          AND n.tag IN ('StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 'Assets', 'Liabilities')
          AND (n.qtrs = 0 OR n.qtrs = 4)
        ORDER BY n.tag, n.ddate DESC
    """
    )
    results = conn.execute(query, {"cik": amt_cik}).fetchall()
    print("AMT Balance Sheet (latest values):")
    for row in results:
        print(f"  {row[0]}: ${row[1]:,.0f}")

    # Calculate
    assets = next((r[1] for r in results if r[0] == "Assets"), 0)
    liab = next((r[1] for r in results if r[0] == "Liabilities"), 0)
    equity_simple = next((r[1] for r in results if r[0] == "StockholdersEquity"), 0)
    equity_with_nci = next(
        (r[1] for r in results if r[0] == "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), 0
    )

    print(f"\nBalance Sheet Check:")
    print(f"  Assets: ${assets:,.0f}")
    print(f"  Liabilities: ${liab:,.0f}")
    print(f"  Equity (simple): ${equity_simple:,.0f}")
    print(f"  Equity (with NCI): ${equity_with_nci:,.0f}")
    print(f"  Liab + Equity (simple): ${liab + equity_simple:,.0f} (diff: ${assets - (liab + equity_simple):,.0f})")
    print(
        f"  Liab + Equity (with NCI): ${liab + equity_with_nci:,.0f} (diff: ${assets - (liab + equity_with_nci):,.0f})"
    )

print("\n" + "=" * 80)
print("NEE ANALYSIS")
print("=" * 80)

# Get NEE  revenue tags
nee_cik = dao.get_cik("NEE")
with dao.engine.connect() as conn:
    query = text(
        """
        SELECT DISTINCT ON (n.tag) n.tag, n.value
        FROM sec_num_data n
        JOIN sec_sub_data s ON n.adsh = s.adsh AND n.quarter_id = s.quarter_id
        WHERE s.cik = :cik
          AND s.fy = 2024
          AND s.fp = 'FY'
          AND (n.tag LIKE '%Revenue%' OR n.tag LIKE '%OperatingIncome%')
          AND n.qtrs = 4
        ORDER BY n.tag, n.ddate DESC
        LIMIT 10
    """
    )
    results = conn.execute(query, {"cik": nee_cik}).fetchall()
    print("\nNEE Revenue-Related Tags:")
    for row in results:
        print(f"  {row[0]}: ${row[1]:,.0f}")

# Check NEE balance sheet
with dao.engine.connect() as conn:
    query = text(
        """
        SELECT DISTINCT ON (n.tag) n.tag, n.value
        FROM sec_num_data n
        JOIN sec_sub_data s ON n.adsh = s.adsh AND n.quarter_id = s.quarter_id
        WHERE s.cik = :cik
          AND s.fy = 2024
          AND s.fp = 'FY'
          AND n.tag IN ('StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
                        'Assets', 'Liabilities', 'TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests')
          AND (n.qtrs = 0 OR n.qtrs = 4)
        ORDER BY n.tag, n.ddate DESC
    """
    )
    results = conn.execute(query, {"cik": nee_cik}).fetchall()
    print("\nNEE Balance Sheet (latest values):")
    for row in results:
        print(f"  {row[0]}: ${row[1]:,.0f}")

    # Calculate
    assets = next((r[1] for r in results if r[0] == "Assets"), 0)
    liab = next((r[1] for r in results if r[0] == "Liabilities"), 0)
    equity_simple = next((r[1] for r in results if r[0] == "StockholdersEquity"), 0)
    equity_with_nci = next(
        (r[1] for r in results if r[0] == "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), 0
    )
    temp_equity = next(
        (
            r[1]
            for r in results
            if r[0] == "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests"
        ),
        0,
    )

    print(f"\nBalance Sheet Check:")
    print(f"  Assets: ${assets:,.0f}")
    print(f"  Liabilities: ${liab:,.0f}")
    print(f"  Equity (simple): ${equity_simple:,.0f}")
    print(f"  Equity (with NCI): ${equity_with_nci:,.0f}")
    print(f"  Temporary Equity: ${temp_equity:,.0f}")
    print(f"  Liab + Equity (simple): ${liab + equity_simple:,.0f} (diff: ${assets - (liab + equity_simple):,.0f})")
    if equity_with_nci:
        print(
            f"  Liab + Equity (with NCI): ${liab + equity_with_nci:,.0f} (diff: ${assets - (liab + equity_with_nci):,.0f})"
        )
    print(
        f"  Liab + Equity + Temp Equity: ${liab + equity_simple + temp_equity:,.0f} (diff: ${assets - (liab + equity_simple + temp_equity):,.0f})"
    )
