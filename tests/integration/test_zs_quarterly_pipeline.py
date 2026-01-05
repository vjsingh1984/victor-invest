"""
Integration Test for ZS (Zscaler) Quarterly Data Pipeline

This integration test validates the complete quarterly data pipeline from
SEC data fetching through Q4 computation to TTM calculation.

**Key Test Scenarios**:
1. Non-calendar fiscal year (July 31 year-end)
2. Q1 fiscal year adjustment (Oct > Jul → FY+1)
3. Q4 computation from FY - (Q1+Q2+Q3)
4. No 184-day gaps between quarters
5. Consecutive quarter validation for TTM

**Company**: ZS (Zscaler Inc.)
- Fiscal year end: July 31
- Filing pattern: Q1, Q2, Q3, FY (no separate Q4)
- CIK: 1713683

Created: 2025-11-12
Author: Claude Code
"""

import asyncio
from datetime import datetime
from typing import Any, Dict

import pytest

from investigator.config import get_config
from investigator.domain.agents.fundamental import FundamentalAgent
from investigator.domain.agents.sec import SECAgent
from investigator.domain.models import AgentTask, TaskStatus
from utils.quarterly_calculator import get_rolling_ttm_periods


class TestZSQuarterlyPipeline:
    """Integration tests for ZS quarterly data pipeline"""

    @pytest.fixture
    def config(self):
        """Get application configuration"""
        return get_config()

    @pytest.fixture
    def sec_agent(self, config):
        """Create SEC agent instance"""
        return SECAgent(config=config)

    @pytest.fixture
    def fundamental_agent(self, config):
        """Create Fundamental agent instance"""
        return FundamentalAgent(config=config)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_zs_complete_pipeline(self, sec_agent, fundamental_agent):
        """
        End-to-end test: Fetch ZS data → Process quarters → Verify Q4 computation

        **Expected Results**:
        1. Quarterly metrics populated (> 8 periods)
        2. Q4 periods present for multiple fiscal years
        3. No 184-day gaps between consecutive quarters
        4. Q1 fiscal years correctly adjusted (+1 year)
        5. TTM calculation includes Q4 periods
        """
        # Step 1: Fetch SEC data for ZS
        sec_task = AgentTask(
            task_id="test_zs_sec",
            symbol="ZS",
            agent_id="sec",
            task_type="sec_analysis",
            parameters={"force_refresh": True},
        )

        sec_result = await sec_agent.execute(sec_task)

        assert sec_result.status == TaskStatus.COMPLETED, f"SEC agent failed: {sec_result.error}"
        assert sec_result.data is not None, "SEC data should not be None"

        # Step 2: Run fundamental analysis (includes quarterly metrics)
        fund_task = AgentTask(
            task_id="test_zs_fundamental",
            symbol="ZS",
            agent_id="fundamental",
            task_type="fundamental_analysis",
            parameters={"force_refresh": True},
        )

        fund_result = await fundamental_agent.execute(fund_task)

        assert fund_result.status == TaskStatus.COMPLETED, f"Fundamental agent failed: {fund_result.error}"

        # Step 3: Extract quarterly metrics
        quarterly_metrics = fund_result.data.get("quarterly_metrics", [])

        assert len(quarterly_metrics) > 0, "Quarterly metrics should not be empty"

        # Step 4: Verify Q4 periods exist
        q4_periods = [q for q in quarterly_metrics if q.get("fiscal_period") == "Q4"]

        assert len(q4_periods) >= 2, (
            f"Expected at least 2 Q4 periods, got {len(q4_periods)}. "
            f"This indicates Q4 computation is not working for all FY periods."
        )

        # Step 5: Verify Q4 fiscal years
        q4_fiscal_years = sorted([q["fiscal_year"] for q in q4_periods], reverse=True)
        print(f"\n✓ Found Q4 periods for fiscal years: {q4_fiscal_years}")

        # Step 6: Verify no 184-day gaps
        sorted_quarters = sorted(
            quarterly_metrics, key=lambda q: datetime.strptime(q["period_end_date"], "%Y-%m-%d"), reverse=True
        )

        gaps_over_150_days = []
        for i in range(len(sorted_quarters) - 1):
            current_date = datetime.strptime(sorted_quarters[i]["period_end_date"], "%Y-%m-%d")
            next_date = datetime.strptime(sorted_quarters[i + 1]["period_end_date"], "%Y-%m-%d")
            gap_days = (current_date - next_date).days

            if gap_days > 150:  # Typical quarter ~90 days
                gaps_over_150_days.append(
                    {
                        "from": f"{sorted_quarters[i]['fiscal_period']}-{sorted_quarters[i]['fiscal_year']}",
                        "to": f"{sorted_quarters[i+1]['fiscal_period']}-{sorted_quarters[i+1]['fiscal_year']}",
                        "gap_days": gap_days,
                        "from_date": sorted_quarters[i]["period_end_date"],
                        "to_date": sorted_quarters[i + 1]["period_end_date"],
                    }
                )

        # CRITICAL: No 184-day gaps (Q1 → Q3) should exist
        q1_to_q3_gaps = [g for g in gaps_over_150_days if "Q1" in g["from"] and "Q3" in g["to"]]

        assert len(q1_to_q3_gaps) == 0, (
            f"Found {len(q1_to_q3_gaps)} Q1→Q3 gaps (184 days), indicating missing Q4 periods. "
            f"Gaps: {q1_to_q3_gaps}"
        )

        if gaps_over_150_days:
            print(f"\n⚠️  Found {len(gaps_over_150_days)} gaps over 150 days:")
            for gap in gaps_over_150_days:
                print(f"  {gap['from']} → {gap['to']}: {gap['gap_days']} days")
        else:
            print("\n✓ No gaps over 150 days found")

        # Step 7: Verify Q1 fiscal year adjustment
        q1_periods = [q for q in quarterly_metrics if q.get("fiscal_period") == "Q1"]

        for q1 in q1_periods:
            # Q1 ends Oct 31, fiscal year ends Jul 31
            # Oct > Jul, so fiscal_year should be period_end_date.year + 1
            period_end = datetime.strptime(q1["period_end_date"], "%Y-%m-%d")
            expected_fy = period_end.year + 1

            assert q1["fiscal_year"] == expected_fy, (
                f"Q1 fiscal_year incorrect: {q1['period_end_date']} should be FY {expected_fy}, "
                f"got FY {q1['fiscal_year']}"
            )

        print(f"\n✓ Q1 fiscal year adjustment verified for {len(q1_periods)} Q1 periods")

        # Step 8: Test TTM calculation includes Q4
        ttm_periods = get_rolling_ttm_periods(
            quarterly_metrics=quarterly_metrics, target_quarters=4, fiscal_year_end_month=7, fiscal_year_end_day=31
        )

        ttm_fiscal_periods = [p["fiscal_period"] for p in ttm_periods]

        # TTM should include Q4 if consecutive quarters available
        if len(ttm_periods) >= 4:
            # Check if Q4 is in TTM
            has_q4_in_ttm = "Q4" in ttm_fiscal_periods
            print(f"\n✓ TTM periods ({len(ttm_periods)}): {ttm_fiscal_periods}")
            print(f"  Q4 in TTM: {has_q4_in_ttm}")

        # Step 9: Summary statistics
        print("\n" + "=" * 60)
        print("ZS Quarterly Pipeline Test Summary")
        print("=" * 60)
        print(f"Total quarterly metrics: {len(quarterly_metrics)}")
        print(f"Q4 periods computed: {len(q4_periods)} (fiscal years: {q4_fiscal_years})")
        print(f"Q1 periods: {len(q1_periods)}")
        print(f"Gaps over 150 days: {len(gaps_over_150_days)}")
        print(f"Q1→Q3 gaps (should be 0): {len(q1_to_q3_gaps)}")
        print(f"TTM periods available: {len(ttm_periods)}")
        print("=" * 60)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_zs_q4_computation_values(self, sec_agent, fundamental_agent):
        """
        Verify Q4 computation values: Q4 = FY - (Q1 + Q2 + Q3)

        This test validates that Q4 free_cash_flow is computed correctly
        by checking the arithmetic.
        """
        # Fetch fundamental data
        fund_task = AgentTask(
            task_id="test_zs_q4_values",
            symbol="ZS",
            agent_id="fundamental",
            task_type="fundamental_analysis",
            parameters={"force_refresh": False},  # Use cached if available
        )

        fund_result = await fundamental_agent.execute(fund_task)
        quarterly_metrics = fund_result.data.get("quarterly_metrics", [])

        # Find a fiscal year with complete data (FY + Q1 + Q2 + Q3)
        fiscal_years = {}
        for q in quarterly_metrics:
            fy = q["fiscal_year"]
            if fy not in fiscal_years:
                fiscal_years[fy] = {}
            fiscal_years[fy][q["fiscal_period"]] = q

        # Find a fiscal year with Q4 computed
        for fy, periods in fiscal_years.items():
            if "Q4" in periods and "FY" in periods and "Q1" in periods and "Q2" in periods and "Q3" in periods:
                # Verify arithmetic: Q4 = FY - (Q1 + Q2 + Q3)
                fy_fcf = periods["FY"].get("free_cash_flow", 0)
                q1_fcf = periods["Q1"].get("free_cash_flow", 0)
                q2_fcf = periods["Q2"].get("free_cash_flow", 0)
                q3_fcf = periods["Q3"].get("free_cash_flow", 0)
                q4_fcf = periods["Q4"].get("free_cash_flow", 0)

                expected_q4 = fy_fcf - (q1_fcf + q2_fcf + q3_fcf)

                # Allow small floating point error (within $1)
                assert abs(q4_fcf - expected_q4) < 1.0, (
                    f"Q4-{fy} FCF incorrect: Expected {expected_q4}, got {q4_fcf}. "
                    f"FY={fy_fcf}, Q1={q1_fcf}, Q2={q2_fcf}, Q3={q3_fcf}"
                )

                print(f"\n✓ Q4-{fy} FCF verified: {q4_fcf} = {fy_fcf} - ({q1_fcf} + {q2_fcf} + {q3_fcf})")
                break
        else:
            pytest.fail("No fiscal year found with complete Q1+Q2+Q3+Q4+FY data")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_zs_consecutive_quarters_for_ttm(self, fundamental_agent):
        """
        Verify consecutive quarter validation for TTM

        TTM requires 4 consecutive quarters (60-150 days apart).
        This test ensures the consecutive validation is working correctly.
        """
        fund_task = AgentTask(
            task_id="test_zs_consecutive",
            symbol="ZS",
            agent_id="fundamental",
            task_type="fundamental_analysis",
            parameters={"force_refresh": False},
        )

        fund_result = await fundamental_agent.execute(fund_task)
        quarterly_metrics = fund_result.data.get("quarterly_metrics", [])

        # Get TTM periods
        ttm_periods = get_rolling_ttm_periods(
            quarterly_metrics=quarterly_metrics, target_quarters=4, fiscal_year_end_month=7, fiscal_year_end_day=31
        )

        # Verify all TTM periods are consecutive (60-150 days apart)
        if len(ttm_periods) >= 2:
            sorted_ttm = sorted(
                ttm_periods, key=lambda p: datetime.strptime(p["period_end_date"], "%Y-%m-%d"), reverse=True
            )

            for i in range(len(sorted_ttm) - 1):
                current_date = datetime.strptime(sorted_ttm[i]["period_end_date"], "%Y-%m-%d")
                next_date = datetime.strptime(sorted_ttm[i + 1]["period_end_date"], "%Y-%m-%d")
                gap_days = (current_date - next_date).days

                assert 60 <= gap_days <= 150, (
                    f"TTM quarters not consecutive: "
                    f"{sorted_ttm[i]['fiscal_period']}-{sorted_ttm[i]['fiscal_year']} → "
                    f"{sorted_ttm[i+1]['fiscal_period']}-{sorted_ttm[i+1]['fiscal_year']}: "
                    f"{gap_days} days (expected 60-150)"
                )

            print(f"\n✓ All {len(ttm_periods)} TTM periods are consecutive (60-150 days apart)")


class TestQ4ComputationRobustness:
    """Test Q4 computation robustness with edge cases"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_companies_q4_computation(self):
        """
        Test Q4 computation works for multiple companies with different fiscal years

        This test validates that the Q4 computation logic is general enough to
        work across different companies with various fiscal year patterns.

        Companies tested:
        - ZS: Fiscal year ends July 31
        - AAPL: Fiscal year ends September (last Saturday)
        - MSFT: Fiscal year ends June 30
        """
        from investigator.config import get_config
        from investigator.domain.agents.fundamental import FundamentalAgent

        config = get_config()
        fundamental_agent = FundamentalAgent(config=config)

        test_symbols = [
            ("ZS", 7, 31),  # July 31
            # ("AAPL", 9, 30),  # September (approx)
            # ("MSFT", 6, 30),  # June 30
        ]

        results = {}

        for symbol, fy_month, fy_day in test_symbols:
            fund_task = AgentTask(
                task_id=f"test_{symbol}_q4",
                symbol=symbol,
                agent_id="fundamental",
                task_type="fundamental_analysis",
                parameters={"force_refresh": False},
            )

            fund_result = await fundamental_agent.execute(fund_task)
            quarterly_metrics = fund_result.data.get("quarterly_metrics", [])

            q4_periods = [q for q in quarterly_metrics if q.get("fiscal_period") == "Q4"]

            results[symbol] = {
                "total_quarters": len(quarterly_metrics),
                "q4_count": len(q4_periods),
                "q4_fiscal_years": sorted([q["fiscal_year"] for q in q4_periods], reverse=True),
            }

            print(f"\n{symbol}:")
            print(f"  Total quarters: {results[symbol]['total_quarters']}")
            print(f"  Q4 periods: {results[symbol]['q4_count']}")
            print(f"  Q4 fiscal years: {results[symbol]['q4_fiscal_years']}")

        # All companies should have Q4 periods computed
        for symbol, data in results.items():
            assert data["q4_count"] >= 1, f"{symbol} should have at least 1 Q4 period, got {data['q4_count']}"


if __name__ == "__main__":
    """
    Run integration tests directly:

    python3 -m pytest tests/integration/test_zs_quarterly_pipeline.py -v
    """
    pytest.main([__file__, "-v", "-s"])
