# NVDA Comprehensive Log Analysis - All Issues Identified

**Date**: November 7, 2025
**Log File**: `./logs/NVDA_v2.log`
**Analysis Time**: 12:05:00 - 12:12:20 (7 minutes 20 seconds)
**Overall Status**: ✅ **PARTIALLY SUCCESSFUL** - All agents completed, but SymbolUpdateAgent failed

---

## Executive Summary

The NVDA comprehensive analysis completed all data collection and valuation steps successfully, but **failed at the final database persistence step** (SymbolUpdateAgent). Additionally, there are several data quality warnings and one critical normalization issue. All issues have been categorized by severity and impact.

**Key Metrics**:
- ✅ SEC Agent: 7.10s (65 filings processed)
- ✅ Technical Agent: 237.39s (85 indicators calculated)
- ✅ Fundamental Agent: 337.41s (DCF valuation completed)
- ✅ Market Context Agent: 59.29s (ETF/macro analysis)
- ❌ SymbolUpdate Agent: **FAILED** (database write succeeded, but post-processing error)
- ✅ Synthesis Agent: 66.39s (investment thesis generated)

---

## Priority Classification

- **P0 (Critical - Blocking)**: Issues that prevent data persistence or cause failures
- **P1 (High - Quality Impact)**: Issues that significantly affect analysis quality
- **P2 (Medium - Performance)**: Issues that impact performance or user experience
- **P3 (Low - Informational)**: Minor issues, warnings, or optimization opportunities

---

## P0 ISSUES (CRITICAL - MUST FIX IMMEDIATELY)

### P0-1: SymbolUpdateAgent post_process() Signature Mismatch

**Line**: 350
**Severity**: 🔴 **CRITICAL**
**Status**: ❌ **BLOCKING DATABASE PERSISTENCE**

**Evidence**:
```
2025-11-07 12:11:14,082 - agent.symbol_update_agent_1 - INFO - ✅ Updated symbol table for NVDA: 7 fields, 1 row(s) affected
2025-11-07 12:11:14,131 - AgentOrchestrator - ERROR - Task NVDA_1762538719.558243 (NVDA): step_3 -> symbol_update failed: SymbolUpdateAgent.post_process() takes 2 positional arguments but 3 were given
```

**Impact**:
- Database UPDATE **succeeded** (7 fields updated, 1 row affected)
- Exception occurred during `post_process()` lifecycle hook
- Transaction may have been **rolled back** despite successful UPDATE
- **NO valuation data persisted to database**

**Root Cause**:
The base agent's lifecycle is calling `post_process(self, result, task)` but SymbolUpdateAgent only accepts `post_process(self, result)`.

**Code Analysis** (symbol_update.py):

The agent has a `post_process` method at line 366-376:
```python
def post_process(self, result: AgentResult) -> AgentResult:
    """Post-process results"""
    if result.status == TaskStatus.COMPLETED:
        self.logger.info(
            f"✅ Symbol update completed for {result.result_data.get('symbol', 'unknown')}"
        )
    return result
```

But the base agent (InvestmentAgent) is calling it with TWO parameters: `result` and `task`.

**Expected Behavior**:
```python
def post_process(self, result: AgentResult, task: AgentTask) -> AgentResult:
    """Post-process results with access to task context"""
    # ... implementation
```

**Solution**:
1. Update `post_process()` signature in `src/investigator/domain/agents/symbol_update.py` (line 366)
2. Add `task: AgentTask` parameter to match base agent interface
3. Verify database transaction is properly committed before post-processing

**Files to Fix**:
- `src/investigator/domain/agents/symbol_update.py` (line 366)

**Verification**:
- Re-run NVDA analysis
- Query database to verify valuation fields are populated
- Check for absence of post-process error in log

---

## P1 ISSUES (HIGH PRIORITY - QUALITY IMPACT)

### P1-1: Missing Debt Metrics Data Gap

**Lines**: 99-101
**Severity**: 🟠 **HIGH**
**Status**: ⚠️ **DATA QUALITY IMPACT**

**Evidence**:
```
2025-11-07 12:05:45,550 - agent.fund_agent_1 - WARNING - ⚠️  UPSTREAM DATA GAP for NVDA: Missing debt metrics: totalDebt, shortTermDebt. Debt-related ratios may be unreliable.
2025-11-07 12:05:45,550 - agent.fund_agent_1 - WARNING - ⚠️  UPSTREAM DATA GAP for NVDA: Debt To Equity is 0.0 (likely due to missing financial data). This may affect analysis quality.
2025-11-07 12:05:45,550 - agent.fund_agent_1 - WARNING - ⚠️  UPSTREAM DATA GAP for NVDA: Debt To Assets is 0.0 (likely due to missing financial data). This may affect analysis quality.
```

**Impact**:
- Debt ratios (Debt-to-Equity, Debt-to-Assets) are **incorrectly showing 0.0**
- Solvency analysis is incomplete
- Credit risk assessment may be misleading
- WACC calculation may be using incorrect debt values

**Root Cause**:
1. CanonicalKeyMapper may not have correct XBRL tags for NVDA's debt reporting
2. SEC CompanyFacts API may use different tag names for debt
3. Data normalization may be missing debt metrics

**Investigation Needed**:
1. Check `sec_companyfacts_processed` table for NVDA debt fields
2. Verify XBRL tags in `comprehensive_canonical_mappings_with_derivations.json`
3. Review SEC CompanyFacts API response for NVDA debt items

**Solution Options**:

**Option A: Fix XBRL Tag Mapping**
```python
# Add to comprehensive_canonical_mappings_with_derivations.json
{
  "totalDebt": {
    "xbrl_tags": [
      "LongTermDebt",
      "DebtCurrent",
      "ShortTermBorrowings",
      "LongTermDebtAndCapitalLeaseObligations"
    ],
    "derivation": "sum of long_term_debt and short_term_debt"
  }
}
```

**Option B: Add Fallback to Market Data**
```python
# In fundamental_agent.py
if not debt_metrics or debt_metrics.get('totalDebt', 0) == 0:
    # Fetch from market data API as fallback
    debt_metrics = self._fetch_debt_from_market_data(symbol)
```

**Option C: Use Balance Sheet Liabilities**
```python
# Derive total debt from balance sheet
if 'totalLiabilities' in balance_sheet and 'currentLiabilities' in balance_sheet:
    total_debt = balance_sheet['totalLiabilities'] - balance_sheet['currentLiabilities']
```

**Files to Investigate**:
- `resources/xbrl_mappings/comprehensive_canonical_mappings_with_derivations.json`
- `src/investigator/infrastructure/sec/companyfacts_extractor.py`
- `src/investigator/domain/agents/fundamental.py` (ratio calculation)

---

### P1-2: Stale Bulk SEC Data (163 Days Old)

**Lines**: 85
**Severity**: 🟠 **HIGH**
**Status**: ⚠️ **DATA FRESHNESS**

**Evidence**:
```
2025-11-07 12:05:31,629 - utils.sec_data_strategy - WARNING - Bulk data for NVDA is stale (163 days old). Will attempt CompanyFacts API as fallback.
```

**Impact**:
- **163 days = 5.4 months old** (last bulk load ~May 27, 2025)
- Missing **Q3-2025 and Q2-2026 data** from bulk tables
- System is falling back to CompanyFacts API (correct behavior)
- Performance penalty: API call instead of fast bulk query

**Root Cause**:
- Bulk SEC tables (`sec_sub_data`, `sec_num_data`) not refreshed since May 2025
- Bulk refresh script (`scripts/reprocess_nee_bulk_table.py`) not running on schedule

**Investigation Needed**:
1. Check when bulk tables were last refreshed:
   ```sql
   SELECT MAX(filed) FROM sec_sub_data WHERE cik = '0001045810';  -- NVDA CIK
   ```
2. Verify bulk refresh script exists and is scheduled
3. Check disk space and database connectivity for bulk loader

**Solution**:

**Immediate**: System is already handling this correctly by falling back to API ✅

**Long-term**: Schedule automatic bulk data refresh
```bash
# Add to cron (weekly refresh)
0 2 * * 0 /path/to/scripts/refresh_sec_bulk_data.sh

# Or manual refresh now
python3 scripts/reprocess_nee_bulk_table.py --start-quarter 2025-Q2
```

**Files to Check**:
- `scripts/reprocess_nee_bulk_table.py`
- `src/investigator/infrastructure/sec/data_strategy.py` (lines 85-86)

---

### P1-3: Calendar-Based Fiscal Quarter Fallback

**Lines**: 86
**Severity**: 🟠 **HIGH**
**Status**: ⚠️ **ACCURACY CONCERN**

**Evidence**:
```
2025-11-07 12:05:31,630 - agent.fund_agent_1 - WARNING - Using calendar-based PREVIOUS quarter 2025-Q3 for NVDA. This is a fallback - actual fiscal periods should come from bulk tables.
```

**Impact**:
- Using **calendar Q3-2025** instead of **fiscal period** from SEC filings
- NVDA's fiscal year ends in January, so calendar quarters != fiscal quarters
- Data may be misaligned if fiscal Q3 is actually calendar Q4
- Affects TTM calculations and YoY comparisons

**Root Cause**:
- Bulk data is stale (see P1-2)
- CompanyFacts API may not provide explicit fiscal period labels
- Fallback logic uses calendar-based quarter detection

**Investigation Needed**:
1. Compare NVDA fiscal calendar vs calendar year:
   ```
   NVDA Fiscal Year: Feb 1 - Jan 31
   - Fiscal Q1: Feb-Apr (calendar Q1/Q2)
   - Fiscal Q2: May-Jul (calendar Q2/Q3)
   - Fiscal Q3: Aug-Oct (calendar Q3/Q4)
   - Fiscal Q4: Nov-Jan (calendar Q4/Q1)
   ```
2. Check if CompanyFacts API provides `fy` and `fp` fields
3. Verify fiscal period detection logic in `sec_data_strategy.py`

**Solution**:

**Option A: Use CompanyFacts `fy` and `fp` fields**
```python
# In companyfacts_extractor.py
for fact in facts:
    fiscal_year = fact.get('fy')  # 2025
    fiscal_period = fact.get('fp')  # Q3
    # Use these instead of calendar-based detection
```

**Option B: Maintain Fiscal Calendar Mapping**
```python
# Add to config.json
{
  "fiscal_calendars": {
    "NVDA": {
      "fiscal_year_end": "January",
      "fiscal_q1_months": [2, 3, 4],  # Feb-Apr
      "fiscal_q2_months": [5, 6, 7],  # May-Jul
      "fiscal_q3_months": [8, 9, 10], # Aug-Oct
      "fiscal_q4_months": [11, 12, 1] # Nov-Jan
    }
  }
}
```

**Files to Fix**:
- `src/investigator/infrastructure/sec/companyfacts_extractor.py`
- `utils/sec_data_strategy.py` (fiscal period detection)

---

### P1-4: Dynamic Model Weighting Normalization Failure

**Lines**: 305
**Severity**: 🟠 **HIGH**
**Status**: ⚠️ **USING FALLBACK WEIGHTS**

**Evidence**:
```
2025-11-07 12:09:27,765 - investigator.domain.services.dynamic_model_weighting - WARNING - Failed to normalize weights: All weights are zero or negative, cannot normalize, using fallback
2025-11-07 12:09:27,770 - investigator.domain.services.dynamic_model_weighting - INFO - 🎯 NVDA - Dynamic Weighting: Tier=pre_profit_negative_ebitda | Sector=Technology | Industry=Semiconductors & Semiconductor Equipment | Weights: DCF=30%, PE=25%, PS=15%, PB=10%, EV_EBITDA=20%
```

**Impact**:
- Model weighting system **detected all weights as zero or negative**
- Falling back to hardcoded tier-based weights
- This works BUT suggests tier classification may be incorrect
- **NVDA is NOT "pre-profit with negative EBITDA"** - NVDA has massive profits!

**Root Cause Analysis**:

**NVDA Financial Profile** (from log lines 176-177, 304):
- TTM Revenue: $96.31B (growth: +194.7%)
- TTM FCF: $42.41B
- FCF Margin: **48.4%**
- Rule of 40: **131.9%** (EXCELLENT)
- Debt: $8.46B
- Market Cap: $4,673.41B

**Expected Tier**: `high_growth_profitable` or `mature_high_margin`
**Actual Tier**: `pre_profit_negative_ebitda` ❌

**Why This Happened**:
1. Tier classification logic is checking EBITDA
2. NVDA may have $0 EBITDA in the processed data (similar to missing debt metrics)
3. System defaults to most conservative tier when EBITDA is missing

**Investigation Needed**:
1. Check EBITDA in processed SEC data:
   ```python
   # In fundamental_agent.py
   print(f"EBITDA: {fundamental_data.get('ebitda')}")
   print(f"Operating Income: {fundamental_data.get('operating_income')}")
   ```
2. Review tier classification logic in `dynamic_model_weighting.py`
3. Verify EBITDA calculation/extraction

**Solution**:

**Fix Tier Classification Logic**:
```python
# src/investigator/domain/services/dynamic_model_weighting.py

def _determine_tier(self, company_data: Dict) -> str:
    """Determine company tier with robust fallbacks"""

    # Extract financials
    revenue = company_data.get('revenue', 0)
    ebitda = company_data.get('ebitda', 0)
    fcf = company_data.get('free_cash_flow', 0)
    net_income = company_data.get('net_income', 0)
    fcf_margin = company_data.get('fcf_margin', 0)

    # CRITICAL FIX: Use FCF as fallback if EBITDA is missing
    profitability_metric = ebitda if ebitda != 0 else fcf

    if profitability_metric > 0 and fcf_margin > 0.25:
        return "mature_high_margin"  # CORRECT for NVDA
    elif profitability_metric > 0:
        return "high_growth_profitable"
    # ... rest of logic
```

**Files to Fix**:
- `src/investigator/domain/services/dynamic_model_weighting.py` (tier classification)
- Add unit test for NVDA-like companies with high FCF, missing EBITDA

---

### P1-5: Quarterly Data Normalization Gaps

**Lines**: 74-76
**Severity**: 🟠 **HIGH**
**Status**: ⚠️ **DATA COMPLETENESS**

**Evidence**:
```
2025-11-07 12:05:31,002 - investigator.infrastructure.sec.data_processor - WARNING - Cannot normalize NVDA 2009-Q2: Previous period Q1 not found in processed filings
2025-11-07 12:05:31,003 - investigator.infrastructure.sec.data_processor - WARNING - Cannot normalize NVDA 2015-Q2: Previous period Q1 not found in processed filings
2025-11-07 12:05:31,003 - investigator.infrastructure.sec.data_processor - WARNING - Cannot normalize NVDA 2021-Q3: Previous period Q2 not found in processed filings
```

**Impact**:
- **3 quarters** cannot be normalized from YTD to quarterly values
- Affects historical trend analysis for 2009, 2015, 2021
- TTM calculations skip these quarters
- **Low impact** for current valuation (using recent data)

**Root Cause**:
- SEC filings missing Q1-2009, Q1-2015, Q2-2021
- Company may have skipped filing or filed late
- Data extraction may have missed these periods

**Investigation Needed**:
1. Check SEC EDGAR for missing filings:
   - https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001045810&type=10-Q
2. Verify if CompanyFacts API has data for these periods
3. Check `sec_companyfacts_raw` table for these periods

**Solution**:

**Accept as limitation** (LOW priority fix):
- These are historical quarters (15, 9, and 3 years old)
- Current valuation uses recent 12 quarters
- Mark these quarters as "incomplete" in metadata

**IF needed for historical analysis**:
- Manual data entry from PDF 10-Q filings
- OR use alternative data source (Bloomberg, FactSet)

**Files to Review**:
- `src/investigator/infrastructure/sec/data_processor.py` (normalization logic)
- Database: `sec_companyfacts_processed` table

---

## P2 ISSUES (MEDIUM PRIORITY - PERFORMANCE)

### P2-1: LLM Pool Wait Time on Concurrent Requests

**Lines**: 102
**Severity**: 🟡 **MEDIUM**
**Status**: 📊 **PERFORMANCE OPTIMIZATION**

**Evidence**:
```
2025-11-07 12:05:45,577 - investigator.infrastructure.llm.pool - WARNING - ⏳ POOL_WAITING model=qwen3:30b required_vram=25.62GB summary=http://localhost:11434: 0.0GB used + 25.6GB reserved / 48GB total (53%), 0 models, 1 active; http://192.168.1.12:11434: 0.0GB used + 25.6GB reserved / 36GB total (71%), 0 models, 1 active
```

**Impact**:
- Two servers, both have pending reservations
- localhost: 25.6GB reserved out of 48GB (53% reserved, but 0% used)
- mac-studio: 25.6GB reserved out of 36GB (71% reserved, but 0% used)
- **Issue**: VRAM is RESERVED but not USED (models not loaded yet)
- Causes temporary queueing for new requests

**Root Cause**:
- Model weights take 5-10 seconds to load into VRAM
- During load time, VRAM is reserved but `0.0GB used` shows in /api/ps
- Multiple concurrent requests all reserve VRAM simultaneously

**Impact**: **MINOR** - System is working correctly, just logging wait state

**Optimization Options**:

**Option A: Pre-warm Models**
```python
# On orchestrator startup, load models once
async def _warmup_models(self):
    """Pre-load models to avoid first-request delays"""
    await self.llm_pool.warmup_model("qwen3:30b")
```

**Option B: Adjust Reservation Logic**
```python
# Only reserve 80% of model size until confirmed loaded
reserved_vram = model_size * 0.8
```

**Option C: Accept as designed** ✅ (Recommended)
- This is expected behavior during concurrent startup
- Models load within 5-10 seconds
- Subsequent requests reuse loaded models

**Files to Review** (if optimizing):
- `src/investigator/infrastructure/llm/pool.py` (VRAM management)
- `src/investigator/infrastructure/llm/semaphore.py`

---

### P2-2: Scenario Generation Failed - Using Fallback

**Lines**: 371
**Severity**: 🟡 **MEDIUM**
**Status**: ⚠️ **OUTPUT QUALITY**

**Evidence**:
```
2025-11-07 12:11:53,507 - agent.synth_agent_1 - WARNING - Scenario generation failed for NVDA: Missing required scenario cases. Using fallback scenarios.
```

**Impact**:
- LLM failed to generate bull/base/bear scenarios
- System using generic fallback scenarios
- **Low impact** on final investment thesis
- **User experience**: Scenarios may be less specific

**Root Cause**:
- LLM response didn't include required fields (e.g., `bull_case`, `base_case`, `bear_case`)
- Prompt may need refinement
- OR JSON parsing failed

**Investigation Needed**:
1. Check LLM response for `_generate_scenarios_prompt` (line 367-370)
2. Review expected JSON schema vs actual response
3. Check if thinking-based model needs different prompt format

**Solution**:

**Fix Prompt to Match Reasoning Model**:
```python
# In synthesis_agent.py
scenario_prompt = f"""
You are analyzing {symbol}. Generate three investment scenarios.

Think through your analysis first, then provide JSON with:
{{
  "bull_case": {{
    "description": "...",
    "target_price": 250.00,
    "probability": 0.25
  }},
  "base_case": {{ ... }},
  "bear_case": {{ ... }}
}}
"""
```

**Files to Fix**:
- `src/investigator/domain/agents/synthesis.py` (scenario generation prompt)
- Check `patterns/analysis/scenario_templates.py` if exists

---

### P2-3: No Investment Thesis Available in Report Generator

**Lines**: 394
**Severity**: 🟡 **MEDIUM**
**Status**: ⚠️ **PDF REPORT INCOMPLETE**

**Evidence**:
```
2025-11-07 12:12:20,441 - utils.report_generator - WARNING - No investment thesis available for NVDA
```

**Impact**:
- PDF report missing investment thesis section
- JSON output likely has thesis (line 336 shows it was saved)
- **Issue**: Report generator can't find thesis in data structure

**Root Cause**:
- Data structure mismatch between synthesis agent output and report generator
- Investment thesis saved under different key name
- OR report generator looking in wrong path

**Investigation Needed**:
1. Check synthesis agent output structure (line 336):
   ```python
   # What key was used?
   "fundamental_investment_thesis"  # From line 336
   ```
2. Check report generator code:
   ```python
   # What key is it looking for?
   thesis = data.get('investment_thesis')  # OR
   thesis = data.get('synthesis', {}).get('thesis')
   ```

**Solution**:

**Fix Key Name Consistency**:
```python
# In report_generator.py (utils/report_generator.py)

# OLD (broken):
thesis = data.get('investment_thesis')

# NEW (fixed):
thesis = (
    data.get('investment_thesis') or  # Try direct key first
    data.get('fundamental_analysis', {}).get('investment_thesis') or  # Then nested
    data.get('synthesis', {}).get('investment_thesis')  # Then synthesis
)

if not thesis:
    logger.warning(f"No investment thesis available for {symbol}")
    thesis = {"summary": "Thesis unavailable - see full analysis"}
```

**Files to Fix**:
- `utils/report_generator.py` (thesis extraction logic)

---

## P3 ISSUES (LOW PRIORITY - INFORMATIONAL)

### P3-1: Executive Summary Shows "N/A" Values

**Lines**: 405-423
**Severity**: 🟢 **LOW**
**Status**: 📋 **USER EXPERIENCE**

**Evidence**:
```
============================================================
EXECUTIVE SUMMARY
============================================================
Symbol: NVDA
Recommendation: N/A
Confidence: N/A
Price: $N/A → Target: $N/A (0.0%)
Investment Grade: N/A
Data Quality: N/A (N/A%)
```

**Impact**:
- CLI output shows placeholder values
- **JSON output likely has correct values**
- Issue is in summary formatting, not data

**Root Cause**:
- Summary generator not extracting values from synthesis output
- Key name mismatch (similar to P2-3)

**Solution**: Same fix as P2-3 - fix key name mapping in summary generator

**Files to Fix**:
- `cli_orchestrator.py` (summary formatting)

---

### P3-2: Multiple Database Engine Initializations

**Lines**: 25-30, 384-399
**Severity**: 🟢 **LOW**
**Status**: 📋 **CODE QUALITY**

**Evidence**:
```
2025-11-07 12:05:07,448 - utils.db - INFO - Database engine initialized successfully (6x in a row)
2025-11-07 12:12:20,439 - utils.db - INFO - Database engine initialized successfully (10x in a row)
```

**Impact**:
- **16 engine initializations** during one analysis
- Each agent initializing its own engine
- **No functional issue** (SQLAlchemy reuses connections)
- Minor performance penalty (connection pool setup)

**Root Cause**:
- Each agent calling `get_engine()` independently
- No shared engine instance

**Solution** (Optimization):

**Use Dependency Injection**:
```python
# In orchestrator.py
engine = get_engine()

agents = [
    SECAgent(engine=engine),
    FundamentalAgent(engine=engine),
    TechnicalAgent(engine=engine),
    # ...
]
```

**OR Accept as designed** ✅ - SQLAlchemy handles this efficiently

---

## VERIFICATION CHECKLIST

After fixing P0 and P1 issues:

### 1. Fix P0-1 (post_process signature)
```bash
# Edit symbol_update.py
# Change: def post_process(self, result: AgentResult) -> AgentResult:
# To:     def post_process(self, result: AgentResult, task: AgentTask) -> AgentResult:
```

### 2. Test NVDA Analysis
```bash
# Clear caches
rm -rf data/llm_cache/NVDA
PGPASSWORD=investigator psql -h ${DB_HOST:-localhost} -U investigator -d sec_database \
  -c "DELETE FROM llm_responses WHERE symbol = 'NVDA';"

# Run analysis
python3 cli_orchestrator.py analyze NVDA -m standard --force-refresh

# Verify database
PGPASSWORD=investigator psql -h ${DB_HOST:-localhost} -U investigator -d stock \
  -c "SELECT ticker, fair_value_blended, fair_value_dcf, tier_classification,
      model_confidence, valuation_updated_at FROM symbol WHERE ticker = 'NVDA';"
```

### 3. Expected Results
- ✅ No post_process error in log
- ✅ `fair_value_blended`: ~$165.17
- ✅ `fair_value_dcf`: ~$165.17
- ✅ `tier_classification`: Should be `mature_high_margin` (NOT `pre_profit_negative_ebitda`)
- ✅ `model_confidence`: 0.85+ (high confidence)
- ✅ `valuation_updated_at`: Current timestamp

---

## SUMMARY TABLE

| Priority | Issue | Severity | Status | Impact | ETA |
|----------|-------|----------|--------|--------|-----|
| P0-1 | post_process signature | 🔴 Critical | ❌ Blocking | No DB persistence | 5 min |
| P1-1 | Missing debt metrics | 🟠 High | ⚠️ Quality | Wrong ratios | 2 hours |
| P1-2 | Stale bulk data | 🟠 High | ⚠️ Freshness | Performance penalty | 1 hour |
| P1-3 | Fiscal quarter fallback | 🟠 High | ⚠️ Accuracy | Wrong periods | 1 hour |
| P1-4 | Tier classification | 🟠 High | ⚠️ Quality | Wrong weights | 30 min |
| P1-5 | Normalization gaps | 🟠 High | ⚠️ Completeness | Historical only | N/A |
| P2-1 | LLM pool waiting | 🟡 Medium | 📊 Performance | 5-10s delay | Optional |
| P2-2 | Scenario generation | 🟡 Medium | ⚠️ Output | Generic scenarios | 30 min |
| P2-3 | Report thesis missing | 🟡 Medium | ⚠️ PDF | Incomplete report | 15 min |
| P3-1 | Summary N/A values | 🟢 Low | 📋 UX | Display only | 15 min |
| P3-2 | Multiple DB inits | 🟢 Low | 📋 Code quality | Minor overhead | Optional |

**Total Estimated Fix Time**:
- **P0**: 5 minutes (MUST FIX NOW)
- **P1**: ~5 hours (High priority)
- **P2**: ~1.5 hours (Medium priority)
- **P3**: ~30 minutes (Low priority)

---

## RECOMMENDED FIX ORDER

1. **P0-1** (5 min): Fix post_process signature → Unblocks database persistence
2. **P1-4** (30 min): Fix tier classification → Correct weights for NVDA
3. **P1-1** (2 hours): Fix debt metrics → Accurate solvency ratios
4. **P2-3** (15 min): Fix report generator → Complete PDF reports
5. **P1-3** (1 hour): Fix fiscal period detection → Accurate quarters
6. **P1-2** (1 hour): Refresh bulk data → Better performance
7. **P2-2** (30 min): Fix scenario generation → Better scenarios
8. **P3-1** (15 min): Fix summary display → Better UX

**Total Priority Work**: ~6 hours

---

**Analysis By**: InvestiGator DevOps Team
**Status**: Ready for sequential fixes
**Next Step**: Fix P0-1 immediately
