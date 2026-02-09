# TODOs and FIXMEs

This document tracks all outstanding TODOs and FIXMEs in the codebase.

## Last Updated: 2025-02-09

## Summary

- **Total TODOs**: 19
- **Completed**: 4 ✅
- **Remaining**: 15
- **High Priority**: 0 (All completed!)
- **Medium Priority**: 7 (Feature implementations)
- **Low Priority**: 4 (Future enhancements)
- **Deferred**: 4 (API server)

---

## ✅ Completed Tasks (2025-02-09)

### High Priority: Clean Architecture Migration (ALL COMPLETED!)

**1. ✅ Synthesizer Import Migration (COMPLETED)**
- Moved `llm_facade` from `patterns/llm/` to `investigator.infrastructure.llm/`
- Moved `peer_comparison` from `patterns/analysis/` to `investigator.domain.services.analysis/`
- Updated 7 imports across the codebase
- Removed misleading TODO comments

**2. ✅ Fundamental Agent Import Cleanup (COMPLETED)**
- Removed outdated TODO comments (imports already in correct locations)
- Verified `data_normalizer` correctly in `domain.services`
- Verified `ticker_mapper` correctly in `infrastructure.database`

**3. ✅ Market Context Agent Import Cleanup (COMPLETED)**
- Removed outdated TODO comment (FRED import already in `infrastructure.external`)

**4. ✅ CLI Orchestrator API Note (DEFERRED)**
- Updated TODO to note API server is intentionally disabled during refactoring
- Added guidance for re-enabling API server in future

---

## Medium Priority: Feature Implementation

### 5. Parallel Valuation Orchestrator Implementation (src/investigator/domain/services/parallel_valuation_orchestrator.py)

```python
# Line 156: TODO: Implement confidence scoring
confidence=1.0,

# Line 194: TODO: Call existing DCFValuation with terminal_growth_rate parameter
# TODO: Call existing DCFValuation with fading growth logic
# TODO: Call existing P/E calculator
# TODO: Implement EV/EBITDA calculation
# TODO: Implement P/S ratio calculation
# TODO: Implement PEG ratio calculation
# TODO: Call existing GordonGrowthModel with terminal_growth_rate
```

**Action**: Complete implementation of parallel valuation orchestrator.

**Impact**: Medium - Architecture foundation is ready, needs integration.

---

### 6. SEC Filing Data Extraction (src/investigator/application/synthesizer.py)

```python
# Line 592: TODO: In the future, extract this from actual SEC filing data
```

**Action**: Extract company profile data from SEC filings instead of hardcoded values.

**Impact**: Medium - Would improve data quality.

---

### 7. Email Notification Implementation (src/investigator/application/synthesizer.py)

```python
# Line 617: TODO: Implement email sending
```

**Action**: Implement email notification functionality.

**Impact**: Low - Nice-to-have feature.

---

### 8. Market Context Building (src/investigator/domain/agents/fundamental/agent.py)

```python
# Line 556: TODO: Build MarketContext from technical/market context agent results
```

**Action**: Build market context from agent results.

**Impact**: Medium - Would improve fundamental analysis context.

---

## Low Priority: Future Enhancements

### 9. Bank Valuation ROE Calculation (src/investigator/domain/services/valuation/bank_valuation.py)

```python
# Line 72: TODO: Implement TTM ROE calculation from database
```

**Action**: Implement TTM ROE calculation for bank valuation models.

**Impact**: Low - Enhancement to bank-specific valuation.

---

## Next Steps

1. **Phase 1 (Immediate)**: Complete import migration for clean architecture (TODOs 1-3)
2. **Phase 2 (Short-term)**: Integrate parallel valuation orchestrator (TODO 5)
3. **Phase 3 (Medium-term)**: Implement data extraction improvements (TODOs 6, 8)
4. **Phase 4 (Long-term)**: Add notification and enhancement features (TODOs 7, 9)

---

## Tracking

- **Created**: 2025-02-09
- **Last Reviewed**: 2025-02-09
- **Total TODOs**: 19
- **Completed**: 0
- **In Progress**: 0
- **Pending**: 19
