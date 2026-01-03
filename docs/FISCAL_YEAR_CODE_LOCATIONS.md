# Fiscal Year Initial Assignment - Code Locations & Snippets

## Location 1: Fiscal Year First Assignment

**File**: `/Users/vijaysingh/code/InvestiGator/src/investigator/infrastructure/sec/data_processor.py`

**Method**: `process_raw_data()` 

**Lines**: 1225-1550 (full method), Key lines 1310, 1344-1361, 1366

### Code Snippet - Where fiscal_year is FIRST Set

```python
# Line 1225 - Method signature
def process_raw_data(
    self,
    symbol: str,
    raw_data: Dict,
    raw_data_id: int,
    extraction_version: str = "1.0.0",
    persist: bool = True,
    current_price: Optional[float] = None,
) -> List[Dict]:
    """
    Extract all quarterly/annual filings from raw us-gaap structure
    ...
    """
    
    # ... Earlier code ...
    
    # Line 1286-1291: Detect fiscal year end for all non-calendar year companies
    fiscal_year_end = self._detect_fiscal_year_end(raw_data, symbol)
    if fiscal_year_end:
        logger.info(f"[Fiscal Year End] {symbol}: Detected fiscal year end: {fiscal_year_end}")
    else:
        logger.warning(f"[Fiscal Year End] {symbol}: Could not detect fiscal year end, Q1 fiscal year may be incorrect")
    
    # Line 1293-1296: PHASE 2 begins
    # IMPORTANT: Fiscal year/period are DERIVED from period_end_date and duration
    # NOT from unreliable fy/fp fields (which indicate filing document, not reporting period)
    for entry in best_entries:
        adsh = entry['accn']
        
        # Derive actual fiscal year from period_end (not from fy field!)
        # Line 1300: Extract period_end_str from entry
        period_end_str = entry['end']
        if not period_end_str:
            continue
        
        try:
            # Line 1305: Parse to datetime
            period_end_date = datetime.strptime(period_end_str, '%Y-%m-%d')
        except ValueError:
            logger.warning(f"Invalid period_end_date format: {period_end_str}")
            continue
        
        # ========== LINE 1310: INITIAL FISCAL YEAR ASSIGNMENT ==========
        actual_fiscal_year = period_end_date.year
        # BUG: This assumes calendar year fiscal years!
        # For ORCL (FY ends May 31), Nov 30 = FY2025 but period_end_date.year = 2024
        # ====================================================================
        
        # Derive fiscal period using fp field
        duration = entry.get('duration_days', 999)
        raw_fp = entry.get('fp', '')
        
        # Use fp from entry if available and valid
        if raw_fp == 'FY' or duration >= 330:
            actual_fp = 'FY'
        elif raw_fp in ['Q1', 'Q2', 'Q3', 'Q4']:
            actual_fp = raw_fp
        else:
            # Fallback: derive quarter from end month
            month = period_end_date.month
            if month <= 3:
                actual_fp = 'Q1'
            elif month <= 6:
                actual_fp = 'Q2'
            elif month <= 9:
                actual_fp = 'Q3'
            else:
                actual_fp = 'Q4'
        
        # ========== LINES 1338-1361: Q1-ONLY FISCAL YEAR ADJUSTMENT ==========
        # CRITICAL FIX: Adjust fiscal_year for Q1 periods in non-calendar fiscal years
        # Q1 can cross calendar year boundary. If period_end is after fiscal_year_end,
        # Q1 belongs to the NEXT fiscal year.
        # Example: ZS fiscal year ends July 31
        #   - Q1 ending Oct 31, 2023 is part of FY2024 (Aug 1, 2023 - Jul 31, 2024)
        #   - period_end_date.year = 2023, but fiscal_year should be 2024
        
        # NOTE: THIS ONLY APPLIES TO Q1! 
        # Q2-Q4 PERIODS ARE NOT ADJUSTED, CAUSING THE BUG!
        if actual_fp == 'Q1' and fiscal_year_end:
            try:
                # Extract month and day from fiscal_year_end (format: '-MM-DD')
                fy_end_month, fy_end_day = map(int, fiscal_year_end[1:].split('-'))
                
                # Check if period_end is after fiscal_year_end
                # If so, Q1 belongs to the next fiscal year
                if (period_end_date.month > fy_end_month) or \
                   (period_end_date.month == fy_end_month and period_end_date.day > fy_end_day):
                    original_fy = actual_fiscal_year
                    actual_fiscal_year += 1
                    logger.debug(
                        f"[Q1 Fiscal Year Adjustment] {symbol} Q1 ending {period_end_str}: "
                        f"Adjusted fiscal_year from {original_fy} to {actual_fiscal_year} "
                        f"(fiscal year ends {fiscal_year_end})"
                    )
            except Exception as e:
                logger.warning(f"[Q1 Fiscal Year Adjustment] {symbol}: Failed to adjust Q1 fiscal year: {e}")
        
        # ========== LINE 1363-1378: FILING DICTIONARY CREATED ==========
        # This is where the fiscal_year value persists through the pipeline
        filings[adsh] = {
            'symbol': symbol.upper(),
            'cik': cik,
            'fiscal_year': actual_fiscal_year,  # ✅ LINE 1366: FISCAL_YEAR ASSIGNED
            'fiscal_period': actual_fp,         # ✅ LINE 1367: FISCAL_PERIOD ASSIGNED
            'adsh': adsh,
            'form_type': entry['form'],
            'filed_date': entry['filed'],
            'period_end_date': period_end_str,
            'period_start_date': entry['start'],
            'frame': entry['frame'],
            'duration_days': duration,
            'data': {},
            'raw_data_id': raw_data_id,
            'extraction_version': extraction_version
        }
```

---

## Location 2: Where fiscal_year is Used to Extract Data

**File**: `/Users/vijaysingh/code/InvestiGator/src/investigator/infrastructure/sec/data_processor.py`

**Method**: Same `process_raw_data()`, Lines 1424-1451

```python
    # Line 1424: PHASE 2: Extract all canonical keys using CanonicalKeyMapper
    # Use sector-aware extraction with automatic fallback chains
    extracted_fields = set()
    
    for period_key, filing in filings.items():
        adsh = filing['adsh']  # Extract adsh from filing dict
        for canonical_key in self.CANONICAL_KEYS_TO_EXTRACT:
            # ... other code ...
            
            # Line 1440-1446: Extract using the fiscal_year we set earlier
            value, source_tag = self._extract_from_json_for_filing(
                canonical_key,
                us_gaap,
                adsh,
                fiscal_year=filing['fiscal_year'],  # ✅ USED HERE (Line 1444)
                fiscal_period=filing['fiscal_period'],
                period_end=filing.get('period_end_date') or filing.get('period_end'),
            )
```

---

## Location 3: Where fiscal_year is Written to Database

**File**: `/Users/vijaysingh/code/InvestiGator/src/investigator/infrastructure/sec/data_processor.py`

**Method**: Same `process_raw_data()` or `_persist_processed_filings()`, Lines ~1500-1550

```python
    # Later in process_raw_data (after Phase 3: PHASE 3: Calculate derived metrics)
    # Line ~1534: Persist to database
    if persist:
        self._persist_processed_filings(processed_filings, symbol, raw_data_id, extraction_version)
```

The `_persist_processed_filings()` method writes each filing to `sec_companyfacts_processed` table:

```python
def _persist_processed_filings(self, ...):
    """Persist processed filings to sec_companyfacts_processed table"""
    # Line ~1544 would insert:
    # INSERT INTO sec_companyfacts_processed (
    #     symbol, cik, fiscal_year, fiscal_period, adsh, form_type, ...
    # ) VALUES (
    #     'ORCL', '0001633917', 2024, 'Q2', '...', '10-Q', ...  # fiscal_year=2024 (WRONG!)
    # )
```

---

## Location 4: Fiscal Year End Detection Service

**File**: `/Users/vijaysingh/code/InvestiGator/src/investigator/domain/services/fiscal_period_service.py`

**Method**: `detect_fiscal_year_end()`, Lines 230-301

```python
def detect_fiscal_year_end(self, company_facts: Dict[str, Any]) -> str:
    """
    Detect fiscal year end from CompanyFacts API data.
    
    Returns:
        Fiscal year end suffix (e.g., "-12-31", "-06-30", "-09-30")
    
    Algorithm:
        1. Find FY (10-K) filings in company facts
        2. Extract period_end dates
        3. Determine most common month-day suffix
    """
    if not company_facts or 'facts' not in company_facts:
        raise ValueError("Invalid company facts data: missing 'facts' key")
    
    # Collect all FY period end dates
    fy_period_ends = []
    
    # Iterate through all taxonomies and concepts
    facts = company_facts.get('facts', {})
    for taxonomy in ['us-gaap', 'dei', 'ifrs-full']:
        if taxonomy not in facts:
            continue
        
        for concept, concept_data in facts[taxonomy].items():
            if 'units' not in concept_data:
                continue
            
            for unit_type, unit_data in concept_data['units'].items():
                for entry in unit_data:
                    # Look for fiscal year entries (form 10-K)
                    if entry.get('form') == '10-K' and entry.get('fy'):
                        period_end = entry.get('end')
                        if period_end:
                            fy_period_ends.append(period_end)
    
    if not fy_period_ends:
        raise ValueError("No fiscal year (10-K) data found in company facts")
    
    # Extract month-day suffix from period end dates
    # Format: YYYY-MM-DD → extract -MM-DD
    suffixes = {}
    for period_end in fy_period_ends:
        # Extract last 6 characters: -MM-DD
        if len(period_end) >= 10:
            suffix = period_end[-6:]  # "-MM-DD"
            suffixes[suffix] = suffixes.get(suffix, 0) + 1
    
    if not suffixes:
        raise ValueError("Could not extract fiscal year end from period dates")
    
    # Return most common suffix
    fiscal_year_end = max(suffixes, key=suffixes.get)
    
    self.logger.info(
        f"Detected fiscal year end: {fiscal_year_end} "
        f"(from {len(fy_period_ends)} FY filings)"
    )
    
    return fiscal_year_end  # e.g., "-05-31" for ORCL
```

---

## Location 5: Where fiscal_year_end is Retrieved

**File**: `/Users/vijaysingh/code/InvestiGator/src/investigator/infrastructure/sec/data_processor.py`

**Method**: `_detect_fiscal_year_end()`, Lines 204-224

```python
def _detect_fiscal_year_end(self, company_facts_data: Dict, symbol: str) -> Optional[str]:
    """
    Detect company's fiscal year end from FY periods only.
    
    UPDATED: Now delegates to FiscalPeriodService for centralized fiscal period handling.
    
    Args:
        company_facts_data: Raw CompanyFacts JSON
        symbol: Company symbol (for logging)
    
    Returns:
        Fiscal year end in '-MM-DD' format (e.g., '-12-31')
        None if cannot determine
    """
    try:
        # Use centralized FiscalPeriodService for fiscal year end detection
        fiscal_period_service = get_fiscal_period_service()
        return fiscal_period_service.detect_fiscal_year_end(company_facts_data)
    except Exception as e:
        logger.error(f"[Fiscal Year End] {symbol}: Error detecting fiscal year end: {e}")
        return None
```

---

## Summary of Data Flow

```
1. Raw CompanyFacts Entry
   └─ entry['end'] = "2024-11-30"
   └─ entry['fy'] = 2024 (ignored - unreliable)
   └─ entry['fp'] = "Q2"

2. process_raw_data() - Line 1287
   └─ Calls _detect_fiscal_year_end()
   └─ Returns: fiscal_year_end = "-05-31"

3. process_raw_data() - Line 1310
   └─ actual_fiscal_year = period_end_date.year = 2024

4. process_raw_data() - Line 1344
   └─ if actual_fp == 'Q1' and fiscal_year_end: [Q2 SKIPS THIS!]
   └─ For ORCL Q2: condition FALSE, no adjustment

5. process_raw_data() - Line 1366
   └─ filings[adsh]['fiscal_year'] = 2024 (INCORRECT)

6. process_raw_data() - Line 1444
   └─ _extract_from_json_for_filing(..., fiscal_year=2024, ...)

7. Database
   └─ INSERT INTO sec_companyfacts_processed (...fiscal_year=2024...)
   └─ RESULT: ORCL Q2 stored with fiscal_year=2024 (WRONG!)
```

