# XBRL Tag Alias Mapping System

## Overview

The XBRL Tag Alias Mapping system provides **unified tag resolution** across different data sources to ensure robust financial data extraction regardless of company-specific XBRL tag variations.

## Problem Statement

Different companies use different XBRL tags for the same financial concepts:

| Company | Revenue Tag Used |
|---------|------------------|
| Apple, Amazon, Meta, Alphabet | `RevenueFromContractWithCustomerExcludingAssessedTax` |
| Netflix | `Revenues` |

Without a unified mapping system, extraction code would need company-specific logic, leading to:
- **Fragile code**: Breaks when encountering new tag variations
- **Incomplete data**: Missing values when expected tags aren't present
- **Maintenance burden**: Every new company requires code updates

## Solution

The `XBRLTagAliasMapper` module (`utils/xbrl_tag_aliases.py`) provides:

1. **Canonical naming**: All metrics use snake_case names (`total_revenue`, `net_income`)
2. **Tag aliases**: Maps canonical names to all known XBRL tag variations
3. **Priority fallbacks**: Tries multiple tag variations in priority order
4. **Bidirectional mapping**: Convert XBRL tags → canonical names and vice versa

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                             │
├─────────────────────────────────────────────────────────────┤
│  • SEC CompanyFacts API (camelCase XBRL tags)              │
│  • SEC Bulk DERA Tables (same XBRL tags)                    │
│  • Processed Tables (snake_case canonical names)            │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│             XBRLTagAliasMapper                              │
├─────────────────────────────────────────────────────────────┤
│  • TAG_ALIASES: canonical → [XBRL tags]                     │
│  • REVERSE_MAP: XBRL tag → canonical (auto-built)           │
│                                                              │
│  Methods:                                                    │
│  • get_xbrl_aliases(canonical) → [tags]                    │
│  • resolve_to_canonical(xbrl_tag) → canonical              │
│  • extract_value_with_fallbacks(data, canonical)           │
│  • normalize_xbrl_dict(xbrl_data) → canonical_data         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│             Downstream Agents                                │
├─────────────────────────────────────────────────────────────┤
│  • FundamentalAgent: Extract financials with fallbacks      │
│  • SynthesisAgent: Work with canonical names                │
│  • PDF Generator: Consistent field naming                   │
└─────────────────────────────────────────────────────────────┘
```

## Usage Examples

### Basic Usage

```python
from utils.xbrl_tag_aliases import get_tag_mapper

mapper = get_tag_mapper()

# Get all possible XBRL tags for a metric
revenue_tags = mapper.get_xbrl_aliases('total_revenue')
# Returns: ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet']

# Resolve XBRL tag to canonical name
canonical = mapper.resolve_to_canonical('Revenues')
# Returns: 'total_revenue'
```

### Extracting with Fallbacks

```python
# AAPL data (uses long tag name)
aapl_data = {
    'RevenueFromContractWithCustomerExcludingAssessedTax': 394_328_000_000,
    'NetIncomeLoss': 99_803_000_000,
}

# NFLX data (uses short tag name)
nflx_data = {
    'Revenues': 39_000_966_000,
    'NetIncomeLoss': 8_711_631_000,
}

# Same extraction code works for both!
mapper = get_tag_mapper()

aapl_revenue = mapper.extract_value_with_fallbacks(aapl_data, 'total_revenue')
# Returns: 394_328_000_000 (matches first alias)

nflx_revenue = mapper.extract_value_with_fallbacks(nflx_data, 'total_revenue')
# Returns: 39_000_966_000 (falls back to second alias)
```

### Normalizing XBRL Dictionaries

```python
# Convert XBRL tag dict → canonical snake_case dict
xbrl_data = {
    'Revenues': 39_000_966_000,
    'NetIncomeLoss': 8_711_631_000,
    'Assets': 53_630_374_000,
    'Liabilities': 28_886_807_000,
}

mapper = get_tag_mapper()
normalized = mapper.normalize_xbrl_dict(xbrl_data)

# Result:
# {
#     'total_revenue': 39_000_966_000,
#     'net_income': 8_711_631_000,
#     'total_assets': 53_630_374_000,
#     'total_liabilities': 28_886_807_000,
# }
```

### Convenience Functions

```python
from utils.xbrl_tag_aliases import (
    get_xbrl_aliases,
    resolve_to_canonical,
    extract_with_fallbacks,
    normalize_xbrl_dict,
)

# Quick access without instantiating mapper
aliases = get_xbrl_aliases('total_revenue')
canonical = resolve_to_canonical('Revenues')
value = extract_with_fallbacks(data, 'total_revenue')
normalized = normalize_xbrl_dict(xbrl_data)
```

## Supported Metrics

The system currently maps **50+ canonical metrics** across:

### Income Statement
- `total_revenue`, `cost_of_revenue`, `gross_profit`
- `operating_income`, `net_income`
- `eps_basic`, `eps_diluted`
- `research_and_development`, `selling_general_administrative`

### Balance Sheet
- `total_assets`, `current_assets`, `noncurrent_assets`
- `cash_and_equivalents`, `accounts_receivable`, `inventory`
- `property_plant_equipment`, `goodwill`, `intangible_assets`
- `total_liabilities`, `current_liabilities`, `noncurrent_liabilities`
- `accounts_payable`, `accrued_liabilities`, `deferred_revenue`
- `long_term_debt`, `short_term_debt`, `total_debt`
- `stockholders_equity`, `retained_earnings`, `common_stock`

### Cash Flow Statement
- `operating_cash_flow`, `investing_cash_flow`, `financing_cash_flow`
- `capital_expenditures`, `free_cash_flow`
- `depreciation_amortization`, `stock_based_compensation`
- `dividends_paid`, `share_repurchases`

### Shares Outstanding
- `shares_outstanding`, `shares_issued`
- `weighted_average_shares_basic`, `weighted_average_shares_diluted`

## Tag Coverage Analysis

### FAANG Companies Tested

The system was built by analyzing **actual XBRL tag usage** across FAANG companies:

| Company | CIK | Tags Analyzed | Data Source |
|---------|-----|---------------|-------------|
| Apple (AAPL) | 0000320193 | 2024-FY, 2025-Q1 | SEC Bulk Tables |
| Amazon (AMZN) | 0001018724 | 2024-FY, 2025-Q1 | SEC Bulk Tables |
| Meta (META) | 0001326801 | 2024-FY, 2025-Q1 | SEC Bulk Tables |
| Alphabet (GOOGL) | 0001652044 | 2024-FY, 2025-Q1 | SEC Bulk Tables |
| Netflix (NFLX) | 0001065280 | 2024-FY, 2025-Q1 | SEC Bulk Tables |

### Key Tag Variations Discovered

**Revenue Tags:**
- `RevenueFromContractWithCustomerExcludingAssessedTax` (AAPL, AMZN, META, GOOGL)
- `Revenues` (NFLX)
- `SalesRevenueNet` (Alternative)

**Cost of Revenue Tags:**
- `CostOfRevenue` (GOOGL, META, NFLX)
- `CostOfGoodsAndServicesSold` (AAPL)
- `CostOfSales` (Alternative)

**Capital Expenditures Tags:**
- `PaymentsToAcquirePropertyPlantAndEquipment` (AAPL)
- `PaymentsToAcquireProductiveAssets` (AMZN)
- `CapitalExpendituresIncurredButNotYetPaid` (META, accrual basis)

**Accounts Payable Tags:**
- `AccountsPayableCurrent` (AAPL, AMZN, GOOGL, NFLX)
- `AccountsPayableTradeCurrent` (META)

## Integration with Fundamental Agent

### Before (Fragile Code)

```python
# agents/fundamental_agent.py (OLD - DO NOT USE)
def extract_revenue(self, data: Dict) -> Optional[float]:
    """Extract revenue - breaks on Netflix!"""
    # Only checks one tag - fails for NFLX
    return data.get('RevenueFromContractWithCustomerExcludingAssessedTax')
```

### After (Robust Code)

```python
# agents/fundamental_agent.py (NEW - RECOMMENDED)
from utils.xbrl_tag_aliases import get_tag_mapper

def extract_financials(self, xbrl_data: Dict) -> Dict[str, float]:
    """Extract financial metrics using unified tag mapper."""
    mapper = get_tag_mapper()

    # Automatically tries all known aliases in priority order
    return {
        'total_revenue': mapper.extract_value_with_fallbacks(xbrl_data, 'total_revenue'),
        'net_income': mapper.extract_value_with_fallbacks(xbrl_data, 'net_income'),
        'total_assets': mapper.extract_value_with_fallbacks(xbrl_data, 'total_assets'),
        'current_liabilities': mapper.extract_value_with_fallbacks(xbrl_data, 'current_liabilities'),
        # ... extract all metrics with automatic fallback logic
    }

    # OR use normalize_xbrl_dict for bulk conversion
    # return mapper.normalize_xbrl_dict(xbrl_data)
```

## Testing

Comprehensive test suite covers:
- **Tag resolution**: All canonical names → XBRL tags
- **Reverse mapping**: All XBRL tags → canonical names
- **Fallback logic**: Priority-ordered alias matching
- **Real-world cases**: Actual FAANG company data patterns
- **Edge cases**: Empty data, None values, zero values, case sensitivity

Run tests:
```bash
pytest tests/utils/test_xbrl_tag_aliases.py -v
```

Test coverage:
- **36 test cases**
- **100% pass rate**
- **Covers all FAANG company patterns**

## Performance Characteristics

- **Initialization**: O(n) where n = number of aliases (happens once, ~150 mappings)
- **Lookup**: O(1) for reverse mapping (XBRL tag → canonical)
- **Extraction with fallbacks**: O(k) where k = number of aliases for metric (typically 1-4)
- **Bulk normalization**: O(m * k) where m = number of XBRL tags in data

**Memory Usage**: ~50KB for all mappings (negligible)

## Extending the System

### Adding New Metrics

```python
# utils/xbrl_tag_aliases.py

TAG_ALIASES = {
    # ... existing mappings ...

    'new_metric_name': [
        'PrimaryXBRLTag',  # Most common tag (highest priority)
        'AlternativeXBRLTag',  # Fallback tag
        'LegacyXBRLTag',  # Legacy tag (lowest priority)
    ],
}
```

### Adding New Tag Aliases for Existing Metrics

```python
'total_revenue': [
    'RevenueFromContractWithCustomerExcludingAssessedTax',
    'Revenues',
    'SalesRevenueNet',
    'NewCompanySpecificRevenueTag',  # Add new alias at end
],
```

### Testing New Mappings

```python
# tests/utils/test_xbrl_tag_aliases.py

def test_new_metric():
    """Test new metric mapping."""
    mapper = XBRLTagAliasMapper()

    # Test aliases list
    aliases = mapper.get_xbrl_aliases('new_metric_name')
    assert 'PrimaryXBRLTag' in aliases

    # Test reverse mapping
    canonical = mapper.resolve_to_canonical('PrimaryXBRLTag')
    assert canonical == 'new_metric_name'

    # Test extraction
    data = {'PrimaryXBRLTag': 100_000}
    value = mapper.extract_value_with_fallbacks(data, 'new_metric_name')
    assert value == 100_000
```

## Best Practices

### ✅ DO

1. **Use canonical names everywhere**: All internal code should use `total_revenue`, not `Revenues`
2. **Extract at the boundary**: Convert XBRL tags → canonical names as early as possible (in extractors/DAOs)
3. **Test with real data**: Validate mappings with actual company filings, not synthetic data
4. **Log fallback usage**: Log which alias was used for debugging (`logger.debug` in `extract_value_with_fallbacks`)
5. **Update tests**: Add test cases when adding new metrics or aliases

### ❌ DON'T

1. **Don't hardcode XBRL tags**: Use the mapper instead of direct dictionary access
2. **Don't assume all companies use same tags**: Always use fallback logic
3. **Don't skip None values**: The mapper automatically skips None and tries next alias
4. **Don't mix camelCase and snake_case**: Standardize on snake_case for internal use
5. **Don't forget priority order**: Most common tags should be listed first

## Migration Guide

### For Existing Code

**Step 1: Identify hardcoded XBRL tag access**

```python
# BEFORE (fragile)
revenue = data.get('RevenueFromContractWithCustomerExcludingAssessedTax')
net_income = data.get('NetIncomeLoss')
```

**Step 2: Replace with mapper-based extraction**

```python
# AFTER (robust)
from utils.xbrl_tag_aliases import get_tag_mapper

mapper = get_tag_mapper()
revenue = mapper.extract_value_with_fallbacks(data, 'total_revenue')
net_income = mapper.extract_value_with_fallbacks(data, 'net_income')
```

**Step 3: Update tests to verify fallback behavior**

```python
def test_extraction_handles_netflix_tags():
    """Test that extraction works with NFLX's 'Revenues' tag."""
    nflx_data = {'Revenues': 39_000_966_000}
    mapper = get_tag_mapper()
    revenue = mapper.extract_value_with_fallbacks(nflx_data, 'total_revenue')
    assert revenue == 39_000_966_000
```

## Troubleshooting

### Issue: Extraction returns None unexpectedly

**Cause**: XBRL tag not in alias list

**Solution**:
1. Check actual XBRL tags in data: `print(list(xbrl_data.keys()))`
2. Add missing alias to `TAG_ALIASES` mapping
3. Run tests to verify

### Issue: Wrong canonical name resolved

**Cause**: XBRL tag maps to multiple canonical names (duplicate in `TAG_ALIASES`)

**Solution**:
1. Check logs for warning: `"Duplicate XBRL tag ... maps to both ..."`
2. Remove duplicate from lower-priority metric
3. Priority is determined by reverse map build order (first occurrence wins)

### Issue: Different companies return different values

**Cause**: Companies report same concept with different precision or accounting methods

**Solution**:
- This is expected behavior (not a mapping issue)
- Use data normalization (`utils/data_normalizer.py`) to standardize precision
- Document company-specific quirks in comments

## Future Enhancements

1. **Auto-discovery**: Scan new filings to automatically detect new tag variations
2. **Confidence scoring**: Track which aliases are most commonly used
3. **Company profiles**: Maintain company-specific tag preferences to optimize lookup order
4. **XBRL taxonomy integration**: Map to official XBRL US GAAP taxonomy concepts
5. **Validation rules**: Flag suspicious values based on expected ranges per metric

## References

- **SEC EDGAR**: https://www.sec.gov/edgar
- **XBRL US GAAP Taxonomy**: https://www.xbrl.org/us-gaap-taxonomy
- **SEC Bulk Data**: https://www.sec.gov/dera/data/financial-statement-data-sets
- **Project CLAUDE.md**: `/.claude/CLAUDE.md` (data normalization standards)

## Version History

- **v1.0.0** (2025-11-03): Initial implementation based on FAANG analysis
  - 50+ canonical metrics
  - 150+ XBRL tag aliases
  - Comprehensive test suite (36 tests)
  - Validated against 5 FAANG companies (10 filings)
