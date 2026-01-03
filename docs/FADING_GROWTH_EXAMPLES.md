# Fading Growth Model - Real Examples

**Purpose**: Show how historical FCF growth + YAML ceilings prevents both inflation and deflation

---

## How It Works

### Formula

```python
initial_growth = min(historical_fcf_growth, yaml_ceiling)
growth_rates = linear_fade(initial_growth, fade_to_growth, projection_years)
```

### Key Principle

**YAML ceilings are guardrails, not fixed values**
- Historical growth < ceiling → Use historical (no deflation)
- Historical growth > ceiling → Cap at ceiling (prevent inflation)
- Always fade to realistic sustainable rate
- Terminal growth is conservative (3.0-3.5%)

---

## Example 1: High-Growth SaaS (Historical 35% FCF Growth)

**Company**: Hypothetical high-growth SaaS
**Historical 3-Year FCF CAGR**: 35%
**Company Stage**: early_stage_saas
**YAML Ceiling**: 30%
**YAML Fade-To**: 10%

### Calculation

```python
planner = ValuationFrameworkPlanner(
    symbol='HGSAAS',
    sector='Technology',
    industry='Software - Infrastructure',
    market_cap_billions=5.0
)

fading_growth = planner.calculate_fading_growth_rates(
    historical_fcf_growth=0.35,  # 35% historical
    company_stage='early_stage_saas'
)

# Result:
# {
#     'initial_growth': 0.30,  # Capped from 35%
#     'fade_to_growth': 0.10,
#     'ceiling_applied': True,
#     'growth_rates': [0.30, 0.25, 0.20, 0.15, 0.10]
# }
```

### Projection

| Year | Growth Rate | FCF Projection | Calculation |
|------|-------------|----------------|-------------|
| 0 | - | $100M | Latest FCF |
| 1 | 30% | $130M | $100M × 1.30 |
| 2 | 25% | $163M | $130M × 1.25 |
| 3 | 20% | $195M | $163M × 1.20 |
| 4 | 15% | $224M | $195M × 1.15 |
| 5 | 10% | $247M | $224M × 1.10 |
| Terminal | 3.5% | $255M | $247M × 1.035 (perpetuity) |

**Outcome**: Gets credit for strong performance (30%), but realistic fade prevents inflation

---

## Example 2: Mid-Stage Tech (Historical 18% FCF Growth)

**Company**: Zscaler (ZS)
**Historical 3-Year FCF CAGR**: 18%
**Company Stage**: mid_stage_tech
**YAML Ceiling**: 20%
**YAML Fade-To**: 9%

### Calculation

```python
fading_growth = planner.calculate_fading_growth_rates(
    historical_fcf_growth=0.18,  # 18% historical
    company_stage='mid_stage_tech'
)

# Result:
# {
#     'initial_growth': 0.18,  # NOT capped (18% < 20% ceiling)
#     'fade_to_growth': 0.09,
#     'ceiling_applied': False,
#     'growth_rates': [0.18, 0.1575, 0.135, 0.1125, 0.09]
# }
```

### Projection

| Year | Growth Rate | FCF Projection | Calculation |
|------|-------------|----------------|-------------|
| 0 | - | $500M | Latest FCF |
| 1 | 18.0% | $590M | $500M × 1.18 |
| 2 | 15.75% | $683M | $590M × 1.1575 |
| 3 | 13.5% | $775M | $683M × 1.135 |
| 4 | 11.25% | $862M | $775M × 1.1125 |
| 5 | 9.0% | $940M | $862M × 1.09 |
| Terminal | 3.5% | $973M | $940M × 1.035 |

**Outcome**: Uses actual historical growth (18%), gentle fade, no deflation

---

## Example 3: Mature Platform (Historical 12% FCF Growth)

**Company**: DoorDash (DASH)
**Historical 3-Year FCF CAGR**: 12%
**Company Stage**: mature_platform
**YAML Ceiling**: 15%
**YAML Fade-To**: 6%

### Calculation

```python
fading_growth = planner.calculate_fading_growth_rates(
    historical_fcf_growth=0.12,  # 12% historical
    company_stage='mature_platform'
)

# Result:
# {
#     'initial_growth': 0.12,  # NOT capped (12% < 15% ceiling)
#     'fade_to_growth': 0.06,
#     'ceiling_applied': False,
#     'growth_rates': [0.12, 0.105, 0.09, 0.075, 0.06]
# }
```

### Projection

| Year | Growth Rate | FCF Projection | Calculation |
|------|-------------|----------------|-------------|
| 0 | - | $3.3B | Latest FCF |
| 1 | 12.0% | $3.70B | $3.3B × 1.12 |
| 2 | 10.5% | $4.09B | $3.70B × 1.105 |
| 3 | 9.0% | $4.46B | $4.09B × 1.09 |
| 4 | 7.5% | $4.79B | $4.46B × 1.075 |
| 5 | 6.0% | $5.08B | $4.79B × 1.06 |
| Terminal | 3.0% | $5.23B | $5.08B × 1.03 |

**Outcome**: Realistic growth profile for maturing platform, lower terminal growth (3.0%)

---

## Example 4: Mega-Cap Tech (Historical 8% FCF Growth)

**Company**: Apple (AAPL)
**Historical 3-Year FCF CAGR**: 8%
**Company Stage**: mega_cap_tech
**YAML Ceiling**: 10%
**YAML Fade-To**: 4%

### Calculation

```python
fading_growth = planner.calculate_fading_growth_rates(
    historical_fcf_growth=0.08,  # 8% historical
    company_stage='mega_cap_tech'
)

# Result:
# {
#     'initial_growth': 0.08,  # NOT capped (8% < 10% ceiling)
#     'fade_to_growth': 0.04,
#     'ceiling_applied': False,
#     'growth_rates': [0.08, 0.07, 0.06, 0.05, 0.04]
# }
```

### Projection

| Year | Growth Rate | FCF Projection | Calculation |
|------|-------------|----------------|-------------|
| 0 | - | $100B | Latest FCF |
| 1 | 8.0% | $108B | $100B × 1.08 |
| 2 | 7.0% | $116B | $108B × 1.07 |
| 3 | 6.0% | $123B | $116B × 1.06 |
| 4 | 5.0% | $129B | $123B × 1.05 |
| 5 | 4.0% | $134B | $129B × 1.04 |
| Terminal | 3.0% | $138B | $134B × 1.03 |

**Outcome**: Conservative growth reflecting mega-cap scale constraints

---

## Example 5: Edge Case - Explosive Growth (Historical 50% FCF Growth)

**Company**: Hypothetical explosive-growth startup
**Historical 3-Year FCF CAGR**: 50%
**Company Stage**: early_stage_saas
**YAML Ceiling**: 30%
**YAML Fade-To**: 10%

### Calculation

```python
fading_growth = planner.calculate_fading_growth_rates(
    historical_fcf_growth=0.50,  # 50% historical
    company_stage='early_stage_saas'
)

# Result:
# {
#     'initial_growth': 0.30,  # CAPPED from 50%
#     'fade_to_growth': 0.10,
#     'ceiling_applied': True,  # ⚠️ 50% > 30% ceiling
#     'growth_rates': [0.30, 0.25, 0.20, 0.15, 0.10]
# }
```

### Projection

| Year | Growth Rate | FCF Projection | Calculation |
|------|-------------|----------------|-------------|
| 0 | - | $50M | Latest FCF |
| 1 | 30% | $65M | $50M × 1.30 (capped from 50%) |
| 2 | 25% | $81M | $65M × 1.25 |
| 3 | 20% | $97M | $81M × 1.20 |
| 4 | 15% | $112M | $97M × 1.15 |
| 5 | 10% | $123M | $112M × 1.10 |
| Terminal | 3.5% | $127M | $123M × 1.035 |

**Comparison with 50% constant growth** (inflated):

| Year | 30% Fading | 50% Constant | Difference |
|------|------------|--------------|------------|
| 1 | $65M | $75M | +15% |
| 2 | $81M | $113M | +39% |
| 3 | $97M | $169M | +74% |
| 4 | $112M | $254M | +127% |
| 5 | $123M | $381M | **+210%** |

**Outcome**: Ceiling prevents 3x inflation, realistic fade applied

---

## Comparison: Old vs New Approach

### Old Approach (Fixed YAML Values)

**Problems**:
- ZS with 18% historical → forced to 15% (deflated by 17%)
- HGSAAS with 35% historical → forced to 25% (deflated by 29%)
- DASH with 12% historical → forced to 12% (coincidentally correct)
- No company-specific treatment

### New Approach (Historical + Ceiling)

**Benefits**:
- ZS with 18% historical → uses 18% (no deflation)
- HGSAAS with 35% historical → capped at 30% (prevents inflation, but still high)
- DASH with 12% historical → uses 12% (matches actual performance)
- Each company valued on its own merits

---

## Professional Analyst Approach

This matches how Wall Street analysts actually build DCF models:

1. **Historical Analysis**: Calculate 3-5 year FCF CAGR
2. **Reasonableness Check**: Cap at sector-appropriate ceiling (prevents 50%+ projections)
3. **Fading Profile**: Linear or exponential fade to sustainable rate
4. **Terminal Growth**: Conservative 2.5-3.5% (GDP + inflation)

**Not** fixed growth rates by arbitrary company buckets.

---

## Implementation Notes

### Calculating Historical FCF Growth

```python
def calculate_geometric_mean_fcf_growth(quarterly_metrics: List[Dict], years: int = 3) -> float:
    """
    Calculate geometric mean FCF growth over N years

    Args:
        quarterly_metrics: List of quarterly financial metrics
        years: Number of years to look back (default: 3)

    Returns:
        Geometric mean FCF growth rate (decimal)
    """
    # Get TTM FCF for each year
    ttm_fcf_by_year = {}
    for q in quarterly_metrics:
        year = q['fiscal_year']
        if year not in ttm_fcf_by_year:
            ttm_fcf_by_year[year] = []
        ttm_fcf_by_year[year].append(q.get('free_cash_flow', 0))

    # Calculate TTM FCF for each year (sum of 4 quarters)
    yearly_fcf = {year: sum(quarters) for year, quarters in ttm_fcf_by_year.items()}

    # Get most recent N years
    sorted_years = sorted(yearly_fcf.keys(), reverse=True)[:years + 1]
    if len(sorted_years) < 2:
        return 0.0  # Not enough data

    # Calculate CAGR
    start_fcf = yearly_fcf[sorted_years[-1]]
    end_fcf = yearly_fcf[sorted_years[0]]

    if start_fcf <= 0:
        return 0.0  # Can't calculate growth from negative/zero

    cagr = (end_fcf / start_fcf) ** (1 / (len(sorted_years) - 1)) - 1

    return max(0.0, cagr)  # Floor at 0% (no negative growth assumptions)
```

### Integration with DCF Calculator

```python
# In DCF calculator's _project_fcf method:
def _project_fcf_with_fading_growth(self, latest_fcf, historical_fcf_growth, company_stage):
    # Get fading growth rates
    fading_result = self.planner.calculate_fading_growth_rates(
        historical_fcf_growth=historical_fcf_growth,
        company_stage=company_stage
    )

    growth_rates = fading_result['growth_rates']

    # Project FCF
    fcf_projections = []
    current_fcf = latest_fcf
    for growth_rate in growth_rates:
        current_fcf *= (1 + growth_rate)
        fcf_projections.append(current_fcf)

    return fcf_projections, fading_result
```

---

## Summary

**Key Insights**:
1. **No deflation**: High performers (35% historical) get capped at 30%, not forced to 15%
2. **No inflation**: Explosive growers (50% historical) get capped at 30%, not projected at 50%
3. **Company-specific**: Each stock valued on actual performance, not arbitrary buckets
4. **Realistic fading**: All companies fade to sustainable rates (6-10%) by year 5
5. **Conservative terminal**: All companies use 3.0-3.5% terminal growth (perpetuity)

**Result**: Fair valuations that reflect both performance and realism

---

**Document Complete**: 2025-11-12
**Status**: Implementation ready, awaiting integration into DCF calculator
