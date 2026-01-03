# Valuation Configuration Granularity Analysis
**Date**: 2025-11-13
**Purpose**: Ensure realistic, granular growth assumptions to avoid costly valuation errors

---

## Executive Summary

**Problem**: Current config uses broad sector-level growth assumptions. Real markets have significant variance within sectors based on:
1. Company size (mega-cap vs mid-cap vs small-cap)
2. Industry sub-segments (Cloud SaaS vs Semiconductors vs Hardware)
3. Business model (Platform vs Product vs Services)
4. Market maturity (Early-stage vs growth vs mature)

**Impact**: Without granularity, we risk:
- ❌ **Overvaluing mature mega-caps** (e.g., META at $722 when realistic is ~$450)
- ❌ **Undervaluing high-growth mid-caps** (e.g., NVDA, exceptional AI winners)
- ❌ **Missing sector nuances** (FinTech grows 25%, traditional banks 5%)

---

## Current State Analysis

### Technology Sector Breakdown (Most Critical)

**Current Config**: Single `Technology` bucket with `industry_median_growth: 0.08`

**Real Market Segments** (require different assumptions):

#### 1. **Semiconductors** (Cyclical, Capital-Intensive)
**Examples**: NVDA, AMD, INTC, TSM, AVGO
**Market Cap Range**: $50B - $3T
**Typical Growth**:
- Peak cycle (AI boom, 2023-2024): 30-60% revenue growth
- Mature cycle: 8-12% revenue growth
- Downcycle: -20% to -30% revenue decline

**Recommended Config**:
```yaml
Technology_Semiconductors:
  max_growth: 0.40          # AI winners like NVDA can sustain 40% in boom
  industry_median_growth: 0.12  # Mature semiconductor median (Intel, QCOM)
  terminal_growth: 0.030
  size_adjustments:
    mega_cap: {ceiling: 0.15, fade_to: 0.05}  # NVDA, TSM (>$500B)
    large_cap: {ceiling: 0.25, fade_to: 0.08}  # AMD, AVGO ($50B-$500B)
    mid_cap: {ceiling: 0.35, fade_to: 0.12}    # Emerging semis ($10B-$50B)
```

#### 2. **Cloud/SaaS** (High Growth, Low CapEx)
**Examples**: CRM, NOW, SNOW, DDOG, NET, CRWD
**Market Cap Range**: $5B - $300B
**Typical Growth**:
- Hyper-growth (ARR <$500M): 50-100% revenue growth
- Growth (ARR $500M-$5B): 30-50% revenue growth
- Scale (ARR >$5B): 20-30% revenue growth

**Recommended Config**:
```yaml
Technology_CloudSaaS:
  max_growth: 0.50          # Early hypergrowth SaaS
  industry_median_growth: 0.25  # Scaled SaaS at $5B+ ARR
  terminal_growth: 0.035
  size_adjustments:
    mega_cap: {ceiling: 0.20, fade_to: 0.08}  # CRM, NOW (>$50B)
    large_cap: {ceiling: 0.35, fade_to: 0.15}  # DDOG, CRWD ($20B-$50B)
    mid_cap: {ceiling: 0.45, fade_to: 0.20}    # Emerging SaaS ($5B-$20B)
```

#### 3. **Internet Platforms** (Network Effects, Mature)
**Examples**: META, GOOGL, AMZN (ads/retail), NFLX
**Market Cap Range**: $200B - $2T
**Typical Growth**:
- Mature platforms: 5-15% revenue growth
- Reinvestment phase (AI, new products): 15-25% growth
- Declining: 0-5% growth

**Recommended Config**:
```yaml
Technology_InternetPlatforms:
  max_growth: 0.15          # Even mega-caps can't sustain >15% at $1T+
  industry_median_growth: 0.08  # Realistic for mature platforms
  terminal_growth: 0.030
  size_adjustments:
    mega_cap: {ceiling: 0.10, fade_to: 0.04}  # META, GOOGL (>$500B)
    large_cap: {ceiling: 0.15, fade_to: 0.08}  # NFLX, Uber ($50B-$500B)
```

#### 4. **Enterprise Software** (Traditional, High Margins)
**Examples**: MSFT, ORCL, SAP, ADBE, INTU
**Market Cap Range**: $50B - $3T
**Typical Growth**:
- Cloud transition: 15-25% revenue growth
- Mature enterprise: 8-12% revenue growth
- Legacy decline: 0-5% growth

**Recommended Config**:
```yaml
Technology_EnterpriseSoftware:
  max_growth: 0.20          # Cloud migration tailwind
  industry_median_growth: 0.10  # Mature enterprise SW
  terminal_growth: 0.030
  size_adjustments:
    mega_cap: {ceiling: 0.12, fade_to: 0.05}  # MSFT, ORCL (>$500B)
    large_cap: {ceiling: 0.18, fade_to: 0.08}  # ADBE, INTU ($50B-$500B)
```

#### 5. **Hardware/Devices** (Low Margin, Mature)
**Examples**: AAPL (devices), DELL, HPQ
**Market Cap Range**: $20B - $3T
**Typical Growth**:
- Services pivot (Apple): 8-12% revenue growth
- Mature hardware: 3-8% revenue growth
- Declining PC: -5% to 0% growth

**Recommended Config**:
```yaml
Technology_Hardware:
  max_growth: 0.12          # Limited by replacement cycles
  industry_median_growth: 0.06  # Mature hardware market
  terminal_growth: 0.025
  size_adjustments:
    mega_cap: {ceiling: 0.08, fade_to: 0.03}  # AAPL (>$1T, services pivot)
    large_cap: {ceiling: 0.10, fade_to: 0.05}  # DELL ($50B-$500B)
```

---

### Healthcare Sector Breakdown

**Current Config**: Single `Healthcare` bucket with `industry_median_growth: 0.20`

#### 1. **Biotechnology** (High Risk, High Reward)
**Examples**: VRTX, REGN, MRNA, BIIB
**Growth**: Pipeline-dependent, 0-50% based on drug approvals

```yaml
Healthcare_Biotech:
  max_growth: 0.40
  industry_median_growth: 0.15
  terminal_growth: 0.030
```

#### 2. **Pharmaceuticals** (Mature, Patent-Driven)
**Examples**: PFE, MRK, JNJ, LLY
**Growth**: 5-12% steady with patent cliff risks

```yaml
Healthcare_Pharma:
  max_growth: 0.15
  industry_median_growth: 0.08
  terminal_growth: 0.025
```

#### 3. **Medical Devices** (Stable, Regulated)
**Examples**: ISRG, ABT, MDT, BSX
**Growth**: 8-15% from innovation + aging demographics

```yaml
Healthcare_MedicalDevices:
  max_growth: 0.18
  industry_median_growth: 0.12
  terminal_growth: 0.030
```

#### 4. **Health Insurance** (Regulatory, Mature)
**Examples**: UNH, CVS, CI, HUM
**Growth**: 6-10% from membership + pricing

```yaml
Healthcare_Insurance:
  max_growth: 0.12
  industry_median_growth: 0.08
  terminal_growth: 0.025
```

---

### Financials Sector Breakdown

**Current Config**: Single `Financials` bucket with `industry_median_growth: 0.15`

#### 1. **FinTech** (Disruption, High Growth)
**Examples**: V, MA, PYPL, SQ, COIN
**Growth**: 15-30% from digital payments shift

```yaml
Financials_FinTech:
  max_growth: 0.30
  industry_median_growth: 0.18
  terminal_growth: 0.035
```

#### 2. **Traditional Banks** (Mature, Regulated)
**Examples**: JPM, BAC, WFC, C
**Growth**: 3-8% from GDP + credit growth

```yaml
Financials_Banks:
  max_growth: 0.10
  industry_median_growth: 0.06
  terminal_growth: 0.025
```

#### 3. **Asset Managers** (AUM-Driven)
**Examples**: BLK, BX, KKR, AMG
**Growth**: 8-15% from market appreciation + flows

```yaml
Financials_AssetManagers:
  max_growth: 0.18
  industry_median_growth: 0.12
  terminal_growth: 0.030
```

---

## Size-Based Growth Reality Check

### Mega-Caps ($200B+)
**Physics of Scale**: Hard to sustain >10% growth at $500B+ market cap
- To grow 15%: Need to add $75B revenue (= entire Fortune 500 company)
- To grow 25%: Need to add $125B revenue (= top 50 company)

**Realistic Mega-Cap Ceilings**:
- Exceptional (NVDA in AI boom): 15-20% for 3-5 years, fade to 8%
- Strong (META, GOOGL): 10% capped, fade to 4-6%
- Mature (AAPL): 6-8% capped, fade to 3-4%

### Large-Caps ($10B-$200B)
**Sweet Spot for Growth**: Large enough to execute, small enough to grow
- High-growth SaaS: 25-40% sustainable for 5 years
- Traditional enterprise: 12-18% sustainable
- Cyclicals: 8-15% mid-cycle

### Mid-Caps ($2B-$10B)
**Highest Growth Potential**: Proven product-market fit, scaling phase
- Category leaders: 35-50% for 3-5 years
- Competitive markets: 20-30%
- Mature niches: 12-20%

---

## Recommended Implementation Strategy

### Phase 1: Industry Sub-Segmentation (Immediate)
Add granular configs for:
1. Technology → 5 sub-industries (Semiconductors, Cloud/SaaS, Platforms, Enterprise SW, Hardware)
2. Healthcare → 4 sub-industries (Biotech, Pharma, Devices, Insurance)
3. Financials → 3 sub-industries (FinTech, Banks, Asset Managers)

### Phase 2: Size-Based Multipliers (Week 2)
Implement dynamic adjustments in `ValuationFrameworkPlanner.classify_company_stage()`:
```python
def get_growth_params(self, sector, industry, market_cap):
    base = self.get_industry_config(sector, industry)
    size_mult = self.get_size_multiplier(market_cap)

    return {
        'ceiling': base['max_growth'] * size_mult['ceiling_factor'],
        'fade_to': base['industry_median'] * size_mult['fade_factor'],
        'terminal': base['terminal_growth']
    }
```

### Phase 3: Business Model Adjustments (Week 3)
Add modifiers for:
- SaaS vs Product (SaaS gets +20% due to recurring revenue)
- Platform vs Linear (Platform gets +15% due to network effects)
- Asset-light vs Capital-intensive (Light gets +10% due to scalability)

---

## Validation & Backtesting

**Before deploying**, backtest against known valuations:
1. **NVDA**: Should value at $120-140 (actual: $145) ✅ Not $200+
2. **META**: Should value at $400-500 (actual: $609) ✅ Not $722
3. **AAPL**: Should value at $180-200 (actual: $226) ✅ Not $150 or $300
4. **SNOW**: Should value at $140-180 (actual: $165) ✅ Not $100 or $250

**Acceptance Criteria**: Fair value within ±20% of actual price for 80% of test cases

---

## Risk of Getting This Wrong

### Overvaluation Example (Current State)
- META valued at $722 with 30% Year 5 growth assumption
- Actual: $609 (16% overvalued)
- **Loss**: Buying 1000 shares = $113K overpaid

### Undervaluation Example (Too Conservative)
- NVDA capped at 15% when sustaining 40% (AI boom)
- Fair value: $90 vs Actual: $145 (38% undervalued)
- **Missed**: 1000 shares = $55K opportunity cost

**Total Impact on $1M Portfolio**: ±$200K based on valuation accuracy

---

## Action Items

1. [ ] Implement Technology sub-industry mapping (5 categories)
2. [ ] Add Healthcare granularity (4 categories)
3. [ ] Add Financials granularity (3 categories)
4. [ ] Wire size-based multipliers into ValuationFrameworkPlanner
5. [ ] Backtest against 20-stock validation set
6. [ ] Document industry classification rules
7. [ ] Migrate DCF code from config.json → config.yaml

**Priority**: HIGH - directly impacts investment returns
