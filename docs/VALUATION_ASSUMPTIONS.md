# Valuation Framework Assumptions & Methodology

This document provides transparency on all key assumptions in the valuation framework.
For open-source credibility, we explicitly document what we assume, why, and known limitations.

## Core Philosophy

1. **Multi-Model Approach**: No single valuation method is reliable. We blend 6+ methods weighted by applicability.
2. **Sector-Specific Logic**: Different business models require different valuation approaches.
3. **Conservative Bias**: When uncertain, we err toward conservative valuations.
4. **Transparency Over Precision**: We prefer honest uncertainty ranges over false precision.

---

## 1. Growth Profile Assumptions

### 1.1 Growth Classification Thresholds

| Profile | Revenue Growth | Rationale |
|---------|----------------|-----------|
| Hyper-Growth | >50% | Unsustainable long-term, requires heavy discounting |
| High-Growth | 25-50% | Strong but decelerating typical within 3-5 years |
| Moderate-Growth | 10-25% | Above GDP, sustainable for quality companies |
| Low-Growth | 0-10% | Near GDP, typical for mature companies |
| Stable | ~0% | Zero growth but consistent earnings |
| Declining | <0% | Structural decline, value trap risk |

**Limitation**: Growth classification uses trailing data. Forward growth estimates would be more predictive but less reliable.

### 1.2 Sustainability Discounts

| Profile | Discount | Rationale |
|---------|----------|-----------|
| Hyper-Growth | 15% | >50% growth rarely sustains beyond 2-3 years |
| High-Growth | 10% | 25-50% typically decelerates to 15-25% |
| Moderate-Growth | 5% | Above-market growth faces competition |
| Low-Growth | 0% | Already normalized to market expectations |

**Source**: Based on historical analysis of growth deceleration patterns (McKinsey Growth Studies).

---

## 2. P/E Multiple Assumptions

### 2.1 Base P/E by Growth Profile

| Profile | Base P/E | Rationale |
|---------|----------|-----------|
| Hyper-Growth | 35x | Premium for high growth, capped for sustainability |
| High-Growth | 30x | Above market, reflecting growth trajectory |
| Moderate-Growth | 25x | Near S&P 500 historical average |
| Low-Growth | 20x | Below market, limited growth premium |
| Stable | 18x | Dividend-yield focused, lower multiple |
| Declining | 12x | Value trap risk, asset-based floor |

**Assumption**: S&P 500 trades at ~20-25x P/E in normal environments. Premiums/discounts relative to this baseline.

### 2.2 PEG Targets

| Profile | PEG Target | Rationale |
|---------|------------|-----------|
| Hyper-Growth | 0.8 | Discount for unsustainability |
| High-Growth | 0.9 | Slight discount for deceleration risk |
| Moderate-Growth | 1.0 | Peter Lynch's "fairly valued" benchmark |
| Low-Growth | 1.2 | Premium for stability and dividends |

**Source**: PEG = 1.0 is the classic Lynch rule; we adjust for sustainability.

### 2.3 P/E Caps by Sector

| Sector | Max P/E | Rationale |
|--------|---------|-----------|
| Technology | 100x | Allows for exceptional growth stories (NVDA, etc.) |
| Healthcare | 60x | Pipeline optionality for pharma |
| Consumer Cyclical | 60x | Premium brands can command high multiples |
| Communication Services | 50x | Mix of growth (streaming) and mature (telecom) |
| Industrials | 40x | Cyclical, asset-intensive businesses |
| Consumer Defensive | 35x | Stable earnings, dividend focus |
| Real Estate | 35x | FFO-based, NAV anchored |
| Materials | 30x | Commodity cyclical |
| Financials | 25x | ROE-driven, leverage constraints |
| Energy | 25x | Commodity cyclical, reserve depletion |
| Utilities | 25x | Regulated returns, low growth ceiling |

**Limitation**: Caps may be too restrictive for exceptional companies within sectors.

---

## 3. Sector-Specific Assumptions

### 3.1 Banks (P/B Methodology)

| Metric | Target | Rationale |
|--------|--------|-----------|
| ROE >= 15% | P/B 1.75x | Exceptional profitability justifies premium |
| ROE 10-15% | P/B 1.25x | Good bank, above book value |
| ROE < 10% | P/B 0.90x | Challenged, may trade below book |

**Key Assumption**: Banks' intrinsic value is driven by ROE sustainability. P/B reflects ability to generate returns on equity.

**Limitation**: No adjustment for:
- Deposit mix (core vs. brokered)
- Interest rate sensitivity (duration mismatch)
- Loan portfolio composition (CRE concentration)

### 3.2 Insurance (Combined Ratio Methodology)

| Quality | Combined Ratio | P/BV Multiple |
|---------|----------------|---------------|
| Excellent | <90% | 1.3-1.5x |
| Good | 90-95% | 1.1-1.3x |
| Acceptable | 95-100% | 0.9-1.1x |
| Weak | 100-105% | 0.7-0.9x |
| Poor | >105% | <0.7x |

**Key Assumption**: Combined ratio is primary quality signal. Underwriting discipline drives long-term value.

**Limitation**: No adjustment for:
- Reserve adequacy (frequency vs. severity)
- Investment portfolio quality
- Reinsurance program effectiveness

### 3.3 REITs (FFO Methodology)

| Property Type | FFO Multiple Range | Rationale |
|---------------|-------------------|-----------|
| Data Centers | 20-24x | Cloud growth, power-constrained supply |
| Industrial/Logistics | 22-25x | E-commerce tailwinds |
| Cell Towers | 22-26x | Recurring 5G revenue |
| Self-Storage | 18-21x | Recession-resistant |
| Sunbelt Apartments | 18-20x | Population migration |
| Healthcare | 14-16x | Aging demographics, reimbursement risk |
| Office (Class A) | 12-14x | Post-COVID flight to quality |
| Office (Class B) | 8-10x | Structural headwinds |
| Regional Malls | 6-10x | E-commerce disruption |

**Key Assumption**: FFO multiples should vary by property type risk/return profile.

**Limitation**: No adjustment for:
- Occupancy rates
- Lease expiration schedules
- Development pipeline risk
- Interest rate sensitivity

### 3.4 Biotech Pre-Revenue (Pipeline Methodology)

| Phase | Approval Probability | Source |
|-------|---------------------|--------|
| Preclinical | 5% | FDA/BIO historical data |
| Phase 1 | 10% | Safety trials, high attrition |
| Phase 2 | 15% | Efficacy trials, highest drop-off |
| Phase 3 | 50% | Pivotal trials |
| Filed NDA | 85% | Regulatory review |
| Approved | 100% | Already on market |

**Key Assumptions**:
- 30% discount to TAM estimates (conservative market sizing)
- 15% discount rate for biotech risk
- Cash runway > 18 months = adequate

**Limitation**:
- Comparable deals component (15% weight) NOT IMPLEMENTED
- No therapeutic area adjustment (oncology vs. rare disease have different probabilities)
- No patent cliff consideration for approved drugs

### 3.5 Defense Contractors (Backlog Methodology)

| Backlog Ratio | Premium/Discount | Rationale |
|---------------|------------------|-----------|
| >= 3x Revenue | +10% | Exceptional visibility |
| >= 2x Revenue | +5% | Strong visibility |
| >= 1x Revenue | 0% | Normal |
| < 1x Revenue | -5% | Weak visibility |

**Key Assumptions**:
- Base EV/EBITDA: 12x (mid-range for defense)
- Terminal growth: 2.5% (government spending grows slowly)
- Discount rate: -0.5% adjustment (government customer creditworthiness)

**Limitation**: No distinction between:
- Funded vs. unfunded backlog
- Cost-plus vs. fixed-price contract mix
- Single-customer concentration risk

### 3.6 Semiconductors (Cycle-Adjusted Methodology)

| Cycle Position | Margin Assumption | Valuation Adjustment |
|----------------|-------------------|---------------------|
| Peak | 35% operating margin | -20% discount |
| Peak→Normal | Normalizing | -10% discount |
| Normal | 25% operating margin | 0% |
| Normal→Trough | Declining | +5% premium |
| Trough | 15% operating margin | +15% premium |

**Cycle Detection Signals**:
- Inventory days > 15% of sales = likely peak
- Inventory days < 8% of sales = likely trough
- Book-to-bill > 1.15 = expansion
- Book-to-bill < 0.85 = contraction

**Key Assumptions**:
- Terminal growth: 4% (secular AI/digitization tailwinds)
- Discount rate: +1% for cyclicality risk

**Limitation**:
- Terminal growth may not apply equally to memory (MU) vs. logic (NVDA)
- Chip type (logic vs. memory) cycles can diverge
- No technology node transition risk adjustment

---

## 4. DCF Assumptions

### 4.1 Standard DCF Parameters

| Parameter | Default Value | Rationale |
|-----------|---------------|-----------|
| Risk-Free Rate | 10Y Treasury | Current market benchmark |
| Equity Risk Premium | 5.5% | Historical average |
| Terminal Growth | 3.0% | GDP + inflation approximation |
| Projection Period | 5 years | Balance of visibility vs. uncertainty |

### 4.2 Sector-Specific Terminal Growth

| Sector | Terminal Growth | Rationale |
|--------|-----------------|-----------|
| Technology | 4.0% | Secular digitization tailwinds |
| Semiconductors | 4.0% | AI/compute demand growth |
| Healthcare | 4.0% | Aging demographics |
| Defense | 2.5% | Government spending constraints |
| Utilities | 2.0% | Regulated, low growth |
| Energy | 2.0% | Commodity with reserve depletion |
| General | 3.0% | GDP proxy |

**Limitation**: Terminal value typically represents 60-80% of DCF value. Small changes in terminal growth significantly impact fair value.

---

## 5. Data Quality Adjustments

### 5.1 Quality Tiers

| Quality | Min Quarters | Max Missing | Confidence Penalty |
|---------|--------------|-------------|-------------------|
| Excellent | 12 | 0 | 0% |
| Good | 8 | 1 | 0% |
| Fair | 4 | 2 | -10% |
| Poor | 2 | 4 | -25% |
| Insufficient | <2 | >4 | Exclude model |

### 5.2 Model Applicability Rules

| Model | Required | Condition |
|-------|----------|-----------|
| PE | Net Income > 0 | TTM positive earnings |
| EV/EBITDA | EBITDA > 0 | TTM positive EBITDA |
| DCF | FCF available | At least 4 quarters |
| GGM | Dividends + NI > 0 | Payout ratio >= 40% |
| PS | Revenue > 0 | Any positive revenue |
| PB | Book Value > 0 | Positive equity |

---

## 6. Known Limitations

### 6.1 What We DON'T Do

1. **Earnings Quality Analysis**: We use reported GAAP/non-GAAP financials without adjusting for:
   - Non-recurring items
   - Accounting policy changes
   - Revenue recognition timing

2. **Macro Adjustments**: We don't adjust for:
   - Interest rate environment
   - Credit cycle position
   - Currency effects

3. **Competitive Moat Scoring**: We don't quantify:
   - Switching costs
   - Network effects
   - Intangible assets (brands, patents)

4. **Management Quality**: We don't factor:
   - Capital allocation track record
   - Insider ownership
   - Corporate governance

### 6.2 Unimplemented Features

| Feature | Weight Listed | Status | Notes |
|---------|---------------|--------|-------|
| Biotech Comparable Deals | 15% | NOT IMPLEMENTED | Requires M&A database |
| Earnings Quality Filter | - | NOT IMPLEMENTED | Would flag non-recurring items |
| Confidence Intervals | - | NOT IMPLEMENTED | Point estimates only |

---

## 7. Validation & Backtesting

### 7.1 Current Validation

- **Unit Tests**: Tier classification logic tested
- **Integration Tests**: End-to-end valuation for sample stocks
- **Manual Review**: Spot-check against analyst reports

### 7.2 Future Validation Needed

- [ ] Historical backtest against actual prices
- [ ] Comparison to analyst consensus
- [ ] Out-of-sample accuracy measurement
- [ ] Sector-specific error analysis

---

## 8. Changelog

| Date | Change | Rationale |
|------|--------|-----------|
| 2025-12-30 | Added sustainability discounts for hyper-growth | Prevent over-valuation of unsustainable growth |
| 2025-12-30 | Added EPS anomaly detection | Handle depressed trailing earnings |
| 2025-12-30 | Added market premium analysis | Show what market pays beyond fundamentals |
| 2025-12-30 | Created this documentation | Open-source credibility |

---

## 9. Contribution Guidelines

When modifying valuation assumptions:

1. **Document the change** in this file
2. **Cite sources** for any thresholds or multiples
3. **Add tests** for edge cases
4. **Consider unintended consequences** on other tiers/sectors
5. **Update the changelog** with rationale

---

*This framework provides fundamental fair value estimates. Market prices may differ significantly due to sentiment, optionality, or factors not captured in financial data. Use as one input among many in investment decisions.*
