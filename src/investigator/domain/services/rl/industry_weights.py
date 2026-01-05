"""
Industry-Specific Model Weight Configuration

Provides granular weight profiles at the industry level within each sector.
This enables the RL policy to learn more nuanced model preferences based on
business characteristics unique to each industry.

Structure:
- SECTOR_INDUSTRY_MAP: Maps sectors to their industries
- INDUSTRY_WEIGHT_PROFILES: Default weights per industry
- INDUSTRY_METRICS: Key metrics that matter for each industry
- INDUSTRY_HOLDING_PERIODS: Optimal holding periods by industry
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class IndustryCategory(Enum):
    """Industry categories for granular classification."""

    # Technology
    SOFTWARE_SAAS = "software_saas"
    SOFTWARE_ENTERPRISE = "software_enterprise"
    SEMICONDUCTORS = "semiconductors"
    SEMICONDUCTOR_EQUIPMENT = "semiconductor_equipment"
    HARDWARE_CONSUMER = "hardware_consumer"
    HARDWARE_ENTERPRISE = "hardware_enterprise"
    CLOUD_INFRASTRUCTURE = "cloud_infrastructure"
    CYBERSECURITY = "cybersecurity"
    FINTECH = "fintech"

    # Healthcare
    PHARMA_LARGE = "pharma_large"
    PHARMA_SPECIALTY = "pharma_specialty"
    BIOTECH_COMMERCIAL = "biotech_commercial"
    BIOTECH_CLINICAL = "biotech_clinical"
    MEDICAL_DEVICES = "medical_devices"
    HEALTHCARE_SERVICES = "healthcare_services"
    MANAGED_CARE = "managed_care"
    DIAGNOSTICS = "diagnostics"

    # Financials
    BANK_MONEY_CENTER = "bank_money_center"
    BANK_REGIONAL = "bank_regional"
    BANK_INVESTMENT = "bank_investment"
    INSURANCE_LIFE = "insurance_life"
    INSURANCE_PC = "insurance_pc"
    INSURANCE_REINSURANCE = "insurance_reinsurance"
    ASSET_MANAGEMENT = "asset_management"
    CREDIT_SERVICES = "credit_services"
    EXCHANGES = "exchanges"

    # Consumer Discretionary
    RETAIL_ECOMMERCE = "retail_ecommerce"
    RETAIL_SPECIALTY = "retail_specialty"
    RETAIL_DISCOUNT = "retail_discount"
    AUTO_OEM = "auto_oem"
    AUTO_EV = "auto_ev"
    AUTO_PARTS = "auto_parts"
    RESTAURANTS = "restaurants"
    HOTELS_LEISURE = "hotels_leisure"
    HOMEBUILDERS = "homebuilders"
    LUXURY_GOODS = "luxury_goods"

    # Consumer Staples
    FOOD_BEVERAGE = "food_beverage"
    HOUSEHOLD_PRODUCTS = "household_products"
    TOBACCO = "tobacco"
    RETAIL_GROCERY = "retail_grocery"
    PERSONAL_CARE = "personal_care"

    # Industrials
    AEROSPACE_DEFENSE = "aerospace_defense"
    AIRLINES = "airlines"
    MACHINERY = "machinery"
    INDUSTRIAL_CONGLOMERATES = "industrial_conglomerates"
    CONSTRUCTION = "construction"
    TRANSPORTATION_RAIL = "transportation_rail"
    TRANSPORTATION_TRUCKING = "transportation_trucking"
    LOGISTICS = "logistics"

    # Energy
    OIL_INTEGRATED = "oil_integrated"
    OIL_EXPLORATION = "oil_exploration"
    OIL_REFINING = "oil_refining"
    OIL_SERVICES = "oil_services"
    RENEWABLE_ENERGY = "renewable_energy"
    UTILITIES_REGULATED = "utilities_regulated"
    UTILITIES_INDEPENDENT = "utilities_independent"

    # Materials
    CHEMICALS_SPECIALTY = "chemicals_specialty"
    CHEMICALS_COMMODITY = "chemicals_commodity"
    MINING_DIVERSIFIED = "mining_diversified"
    MINING_PRECIOUS = "mining_precious"
    STEEL = "steel"
    PACKAGING = "packaging"

    # Real Estate
    REIT_RESIDENTIAL = "reit_residential"
    REIT_OFFICE = "reit_office"
    REIT_RETAIL = "reit_retail"
    REIT_INDUSTRIAL = "reit_industrial"
    REIT_HEALTHCARE = "reit_healthcare"
    REIT_DATA_CENTER = "reit_data_center"
    REIT_CELL_TOWER = "reit_cell_tower"

    # Communication Services
    TELECOM_INTEGRATED = "telecom_integrated"
    TELECOM_WIRELESS = "telecom_wireless"
    MEDIA_ENTERTAINMENT = "media_entertainment"
    MEDIA_STREAMING = "media_streaming"
    ADVERTISING = "advertising"
    GAMING = "gaming"
    SOCIAL_MEDIA = "social_media"

    # Default
    UNKNOWN = "unknown"


@dataclass
class IndustryProfile:
    """Profile defining model weights and characteristics for an industry."""

    category: IndustryCategory
    display_name: str
    # Model weights (must sum to 100)
    dcf_weight: float
    pe_weight: float
    ps_weight: float
    ev_ebitda_weight: float
    pb_weight: float
    ggm_weight: float
    # Key metrics for this industry
    key_metrics: List[str]
    # Preferred holding period
    holding_period: str
    # Volatility profile (affects confidence)
    volatility: str  # "low", "medium", "high", "very_high"
    # Cyclicality
    cyclical: bool
    # Growth vs Value orientation
    orientation: str  # "growth", "value", "blend"


# Industry weight profiles - comprehensive coverage
INDUSTRY_PROFILES: Dict[IndustryCategory, IndustryProfile] = {
    # ========== TECHNOLOGY ==========
    # ADJUSTED: Tech sector underperforms (-0.05 reward) - reduce PS (worst model), boost PE
    IndustryCategory.SOFTWARE_SAAS: IndustryProfile(
        category=IndustryCategory.SOFTWARE_SAAS,
        display_name="Software - SaaS",
        dcf_weight=25,
        pe_weight=25,
        ps_weight=15,
        ev_ebitda_weight=25,
        pb_weight=5,
        ggm_weight=5,
        key_metrics=["revenue_growth", "rule_of_40", "net_retention", "cac_ltv", "gross_margin"],
        holding_period="3m",
        volatility="high",
        cyclical=False,
        orientation="growth",
    ),
    # ADJUSTED: Reduce PS, boost PE
    IndustryCategory.SOFTWARE_ENTERPRISE: IndustryProfile(
        category=IndustryCategory.SOFTWARE_ENTERPRISE,
        display_name="Software - Enterprise",
        dcf_weight=30,
        pe_weight=30,
        ps_weight=10,
        ev_ebitda_weight=20,
        pb_weight=5,
        ggm_weight=5,
        key_metrics=["revenue_growth", "operating_margin", "fcf_margin", "backlog"],
        holding_period="3m",
        volatility="medium",
        cyclical=False,
        orientation="growth",
    ),
    # ADJUSTED: Semiconductors worst performer (-0.13 reward) - reduce DCF/PS, boost PB, shorten holding
    IndustryCategory.SEMICONDUCTORS: IndustryProfile(
        category=IndustryCategory.SEMICONDUCTORS,
        display_name="Semiconductors",
        dcf_weight=15,
        pe_weight=30,
        ps_weight=5,
        ev_ebitda_weight=30,
        pb_weight=20,
        ggm_weight=0,
        key_metrics=["gross_margin", "inventory_days", "capex_intensity", "book_to_bill"],
        holding_period="3m",
        volatility="very_high",
        cyclical=True,
        orientation="blend",
    ),
    # ADJUSTED: Semi equipment also volatile - similar adjustments
    IndustryCategory.SEMICONDUCTOR_EQUIPMENT: IndustryProfile(
        category=IndustryCategory.SEMICONDUCTOR_EQUIPMENT,
        display_name="Semiconductor Equipment",
        dcf_weight=15,
        pe_weight=30,
        ps_weight=5,
        ev_ebitda_weight=30,
        pb_weight=20,
        ggm_weight=0,
        key_metrics=["backlog", "book_to_bill", "gross_margin", "service_revenue_pct"],
        holding_period="3m",
        volatility="very_high",
        cyclical=True,
        orientation="blend",
    ),
    IndustryCategory.HARDWARE_CONSUMER: IndustryProfile(
        category=IndustryCategory.HARDWARE_CONSUMER,
        display_name="Hardware - Consumer Electronics",
        dcf_weight=35,
        pe_weight=25,
        ps_weight=15,
        ev_ebitda_weight=15,
        pb_weight=5,
        ggm_weight=5,
        key_metrics=["revenue_growth", "gross_margin", "inventory_turns", "services_revenue"],
        holding_period="1m",
        volatility="medium",
        cyclical=True,
        orientation="blend",
    ),
    IndustryCategory.CLOUD_INFRASTRUCTURE: IndustryProfile(
        category=IndustryCategory.CLOUD_INFRASTRUCTURE,
        display_name="Cloud Infrastructure",
        dcf_weight=30,
        pe_weight=15,
        ps_weight=30,
        ev_ebitda_weight=20,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["revenue_growth", "gross_margin", "capex_intensity", "remaining_performance_obligations"],
        holding_period="1m",
        volatility="high",
        cyclical=False,
        orientation="growth",
    ),
    IndustryCategory.CYBERSECURITY: IndustryProfile(
        category=IndustryCategory.CYBERSECURITY,
        display_name="Cybersecurity",
        dcf_weight=25,
        pe_weight=10,
        ps_weight=40,
        ev_ebitda_weight=20,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["revenue_growth", "arr_growth", "net_retention", "billings_growth"],
        holding_period="1m",
        volatility="high",
        cyclical=False,
        orientation="growth",
    ),
    IndustryCategory.FINTECH: IndustryProfile(
        category=IndustryCategory.FINTECH,
        display_name="Financial Technology",
        dcf_weight=30,
        pe_weight=15,
        ps_weight=30,
        ev_ebitda_weight=20,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["revenue_growth", "take_rate", "tpv_growth", "gross_margin"],
        holding_period="1m",
        volatility="high",
        cyclical=False,
        orientation="growth",
    ),
    # ========== HEALTHCARE ==========
    IndustryCategory.PHARMA_LARGE: IndustryProfile(
        category=IndustryCategory.PHARMA_LARGE,
        display_name="Pharmaceuticals - Large Cap",
        dcf_weight=35,
        pe_weight=25,
        ps_weight=10,
        ev_ebitda_weight=15,
        pb_weight=5,
        ggm_weight=10,
        key_metrics=["pipeline_value", "patent_cliff", "rd_intensity", "operating_margin"],
        holding_period="3m",
        volatility="medium",
        cyclical=False,
        orientation="blend",
    ),
    IndustryCategory.PHARMA_SPECIALTY: IndustryProfile(
        category=IndustryCategory.PHARMA_SPECIALTY,
        display_name="Pharmaceuticals - Specialty",
        dcf_weight=30,
        pe_weight=20,
        ps_weight=25,
        ev_ebitda_weight=20,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["revenue_growth", "gross_margin", "pipeline_diversity", "pricing_power"],
        holding_period="3m",
        volatility="high",
        cyclical=False,
        orientation="growth",
    ),
    IndustryCategory.BIOTECH_COMMERCIAL: IndustryProfile(
        category=IndustryCategory.BIOTECH_COMMERCIAL,
        display_name="Biotechnology - Commercial Stage",
        dcf_weight=35,
        pe_weight=20,
        ps_weight=25,
        ev_ebitda_weight=15,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["revenue_growth", "gross_margin", "pipeline_value", "cash_runway"],
        holding_period="3m",
        volatility="high",
        cyclical=False,
        orientation="growth",
    ),
    IndustryCategory.BIOTECH_CLINICAL: IndustryProfile(
        category=IndustryCategory.BIOTECH_CLINICAL,
        display_name="Biotechnology - Clinical Stage",
        dcf_weight=20,
        pe_weight=0,
        ps_weight=60,
        ev_ebitda_weight=0,
        pb_weight=20,
        ggm_weight=0,
        key_metrics=["cash_runway", "pipeline_value", "clinical_trial_progress", "partnership_value"],
        holding_period="18m",
        volatility="very_high",
        cyclical=False,
        orientation="growth",
    ),
    IndustryCategory.MEDICAL_DEVICES: IndustryProfile(
        category=IndustryCategory.MEDICAL_DEVICES,
        display_name="Medical Devices",
        dcf_weight=35,
        pe_weight=25,
        ps_weight=15,
        ev_ebitda_weight=20,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["revenue_growth", "gross_margin", "rd_intensity", "procedure_volumes"],
        holding_period="3m",
        volatility="medium",
        cyclical=False,
        orientation="blend",
    ),
    IndustryCategory.MANAGED_CARE: IndustryProfile(
        category=IndustryCategory.MANAGED_CARE,
        display_name="Managed Care / Health Insurance",
        dcf_weight=30,
        pe_weight=30,
        ps_weight=10,
        ev_ebitda_weight=20,
        pb_weight=5,
        ggm_weight=5,
        key_metrics=["medical_loss_ratio", "membership_growth", "operating_margin", "star_ratings"],
        holding_period="3m",
        volatility="medium",
        cyclical=False,
        orientation="blend",
    ),
    # ========== FINANCIALS ==========
    # ADJUSTED: Banks - boost PB further (best model for financials)
    IndustryCategory.BANK_MONEY_CENTER: IndustryProfile(
        category=IndustryCategory.BANK_MONEY_CENTER,
        display_name="Banks - Money Center",
        dcf_weight=10,
        pe_weight=30,
        ps_weight=0,
        ev_ebitda_weight=5,
        pb_weight=45,
        ggm_weight=10,
        key_metrics=["nim", "efficiency_ratio", "tier1_capital", "rote", "npl_ratio"],
        holding_period="1m",
        volatility="medium",
        cyclical=True,
        orientation="value",
    ),
    # ADJUSTED: Regional banks - PB is key metric
    IndustryCategory.BANK_REGIONAL: IndustryProfile(
        category=IndustryCategory.BANK_REGIONAL,
        display_name="Banks - Regional",
        dcf_weight=10,
        pe_weight=25,
        ps_weight=0,
        ev_ebitda_weight=5,
        pb_weight=50,
        ggm_weight=10,
        key_metrics=["nim", "efficiency_ratio", "deposit_growth", "loan_growth", "npl_ratio"],
        holding_period="1m",
        volatility="high",
        cyclical=True,
        orientation="value",
    ),
    # ADJUSTED: Investment banks - boost PB, remove PS
    IndustryCategory.BANK_INVESTMENT: IndustryProfile(
        category=IndustryCategory.BANK_INVESTMENT,
        display_name="Banks - Investment",
        dcf_weight=15,
        pe_weight=30,
        ps_weight=0,
        ev_ebitda_weight=10,
        pb_weight=40,
        ggm_weight=5,
        key_metrics=["trading_revenue", "advisory_fees", "rote", "compensation_ratio", "var"],
        holding_period="3m",
        volatility="high",
        cyclical=True,
        orientation="blend",
    ),
    # ADJUSTED: Insurance performs well (+0.06-0.07 reward) - boost PB significantly
    IndustryCategory.INSURANCE_LIFE: IndustryProfile(
        category=IndustryCategory.INSURANCE_LIFE,
        display_name="Insurance - Life",
        dcf_weight=20,
        pe_weight=20,
        ps_weight=0,
        ev_ebitda_weight=5,
        pb_weight=45,
        ggm_weight=10,
        key_metrics=["embedded_value", "new_business_margin", "roe", "solvency_ratio"],
        holding_period="1m",
        volatility="medium",
        cyclical=False,
        orientation="value",
    ),
    # ADJUSTED: P&C Insurance good performer - boost PB
    IndustryCategory.INSURANCE_PC: IndustryProfile(
        category=IndustryCategory.INSURANCE_PC,
        display_name="Insurance - Property & Casualty",
        dcf_weight=15,
        pe_weight=20,
        ps_weight=0,
        ev_ebitda_weight=5,
        pb_weight=50,
        ggm_weight=10,
        key_metrics=["combined_ratio", "premium_growth", "reserve_ratio", "cat_loss_ratio"],
        holding_period="1m",
        volatility="medium",
        cyclical=True,
        orientation="value",
    ),
    IndustryCategory.ASSET_MANAGEMENT: IndustryProfile(
        category=IndustryCategory.ASSET_MANAGEMENT,
        display_name="Asset Management",
        dcf_weight=30,
        pe_weight=25,
        ps_weight=15,
        ev_ebitda_weight=20,
        pb_weight=5,
        ggm_weight=5,
        key_metrics=["aum_growth", "fee_rate", "flows", "operating_margin"],
        holding_period="1m",
        volatility="high",
        cyclical=True,
        orientation="blend",
    ),
    IndustryCategory.EXCHANGES: IndustryProfile(
        category=IndustryCategory.EXCHANGES,
        display_name="Financial Exchanges",
        dcf_weight=35,
        pe_weight=25,
        ps_weight=15,
        ev_ebitda_weight=20,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["trading_volume", "revenue_per_contract", "market_data_revenue", "operating_margin"],
        holding_period="3m",
        volatility="medium",
        cyclical=True,
        orientation="blend",
    ),
    IndustryCategory.CREDIT_SERVICES: IndustryProfile(
        category=IndustryCategory.CREDIT_SERVICES,
        display_name="Credit Services / Card Networks",
        dcf_weight=35,
        pe_weight=25,
        ps_weight=15,
        ev_ebitda_weight=20,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["payment_volume", "take_rate", "cross_border_volume", "operating_margin"],
        holding_period="3m",
        volatility="medium",
        cyclical=True,
        orientation="growth",
    ),
    # ========== CONSUMER DISCRETIONARY ==========
    IndustryCategory.RETAIL_ECOMMERCE: IndustryProfile(
        category=IndustryCategory.RETAIL_ECOMMERCE,
        display_name="Retail - E-Commerce",
        dcf_weight=30,
        pe_weight=15,
        ps_weight=30,
        ev_ebitda_weight=20,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["gmv_growth", "take_rate", "customer_acquisition_cost", "repeat_rate"],
        holding_period="1m",
        volatility="high",
        cyclical=True,
        orientation="growth",
    ),
    IndustryCategory.RETAIL_SPECIALTY: IndustryProfile(
        category=IndustryCategory.RETAIL_SPECIALTY,
        display_name="Retail - Specialty",
        dcf_weight=30,
        pe_weight=25,
        ps_weight=15,
        ev_ebitda_weight=25,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["same_store_sales", "inventory_turns", "gross_margin", "store_count"],
        holding_period="1m",
        volatility="high",
        cyclical=True,
        orientation="blend",
    ),
    IndustryCategory.RETAIL_DISCOUNT: IndustryProfile(
        category=IndustryCategory.RETAIL_DISCOUNT,
        display_name="Retail - Discount",
        dcf_weight=35,
        pe_weight=25,
        ps_weight=10,
        ev_ebitda_weight=20,
        pb_weight=5,
        ggm_weight=5,
        key_metrics=["same_store_sales", "inventory_turns", "membership_revenue", "ecommerce_penetration"],
        holding_period="3m",
        volatility="medium",
        cyclical=True,
        orientation="value",
    ),
    IndustryCategory.AUTO_OEM: IndustryProfile(
        category=IndustryCategory.AUTO_OEM,
        display_name="Automotive - Traditional OEM",
        dcf_weight=25,
        pe_weight=30,
        ps_weight=10,
        ev_ebitda_weight=25,
        pb_weight=10,
        ggm_weight=0,
        key_metrics=["unit_sales", "average_selling_price", "warranty_costs", "ev_mix"],
        holding_period="3m",
        volatility="high",
        cyclical=True,
        orientation="value",
    ),
    IndustryCategory.AUTO_EV: IndustryProfile(
        category=IndustryCategory.AUTO_EV,
        display_name="Automotive - EV Pure-Play",
        dcf_weight=25,
        pe_weight=10,
        ps_weight=40,
        ev_ebitda_weight=20,
        pb_weight=5,
        ggm_weight=0,
        key_metrics=["deliveries_growth", "gross_margin", "battery_cost", "regulatory_credits"],
        holding_period="3m",
        volatility="very_high",
        cyclical=True,
        orientation="growth",
    ),
    IndustryCategory.RESTAURANTS: IndustryProfile(
        category=IndustryCategory.RESTAURANTS,
        display_name="Restaurants",
        dcf_weight=30,
        pe_weight=25,
        ps_weight=15,
        ev_ebitda_weight=25,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["same_store_sales", "unit_growth", "average_check", "digital_mix"],
        holding_period="1m",
        volatility="medium",
        cyclical=True,
        orientation="blend",
    ),
    IndustryCategory.HOMEBUILDERS: IndustryProfile(
        category=IndustryCategory.HOMEBUILDERS,
        display_name="Homebuilders",
        dcf_weight=25,
        pe_weight=30,
        ps_weight=10,
        ev_ebitda_weight=20,
        pb_weight=15,
        ggm_weight=0,
        key_metrics=["orders", "backlog", "gross_margin", "cancellation_rate", "land_inventory"],
        holding_period="3m",
        volatility="high",
        cyclical=True,
        orientation="value",
    ),
    IndustryCategory.LUXURY_GOODS: IndustryProfile(
        category=IndustryCategory.LUXURY_GOODS,
        display_name="Luxury Goods",
        dcf_weight=30,
        pe_weight=25,
        ps_weight=20,
        ev_ebitda_weight=20,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["organic_growth", "gross_margin", "brand_value", "china_exposure"],
        holding_period="3m",
        volatility="medium",
        cyclical=True,
        orientation="blend",
    ),
    # ========== CONSUMER STAPLES ==========
    # ADJUSTED: Best performing sector (+0.09 reward) - boost PB and GGM significantly
    IndustryCategory.FOOD_BEVERAGE: IndustryProfile(
        category=IndustryCategory.FOOD_BEVERAGE,
        display_name="Food & Beverage",
        dcf_weight=20,
        pe_weight=25,
        ps_weight=5,
        ev_ebitda_weight=15,
        pb_weight=15,
        ggm_weight=20,
        key_metrics=["organic_growth", "price_mix", "gross_margin", "market_share"],
        holding_period="1m",
        volatility="low",
        cyclical=False,
        orientation="value",
    ),
    # ADJUSTED: Consumer Staples outperforms - boost PB/GGM
    IndustryCategory.HOUSEHOLD_PRODUCTS: IndustryProfile(
        category=IndustryCategory.HOUSEHOLD_PRODUCTS,
        display_name="Household Products",
        dcf_weight=20,
        pe_weight=25,
        ps_weight=5,
        ev_ebitda_weight=15,
        pb_weight=15,
        ggm_weight=20,
        key_metrics=["organic_growth", "gross_margin", "advertising_spend", "market_share"],
        holding_period="1m",
        volatility="low",
        cyclical=False,
        orientation="value",
    ),
    # ADJUSTED: Tobacco - high dividend sector, boost GGM
    IndustryCategory.TOBACCO: IndustryProfile(
        category=IndustryCategory.TOBACCO,
        display_name="Tobacco",
        dcf_weight=15,
        pe_weight=20,
        ps_weight=0,
        ev_ebitda_weight=10,
        pb_weight=20,
        ggm_weight=35,
        key_metrics=["volume_decline", "pricing_power", "reduced_risk_products", "dividend_coverage"],
        holding_period="3m",
        volatility="low",
        cyclical=False,
        orientation="value",
    ),
    # ========== INDUSTRIALS ==========
    # ADJUSTED: Industrials underperform (-0.026 reward) - reduce DCF, boost PB, shorten holding
    IndustryCategory.AEROSPACE_DEFENSE: IndustryProfile(
        category=IndustryCategory.AEROSPACE_DEFENSE,
        display_name="Aerospace & Defense",
        dcf_weight=20,
        pe_weight=30,
        ps_weight=5,
        ev_ebitda_weight=25,
        pb_weight=15,
        ggm_weight=5,
        key_metrics=["backlog", "book_to_bill", "program_margins", "fcf_conversion"],
        holding_period="1m",
        volatility="medium",
        cyclical=False,
        orientation="blend",
    ),
    # ADJUSTED: Airlines very volatile - shorten holding, boost EV/EBITDA
    IndustryCategory.AIRLINES: IndustryProfile(
        category=IndustryCategory.AIRLINES,
        display_name="Airlines",
        dcf_weight=10,
        pe_weight=25,
        ps_weight=5,
        ev_ebitda_weight=40,
        pb_weight=20,
        ggm_weight=0,
        key_metrics=["rasm", "casm", "load_factor", "available_seat_miles", "fuel_hedge"],
        holding_period="1m",
        volatility="very_high",
        cyclical=True,
        orientation="value",
    ),
    # ADJUSTED: Machinery - reduce DCF, boost PB
    IndustryCategory.MACHINERY: IndustryProfile(
        category=IndustryCategory.MACHINERY,
        display_name="Industrial Machinery",
        dcf_weight=20,
        pe_weight=30,
        ps_weight=5,
        ev_ebitda_weight=25,
        pb_weight=15,
        ggm_weight=5,
        key_metrics=["orders", "backlog", "aftermarket_revenue", "margin_expansion"],
        holding_period="1m",
        volatility="medium",
        cyclical=True,
        orientation="blend",
    ),
    # ADJUSTED: Conglomerates - reduce DCF, boost PB
    IndustryCategory.INDUSTRIAL_CONGLOMERATES: IndustryProfile(
        category=IndustryCategory.INDUSTRIAL_CONGLOMERATES,
        display_name="Industrial Conglomerates",
        dcf_weight=25,
        pe_weight=30,
        ps_weight=5,
        ev_ebitda_weight=20,
        pb_weight=15,
        ggm_weight=5,
        key_metrics=["organic_growth", "margin", "fcf_conversion", "portfolio_value"],
        holding_period="1m",
        volatility="medium",
        cyclical=True,
        orientation="blend",
    ),
    # ADJUSTED: Logistics - reduce PS, boost PB
    IndustryCategory.LOGISTICS: IndustryProfile(
        category=IndustryCategory.LOGISTICS,
        display_name="Logistics & Freight",
        dcf_weight=25,
        pe_weight=30,
        ps_weight=5,
        ev_ebitda_weight=20,
        pb_weight=15,
        ggm_weight=5,
        key_metrics=["volume_growth", "yield", "operating_ratio", "network_density"],
        holding_period="1m",
        volatility="medium",
        cyclical=True,
        orientation="blend",
    ),
    # ========== ENERGY ==========
    IndustryCategory.OIL_INTEGRATED: IndustryProfile(
        category=IndustryCategory.OIL_INTEGRATED,
        display_name="Oil & Gas - Integrated",
        dcf_weight=30,
        pe_weight=20,
        ps_weight=5,
        ev_ebitda_weight=25,
        pb_weight=10,
        ggm_weight=10,
        key_metrics=["production_growth", "reserve_replacement", "upstream_margin", "downstream_margin"],
        holding_period="3m",
        volatility="high",
        cyclical=True,
        orientation="value",
    ),
    IndustryCategory.OIL_EXPLORATION: IndustryProfile(
        category=IndustryCategory.OIL_EXPLORATION,
        display_name="Oil & Gas - E&P",
        dcf_weight=25,
        pe_weight=15,
        ps_weight=5,
        ev_ebitda_weight=35,
        pb_weight=15,
        ggm_weight=5,
        key_metrics=["production_growth", "reserve_life", "finding_costs", "breakeven_price"],
        holding_period="3m",
        volatility="very_high",
        cyclical=True,
        orientation="value",
    ),
    IndustryCategory.OIL_SERVICES: IndustryProfile(
        category=IndustryCategory.OIL_SERVICES,
        display_name="Oil & Gas Services",
        dcf_weight=25,
        pe_weight=20,
        ps_weight=15,
        ev_ebitda_weight=30,
        pb_weight=10,
        ggm_weight=0,
        key_metrics=["rig_count", "day_rates", "backlog", "utilization"],
        holding_period="3m",
        volatility="very_high",
        cyclical=True,
        orientation="value",
    ),
    IndustryCategory.RENEWABLE_ENERGY: IndustryProfile(
        category=IndustryCategory.RENEWABLE_ENERGY,
        display_name="Renewable Energy",
        dcf_weight=35,
        pe_weight=15,
        ps_weight=20,
        ev_ebitda_weight=25,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["capacity_additions", "ppa_prices", "capacity_factor", "development_pipeline"],
        holding_period="18m",
        volatility="high",
        cyclical=False,
        orientation="growth",
    ),
    # ADJUSTED: Utilities good performer (+0.044 reward) - boost GGM/PB, shorten holding
    IndustryCategory.UTILITIES_REGULATED: IndustryProfile(
        category=IndustryCategory.UTILITIES_REGULATED,
        display_name="Utilities - Regulated",
        dcf_weight=15,
        pe_weight=20,
        ps_weight=0,
        ev_ebitda_weight=15,
        pb_weight=20,
        ggm_weight=30,
        key_metrics=["rate_base_growth", "allowed_roe", "regulatory_lag", "renewable_mix"],
        holding_period="1m",
        volatility="low",
        cyclical=False,
        orientation="value",
    ),
    # ADJUSTED: Independent utilities - less GGM but still boost PB
    IndustryCategory.UTILITIES_INDEPENDENT: IndustryProfile(
        category=IndustryCategory.UTILITIES_INDEPENDENT,
        display_name="Utilities - Independent Power",
        dcf_weight=25,
        pe_weight=25,
        ps_weight=5,
        ev_ebitda_weight=20,
        pb_weight=15,
        ggm_weight=10,
        key_metrics=["spark_spread", "capacity_payments", "hedge_ratio", "fleet_efficiency"],
        holding_period="1m",
        volatility="high",
        cyclical=True,
        orientation="blend",
    ),
    # ========== MATERIALS ==========
    IndustryCategory.CHEMICALS_SPECIALTY: IndustryProfile(
        category=IndustryCategory.CHEMICALS_SPECIALTY,
        display_name="Chemicals - Specialty",
        dcf_weight=35,
        pe_weight=25,
        ps_weight=10,
        ev_ebitda_weight=25,
        pb_weight=0,
        ggm_weight=5,
        key_metrics=["organic_growth", "gross_margin", "innovation_revenue", "pricing_power"],
        holding_period="3m",
        volatility="medium",
        cyclical=True,
        orientation="blend",
    ),
    IndustryCategory.CHEMICALS_COMMODITY: IndustryProfile(
        category=IndustryCategory.CHEMICALS_COMMODITY,
        display_name="Chemicals - Commodity",
        dcf_weight=25,
        pe_weight=20,
        ps_weight=10,
        ev_ebitda_weight=30,
        pb_weight=10,
        ggm_weight=5,
        key_metrics=["utilization", "feedstock_advantage", "cost_position", "capacity_additions"],
        holding_period="3m",
        volatility="high",
        cyclical=True,
        orientation="value",
    ),
    IndustryCategory.MINING_DIVERSIFIED: IndustryProfile(
        category=IndustryCategory.MINING_DIVERSIFIED,
        display_name="Mining - Diversified",
        dcf_weight=25,
        pe_weight=20,
        ps_weight=5,
        ev_ebitda_weight=30,
        pb_weight=15,
        ggm_weight=5,
        key_metrics=["production_costs", "reserve_life", "commodity_exposure", "capex_intensity"],
        holding_period="3m",
        volatility="high",
        cyclical=True,
        orientation="value",
    ),
    IndustryCategory.STEEL: IndustryProfile(
        category=IndustryCategory.STEEL,
        display_name="Steel",
        dcf_weight=20,
        pe_weight=20,
        ps_weight=5,
        ev_ebitda_weight=35,
        pb_weight=15,
        ggm_weight=5,
        key_metrics=["capacity_utilization", "spread", "raw_material_costs", "trade_policy"],
        holding_period="1m",
        volatility="very_high",
        cyclical=True,
        orientation="value",
    ),
    # ========== REAL ESTATE ==========
    # ADJUSTED: Real Estate good performer (+0.058 reward) - boost GGM/PB for dividend-paying REITs
    IndustryCategory.REIT_RESIDENTIAL: IndustryProfile(
        category=IndustryCategory.REIT_RESIDENTIAL,
        display_name="REIT - Residential",
        dcf_weight=15,
        pe_weight=20,
        ps_weight=0,
        ev_ebitda_weight=15,
        pb_weight=25,
        ggm_weight=25,
        key_metrics=["ffo", "same_store_noi", "occupancy", "rent_growth", "turnover"],
        holding_period="1m",
        volatility="medium",
        cyclical=True,
        orientation="blend",
    ),
    # ADJUSTED: Industrial REITs - boost GGM/PB
    IndustryCategory.REIT_INDUSTRIAL: IndustryProfile(
        category=IndustryCategory.REIT_INDUSTRIAL,
        display_name="REIT - Industrial/Logistics",
        dcf_weight=20,
        pe_weight=20,
        ps_weight=0,
        ev_ebitda_weight=15,
        pb_weight=20,
        ggm_weight=25,
        key_metrics=["ffo", "same_store_noi", "occupancy", "rent_spreads", "development_pipeline"],
        holding_period="1m",
        volatility="medium",
        cyclical=True,
        orientation="growth",
    ),
    # ADJUSTED: Data Center REITs - more growth-oriented but still boost PB
    IndustryCategory.REIT_DATA_CENTER: IndustryProfile(
        category=IndustryCategory.REIT_DATA_CENTER,
        display_name="REIT - Data Centers",
        dcf_weight=25,
        pe_weight=25,
        ps_weight=5,
        ev_ebitda_weight=20,
        pb_weight=15,
        ggm_weight=10,
        key_metrics=["ffo", "interconnection_revenue", "power_cost", "bookings", "churn"],
        holding_period="1m",
        volatility="medium",
        cyclical=False,
        orientation="growth",
    ),
    # ADJUSTED: Cell Tower REITs - stable dividends, boost GGM
    IndustryCategory.REIT_CELL_TOWER: IndustryProfile(
        category=IndustryCategory.REIT_CELL_TOWER,
        display_name="REIT - Cell Towers",
        dcf_weight=25,
        pe_weight=20,
        ps_weight=5,
        ev_ebitda_weight=15,
        pb_weight=15,
        ggm_weight=20,
        key_metrics=["affo", "tenant_billings", "colocation", "escalators", "churn"],
        holding_period="1m",
        volatility="low",
        cyclical=False,
        orientation="growth",
    ),
    # ========== COMMUNICATION SERVICES ==========
    # ADJUSTED: Telecom good for dividends - boost GGM/PB
    IndustryCategory.TELECOM_INTEGRATED: IndustryProfile(
        category=IndustryCategory.TELECOM_INTEGRATED,
        display_name="Telecom - Integrated",
        dcf_weight=15,
        pe_weight=20,
        ps_weight=0,
        ev_ebitda_weight=15,
        pb_weight=20,
        ggm_weight=30,
        key_metrics=["arpu", "churn", "subscriber_growth", "fiber_penetration", "5g_coverage"],
        holding_period="1m",
        volatility="medium",
        cyclical=False,
        orientation="value",
    ),
    # ADJUSTED: Wireless telecom - dividend payers, boost GGM
    IndustryCategory.TELECOM_WIRELESS: IndustryProfile(
        category=IndustryCategory.TELECOM_WIRELESS,
        display_name="Telecom - Wireless",
        dcf_weight=20,
        pe_weight=20,
        ps_weight=0,
        ev_ebitda_weight=15,
        pb_weight=15,
        ggm_weight=30,
        key_metrics=["arpu", "churn", "postpaid_net_adds", "spectrum_holdings"],
        holding_period="1m",
        volatility="medium",
        cyclical=False,
        orientation="blend",
    ),
    # ADJUSTED: Streaming - reduce PS (worst model), boost PE
    IndustryCategory.MEDIA_STREAMING: IndustryProfile(
        category=IndustryCategory.MEDIA_STREAMING,
        display_name="Media - Streaming",
        dcf_weight=30,
        pe_weight=30,
        ps_weight=10,
        ev_ebitda_weight=20,
        pb_weight=5,
        ggm_weight=5,
        key_metrics=["subscriber_growth", "arpu", "churn", "content_spend", "engagement"],
        holding_period="3m",
        volatility="high",
        cyclical=False,
        orientation="growth",
    ),
    # ADJUSTED: Advertising - reduce PS, boost PE
    IndustryCategory.ADVERTISING: IndustryProfile(
        category=IndustryCategory.ADVERTISING,
        display_name="Digital Advertising",
        dcf_weight=30,
        pe_weight=35,
        ps_weight=5,
        ev_ebitda_weight=20,
        pb_weight=5,
        ggm_weight=5,
        key_metrics=["revenue_growth", "arpu", "dau_mau", "ad_pricing", "engagement"],
        holding_period="3m",
        volatility="high",
        cyclical=True,
        orientation="growth",
    ),
    # ADJUSTED: Gaming - reduce PS, boost PE
    IndustryCategory.GAMING: IndustryProfile(
        category=IndustryCategory.GAMING,
        display_name="Gaming / Interactive Entertainment",
        dcf_weight=25,
        pe_weight=35,
        ps_weight=10,
        ev_ebitda_weight=20,
        pb_weight=5,
        ggm_weight=5,
        key_metrics=["bookings", "mau", "arpu", "live_services_revenue", "pipeline"],
        holding_period="3m",
        volatility="high",
        cyclical=True,
        orientation="growth",
    ),
    # ADJUSTED: Social Media - reduce PS significantly
    IndustryCategory.SOCIAL_MEDIA: IndustryProfile(
        category=IndustryCategory.SOCIAL_MEDIA,
        display_name="Social Media",
        dcf_weight=30,
        pe_weight=35,
        ps_weight=5,
        ev_ebitda_weight=20,
        pb_weight=5,
        ggm_weight=5,
        key_metrics=["dau_mau", "arpu", "engagement_time", "ad_impressions", "creator_economy"],
        holding_period="3m",
        volatility="high",
        cyclical=True,
        orientation="growth",
    ),
    # ========== DEFAULT ==========
    # ADJUSTED: Default profile - boost PB (best model), reduce PS (worst), add GGM
    # HOLDING PERIOD: Changed to 1m based on backtest (1m optimal reward: 0.8239 vs 0.58 for 6m)
    IndustryCategory.UNKNOWN: IndustryProfile(
        category=IndustryCategory.UNKNOWN,
        display_name="Unknown/Other",
        dcf_weight=20,
        pe_weight=30,
        ps_weight=5,
        ev_ebitda_weight=15,
        pb_weight=20,
        ggm_weight=10,
        key_metrics=["revenue_growth", "operating_margin", "fcf_margin", "roe"],
        holding_period="1m",
        volatility="medium",
        cyclical=False,
        orientation="blend",
    ),
}


# Mapping from sector/industry strings to IndustryCategory
INDUSTRY_CLASSIFICATION: Dict[str, Dict[str, IndustryCategory]] = {
    # Handle variations of sector names
    "Information Technology": {
        "default": IndustryCategory.SOFTWARE_ENTERPRISE,
        "Semiconductors & Semiconductor Equipment": IndustryCategory.SEMICONDUCTORS,
        "Software": IndustryCategory.SOFTWARE_ENTERPRISE,
        "Technology Hardware, Storage & Peripherals": IndustryCategory.HARDWARE_CONSUMER,
        "IT Services": IndustryCategory.CLOUD_INFRASTRUCTURE,
        "Communications Equipment": IndustryCategory.HARDWARE_ENTERPRISE,
        "Electronic Equipment, Instruments & Components": IndustryCategory.HARDWARE_ENTERPRISE,
    },
    "Finance": {
        "default": IndustryCategory.BANK_REGIONAL,
        "Major Banks": IndustryCategory.BANK_MONEY_CENTER,
        "Regional Banks": IndustryCategory.BANK_REGIONAL,
        "Investment Banks/Brokers": IndustryCategory.BANK_INVESTMENT,
        "Insurance": IndustryCategory.INSURANCE_PC,
        "Life Insurance": IndustryCategory.INSURANCE_LIFE,
        "Property & Casualty Insurance": IndustryCategory.INSURANCE_PC,
        "Asset Management": IndustryCategory.ASSET_MANAGEMENT,
        "Financial Services": IndustryCategory.CREDIT_SERVICES,
        "Consumer Finance": IndustryCategory.CREDIT_SERVICES,
        "Diversified Financial Services": IndustryCategory.CREDIT_SERVICES,
    },
    "Technology": {
        "default": IndustryCategory.SOFTWARE_ENTERPRISE,
        "Software—Application": IndustryCategory.SOFTWARE_SAAS,
        "Software—Infrastructure": IndustryCategory.SOFTWARE_ENTERPRISE,
        "Semiconductors": IndustryCategory.SEMICONDUCTORS,
        "Semiconductor Equipment & Materials": IndustryCategory.SEMICONDUCTOR_EQUIPMENT,
        "Consumer Electronics": IndustryCategory.HARDWARE_CONSUMER,
        "Computer Hardware": IndustryCategory.HARDWARE_ENTERPRISE,
        "Information Technology Services": IndustryCategory.CLOUD_INFRASTRUCTURE,
        "Electronic Components": IndustryCategory.SEMICONDUCTORS,
        "Scientific & Technical Instruments": IndustryCategory.HARDWARE_ENTERPRISE,
        "Communication Equipment": IndustryCategory.HARDWARE_ENTERPRISE,
    },
    "Healthcare": {
        "default": IndustryCategory.PHARMA_LARGE,
        "Drug Manufacturers—General": IndustryCategory.PHARMA_LARGE,
        "Drug Manufacturers—Specialty & Generic": IndustryCategory.PHARMA_SPECIALTY,
        "Biotechnology": IndustryCategory.BIOTECH_COMMERCIAL,
        "Medical Devices": IndustryCategory.MEDICAL_DEVICES,
        "Medical Instruments & Supplies": IndustryCategory.MEDICAL_DEVICES,
        "Healthcare Plans": IndustryCategory.MANAGED_CARE,
        "Health Information Services": IndustryCategory.HEALTHCARE_SERVICES,
        "Medical Care Facilities": IndustryCategory.HEALTHCARE_SERVICES,
        "Diagnostics & Research": IndustryCategory.DIAGNOSTICS,
        "Pharmaceutical Retailers": IndustryCategory.RETAIL_SPECIALTY,
    },
    "Financials": {
        "default": IndustryCategory.BANK_REGIONAL,
        "Banks—Diversified": IndustryCategory.BANK_MONEY_CENTER,
        "Banks—Regional": IndustryCategory.BANK_REGIONAL,
        "Capital Markets": IndustryCategory.BANK_INVESTMENT,
        "Insurance—Life": IndustryCategory.INSURANCE_LIFE,
        "Insurance—Property & Casualty": IndustryCategory.INSURANCE_PC,
        "Insurance—Reinsurance": IndustryCategory.INSURANCE_REINSURANCE,
        "Insurance—Diversified": IndustryCategory.INSURANCE_PC,
        "Asset Management": IndustryCategory.ASSET_MANAGEMENT,
        "Credit Services": IndustryCategory.CREDIT_SERVICES,
        "Financial Data & Stock Exchanges": IndustryCategory.EXCHANGES,
        "Insurance Brokers": IndustryCategory.INSURANCE_PC,
        "Mortgage Finance": IndustryCategory.BANK_REGIONAL,
    },
    "Consumer Discretionary": {
        "default": IndustryCategory.RETAIL_SPECIALTY,
        "Internet Retail": IndustryCategory.RETAIL_ECOMMERCE,
        "Specialty Retail": IndustryCategory.RETAIL_SPECIALTY,
        "Discount Stores": IndustryCategory.RETAIL_DISCOUNT,
        "Auto Manufacturers": IndustryCategory.AUTO_OEM,
        "Auto Parts": IndustryCategory.AUTO_PARTS,
        "Restaurants": IndustryCategory.RESTAURANTS,
        "Lodging": IndustryCategory.HOTELS_LEISURE,
        "Resorts & Casinos": IndustryCategory.HOTELS_LEISURE,
        "Residential Construction": IndustryCategory.HOMEBUILDERS,
        "Luxury Goods": IndustryCategory.LUXURY_GOODS,
        "Apparel Retail": IndustryCategory.RETAIL_SPECIALTY,
        "Apparel Manufacturing": IndustryCategory.LUXURY_GOODS,
        "Home Improvement Retail": IndustryCategory.RETAIL_SPECIALTY,
        "Footwear & Accessories": IndustryCategory.LUXURY_GOODS,
        "Travel Services": IndustryCategory.HOTELS_LEISURE,
        "Leisure": IndustryCategory.HOTELS_LEISURE,
    },
    "Consumer Staples": {
        "default": IndustryCategory.FOOD_BEVERAGE,
        "Beverages—Non-Alcoholic": IndustryCategory.FOOD_BEVERAGE,
        "Beverages—Alcoholic": IndustryCategory.FOOD_BEVERAGE,
        "Packaged Foods": IndustryCategory.FOOD_BEVERAGE,
        "Household & Personal Products": IndustryCategory.HOUSEHOLD_PRODUCTS,
        "Tobacco": IndustryCategory.TOBACCO,
        "Grocery Stores": IndustryCategory.RETAIL_GROCERY,
        "Farm Products": IndustryCategory.FOOD_BEVERAGE,
        "Confectioners": IndustryCategory.FOOD_BEVERAGE,
        "Food Distribution": IndustryCategory.FOOD_BEVERAGE,
    },
    "Industrials": {
        "default": IndustryCategory.MACHINERY,
        "Aerospace & Defense": IndustryCategory.AEROSPACE_DEFENSE,
        "Airlines": IndustryCategory.AIRLINES,
        "Farm & Heavy Construction Machinery": IndustryCategory.MACHINERY,
        "Specialty Industrial Machinery": IndustryCategory.MACHINERY,
        "Industrial Distribution": IndustryCategory.LOGISTICS,
        "Conglomerates": IndustryCategory.INDUSTRIAL_CONGLOMERATES,
        "Railroads": IndustryCategory.TRANSPORTATION_RAIL,
        "Trucking": IndustryCategory.TRANSPORTATION_TRUCKING,
        "Integrated Freight & Logistics": IndustryCategory.LOGISTICS,
        "Marine Shipping": IndustryCategory.LOGISTICS,
        "Building Products & Equipment": IndustryCategory.CONSTRUCTION,
        "Engineering & Construction": IndustryCategory.CONSTRUCTION,
        "Rental & Leasing Services": IndustryCategory.LOGISTICS,
        "Electrical Equipment & Parts": IndustryCategory.MACHINERY,
        "Waste Management": IndustryCategory.INDUSTRIAL_CONGLOMERATES,
        "Staffing & Employment Services": IndustryCategory.INDUSTRIAL_CONGLOMERATES,
        "Security & Protection Services": IndustryCategory.INDUSTRIAL_CONGLOMERATES,
        "Consulting Services": IndustryCategory.INDUSTRIAL_CONGLOMERATES,
    },
    "Energy": {
        "default": IndustryCategory.OIL_INTEGRATED,
        "Oil & Gas Integrated": IndustryCategory.OIL_INTEGRATED,
        "Oil & Gas E&P": IndustryCategory.OIL_EXPLORATION,
        "Oil & Gas Refining & Marketing": IndustryCategory.OIL_REFINING,
        "Oil & Gas Equipment & Services": IndustryCategory.OIL_SERVICES,
        "Oil & Gas Midstream": IndustryCategory.OIL_INTEGRATED,
        "Uranium": IndustryCategory.RENEWABLE_ENERGY,
        "Solar": IndustryCategory.RENEWABLE_ENERGY,
        "Thermal Coal": IndustryCategory.OIL_EXPLORATION,
    },
    "Materials": {
        "default": IndustryCategory.CHEMICALS_SPECIALTY,
        "Specialty Chemicals": IndustryCategory.CHEMICALS_SPECIALTY,
        "Chemicals": IndustryCategory.CHEMICALS_COMMODITY,
        "Agricultural Inputs": IndustryCategory.CHEMICALS_COMMODITY,
        "Gold": IndustryCategory.MINING_PRECIOUS,
        "Silver": IndustryCategory.MINING_PRECIOUS,
        "Copper": IndustryCategory.MINING_DIVERSIFIED,
        "Other Industrial Metals & Mining": IndustryCategory.MINING_DIVERSIFIED,
        "Steel": IndustryCategory.STEEL,
        "Aluminum": IndustryCategory.MINING_DIVERSIFIED,
        "Paper & Paper Products": IndustryCategory.PACKAGING,
        "Packaging & Containers": IndustryCategory.PACKAGING,
        "Building Materials": IndustryCategory.CHEMICALS_SPECIALTY,
        "Lumber & Wood Production": IndustryCategory.PACKAGING,
    },
    "Real Estate": {
        "default": IndustryCategory.REIT_RESIDENTIAL,
        "REIT—Residential": IndustryCategory.REIT_RESIDENTIAL,
        "REIT—Retail": IndustryCategory.REIT_RETAIL,
        "REIT—Office": IndustryCategory.REIT_OFFICE,
        "REIT—Industrial": IndustryCategory.REIT_INDUSTRIAL,
        "REIT—Healthcare Facilities": IndustryCategory.REIT_HEALTHCARE,
        "REIT—Diversified": IndustryCategory.REIT_RESIDENTIAL,
        "REIT—Hotel & Motel": IndustryCategory.REIT_RETAIL,
        "REIT—Specialty": IndustryCategory.REIT_DATA_CENTER,
        "Real Estate Services": IndustryCategory.REIT_RESIDENTIAL,
        "Real Estate—Development": IndustryCategory.REIT_RESIDENTIAL,
        "Real Estate—Diversified": IndustryCategory.REIT_RESIDENTIAL,
    },
    "Utilities": {
        "default": IndustryCategory.UTILITIES_REGULATED,
        "Utilities—Regulated Electric": IndustryCategory.UTILITIES_REGULATED,
        "Utilities—Regulated Gas": IndustryCategory.UTILITIES_REGULATED,
        "Utilities—Diversified": IndustryCategory.UTILITIES_REGULATED,
        "Utilities—Regulated Water": IndustryCategory.UTILITIES_REGULATED,
        "Utilities—Independent Power Producers": IndustryCategory.UTILITIES_INDEPENDENT,
        "Utilities—Renewable": IndustryCategory.RENEWABLE_ENERGY,
    },
    "Communication Services": {
        "default": IndustryCategory.TELECOM_INTEGRATED,
        "Telecom Services": IndustryCategory.TELECOM_INTEGRATED,
        "Internet Content & Information": IndustryCategory.SOCIAL_MEDIA,
        "Entertainment": IndustryCategory.MEDIA_ENTERTAINMENT,
        "Electronic Gaming & Multimedia": IndustryCategory.GAMING,
        "Advertising Agencies": IndustryCategory.ADVERTISING,
        "Broadcasting": IndustryCategory.MEDIA_ENTERTAINMENT,
        "Publishing": IndustryCategory.MEDIA_ENTERTAINMENT,
    },
}


def classify_industry(sector: str, industry: str) -> IndustryCategory:
    """
    Classify a company into an IndustryCategory based on sector and industry.

    Args:
        sector: The sector (e.g., "Technology")
        industry: The industry (e.g., "Semiconductors")

    Returns:
        The most specific IndustryCategory match
    """
    sector_map = INDUSTRY_CLASSIFICATION.get(sector, {})

    # Try exact match first
    if industry in sector_map:
        return sector_map[industry]

    # Try partial match
    industry_lower = industry.lower()
    for key, category in sector_map.items():
        if key != "default" and key.lower() in industry_lower:
            return category
        if key != "default" and industry_lower in key.lower():
            return category

    # Fall back to sector default
    if "default" in sector_map:
        return sector_map["default"]

    return IndustryCategory.UNKNOWN


def get_industry_profile(sector: str, industry: str) -> IndustryProfile:
    """
    Get the complete industry profile for weight determination.

    Args:
        sector: The sector
        industry: The industry

    Returns:
        IndustryProfile with weights and characteristics
    """
    category = classify_industry(sector, industry)
    return INDUSTRY_PROFILES.get(category, INDUSTRY_PROFILES[IndustryCategory.UNKNOWN])


def get_industry_weights(sector: str, industry: str) -> Dict[str, float]:
    """
    Get model weights for a specific industry.

    Args:
        sector: The sector
        industry: The industry

    Returns:
        Dict with model weights (dcf, pe, ps, ev_ebitda, pb, ggm)
    """
    profile = get_industry_profile(sector, industry)
    return {
        "dcf": profile.dcf_weight,
        "pe": profile.pe_weight,
        "ps": profile.ps_weight,
        "ev_ebitda": profile.ev_ebitda_weight,
        "pb": profile.pb_weight,
        "ggm": profile.ggm_weight,
    }


def get_industry_holding_period(sector: str, industry: str) -> str:
    """Get recommended holding period for an industry."""
    profile = get_industry_profile(sector, industry)
    return profile.holding_period


def get_industry_volatility(sector: str, industry: str) -> str:
    """Get volatility profile for an industry."""
    profile = get_industry_profile(sector, industry)
    return profile.volatility


def get_industry_orientation(sector: str, industry: str) -> str:
    """Get growth/value orientation for an industry."""
    profile = get_industry_profile(sector, industry)
    return profile.orientation
