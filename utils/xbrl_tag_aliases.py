"""
XBRL Tag Alias Mapping Module

Provides unified tag resolution across:
1. SEC CompanyFacts API (camelCase tags like 'RevenueFromContractWithCustomerExcludingAssessedTax')
2. SEC Bulk DERA tables (same XBRL tags)
3. Processed tables (snake_case like 'total_revenue')

Based on comprehensive analysis of FAANG companies' actual tag usage across:
- AAPL (Apple)
- AMZN (Amazon)
- META (Facebook/Meta)
- GOOGL (Alphabet)
- NFLX (Netflix)

This ensures robust extraction regardless of tag variations across companies.
"""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class XBRLTagAliasMapper:
    """
    Maps canonical financial metrics to all possible XBRL tag variations
    found across major companies.

    Usage:
        mapper = XBRLTagAliasMapper()

        # Get all possible XBRL tags for a metric
        revenue_tags = mapper.get_xbrl_aliases('total_revenue')
        # Returns: ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet', ...]

        # Resolve XBRL tag to canonical name
        canonical = mapper.resolve_to_canonical('Revenues')
        # Returns: 'total_revenue'
    """

    # Canonical name → List of XBRL tag aliases (priority order)
    TAG_ALIASES: Dict[str, List[str]] = {
        # ====================
        # INCOME STATEMENT
        # ====================

        'revenues': [
            'RevenueFromContractWithCustomerExcludingAssessedTax',  # AAPL, AMZN, META, GOOGL
            'Revenues',  # NFLX
            'SalesRevenueNet',  # Alternative
            'RevenueFromContractWithCustomerIncludingAssessedTax',  # Some companies
            'InterestAndDividendIncomeOperating',  # Financial institutions (banks, insurance)
            'RevenuesNetOfInterestExpense',  # Some banks
            'InterestIncomeOperating',  # Banks
            'RegulatedAndUnregulatedOperatingRevenue',  # Utilities (NEE)
        ],

        'cost_of_revenue': [
            'CostOfRevenue',  # GOOGL, META, NFLX
            'CostOfGoodsAndServicesSold',  # AAPL
            'CostOfSales',  # Alternative
        ],

        'gross_profit': [
            'GrossProfit',  # AAPL, common
            'RevenueMinusCostOfRevenue',  # Calculated fallback
        ],

        'operating_income': [
            'OperatingIncomeLoss',  # AAPL, AMZN, META, GOOGL, NFLX
            'OperatingIncome',  # Alternative name
            'IncomeLossFromOperations',  # Some companies
        ],

        'net_income': [
            'NetIncomeLoss',  # All FAANG use this
            'NetIncome',  # Alternative
            'ProfitLoss',  # IFRS equivalent
        ],

        'income_tax_expense': [
            'IncomeTaxExpenseBenefit',  # All companies (14/14)
            'IncomeTaxesPaid',  # Cash basis
        ],

        'pretax_income': [
            'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
            'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
            'IncomeLossFromContinuingOperationsBeforeIncomeTaxes',
        ],

        'comprehensive_income': [
            'ComprehensiveIncomeNetOfTax',  # 13/14 companies
            'ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest',
        ],

        'other_income_expense': [
            'NonoperatingIncomeExpense',  # 7/14 companies
            'OtherNonoperatingIncomeExpense',
        ],

        # Earnings Per Share
        'eps_basic': [
            'EarningsPerShareBasic',  # All FAANG
        ],

        'eps_diluted': [
            'EarningsPerShareDiluted',  # All FAANG
        ],

        # Operating Expenses
        'research_and_development': [
            'ResearchAndDevelopmentExpense',
            'ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost',
        ],

        'selling_general_administrative': [
            'SellingGeneralAndAdministrativeExpense',
            'GeneralAndAdministrativeExpense',  # AMZN, META, GOOGL separate
        ],

        # ====================
        # BALANCE SHEET
        # ====================

        'total_assets': [
            'Assets',  # All FAANG
        ],

        'current_assets': [
            'AssetsCurrent',  # All FAANG
        ],

        'noncurrent_assets': [
            'AssetsNoncurrent',  # AAPL
            'AssetsPermanentEndOfPeriod',  # Alternative
        ],

        'cash_and_equivalents': [
            'CashAndCashEquivalentsAtCarryingValue',  # All FAANG
            'Cash',  # Simple form
            'CashEquivalentsAtCarryingValue',
        ],

        'cash_and_short_term_investments': [
            'CashCashEquivalentsAndShortTermInvestments',  # GOOGL
            'CashAndCashEquivalentsAtCarryingValue',  # Fallback
        ],

        'cash_and_restricted_cash': [
            'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',  # 14/14 companies
        ],

        'accounts_receivable': [
            'AccountsReceivableNetCurrent',  # All FAANG
            'ReceivablesNetCurrent',  # Alternative
        ],

        'inventory': [
            'InventoryNet',
            'Inventory',
        ],

        'property_plant_equipment': [
            'PropertyPlantAndEquipmentNet',
            'PropertyPlantAndEquipmentGross',  # Before depreciation
        ],

        'goodwill': [
            'Goodwill',  # AMZN and others
        ],

        'intangible_assets': [
            'IntangibleAssetsNetExcludingGoodwill',
            'FiniteLivedIntangibleAssetsNet',
        ],

        'other_assets_current': [
            'OtherAssetsCurrent',  # AAPL, GOOGL, NFLX
            'PrepaidExpenseAndOtherAssetsCurrent',  # META
        ],

        'other_assets_noncurrent': [
            'OtherAssetsNoncurrent',  # All FAANG
        ],

        # Balance Sheet Validation (Assets = Liabilities + Equity)
        'liabilities_and_equity': [
            'LiabilitiesAndStockholdersEquity',  # 14/14 companies - should equal Assets
        ],

        # Liabilities
        'total_liabilities': [
            'Liabilities',  # AAPL, META, NFLX, GOOGL
        ],

        'current_liabilities': [
            'LiabilitiesCurrent',  # All FAANG
        ],

        'noncurrent_liabilities': [
            'LiabilitiesNoncurrent',  # AAPL
            'LiabilitiesOtherThanLongtermDebtNoncurrent',  # Alternative
        ],

        'accounts_payable': [
            'AccountsPayableCurrent',  # AAPL, AMZN, GOOGL, NFLX
            'AccountsPayableTradeCurrent',  # META
        ],

        'accrued_liabilities': [
            'AccruedLiabilitiesCurrent',  # All FAANG except AAPL
            'OtherLiabilitiesCurrent',  # AAPL
        ],

        'other_liabilities_noncurrent': [
            'OtherLiabilitiesNoncurrent',  # 12/14 companies
        ],

        'deferred_revenue': [
            'ContractWithCustomerLiabilityCurrent',  # All FAANG
            'DeferredRevenueCurrent',  # Legacy name
        ],

        'long_term_debt': [
            'LongTermDebtNoncurrent',
            'LongTermDebt',
            'DebtLongTermAndShortTermCombinedAmount',  # Combined
        ],

        'short_term_debt': [
            'ShortTermBorrowings',
            'DebtCurrent',
            'CommercialPaper',  # AAPL uses this
        ],

        'total_debt': [
            'DebtLongTermAndShortTermCombinedAmount',
            'LongTermDebtAndCapitalLeaseObligations',  # With leases
        ],

        # Equity
        'stockholders_equity': [
            'StockholdersEquity',
            'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        ],

        'retained_earnings': [
            'RetainedEarningsAccumulatedDeficit',  # All FAANG
        ],

        'common_stock': [
            'CommonStockValue',  # AMZN, META, NFLX
            'CommonStocksIncludingAdditionalPaidInCapital',  # AAPL, GOOGL
        ],

        'additional_paid_in_capital': [
            'AdditionalPaidInCapital',  # AMZN, META
            'CommonStocksIncludingAdditionalPaidInCapital',  # AAPL, GOOGL (combined)
        ],

        'treasury_stock': [
            'TreasuryStockValue',
            'TreasuryStockCommonValue',
        ],

        'accumulated_other_comprehensive_income': [
            'AccumulatedOtherComprehensiveIncomeLossNetOfTax',  # All FAANG
        ],

        'minority_interest': [
            'MinorityInterest',  # 7/14 companies
            'MinorityInterestInNetIncomeLossOfConsolidatedEntities',
        ],

        # ====================
        # CASH FLOW STATEMENT
        # ====================

        'operating_cash_flow': [
            'NetCashProvidedByUsedInOperatingActivities',  # AAPL
            'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',  # Long form
        ],

        'investing_cash_flow': [
            'NetCashProvidedByUsedInInvestingActivities',
            'NetCashProvidedByUsedInInvestingActivitiesContinuingOperations',
        ],

        'financing_cash_flow': [
            'NetCashProvidedByUsedInFinancingActivities',
            'NetCashProvidedByUsedInFinancingActivitiesContinuingOperations',
        ],

        'capital_expenditures': [
            'PaymentsToAcquirePropertyPlantAndEquipment',  # AAPL (most common)
            'PaymentsToAcquireProductiveAssets',  # AMZN
            'CapitalExpendituresIncurredButNotYetPaid',  # META (accrual basis)
            'PaymentsForCapitalImprovements',  # Alternative
        ],

        'free_cash_flow': [
            'FreeCashFlow',  # If directly reported
            # Otherwise calculated: operating_cash_flow - capital_expenditures
        ],

        'depreciation_amortization': [
            'DepreciationDepletionAndAmortization',  # All FAANG
            'Depreciation',  # GOOGL uses separate
            'DepreciationAndAmortization',  # Alternative
        ],

        # Working Capital Changes (Cash Flow)
        'change_in_accounts_receivable': [
            'IncreaseDecreaseInAccountsReceivable',  # 8/14 companies
            'IncreaseDecreaseInReceivables',
        ],

        'change_in_inventory': [
            'IncreaseDecreaseInInventories',  # 8/14 companies
        ],

        'change_in_accounts_payable': [
            'IncreaseDecreaseInAccountsPayable',  # 7/14 companies
        ],

        'other_cash_flow_changes': [
            'OtherNoncashIncomeExpense',  # 9/14 companies
        ],

        'stock_based_compensation': [
            'AllocatedShareBasedCompensationExpense',
            'ShareBasedCompensation',
            'AdjustmentsToAdditionalPaidInCapitalSharebasedCompensationRequisiteServicePeriodRecognitionValue',  # All FAANG (long!)
        ],

        'dividends_paid': [
            'PaymentsOfDividends',  # Most common (all companies use this)
            'PaymentsOfDividendsCommonStock',  # Specific to common stock
            'Dividends',  # AAPL (simple form)
            'DividendsCommonStock',  # META
            'PaymentsOfDividendsPreferredStockAndPreferenceStock',  # Preferred dividends
        ],

        'share_repurchases': [
            'PaymentsForRepurchaseOfCommonStock',
            'TreasuryStockValueAcquiredCostMethod',
        ],

        # ====================
        # INSURANCE-SPECIFIC METRICS
        # ====================

        # Insurance Premiums (Revenue)
        'premiums_earned': [
            'PremiumsEarnedNet',  # Most common - net premiums earned after reinsurance
            'NetPremiumsEarned',  # Alternative naming
            'InsurancePremiumsAndFeesNet',  # Includes fees
            'PremiumsWrittenNet',  # Written vs earned
            'PremiumsEarnedNetPropertyAndCasualtyInsurance',  # P&C specific
            'PremiumsEarnedNetLifeInsurance',  # Life insurance specific
            'GrossPremiumsWritten',  # Gross before reinsurance
            'NetPremiumsWritten',  # Net written premiums
        ],

        # Insurance Claims and Losses
        'claims_incurred': [
            'PolicyholderBenefitsAndClaimsIncurred',  # Primary - total claims incurred
            'PolicyholderBenefitsAndClaimsIncurredNet',  # Net of reinsurance
            'PolicyholderBenefits',  # Life insurance benefits
            'BenefitsLossesAndExpenses',  # Combined benefits and losses
            'InsuranceClaimsAndLosses',  # Alternative naming
            'LiabilityForClaimsAndClaimsAdjustmentExpense',  # Reserve-based
            'PolicyholderBenefitsAndClaimsIncurredGross',  # Gross before reinsurance
            'IncurredClaimsPropertyCasualtyAndLiability',  # P&C specific
            'BenefitsAndExpenses',  # Life insurance
            'InsuranceLossesAndLossAdjustmentExpense',  # Loss adjustment expenses
        ],

        # Policy Acquisition Costs (LAE and Underwriting Expenses)
        'policy_acquisition_costs': [
            'DeferredPolicyAcquisitionCostAmortizationExpense',  # DAC amortization
            'PolicyAcquisitionCosts',  # Direct acquisition costs
            'DeferredPolicyAcquisitionCosts',  # Deferred costs (balance sheet)
            'AmortizationOfDeferredPolicyAcquisitionCosts',  # Amortization expense
            'OtherUnderwritingExpensesIncurred',  # Other underwriting expenses
            'UnderwritingExpenses',  # General underwriting
        ],

        # Insurance Operating Expenses
        'insurance_operating_expenses': [
            'OperatingCostsAndExpenses',  # General operating
            'GeneralAndAdministrativeExpense',  # G&A
            'OtherExpenses',  # Other expenses
            'InsuranceCommissions',  # Agent/broker commissions
            'InsuranceCommissionsAndFees',  # Commission and fees
        ],

        # Insurance Reserves (Balance Sheet)
        'loss_reserves': [
            'LiabilityForUnpaidClaimsAndClaimsAdjustmentExpense',  # Claims reserve
            'LiabilityForFuturePolicyBenefits',  # Future policy benefits (life)
            'PolicyholderContractDeposits',  # Contract deposits
            'UnearnedPremiums',  # Unearned premium reserve
            'LossAndLossAdjustmentExpenseReserve',  # Loss reserves
        ],

        # Reinsurance
        'reinsurance_recoverables': [
            'ReinsuranceRecoverables',  # Amounts recoverable from reinsurers
            'ReinsuranceRecoverablesOnPaidAndUnpaidLosses',  # Specific breakdown
            'DeferredReinsurancePremiumsAssumed',  # Assumed reinsurance
            'PrepaidReinsurancePremiums',  # Prepaid to reinsurers
        ],

        # Investment Income (Insurance Float)
        'insurance_investment_income': [
            'NetInvestmentIncome',  # Primary - net investment income
            'InvestmentIncomeInterest',  # Interest income
            'InvestmentIncomeInterestAndDividend',  # Interest and dividends
            'GainLossOnInvestments',  # Investment gains/losses
            'RealizedInvestmentGainsLosses',  # Realized gains/losses
        ],

        # ====================
        # SHARES OUTSTANDING
        # ====================

        'shares_outstanding': [
            'EntityCommonStockSharesOutstanding',  # DEI namespace - most authoritative (actual shares, not millions)
            'CommonStockSharesOutstanding',  # All FAANG - us-gaap namespace
            'WeightedAverageNumberOfSharesOutstandingBasic',  # For EPS calc - fallback
            'SharesOutstanding',  # Alternative naming
        ],

        'shares_issued': [
            'CommonStockSharesIssued',  # All FAANG
        ],

        'weighted_average_shares_basic': [
            'WeightedAverageNumberOfSharesOutstandingBasic',
        ],

        'weighted_average_shares_diluted': [
            'WeightedAverageNumberOfDilutedSharesOutstanding',
        ],

        # ====================
        # DEFENSE CONTRACTOR / BACKLOG METRICS (P2-B)
        # ====================
        # Defense contractors have multi-year contract visibility that drives revenue
        # Backlog metrics indicate future revenue certainty

        'order_backlog': [
            'OrderBacklog',  # Primary - total orders not yet delivered
            'BacklogOfOrders',  # Alternative naming
            'UnfulfilledOrders',  # Some companies use this
            'ContractBacklog',  # Contract-specific backlog
        ],

        'contract_liability': [
            'ContractWithCustomerLiability',  # Primary ASC 606 - advances received
            'ContractWithCustomerLiabilityCurrent',  # Current portion
            'ContractWithCustomerLiabilityNoncurrent',  # Non-current portion
            'ContractLiability',  # Simplified form
        ],

        'deferred_revenue_backlog': [
            'DeferredRevenue',  # Total deferred revenue
            'DeferredRevenueCurrent',  # Current portion
            'DeferredRevenueNoncurrent',  # Non-current portion
            'UnearnedRevenue',  # Alternative naming
            'DeferredIncome',  # Some companies use this
        ],

        'unbilled_contracts_receivable': [
            'UnbilledContractsReceivable',  # Primary - unbilled work performed
            'UnbilledReceivables',  # Simplified
            'ContractWithCustomerAssetNetCurrent',  # ASC 606 contract assets
            'UnbilledRevenue',  # Alternative
            'ContractReceivablesUnbilled',  # Some defense contractors
        ],

        'contract_assets': [
            'ContractWithCustomerAsset',  # Primary ASC 606
            'ContractWithCustomerAssetNet',  # Net of allowances
            'ContractWithCustomerAssetNetCurrent',  # Current portion
            'ContractWithCustomerAssetNetNoncurrent',  # Non-current portion
            'ContractAssets',  # Simplified form
        ],

        # Government contract-specific metrics
        'funded_backlog': [
            'FundedBacklog',  # Government contracts with approved funding
            'ContractBacklogFunded',  # Alternative
        ],

        'unfunded_backlog': [
            'UnfundedBacklog',  # Government contracts awaiting funding
            'ContractBacklogUnfunded',  # Alternative
        ],

        # Contract-related costs
        'contract_costs': [
            'ContractCosts',  # Costs incurred on contracts
            'ContractWithCustomerCost',  # ASC 606 contract costs
            'CostOfContractRevenue',  # Cost of goods sold on contracts
        ],

        # ====================
        # AUTO MANUFACTURING METRICS
        # ====================
        # Auto manufacturing industry-specific tags for EV transition analysis
        # Used by dynamic_model_weighting.py for tier classification

        'ev_sales_mix_pct': [
            'ElectricVehicleRevenuePct',  # Direct EV percentage if reported
            'EVtoTotalRevenueRatio',  # Alternative naming
            'ElectricVehicleSalesPercentage',  # Percentage of EV sales
        ],

        'vehicle_unit_sales': [
            'UnitsSold',  # Generic units sold
            'VehicleUnitsSold',  # Vehicle-specific
            'UnitsShipped',  # Alternative: shipped vs sold
            'VehicleDeliveries',  # Deliveries terminology (Tesla uses this)
        ],

        'average_vehicle_selling_price': [
            'AverageVehicleSellingPrice',  # Direct ASP if reported
            'MeanTransactionPrice',  # Alternative naming
            'AverageSellingPricePerUnit',  # Per-unit ASP
        ],

        'warranty_reserve_ratio': [
            'WarrantyReserveRatio',  # Ratio if directly reported
            'AccrualForWarranties',  # Warranty accrual (balance sheet)
            'ProductWarrantyAccrual',  # Product warranty reserve
        ],

        'dealer_inventory_days': [
            'DealerInventoryDays',  # Days inventory outstanding at dealers
            'VehicleInventoryDaysOutstanding',  # Alternative naming
        ],

        # ====================
        # BANK-SPECIFIC METRICS
        # ====================
        # Bank-specific metrics for P/B valuation
        # Used by bank_valuation.py for ROE-based P/B determination

        # Net Interest Margin (NIM)
        # NIM = (Interest Income - Interest Expense) / Average Earning Assets
        'net_interest_margin': [
            'NetInterestMargin',  # Primary - if reported directly
            'InterestMarginNet',  # Alternative naming
            'NetInterestIncomeAsPercentOfAverageEarningAssets',  # Detailed naming
        ],

        # Tier 1 Capital Ratio
        # Tier 1 Capital / Risk-Weighted Assets
        'tier_1_capital_ratio': [
            'Tier1CapitalRatio',  # Primary - reported by most banks
            'CoreCapitalRatio',  # Alternative naming (Core = Tier 1)
            'Tier1CapitalToRiskWeightedAssets',  # Detailed naming
            'Tier1RiskBasedCapitalRatio',  # Some banks use this
            'CommonEquityTier1CapitalRatio',  # CET1 ratio (subset of Tier 1)
        ],

        # Efficiency Ratio
        # Non-Interest Expense / (Net Interest Income + Non-Interest Income)
        'efficiency_ratio': [
            'EfficiencyRatio',  # Primary - if reported directly
            'OperatingExpenseToOperatingIncome',  # Alternative calculation
            'NoninterestExpenseToOperatingIncome',  # Bank-specific naming
        ],

        # Non-Performing Loans (NPL) Ratio
        # Non-Performing Loans / Total Loans
        'npl_ratio': [
            'NonPerformingLoansRatio',  # Primary - if reported directly
            'NonAccruingLoansRatio',  # Alternative naming (non-accrual = non-performing)
            'NonPerformingAssetsToTotalAssets',  # Broader measure including other assets
            'NonPerformingLoansToTotalLoans',  # Detailed naming
            'NonaccrualLoansToTotalLoans',  # Some banks use this
        ],

        # Loan to Deposit Ratio
        # Total Loans / Total Deposits
        'loan_to_deposit_ratio': [
            'LoanToDepositRatio',  # Primary - if reported directly
            'LoansAndLeasesToDeposits',  # Including leases
            'LoansToDepositsRatio',  # Alternative naming
            'NetLoansToTotalDeposits',  # Net of allowances
        ],

        # Supporting bank metrics for calculations
        'net_interest_income': [
            'InterestIncomeExpenseNet',  # Primary - net interest income
            'NetInterestIncome',  # Alternative naming
            'InterestIncomeNet',  # Simplified
            'InterestAndDividendIncomeOperating',  # Some banks
        ],

        'non_interest_income': [
            'NoninterestIncome',  # Primary
            'FeesAndCommissions',  # Component
            'ServiceChargesOnDepositAccounts',  # Component
            'TradingRevenue',  # Investment banks
        ],

        'non_interest_expense': [
            'NoninterestExpense',  # Primary
            'OtherNoninterestExpense',  # Alternative
            'SalariesAndEmployeeBenefits',  # Component
        ],

        'total_loans': [
            'LoansAndLeasesReceivableNetReportedAmount',  # Primary - net of allowances
            'LoansAndLeasesReceivableGross',  # Gross loans
            'LoansReceivableNet',  # Simplified
            'TotalLoans',  # Alternative naming
            'FinancingReceivable',  # FASB terminology
        ],

        'total_deposits': [
            'Deposits',  # Primary
            'DepositsTotalAmount',  # Detailed naming
            'TotalDeposits',  # Alternative
            'InterestBearingDeposits',  # Subset
            'NoninterestBearingDeposits',  # Subset
        ],

        'non_performing_loans': [
            'NonaccrualLoans',  # Primary - loans not accruing interest
            'NonPerformingLoans',  # Direct naming
            'LoansAndLeasesReceivableNonaccrual',  # Detailed
            'ImpairedLoans',  # Alternative classification
        ],

        'allowance_for_loan_losses': [
            'AllowanceForLoanAndLeaseLossesRealEstate',  # Real estate specific
            'AllowanceForCreditLosses',  # Current terminology (CECL)
            'AllowanceForLoanAndLeaseLosses',  # Legacy terminology
            'FinancingReceivableAllowanceForCreditLosses',  # FASB terminology
        ],

        'provision_for_loan_losses': [
            'ProvisionForLoanAndLeaseLosses',  # Primary
            'ProvisionForCreditLosses',  # Current terminology
            'ProvisionForLoanLossesExpensed',  # Expensed portion
        ],

        'tier_1_capital': [
            'Tier1Capital',  # Primary
            'CoreCapital',  # Alternative naming
            'CommonEquityTier1Capital',  # CET1 component
        ],

        'risk_weighted_assets': [
            'RiskWeightedAssets',  # Primary
            'TotalRiskWeightedAssets',  # Alternative naming
        ],

        # ====================
        # SEMICONDUCTOR METRICS (P2-C)
        # ====================
        # Semiconductor companies require cycle-aware valuation
        # These metrics help detect cycle position (peak/trough)

        'inventory_days': [
            'DaysInventoryOutstanding',  # Primary - days inventory on hand
            'InventoryDaysOfSalesOutstanding',  # Alternative naming
            'InventoryTurnoverDays',  # Some companies use this
        ],

        'book_to_bill_ratio': [
            'BookToOrderRatio',  # Primary - orders vs shipments ratio
            'OrderBacklogToCurrentRevenue',  # Backlog-based calculation
            'BookToShipRatio',  # Alternative naming
        ],

        'inventory_to_sales': [
            'InventoryToRevenue',  # Primary - inventory as % of revenue
            'InventoryToSalesRatio',  # Alternative naming
        ],
    }

    # Reverse mapping: XBRL tag → canonical name
    REVERSE_MAP: Dict[str, str] = {}

    def __init__(self):
        """Initialize reverse mapping for fast lookups."""
        self._build_reverse_map()

    def _build_reverse_map(self):
        """Build reverse mapping from XBRL tags to canonical names."""
        self.REVERSE_MAP.clear()
        for canonical, aliases in self.TAG_ALIASES.items():
            for alias in aliases:
                if alias in self.REVERSE_MAP:
                    # Tag maps to multiple canonical names - keep first (highest priority)
                    logger.debug(f"Duplicate XBRL tag '{alias}' maps to both '{self.REVERSE_MAP[alias]}' and '{canonical}', keeping first")
                else:
                    self.REVERSE_MAP[alias] = canonical

    def get_xbrl_aliases(self, canonical_name: str) -> List[str]:
        """
        Get all XBRL tag aliases for a canonical metric name.

        Args:
            canonical_name: Snake_case canonical name (e.g., 'total_revenue')

        Returns:
            List of XBRL tag aliases in priority order

        Example:
            >>> mapper.get_xbrl_aliases('total_revenue')
            ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet']
        """
        return self.TAG_ALIASES.get(canonical_name, [])

    def resolve_to_canonical(self, xbrl_tag: str) -> Optional[str]:
        """
        Resolve an XBRL tag to its canonical snake_case name.

        Args:
            xbrl_tag: XBRL tag (e.g., 'Revenues')

        Returns:
            Canonical snake_case name or None if not found

        Example:
            >>> mapper.resolve_to_canonical('Revenues')
            'total_revenue'
        """
        return self.REVERSE_MAP.get(xbrl_tag)

    def extract_value_with_fallbacks(
        self,
        data: Dict[str, Any],
        canonical_name: str,
        default: Any = None
    ) -> Any:
        """
        Extract value from data dict trying all XBRL tag aliases in priority order.

        Args:
            data: Dictionary with XBRL tags as keys
            canonical_name: Canonical metric name to extract
            default: Default value if no alias found

        Returns:
            First matching value or default

        Example:
            >>> data = {'Revenues': 100000, 'NetIncomeLoss': 20000}
            >>> mapper.extract_value_with_fallbacks(data, 'total_revenue')
            100000
        """
        aliases = self.get_xbrl_aliases(canonical_name)
        for alias in aliases:
            if alias in data and data[alias] is not None:
                logger.debug(f"Resolved '{canonical_name}' using XBRL tag '{alias}' → value: {data[alias]}")
                return data[alias]

        logger.warning(f"No value found for '{canonical_name}' (tried {len(aliases)} aliases: {aliases[:3]}...)")
        return default

    def normalize_xbrl_dict(
        self,
        xbrl_data: Dict[str, Any],
        include_unmatched: bool = False
    ) -> Dict[str, Any]:
        """
        Convert XBRL tag dictionary to canonical snake_case dictionary.

        Args:
            xbrl_data: Dict with XBRL tags as keys (e.g., {'Revenues': 100000})
            include_unmatched: If True, include XBRL tags that don't map to canonical names

        Returns:
            Dict with canonical snake_case keys (e.g., {'total_revenue': 100000})

        Example:
            >>> xbrl_data = {'Revenues': 100000, 'NetIncomeLoss': 20000}
            >>> mapper.normalize_xbrl_dict(xbrl_data)
            {'total_revenue': 100000, 'net_income': 20000}
        """
        normalized = {}

        for xbrl_tag, value in xbrl_data.items():
            canonical = self.resolve_to_canonical(xbrl_tag)
            if canonical:
                # Only keep if not already set (priority to first match)
                if canonical not in normalized:
                    normalized[canonical] = value
            elif include_unmatched:
                # Keep original XBRL tag if no mapping found
                normalized[xbrl_tag] = value

        return normalized

    def get_all_canonical_names(self) -> List[str]:
        """Get list of all canonical metric names."""
        return list(self.TAG_ALIASES.keys())

    def get_coverage_stats(self, xbrl_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze coverage of canonical metrics in provided XBRL data.

        Args:
            xbrl_data: Dict with XBRL tags as keys

        Returns:
            Dict with coverage statistics

        Example:
            >>> stats = mapper.get_coverage_stats(xbrl_data)
            >>> print(f"Coverage: {stats['coverage_pct']:.1f}%")
        """
        all_canonical = self.get_all_canonical_names()
        found_canonical = set()

        for xbrl_tag in xbrl_data.keys():
            canonical = self.resolve_to_canonical(xbrl_tag)
            if canonical:
                found_canonical.add(canonical)

        missing = set(all_canonical) - found_canonical

        return {
            'total_canonical_metrics': len(all_canonical),
            'found_metrics': len(found_canonical),
            'missing_metrics': len(missing),
            'coverage_pct': (len(found_canonical) / len(all_canonical)) * 100 if all_canonical else 0,
            'found_list': sorted(found_canonical),
            'missing_list': sorted(missing),
        }


# Global singleton instance
_mapper_instance: Optional[XBRLTagAliasMapper] = None


def get_tag_mapper() -> XBRLTagAliasMapper:
    """Get singleton instance of XBRLTagAliasMapper."""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = XBRLTagAliasMapper()
    return _mapper_instance


# Convenience functions for quick access
def get_xbrl_aliases(canonical_name: str) -> List[str]:
    """Get XBRL tag aliases for a canonical metric name."""
    return get_tag_mapper().get_xbrl_aliases(canonical_name)


def resolve_to_canonical(xbrl_tag: str) -> Optional[str]:
    """Resolve XBRL tag to canonical name."""
    return get_tag_mapper().resolve_to_canonical(xbrl_tag)


def extract_with_fallbacks(data: Dict[str, Any], canonical_name: str, default: Any = None) -> Any:
    """Extract value from data trying all XBRL tag aliases."""
    return get_tag_mapper().extract_value_with_fallbacks(data, canonical_name, default)


def normalize_xbrl_dict(xbrl_data: Dict[str, Any], include_unmatched: bool = False) -> Dict[str, Any]:
    """Convert XBRL dict to canonical snake_case dict."""
    return get_tag_mapper().normalize_xbrl_dict(xbrl_data, include_unmatched)
