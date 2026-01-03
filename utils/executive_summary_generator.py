#!/usr/bin/env python3
"""
Executive Summary Generator

Reduces 4.5MB analysis JSON to concise executive summary (~50KB) for PDF generation.
Uses LLM to synthesize key insights from full analysis.

Usage:
    from utils.executive_summary_generator import ExecutiveSummaryGenerator

    generator = ExecutiveSummaryGenerator(ollama_client)
    summary = await generator.generate_summary(full_analysis_json)

    # Save summary for PDF generation
    with open('results/AAPL_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import psycopg2
from psycopg2.extras import Json


class ExecutiveSummaryGenerator:
    """Generate concise executive summaries from full analysis JSON"""

    def __init__(self, ollama_client, logger: Optional[logging.Logger] = None):
        """
        Initialize summary generator

        Args:
            ollama_client: Ollama client for LLM synthesis
            logger: Optional logger instance
        """
        self.ollama = ollama_client
        self.logger = logger or logging.getLogger(__name__)

        # Summary size targets (in tokens)
        self.MAX_SUMMARY_TOKENS = 2000  # ~1.5 pages
        self.MAX_INPUT_TOKENS = 16000   # Model context window

    async def generate_summary(self, full_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate executive summary from full analysis

        Args:
            full_analysis: Complete analysis JSON (4.5MB)

        Returns:
            Concise executive summary dict (~50KB)
        """
        symbol = full_analysis.get('symbol', 'UNKNOWN')
        self.logger.info(f"Generating executive summary for {symbol}")

        try:
            # Step 1: Extract key data points (no LLM)
            extracted_data = self._extract_key_data(full_analysis)

            # Step 2: Synthesize with LLM
            llm_summary = await self._synthesize_with_llm(symbol, extracted_data)

            # Step 3: Combine into final summary
            executive_summary = {
                'symbol': symbol,
                'generated_at': datetime.now().isoformat(),
                'summary_version': '1.0',
                'original_size_mb': self._calculate_json_size(full_analysis),
                'summary_size_kb': 0,  # Will be calculated after serialization

                # Executive summary (LLM-generated)
                'executive_summary': llm_summary.get('executive_summary', 'N/A'),

                # Investment recommendation
                'recommendation': {
                    'action': llm_summary.get('recommendation', 'HOLD'),
                    'conviction': llm_summary.get('conviction', 'MEDIUM'),
                    'price_target': llm_summary.get('price_target', 0),
                    'current_price': extracted_data.get('current_price', 0),
                    'upside_potential': llm_summary.get('upside_potential', 0),
                    'time_horizon': llm_summary.get('time_horizon', '12 months'),
                },

                # Key metrics (extracted)
                'key_metrics': extracted_data.get('key_metrics', {}),

                # Investment thesis (LLM-generated)
                'investment_thesis': {
                    'bull_case': llm_summary.get('bull_case', []),
                    'bear_case': llm_summary.get('bear_case', []),
                    'catalysts': llm_summary.get('catalysts', []),
                    'risks': llm_summary.get('risks', []),
                },

                # Valuation summary
                'valuation': extracted_data.get('valuation', {}),

                # Financial health (extracted)
                'financial_health': extracted_data.get('financial_health', {}),

                # Technical summary (extracted)
                'technical_summary': extracted_data.get('technical_summary', {}),

                # Metadata
                'metadata': {
                    'analysis_date': full_analysis.get('completed_at', datetime.now().isoformat()),
                    'analysis_mode': full_analysis.get('mode', 'standard'),
                    'agents_used': list(full_analysis.get('agents', {}).keys()),
                    'data_quality': extracted_data.get('data_quality', 'good'),
                }
            }

            # Calculate summary size
            summary_json = json.dumps(executive_summary)
            executive_summary['summary_size_kb'] = round(len(summary_json) / 1024, 2)

            self.logger.info(
                f"Executive summary generated for {symbol}: "
                f"{executive_summary['original_size_mb']:.2f}MB → "
                f"{executive_summary['summary_size_kb']:.2f}KB "
                f"({100 * executive_summary['summary_size_kb'] / (executive_summary['original_size_mb'] * 1024):.1f}% of original)"
            )

            return executive_summary

        except Exception as e:
            self.logger.error(f"Failed to generate executive summary for {symbol}: {e}")
            raise

    def _extract_key_data(self, full_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key data points from full analysis (no LLM)

        This is a deterministic extraction of critical metrics.
        """
        agents = full_analysis.get('agents', {})

        extracted = {
            'current_price': 0,
            'key_metrics': {},
            'valuation': {},
            'financial_health': {},
            'technical_summary': {},
            'data_quality': 'unknown',
            'investment_thesis': {}
        }

        # Extract from fundamental agent - CORRECTED PATHS
        fundamental = agents.get('fundamental', {})
        if fundamental.get('status') == 'success':
            # Navigate to the LLM response structure
            fund_response = fundamental.get('analysis', {}).get('response', {})
            company_data = fund_response.get('company_data', {})
            ratios = fund_response.get('ratios', {})
            health = fund_response.get('health_analysis', {})
            growth = fund_response.get('growth_analysis', {})

            # Key metrics from company_data and ratios
            extracted['key_metrics'] = {
                'market_cap': company_data.get('market_cap', 0),
                'revenue': company_data.get('revenue', 0),
                'net_income': company_data.get('net_income', 0),
                'total_assets': company_data.get('total_assets', 0),
                'total_equity': company_data.get('total_equity', 0),
                'pe_ratio': ratios.get('pe_ratio', 0),
                'roe': ratios.get('roe', 0),  # Already in %
                'roa': ratios.get('roa', 0),  # Already in %
                'debt_to_equity': ratios.get('debt_to_equity', 0),
                'free_cash_flow': ratios.get('free_cash_flow', 0),
                'eps': ratios.get('eps', 0),
                'gross_margin': ratios.get('gross_margin', 0),
                'operating_margin': ratios.get('operating_margin', 0),
                'net_margin': ratios.get('net_margin', 0),
            }

            # Try to get current price from company_data (fallback if technical not available)
            current_price_from_fund = company_data.get('price', 0)

            # Valuation - extract from professional DCF and GGM
            # Navigate to valuation_analysis → valuation_methods
            valuation_analysis = fund_response.get('valuation_analysis', {})
            valuation_methods = valuation_analysis.get('valuation_methods', {})

            # Get DCF results (always present)
            dcf_result = valuation_methods.get('dcf', {})
            dcf_fair_value = dcf_result.get('fair_value_per_share', 0)
            dcf_upside = dcf_result.get('upside_downside_pct', 0)

            # Get GGM results (only for dividend stocks with ≥20% payout)
            ggm_result = valuation_methods.get('ggm', {})
            ggm_applicable = ggm_result.get('applicable', False)
            ggm_fair_value = ggm_result.get('fair_value_per_share', 0) if ggm_applicable else 0
            ggm_upside = ggm_result.get('upside_downside_pct', 0) if ggm_applicable else 0

            # Primary fair value: Use GGM if applicable (dividend stocks), otherwise DCF
            primary_fair_value = ggm_fair_value if ggm_applicable and ggm_fair_value > 0 else dcf_fair_value
            primary_method = 'Gordon Growth Model' if (ggm_applicable and ggm_fair_value > 0) else 'DCF'
            primary_upside = ggm_upside if (ggm_applicable and ggm_fair_value > 0) else dcf_upside

            # Fallback to legacy valuation if professional valuations not available
            if not primary_fair_value:
                if isinstance(health, dict):
                    primary_fair_value = health.get('fair_value', 0) or health.get('intrinsic_value', 0)
                if not primary_fair_value and isinstance(growth, dict):
                    primary_fair_value = growth.get('fair_value', 0) or growth.get('target_price', 0)
                primary_method = 'Legacy'

            # Store preliminary valuation (will be updated with technical price if available)
            extracted['valuation'] = {
                'fair_value': primary_fair_value,
                'valuation_method': primary_method,
                'upside_potential': primary_upside,
                'current_price': current_price_from_fund,  # Will be overridden by technical
                'discount_premium': 0,  # Will calculate after getting technical price
                # Store both DCF and GGM for reference
                'dcf_fair_value': dcf_fair_value,
                'dcf_upside': dcf_upside,
                'ggm_fair_value': ggm_fair_value if ggm_applicable else None,
                'ggm_upside': ggm_upside if ggm_applicable else None,
                'ggm_applicable': ggm_applicable,
            }

            # Store fallback current_price
            extracted['current_price'] = current_price_from_fund

            # Financial health
            health_score = 50.0  # default
            if isinstance(health, dict):
                health_score = health.get('quality_score', health.get('health_score', 50.0))

            extracted['financial_health'] = {
                'quality_score': health_score,
                'data_quality': fundamental.get('data_quality', {}),
                'confidence': fundamental.get('confidence', {}),
            }
            extracted['data_quality'] = fundamental.get('data_quality', {})

        # Extract from technical agent - CORRECTED PATHS
        technical = agents.get('technical', {})
        if technical.get('status') == 'success':
            # Navigate to the LLM response structure
            tech_response = technical.get('analysis', {}).get('response', {})

            # Get current price from technical
            current_price = tech_response.get('current_price', 0)

            # Get support/resistance from nested structure
            support_resistance = tech_response.get('support_resistance', {})
            support = support_resistance.get('support_1', 0)
            resistance = support_resistance.get('resistance_1', 0)

            # Get RSI from indicators
            indicators = tech_response.get('indicators', {})
            rsi = indicators.get('rsi', 50)

            extracted['technical_summary'] = {
                'trend': tech_response.get('trend', 'neutral'),
                'recommendation': tech_response.get('recommendation', 'hold'),
                'current_price': current_price,
                'support': support,
                'resistance': resistance,
                'rsi': rsi,
            }

            # Store current_price at top level for valuation calculations
            extracted['current_price'] = current_price

            # Update valuation with better current_price from technical
            if current_price and 'valuation' in extracted:
                extracted['valuation']['current_price'] = current_price
                fair_value = extracted['valuation'].get('fair_value', 0)
                if fair_value and current_price:
                    extracted['valuation']['discount_premium'] = (
                        (fair_value - current_price) / current_price * 100
                    )

        # Extract from synthesis agent - CORRECTED PATHS
        synthesis = agents.get('synthesis', {})
        if synthesis.get('status') == 'success':
            # Get recommendation from top-level synthesis keys
            if 'recommendation' in synthesis:
                rec_response = synthesis['recommendation'].get('response', {})
                extracted['synthesis_recommendation'] = {
                    'action': rec_response.get('final_recommendation', 'hold'),
                    'conviction': rec_response.get('conviction_level', 'medium'),
                    'confidence': synthesis.get('confidence', 50),
                    'key_reasons': rec_response.get('key_reasons', []),
                    'main_risks': rec_response.get('main_risks', []),
                }

                # Populate investment thesis from synthesis
                extracted['investment_thesis'] = {
                    'bull_case': rec_response.get('key_reasons', []),
                    'bear_case': rec_response.get('main_risks', []),
                    'catalysts': [],  # TODO: Find where catalysts are stored
                    'risks': rec_response.get('main_risks', [])
                }

        return extracted

    async def _synthesize_with_llm(self, symbol: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to synthesize executive summary from extracted data

        This generates narrative insights from the numerical data.
        """
        # Prepare concise prompt (fit within token limits)
        prompt = f"""
Generate an executive investment summary for {symbol}:

KEY METRICS:
- Market Cap: ${extracted_data['key_metrics'].get('market_cap', 0):,.0f}
- Revenue: ${extracted_data['key_metrics'].get('revenue', 0):,.0f}
- Net Income: ${extracted_data['key_metrics'].get('net_income', 0):,.0f}
- P/E Ratio: {extracted_data['key_metrics'].get('pe_ratio', 0):.2f}
- ROE: {extracted_data['key_metrics'].get('roe', 0):.2f}%
- Debt/Equity: {extracted_data['key_metrics'].get('debt_to_equity', 0):.2f}

VALUATION:
- Current Price: ${extracted_data.get('current_price', 0):.2f}
- Primary Fair Value ({extracted_data['valuation'].get('valuation_method', 'N/A')}): ${extracted_data['valuation'].get('fair_value', 0):.2f}
- Upside Potential: {extracted_data['valuation'].get('upside_potential', 0):.1f}%
- DCF Fair Value: ${extracted_data['valuation'].get('dcf_fair_value', 0):.2f} (Upside: {extracted_data['valuation'].get('dcf_upside', 0):.1f}%)
{f"- GGM Fair Value: ${extracted_data['valuation'].get('ggm_fair_value', 0):.2f} (Upside: {extracted_data['valuation'].get('ggm_upside', 0):.1f}%)" if extracted_data['valuation'].get('ggm_applicable') else "- GGM: Not applicable (non-dividend or low payout <20%)"}

TECHNICAL:
- Trend: {extracted_data['technical_summary'].get('trend', 'neutral')}
- Recommendation: {extracted_data['technical_summary'].get('recommendation', 'hold')}
- RSI: {extracted_data['technical_summary'].get('rsi', 50):.1f}

QUALITY:
- Quality Score: {extracted_data['financial_health'].get('quality_score', 50)}/100
- Data Quality: {extracted_data['financial_health'].get('data_quality', 'unknown')}
- Confidence: {extracted_data['financial_health'].get('confidence', 50)}%

Generate a concise executive summary with:
1. Executive Summary (3-4 sentences)
2. Investment Recommendation (BUY/HOLD/SELL)
3. Conviction Level (HIGH/MEDIUM/LOW)
4. Price Target (12-month)
5. Upside Potential (%)
6. Bull Case (3 key points)
7. Bear Case (3 key points)
8. Top 3 Catalysts
9. Top 3 Risks

Return as structured JSON with these exact keys.
"""

        try:
            response = await self.ollama.generate(
                model='deepseek-r1:32b',  # Use reasoning model for synthesis
                prompt=prompt,
                system="You are an institutional investment analyst creating executive summaries for portfolio managers.",
                format="json",
                options={
                    'temperature': 0.3,  # Low temperature for consistency
                    'top_p': 0.9,
                }
            )

            # Parse response
            if isinstance(response, str):
                summary = json.loads(response)
            elif isinstance(response, dict):
                summary = response
            else:
                raise ValueError(f"Unexpected LLM response type: {type(response)}")

            return summary

        except Exception as e:
            self.logger.error(f"LLM synthesis failed for {symbol}: {e}")

            # Fallback to rule-based summary
            return self._generate_fallback_summary(symbol, extracted_data)

    def _generate_fallback_summary(self, symbol: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback summary if LLM fails"""

        # Simple rule-based recommendation
        quality_score = extracted_data['financial_health'].get('quality_score', 50)
        discount_premium = extracted_data['valuation'].get('discount_premium', 0)

        if quality_score > 70 and discount_premium > 15:
            recommendation = 'BUY'
            conviction = 'HIGH'
        elif quality_score > 60 and discount_premium > 0:
            recommendation = 'BUY'
            conviction = 'MEDIUM'
        elif quality_score < 40 or discount_premium < -20:
            recommendation = 'SELL'
            conviction = 'MEDIUM'
        else:
            recommendation = 'HOLD'
            conviction = 'MEDIUM'

        current_price = extracted_data.get('current_price', 0)
        fair_value = extracted_data['valuation'].get('fair_value', current_price)

        return {
            'executive_summary': f"{symbol} shows {'strong' if quality_score > 70 else 'moderate' if quality_score > 50 else 'weak'} fundamentals with quality score of {quality_score}/100. Currently trading at {'a discount' if discount_premium > 0 else 'a premium'} to fair value.",
            'recommendation': recommendation,
            'conviction': conviction,
            'price_target': fair_value,
            'upside_potential': discount_premium,
            'time_horizon': '12 months',
            'bull_case': ['Strong fundamentals', 'Attractive valuation', 'Positive trend'],
            'bear_case': ['Market volatility', 'Execution risk', 'Competition'],
            'catalysts': ['Earnings growth', 'Market expansion', 'Product innovation'],
            'risks': ['Economic downturn', 'Regulatory changes', 'Competitive pressure'],
        }

    def _calculate_json_size(self, data: Dict) -> float:
        """Calculate JSON size in MB"""
        json_str = json.dumps(data)
        return len(json_str) / (1024 * 1024)

    def save_to_database(
        self,
        summary: Dict[str, Any],
        db_config: Dict[str, str],
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None
    ) -> bool:
        """
        Save executive summary to quarterly_ai_summaries table

        Args:
            summary: Executive summary dict from generate_summary()
            db_config: Database connection config (host, user, password, database)
            fiscal_year: Fiscal year (e.g., 2024). If None, extracted from metadata
            fiscal_period: Fiscal period (e.g., 'Q3'). If None, extracted from metadata

        Returns:
            True if saved successfully, False otherwise

        Example:
            >>> db_config = {
            ...     'host': '${DB_HOST:-localhost}',
            ...     'user': 'investigator',
            ...     'password': 'investigator',
            ...     'database': 'sec_database'
            ... }
            >>> generator.save_to_database(summary, db_config)
        """
        symbol = summary.get('symbol', 'UNKNOWN')

        try:
            # Import here to avoid dependency if not using database persistence
            from dao.sec_bulk_dao import SECBulkDAO

            # Get CIK for symbol
            dao = SECBulkDAO(db_config)
            cik = dao.get_cik(symbol)

            if not cik:
                self.logger.error(f"Cannot save summary to database: CIK not found for {symbol}")
                return False

            # Extract fiscal period from metadata if not provided
            if not fiscal_year or not fiscal_period:
                analysis_date = summary.get('metadata', {}).get('analysis_date', '')
                # Try to parse fiscal period from analysis date or determine from current date
                # For now, use current date if not provided
                from datetime import datetime
                now = datetime.now()
                fiscal_year = fiscal_year or now.year
                fiscal_period = fiscal_period or f"Q{(now.month - 1) // 3 + 1}"

            # Prepare financial summary text (for backward compatibility)
            exec_summary_text = summary.get('executive_summary', 'N/A')
            recommendation = summary.get('recommendation', {})

            financial_summary = f"""
Executive Summary: {exec_summary_text}

Recommendation: {recommendation.get('action', 'HOLD')}
Conviction: {recommendation.get('conviction', 'MEDIUM')}
Price Target: ${recommendation.get('price_target', 0):.2f}
Current Price: ${recommendation.get('current_price', 0):.2f}
Upside Potential: {recommendation.get('upside_potential', 0):.1f}%
Time Horizon: {recommendation.get('time_horizon', '12 months')}
""".strip()

            # Prepare scores dict
            scores = {
                'key_metrics': summary.get('key_metrics', {}),
                'valuation': summary.get('valuation', {}),
                'financial_health': summary.get('financial_health', {}),
                'technical_summary': summary.get('technical_summary', {})
            }

            # Normalize db_config for psycopg2 (expects 'user', not 'username')
            psycopg2_config = db_config.copy()
            if 'username' in psycopg2_config:
                psycopg2_config['user'] = psycopg2_config.pop('username')

            # Connect to database
            conn = psycopg2.connect(**psycopg2_config)
            cur = conn.cursor()

            # Insert or update summary
            upsert_sql = """
                INSERT INTO quarterly_ai_summaries (
                    cik, ticker, fiscal_year, fiscal_period, form_type,
                    financial_summary, ai_analysis, scores, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (cik, fiscal_year, fiscal_period)
                DO UPDATE SET
                    financial_summary = EXCLUDED.financial_summary,
                    ai_analysis = EXCLUDED.ai_analysis,
                    scores = EXCLUDED.scores,
                    created_at = NOW()
            """

            cur.execute(upsert_sql, (
                cik,
                symbol,
                fiscal_year,
                fiscal_period,
                '10-Q' if fiscal_period.startswith('Q') else '10-K',
                financial_summary,
                Json(summary),  # Full executive summary as JSONB
                Json(scores)    # Scores as JSONB
            ))

            conn.commit()
            cur.close()
            conn.close()

            self.logger.info(
                f"✅ Saved executive summary to database for {symbol} "
                f"(fiscal_year={fiscal_year}, fiscal_period={fiscal_period})"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to save summary to database for {symbol}: {e}")
            return False


async def generate_executive_summary_from_file(
    input_file: str,
    output_file: str,
    ollama_client,
    db_config: Optional[Dict[str, str]] = None,
    fiscal_year: Optional[int] = None,
    fiscal_period: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate summary from file

    Args:
        input_file: Path to full analysis JSON (e.g., results/NEE_FINAL_FIX.json)
        output_file: Path to save summary JSON (e.g., results/NEE_summary.json)
        ollama_client: Ollama client instance
        db_config: Optional database config for persistence (host, user, password, database)
        fiscal_year: Optional fiscal year for database persistence
        fiscal_period: Optional fiscal period for database persistence

    Returns:
        Executive summary dict

    Example:
        >>> from core.ollama_client import OllamaClient
        >>> ollama = OllamaClient()
        >>>
        >>> # Basic usage (file only)
        >>> summary = await generate_executive_summary_from_file(
        ...     'results/NEE_FINAL_FIX.json',
        ...     'results/NEE_summary.json',
        ...     ollama
        ... )
        >>>
        >>> # With database persistence
        >>> db_config = {
        ...     'host': '${DB_HOST:-localhost}',
        ...     'user': 'investigator',
        ...     'password': 'investigator',
        ...     'database': 'sec_database'
        ... }
        >>> summary = await generate_executive_summary_from_file(
        ...     'results/NEE_FINAL_FIX.json',
        ...     'results/NEE_summary.json',
        ...     ollama,
        ...     db_config=db_config,
        ...     fiscal_year=2024,
        ...     fiscal_period='Q3'
        ... )
    """
    # Load full analysis
    with open(input_file) as f:
        full_analysis = json.load(f)

    # Generate summary
    generator = ExecutiveSummaryGenerator(ollama_client)
    summary = await generator.generate_summary(full_analysis)

    # Save summary to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Executive summary saved to {output_file}")

    # Save to database if config provided
    if db_config:
        success = generator.save_to_database(
            summary,
            db_config,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period
        )
        if success:
            logging.info("Executive summary saved to database")
        else:
            logging.warning("Failed to save executive summary to database")

    return summary
