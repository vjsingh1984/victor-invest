"""
Insider Trading Analysis Module

Analyzes Form 4 filings to track insider buying/selling patterns
and calculate insider sentiment scores.
"""
import logging
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class InsiderTradingAnalyzer:
    """Analyze insider trading patterns from Form 4 filings"""

    def __init__(self, db_manager=None):
        """
        Initialize insider trading analyzer

        Args:
            db_manager: Database manager (optional)
        """
        self.db_manager = db_manager
        self.sec_base_url = "https://www.sec.gov"

    def analyze_insider_activity(self, symbol: str, days: int = 180) -> Dict:
        """
        Analyze insider trading activity for a symbol

        Args:
            symbol: Stock symbol
            days: Number of days to look back (default 180)

        Returns:
            Dictionary with insider sentiment analysis
        """
        try:
            logger.info(f"Analyzing insider trading for {symbol} over last {days} days")

            # For initial implementation, use simplified approach
            # In production, would fetch real Form 4 data from SEC EDGAR

            # Placeholder for demo - would be replaced with real SEC data
            insider_sentiment = self._calculate_mock_insider_sentiment(symbol)

            return insider_sentiment

        except Exception as e:
            logger.error(f"Error analyzing insider trading for {symbol}: {e}")
            return {}

    def _calculate_mock_insider_sentiment(self, symbol: str) -> Dict:
        """
        Calculate insider sentiment (mock implementation for now)

        In production, this would parse actual Form 4 filings

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with insider sentiment metrics
        """
        # Placeholder implementation
        # In production: fetch_form4_filings() -> parse_transactions() -> calculate_sentiment()

        return {
            'sentiment_score': 5.0,  # Neutral by default
            'sentiment_rating': 'Neutral',
            'buy_count': 0,
            'sell_count': 0,
            'total_buy_value': 0.0,
            'total_sell_value': 0.0,
            'unusual_patterns': [],
            'period_days': 180,
            'data_available': False,  # Mark as not yet implemented
            'note': 'Insider trading analysis requires SEC EDGAR API integration - coming soon'
        }

    def fetch_form4_filings(self, symbol: str, days: int = 180) -> List[Dict]:
        """
        Fetch recent Form 4 filings for symbol

        Args:
            symbol: Stock symbol
            days: Number of days to look back

        Returns:
            List of Form 4 filings
        """
        try:
            # Get CIK for symbol
            cik = self._get_cik(symbol)
            if not cik:
                logger.warning(f"Could not find CIK for {symbol}")
                return []

            # Fetch filing list from SEC EDGAR
            # This is a placeholder - full implementation would parse SEC response
            logger.info(f"Would fetch Form 4 filings for CIK {cik}")

            return []

        except Exception as e:
            logger.error(f"Error fetching Form 4 filings: {e}")
            return []

    def _get_cik(self, symbol: str) -> Optional[str]:
        """
        Get CIK for symbol from database

        Args:
            symbol: Stock symbol

        Returns:
            CIK string or None
        """
        if not self.db_manager:
            return None

        try:
            from sqlalchemy import text

            query = text("SELECT cik FROM ticker_cik_mapping WHERE ticker = :symbol")

            with self.db_manager.get_session() as session:
                result = session.execute(query, {'symbol': symbol}).fetchone()
                if result:
                    return str(result[0]).zfill(10)  # Pad to 10 digits

        except Exception as e:
            logger.error(f"Error getting CIK for {symbol}: {e}")

        return None

    def calculate_insider_sentiment(self, transactions: List[Dict]) -> Dict:
        """
        Calculate insider sentiment score based on buy/sell patterns

        Args:
            transactions: List of parsed transactions

        Returns:
            Dictionary with sentiment metrics
        """
        if not transactions:
            return {
                'sentiment_score': 5.0,
                'sentiment_rating': 'Neutral',
                'buy_count': 0,
                'sell_count': 0,
                'total_buy_value': 0.0,
                'total_sell_value': 0.0,
                'unusual_patterns': []
            }

        # Separate buys and sells
        buys = [t for t in transactions if t.get('transaction_type') == 'Purchase']
        sells = [t for t in transactions if t.get('transaction_type') == 'Sale']

        # Calculate metrics
        total_buy_value = sum(t.get('shares', 0) * t.get('price_per_share', 0) for t in buys)
        total_sell_value = sum(t.get('shares', 0) * t.get('price_per_share', 0) for t in sells)

        buy_count = len(buys)
        sell_count = len(sells)

        # Insider sentiment score (0-10)
        # High score = More buying than selling
        if (buy_count + sell_count) == 0:
            sentiment_score = 5.0  # Neutral if no activity
        else:
            buy_ratio = buy_count / (buy_count + sell_count)
            value_ratio = total_buy_value / (total_buy_value + total_sell_value) if (total_buy_value + total_sell_value) > 0 else 0.5

            # Weight count (60%) and value (40%)
            combined_ratio = buy_ratio * 0.6 + value_ratio * 0.4
            sentiment_score = combined_ratio * 10

        # Detect unusual patterns
        unusual_patterns = self._detect_unusual_patterns(transactions)

        return {
            'sentiment_score': round(sentiment_score, 1),
            'sentiment_rating': self._get_sentiment_rating(sentiment_score),
            'buy_count': buy_count,
            'sell_count': sell_count,
            'total_buy_value': round(total_buy_value, 2),
            'total_sell_value': round(total_sell_value, 2),
            'unusual_patterns': unusual_patterns,
            'period_days': 180
        }

    def _detect_unusual_patterns(self, transactions: List[Dict]) -> List[str]:
        """
        Detect unusual insider trading patterns

        Args:
            transactions: List of transactions

        Returns:
            List of unusual pattern descriptions
        """
        patterns = []

        # Pattern 1: Cluster of insider buys (3+ in 30 days)
        recent_buys = [
            t for t in transactions
            if t.get('transaction_type') == 'Purchase'
            and self._is_recent(t.get('transaction_date', ''), days=30)
        ]

        if len(recent_buys) >= 3:
            patterns.append(f"Cluster of {len(recent_buys)} insider purchases in last 30 days")

        # Pattern 2: Unusually large transaction
        if transactions and len(transactions) >= 2:
            values = [t.get('shares', 0) * t.get('price_per_share', 0) for t in transactions]
            if values:
                # For small samples, use median-based detection to avoid outlier influence
                if len(values) <= 5:
                    sorted_values = sorted(values)
                    median = sorted_values[len(sorted_values) // 2]
                    # If any transaction is 10x the median, it's unusual
                    threshold = median * 10
                    for t in transactions:
                        value = t.get('shares', 0) * t.get('price_per_share', 0)
                        if value > threshold:
                            trans_type = t.get('transaction_type', 'Transaction')
                            patterns.append(f"Unusually large {trans_type.lower()}: ${value:,.0f}")
                else:
                    # For larger samples, use mean + 2 std dev
                    mean_value = sum(values) / len(values)
                    std_dev = (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5

                    for t in transactions:
                        value = t.get('shares', 0) * t.get('price_per_share', 0)
                        if value > mean_value + 2 * std_dev:
                            trans_type = t.get('transaction_type', 'Transaction')
                            patterns.append(f"Unusually large {trans_type.lower()}: ${value:,.0f}")

        return patterns

    def _is_recent(self, date_str: str, days: int = 30) -> bool:
        """
        Check if date is within last N days

        Args:
            date_str: Date string
            days: Number of days

        Returns:
            True if within last N days
        """
        try:
            trans_date = datetime.strptime(date_str, '%Y-%m-%d')
            cutoff = datetime.now() - timedelta(days=days)
            return trans_date >= cutoff
        except:
            return False

    def _get_sentiment_rating(self, score: float) -> str:
        """
        Convert sentiment score to rating

        Args:
            score: Sentiment score 0-10

        Returns:
            Rating string
        """
        if score >= 7.5:
            return 'Very Bullish'
        elif score >= 6.0:
            return 'Bullish'
        elif score >= 4.0:
            return 'Neutral'
        elif score >= 2.5:
            return 'Bearish'
        else:
            return 'Very Bearish'
