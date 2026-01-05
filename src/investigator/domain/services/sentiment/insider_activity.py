# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Insider Activity Sentiment Service.

This module analyzes SEC Form 4 insider trading data to derive sentiment
signals for investment decisions.

Key Features:
- Sentiment scoring based on buy/sell ratio and transaction values
- Cluster detection for coordinated insider activity
- Key insider tracking (C-suite, directors, 10% owners)
- Historical pattern analysis

Sentiment Interpretation:
- Score > 0.3: Bullish (net buying, especially by key insiders)
- Score 0.1-0.3: Slightly Bullish
- Score -0.1-0.1: Neutral
- Score -0.3--0.1: Slightly Bearish
- Score < -0.3: Bearish (net selling)

Example:
    service = get_insider_activity_service()

    # Get sentiment analysis
    sentiment = await service.analyze_sentiment("AAPL", days=90)

    # Check for cluster activity
    clusters = await service.detect_cluster_activity("AAPL", days=30)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SentimentLevel(Enum):
    """Insider sentiment classification levels."""

    BULLISH = "bullish"
    SLIGHTLY_BULLISH = "slightly_bullish"
    NEUTRAL = "neutral"
    SLIGHTLY_BEARISH = "slightly_bearish"
    BEARISH = "bearish"
    NO_DATA = "no_data"


class ClusterType(Enum):
    """Types of coordinated insider activity."""

    BUYING_CLUSTER = "buying_cluster"
    SELLING_CLUSTER = "selling_cluster"
    MIXED_CLUSTER = "mixed_cluster"
    NO_CLUSTER = "no_cluster"


@dataclass
class InsiderSentiment:
    """Comprehensive insider sentiment analysis result.

    Attributes:
        symbol: Stock ticker symbol
        sentiment_score: Score from -1 (bearish) to +1 (bullish)
        sentiment_level: Classified sentiment level
        purchase_count: Number of purchase transactions
        sale_count: Number of sale transactions
        purchase_value: Total value of purchases ($)
        sale_value: Total value of sales ($)
        net_value: Net insider activity value (positive = buying)
        unique_insiders: Number of unique insiders transacting
        key_insider_activity: Transactions by C-suite/directors
        significant_filings: High-value or notable filings
        cluster_detected: Whether coordinated activity detected
        analysis_period_days: Number of days analyzed
        confidence: Confidence in the sentiment signal (0-1)
        warnings: Any data quality warnings
    """

    symbol: str
    sentiment_score: float = 0.0
    sentiment_level: SentimentLevel = SentimentLevel.NO_DATA
    purchase_count: int = 0
    sale_count: int = 0
    purchase_value: float = 0.0
    sale_value: float = 0.0
    net_value: float = 0.0
    unique_insiders: int = 0
    key_insider_activity: int = 0
    significant_filings: int = 0
    cluster_detected: bool = False
    cluster_type: ClusterType = ClusterType.NO_CLUSTER
    analysis_period_days: int = 90
    confidence: float = 0.0
    analysis_date: Optional[date] = None
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.analysis_date is None:
            self.analysis_date = date.today()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "sentiment_score": round(self.sentiment_score, 4),
            "sentiment_level": self.sentiment_level.value,
            "transactions": {
                "purchase_count": self.purchase_count,
                "sale_count": self.sale_count,
                "total_count": self.purchase_count + self.sale_count,
            },
            "values": {
                "purchase_value": self.purchase_value,
                "sale_value": self.sale_value,
                "net_value": self.net_value,
            },
            "insiders": {
                "unique_count": self.unique_insiders,
                "key_insider_activity": self.key_insider_activity,
            },
            "signals": {
                "cluster_detected": self.cluster_detected,
                "cluster_type": self.cluster_type.value,
                "significant_filings": self.significant_filings,
            },
            "metadata": {
                "analysis_period_days": self.analysis_period_days,
                "analysis_date": str(self.analysis_date),
                "confidence": round(self.confidence, 2),
            },
            "warnings": self.warnings,
        }

    @property
    def is_signal(self) -> bool:
        """Whether this represents an actionable signal."""
        return self.confidence >= 0.5 and self.sentiment_level not in (SentimentLevel.NEUTRAL, SentimentLevel.NO_DATA)

    @property
    def signal_strength(self) -> str:
        """Get signal strength description."""
        if self.confidence >= 0.8:
            return "strong"
        elif self.confidence >= 0.5:
            return "moderate"
        elif self.confidence >= 0.3:
            return "weak"
        return "insufficient_data"


@dataclass
class ClusterActivity:
    """Detected cluster of coordinated insider activity.

    Attributes:
        symbol: Stock ticker symbol
        cluster_type: Type of cluster (buying/selling/mixed)
        start_date: First transaction in cluster
        end_date: Last transaction in cluster
        insider_count: Number of insiders participating
        transaction_count: Total transactions in cluster
        total_value: Combined transaction value
        insiders: List of insider names in cluster
        is_significant: Whether cluster meets significance thresholds
    """

    symbol: str
    cluster_type: ClusterType = ClusterType.NO_CLUSTER
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    insider_count: int = 0
    transaction_count: int = 0
    total_value: float = 0.0
    insiders: List[str] = field(default_factory=list)
    is_significant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "cluster_type": self.cluster_type.value,
            "period": {
                "start_date": str(self.start_date) if self.start_date else None,
                "end_date": str(self.end_date) if self.end_date else None,
                "days": (self.end_date - self.start_date).days + 1 if self.start_date and self.end_date else 0,
            },
            "activity": {
                "insider_count": self.insider_count,
                "transaction_count": self.transaction_count,
                "total_value": self.total_value,
            },
            "insiders": self.insiders,
            "is_significant": self.is_significant,
        }


class InsiderActivityService:
    """Service for analyzing insider trading sentiment.

    Provides sentiment analysis based on SEC Form 4 filings,
    including cluster detection and key insider tracking.

    SOLID: Single Responsibility - only handles sentiment analysis
    """

    # Significance thresholds
    CLUSTER_MIN_INSIDERS = 3
    CLUSTER_MIN_VALUE = 500_000
    KEY_INSIDER_WEIGHT = 1.5  # Weight multiplier for C-suite transactions
    HIGH_VALUE_THRESHOLD = 1_000_000

    def __init__(self, dao=None):
        """Initialize service with optional DAO injection.

        Args:
            dao: InsiderTradingDAO instance (created if not provided)
        """
        self._dao = dao

    def _get_dao(self):
        """Lazy-load DAO to avoid circular imports."""
        if self._dao is None:
            from dao.insider_trading_dao import get_insider_trading_dao

            self._dao = get_insider_trading_dao()
        return self._dao

    async def analyze_sentiment(self, symbol: str, days: int = 90) -> InsiderSentiment:
        """Analyze insider sentiment for a symbol.

        Args:
            symbol: Stock ticker symbol
            days: Analysis period in days (default 90)

        Returns:
            InsiderSentiment with analysis results
        """
        try:
            dao = self._get_dao()

            # Get aggregated sentiment data from DAO
            loop = asyncio.get_event_loop()
            raw_sentiment = await loop.run_in_executor(None, dao.get_insider_sentiment, symbol, days)

            # Get key insider transactions
            key_transactions = await loop.run_in_executor(None, dao.get_key_insider_transactions, symbol, days)

            # Build sentiment result
            sentiment = InsiderSentiment(
                symbol=symbol.upper(),
                analysis_period_days=days,
            )

            if raw_sentiment.get("sentiment_label") == "no_data":
                sentiment.warnings.append("No insider activity found in period")
                return sentiment

            # Map raw data to sentiment object
            sentiment.sentiment_score = raw_sentiment.get("sentiment_score", 0)
            sentiment.purchase_count = raw_sentiment.get("purchase_count", 0)
            sentiment.sale_count = raw_sentiment.get("sale_count", 0)
            sentiment.purchase_value = raw_sentiment.get("purchase_value", 0)
            sentiment.sale_value = raw_sentiment.get("sale_value", 0)
            sentiment.net_value = raw_sentiment.get("net_value", 0)
            sentiment.unique_insiders = raw_sentiment.get("unique_insiders", 0)
            sentiment.significant_filings = raw_sentiment.get("significant_filings", 0)
            sentiment.cluster_detected = raw_sentiment.get("cluster_detected", False)

            # Map sentiment label
            label = raw_sentiment.get("sentiment_label", "no_data")
            sentiment.sentiment_level = SentimentLevel(label)

            # Count key insider activity
            sentiment.key_insider_activity = len(key_transactions)

            # Determine cluster type
            if sentiment.cluster_detected:
                if sentiment.purchase_count > sentiment.sale_count * 2:
                    sentiment.cluster_type = ClusterType.BUYING_CLUSTER
                elif sentiment.sale_count > sentiment.purchase_count * 2:
                    sentiment.cluster_type = ClusterType.SELLING_CLUSTER
                else:
                    sentiment.cluster_type = ClusterType.MIXED_CLUSTER

            # Calculate confidence score
            sentiment.confidence = self._calculate_confidence(sentiment, key_transactions)

            return sentiment

        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            sentiment = InsiderSentiment(symbol=symbol.upper(), analysis_period_days=days)
            sentiment.warnings.append(f"Analysis error: {str(e)}")
            return sentiment

    def _calculate_confidence(self, sentiment: InsiderSentiment, key_transactions: List[Dict]) -> float:
        """Calculate confidence score for sentiment signal.

        Confidence is based on:
        - Number of transactions (more = higher confidence)
        - Number of unique insiders (more = higher confidence)
        - Presence of key insider activity (boosts confidence)
        - Value of transactions (higher = higher confidence)
        - Consistency of direction (all buys or all sells = higher)

        Returns:
            Confidence score from 0 to 1
        """
        confidence = 0.0

        total_count = sentiment.purchase_count + sentiment.sale_count

        # Base confidence from transaction count
        if total_count >= 10:
            confidence += 0.25
        elif total_count >= 5:
            confidence += 0.15
        elif total_count >= 2:
            confidence += 0.05

        # Confidence from unique insiders
        if sentiment.unique_insiders >= 5:
            confidence += 0.25
        elif sentiment.unique_insiders >= 3:
            confidence += 0.15
        elif sentiment.unique_insiders >= 2:
            confidence += 0.05

        # Confidence from key insiders
        if len(key_transactions) >= 3:
            confidence += 0.20
        elif len(key_transactions) >= 1:
            confidence += 0.10

        # Confidence from total value
        total_value = sentiment.purchase_value + sentiment.sale_value
        if total_value >= 10_000_000:
            confidence += 0.15
        elif total_value >= 1_000_000:
            confidence += 0.10
        elif total_value >= 100_000:
            confidence += 0.05

        # Confidence from directional consistency
        if total_count > 0:
            buy_ratio = sentiment.purchase_count / total_count
            sell_ratio = sentiment.sale_count / total_count
            # More consistent direction = higher confidence
            consistency = max(buy_ratio, sell_ratio)
            confidence += 0.15 * consistency

        return min(confidence, 1.0)

    async def detect_cluster_activity(self, symbol: str, days: int = 30, window_days: int = 7) -> List[ClusterActivity]:
        """Detect clusters of coordinated insider activity.

        A cluster is defined as multiple insiders transacting
        in the same direction within a short time window.

        Args:
            symbol: Stock ticker symbol
            days: Period to analyze
            window_days: Rolling window for cluster detection

        Returns:
            List of detected ClusterActivity instances
        """
        try:
            dao = self._get_dao()

            # Get recent activity
            loop = asyncio.get_event_loop()
            filings = await loop.run_in_executor(
                None, dao.get_recent_activity, symbol, days, False  # Include all filings, not just significant
            )

            if not filings:
                return []

            clusters = []

            # Group filings by transaction direction
            buys = [f for f in filings if f.get("transaction_code") == "P"]
            sells = [f for f in filings if f.get("transaction_code") == "S"]

            # Detect buying clusters
            buy_cluster = self._find_cluster(symbol, buys, window_days, ClusterType.BUYING_CLUSTER)
            if buy_cluster:
                clusters.append(buy_cluster)

            # Detect selling clusters
            sell_cluster = self._find_cluster(symbol, sells, window_days, ClusterType.SELLING_CLUSTER)
            if sell_cluster:
                clusters.append(sell_cluster)

            return clusters

        except Exception as e:
            logger.error(f"Error detecting clusters for {symbol}: {e}")
            return []

    def _find_cluster(
        self, symbol: str, filings: List[Dict], window_days: int, cluster_type: ClusterType
    ) -> Optional[ClusterActivity]:
        """Find cluster activity in a list of filings.

        Args:
            symbol: Stock ticker
            filings: List of filing dictionaries
            window_days: Rolling window size
            cluster_type: Type of cluster to detect

        Returns:
            ClusterActivity if found, None otherwise
        """
        if len(filings) < self.CLUSTER_MIN_INSIDERS:
            return None

        # Sort by date
        sorted_filings = sorted(filings, key=lambda f: f.get("filing_date", ""))

        # Use sliding window to find clusters
        best_cluster = None
        best_count = 0

        for i, start_filing in enumerate(sorted_filings):
            start_date_str = start_filing.get("filing_date")
            if not start_date_str:
                continue

            try:
                start_date = date.fromisoformat(start_date_str)
            except ValueError:
                continue

            end_date = start_date + timedelta(days=window_days)

            # Find all filings in window
            window_filings = [
                f
                for f in sorted_filings
                if f.get("filing_date") and start_date_str <= f.get("filing_date") <= str(end_date)
            ]

            # Get unique insiders in window
            insiders = set(f.get("owner_name") for f in window_filings if f.get("owner_name"))

            if len(insiders) >= self.CLUSTER_MIN_INSIDERS:
                total_value = sum(abs(f.get("total_value", 0)) for f in window_filings)

                if len(insiders) > best_count or (
                    len(insiders) == best_count and total_value > (best_cluster.total_value if best_cluster else 0)
                ):
                    best_count = len(insiders)

                    # Find actual end date
                    actual_end = max(
                        date.fromisoformat(f.get("filing_date")) for f in window_filings if f.get("filing_date")
                    )

                    best_cluster = ClusterActivity(
                        symbol=symbol,
                        cluster_type=cluster_type,
                        start_date=start_date,
                        end_date=actual_end,
                        insider_count=len(insiders),
                        transaction_count=len(window_filings),
                        total_value=total_value,
                        insiders=list(insiders),
                        is_significant=total_value >= self.CLUSTER_MIN_VALUE,
                    )

        return best_cluster

    async def get_key_insider_summary(self, symbol: str, days: int = 180) -> Dict[str, Any]:
        """Get summary of key insider (C-suite, directors) activity.

        Args:
            symbol: Stock ticker symbol
            days: Analysis period

        Returns:
            Summary dictionary with key insider metrics
        """
        try:
            dao = self._get_dao()

            loop = asyncio.get_event_loop()
            transactions = await loop.run_in_executor(None, dao.get_key_insider_transactions, symbol, days)

            if not transactions:
                return {
                    "symbol": symbol,
                    "key_insiders": [],
                    "total_transactions": 0,
                    "net_value": 0,
                    "summary": "No key insider activity found",
                }

            # Group by insider
            insider_activity = {}
            for t in transactions:
                name = t.get("owner_name", "Unknown")
                if name not in insider_activity:
                    insider_activity[name] = {
                        "name": name,
                        "title": t.get("owner_title"),
                        "is_director": t.get("is_director", False),
                        "transactions": 0,
                        "net_value": 0,
                    }

                insider_activity[name]["transactions"] += 1
                insider_activity[name]["net_value"] += t.get("total_value", 0)

            # Calculate totals
            total_net = sum(i["net_value"] for i in insider_activity.values())

            # Sort by absolute net value
            sorted_insiders = sorted(insider_activity.values(), key=lambda x: abs(x["net_value"]), reverse=True)

            return {
                "symbol": symbol,
                "analysis_period_days": days,
                "key_insiders": sorted_insiders,
                "total_transactions": len(transactions),
                "net_value": total_net,
                "direction": "buying" if total_net > 0 else "selling" if total_net < 0 else "neutral",
            }

        except Exception as e:
            logger.error(f"Error getting key insider summary for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
            }


# Singleton instance
_insider_activity_service: Optional[InsiderActivityService] = None


def get_insider_activity_service() -> InsiderActivityService:
    """Get or create singleton service instance."""
    global _insider_activity_service
    if _insider_activity_service is None:
        _insider_activity_service = InsiderActivityService()
    return _insider_activity_service
