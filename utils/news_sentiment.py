"""
News Sentiment Analysis Module

Analyzes news sentiment using NewsAPI and LLM-powered sentiment scoring
to provide market sentiment insights and detect sentiment-price divergences.
"""
import logging
import requests
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import Counter

logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """Analyze news sentiment for stocks"""

    def __init__(self, db_manager=None, ollama_client=None, newsapi_key: Optional[str] = None):
        """
        Initialize news sentiment analyzer

        Args:
            db_manager: Database manager (optional)
            ollama_client: Ollama LLM client for sentiment analysis
            newsapi_key: NewsAPI key (optional, uses placeholder if not provided)
        """
        self.db_manager = db_manager
        self.ollama_client = ollama_client
        self.newsapi_key = newsapi_key or "placeholder"  # Will be configured later
        self.newsapi_base_url = "https://newsapi.org/v2"

    def analyze_news_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """
        Analyze news sentiment for a symbol

        Args:
            symbol: Stock symbol
            days: Number of days to look back (default 7)

        Returns:
            Dictionary with sentiment analysis
        """
        try:
            logger.info(f"Analyzing news sentiment for {symbol} over last {days} days")

            # Fetch news articles
            articles = self._fetch_news_articles(symbol, days)

            if not articles or len(articles) == 0:
                logger.warning(f"No news articles found for {symbol}")
                return self._get_default_sentiment(symbol, days, "No news articles found")

            # Filter irrelevant articles
            articles = self._filter_articles(articles)

            if len(articles) == 0:
                return self._get_default_sentiment(symbol, days, "No relevant articles after filtering")

            # Analyze sentiment for each article
            article_sentiments = []
            for article in articles[:20]:  # Limit to 20 most recent
                try:
                    sentiment = self._analyze_article_sentiment(article)
                    if sentiment:
                        sentiment['article'] = article
                        article_sentiments.append(sentiment)
                except Exception as e:
                    logger.warning(f"Error analyzing article sentiment: {e}")
                    continue

            if not article_sentiments:
                return self._get_default_sentiment(symbol, days, "Could not analyze article sentiments")

            # Calculate aggregate sentiment
            aggregate = self._calculate_aggregate_sentiment(article_sentiments)

            # Calculate sentiment trend
            trend = self._calculate_sentiment_trend(article_sentiments)
            aggregate['sentiment_trend'] = trend

            # Add metadata
            aggregate['period_days'] = days
            aggregate['data_available'] = True

            logger.info(f"{symbol} - News sentiment: {aggregate.get('sentiment_rating')} ({aggregate.get('sentiment_score')}/10) from {len(article_sentiments)} articles")

            return aggregate

        except Exception as e:
            logger.error(f"Error analyzing news sentiment for {symbol}: {e}")
            return self._get_default_sentiment(symbol, days, f"Error: {str(e)}")

    def _fetch_news_articles(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Fetch news articles from NewsAPI

        Args:
            symbol: Stock symbol
            days: Number of days to look back

        Returns:
            List of news articles
        """
        try:
            # Check if we have a valid API key
            if not self.newsapi_key or self.newsapi_key == "placeholder":
                logger.info(f"NewsAPI key not configured, skipping news fetch for {symbol}")
                return []

            # Calculate date range
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            # Build API request
            url = f"{self.newsapi_base_url}/everything"
            params = {
                'q': symbol,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.newsapi_key
            }

            # Make API request
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                logger.info(f"Fetched {len(articles)} articles for {symbol}")
                return articles
            else:
                logger.warning(f"NewsAPI returned status: {data.get('status')}")
                return []

        except Exception as e:
            logger.error(f"Error fetching news articles: {e}")
            return []

    def _filter_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Filter irrelevant or duplicate articles

        Args:
            articles: List of articles

        Returns:
            Filtered list of articles
        """
        if not articles:
            return []

        filtered = []
        seen_titles = set()

        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')

            # Skip removed content
            if '[Removed]' in title or not title or not description:
                continue

            # Skip duplicates
            if title in seen_titles:
                continue

            seen_titles.add(title)
            filtered.append(article)

        return filtered

    def _analyze_article_sentiment(self, article: Dict) -> Optional[Dict]:
        """
        Analyze sentiment of a single article using LLM

        Args:
            article: News article

        Returns:
            Sentiment analysis result
        """
        if not self.ollama_client:
            # Without LLM, use basic keyword analysis
            return self._basic_sentiment_analysis(article)

        try:
            title = article.get('title', '')
            description = article.get('description', '')

            prompt = f"""Analyze the sentiment of this news article about a stock.

Title: {title}
Description: {description}

Provide a JSON response with:
- sentiment_score: number from 0-10 (0=very negative, 5=neutral, 10=very positive)
- sentiment_label: "positive", "negative", or "neutral"
- reasoning: brief explanation

Focus on implications for stock price. Be objective and balanced.
"""

            response = self.ollama_client.generate(prompt)
            response_text = response.get('response', '{}')

            # Parse JSON response
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM sentiment response as JSON")
                return self._basic_sentiment_analysis(article)

        except Exception as e:
            logger.error(f"Error in LLM sentiment analysis: {e}")
            return self._basic_sentiment_analysis(article)

    def _basic_sentiment_analysis(self, article: Dict) -> Dict:
        """
        Basic keyword-based sentiment analysis (fallback)

        Args:
            article: News article

        Returns:
            Sentiment result
        """
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        text = f"{title} {description}"

        # Positive keywords
        positive_keywords = [
            'profit', 'growth', 'record', 'beat', 'exceed', 'strong', 'surge',
            'innovation', 'success', 'breakthrough', 'gain', 'rise', 'jump'
        ]

        # Negative keywords
        negative_keywords = [
            'loss', 'decline', 'fall', 'miss', 'lawsuit', 'probe', 'investigation',
            'warning', 'cut', 'layoff', 'concern', 'risk', 'drop', 'plunge'
        ]

        pos_count = sum(1 for keyword in positive_keywords if keyword in text)
        neg_count = sum(1 for keyword in negative_keywords if keyword in text)

        # Calculate score
        if pos_count == 0 and neg_count == 0:
            score = 5.0
            label = 'neutral'
        else:
            # Score based on ratio
            total = pos_count + neg_count
            pos_ratio = pos_count / total
            score = pos_ratio * 10
            if score >= 6.5:
                label = 'positive'
            elif score <= 3.5:
                label = 'negative'
            else:
                label = 'neutral'

        return {
            'sentiment_score': round(score, 1),
            'sentiment_label': label,
            'reasoning': f'Keyword analysis: {pos_count} positive, {neg_count} negative'
        }

    def _calculate_aggregate_sentiment(self, article_sentiments: List[Dict]) -> Dict:
        """
        Calculate aggregate sentiment from multiple articles

        Args:
            article_sentiments: List of article sentiment results

        Returns:
            Aggregate sentiment metrics
        """
        if not article_sentiments:
            return {
                'sentiment_score': 5.0,
                'sentiment_rating': 'Neutral',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'article_count': 0
            }

        # Count sentiment labels
        labels = [s.get('sentiment_label', 'neutral') for s in article_sentiments]
        label_counts = Counter(labels)

        # Calculate weighted average with recency bias
        total_weight = 0
        weighted_sum = 0

        for i, sentiment in enumerate(article_sentiments):
            # More recent articles (earlier in list) get higher weight
            recency_weight = 1.0 + (0.5 * (len(article_sentiments) - i) / len(article_sentiments))
            score = sentiment.get('sentiment_score', 5.0)

            weighted_sum += score * recency_weight
            total_weight += recency_weight

        avg_score = weighted_sum / total_weight if total_weight > 0 else 5.0

        return {
            'sentiment_score': round(avg_score, 1),
            'sentiment_rating': self._get_sentiment_rating(avg_score),
            'positive_count': label_counts.get('positive', 0),
            'negative_count': label_counts.get('negative', 0),
            'neutral_count': label_counts.get('neutral', 0),
            'article_count': len(article_sentiments)
        }

    def _calculate_sentiment_trend(self, article_sentiments: List[Dict]) -> Dict:
        """
        Calculate sentiment trend over time

        Args:
            article_sentiments: List of article sentiments (ordered by date, newest first)

        Returns:
            Trend analysis
        """
        if len(article_sentiments) < 2:
            return {'direction': 'stable', 'strength': 0}

        # Split into recent and older halves
        mid_point = len(article_sentiments) // 2
        recent = article_sentiments[:mid_point]
        older = article_sentiments[mid_point:]

        # Calculate average for each period
        recent_avg = sum(s.get('sentiment_score', 5.0) for s in recent) / len(recent)
        older_avg = sum(s.get('sentiment_score', 5.0) for s in older) / len(older)

        # Determine trend
        diff = recent_avg - older_avg

        if diff > 1.0:
            direction = 'improving'
            strength = min(abs(diff), 5.0)  # Cap at 5
        elif diff < -1.0:
            direction = 'declining'
            strength = min(abs(diff), 5.0)
        else:
            direction = 'stable'
            strength = abs(diff)

        return {
            'direction': direction,
            'strength': round(strength, 1),
            'recent_avg': round(recent_avg, 1),
            'older_avg': round(older_avg, 1)
        }

    def _detect_sentiment_price_divergence(self, sentiment_data: Dict,
                                           price_data: Dict) -> Optional[Dict]:
        """
        Detect divergences between sentiment and price movement

        Args:
            sentiment_data: Sentiment analysis results
            price_data: Price movement data

        Returns:
            Divergence information if detected
        """
        sentiment_score = sentiment_data.get('sentiment_score', 5.0)
        price_change = price_data.get('price_change_pct_7d', 0)

        # Bearish divergence: negative sentiment but price rising
        if sentiment_score < 4.0 and price_change > 5.0:
            return {
                'type': 'bearish_divergence',
                'description': f'Negative news sentiment ({sentiment_score}/10) but price up {price_change:.1f}%',
                'signal': 'Potential overvaluation or market ignoring risks'
            }

        # Bullish divergence: positive sentiment but price falling
        if sentiment_score > 6.5 and price_change < -5.0:
            return {
                'type': 'bullish_divergence',
                'description': f'Positive news sentiment ({sentiment_score}/10) but price down {price_change:.1f}%',
                'signal': 'Potential buying opportunity if fundamentals support sentiment'
            }

        return None

    def _get_sentiment_rating(self, score: float) -> str:
        """
        Convert sentiment score to rating

        Args:
            score: Sentiment score 0-10

        Returns:
            Rating string
        """
        if score >= 8.0:
            return 'Very Positive'
        elif score >= 6.5:
            return 'Positive'
        elif score >= 4.5:
            return 'Neutral'
        elif score >= 2.5:
            return 'Negative'
        else:
            return 'Very Negative'

    def _get_default_sentiment(self, symbol: str, days: int, reason: str) -> Dict:
        """
        Get default sentiment when data is unavailable

        Args:
            symbol: Stock symbol
            days: Analysis period
            reason: Reason for default

        Returns:
            Default sentiment structure
        """
        return {
            'sentiment_score': 5.0,
            'sentiment_rating': 'Neutral',
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'article_count': 0,
            'sentiment_trend': {'direction': 'stable', 'strength': 0},
            'period_days': days,
            'data_available': False,
            'note': f'News sentiment analysis requires NewsAPI integration - {reason}'
        }
