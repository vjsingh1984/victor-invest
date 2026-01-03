"""
Alert Engine Module

Evaluates alert rules and triggers notifications based on
significant changes in investment recommendations.
"""
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AlertEngine:
    """Alert engine for evaluating and triggering investment alerts"""

    def __init__(self, db_manager=None):
        """
        Initialize alert engine

        Args:
            db_manager: Database manager for storing alert history
        """
        self.db_manager = db_manager
        self.alert_rules = self._initialize_alert_rules()

    def _initialize_alert_rules(self) -> Dict:
        """
        Initialize alert rule configuration

        Returns:
            Dictionary of alert rules and their thresholds
        """
        return {
            'score_change': {
                'threshold': 1.0,  # Trigger if score changes by >= 1.0 points
                'enabled': True
            },
            'recommendation_change': {
                'enabled': True
            },
            'technical_breakdown': {
                'technical_score_threshold': 4.0,  # Below this is concerning
                'enabled': True
            },
            'earnings_surprise': {
                'threshold_pct': 10.0,  # 10% beat or miss
                'enabled': True
            },
            '8k_filing': {
                'material_items': ['Item 2.02', 'Item 2.03', 'Item 5.02'],  # Material events
                'enabled': True
            }
        }

    def evaluate_alerts(self, current_recommendation: Dict,
                       previous_recommendation: Optional[Dict] = None) -> List[Dict]:
        """
        Evaluate all alert rules for a recommendation

        Args:
            current_recommendation: Current investment recommendation
            previous_recommendation: Previous recommendation for comparison

        Returns:
            List of triggered alerts
        """
        alerts = []

        try:
            symbol = current_recommendation.get('symbol', 'UNKNOWN')
            logger.info(f"Evaluating alerts for {symbol}")

            # Rule 1: Score change alert
            if previous_recommendation:
                score_alert = self._evaluate_score_change(
                    current_recommendation,
                    previous_recommendation
                )
                if score_alert:
                    alerts.append(score_alert)

                # Rule 2: Recommendation change alert
                rec_alert = self._evaluate_recommendation_change_alert(
                    current_recommendation,
                    previous_recommendation
                )
                if rec_alert:
                    alerts.append(rec_alert)

            # Rule 3: Technical breakdown alert
            tech_alert = self._evaluate_technical_breakdown(current_recommendation)
            if tech_alert:
                alerts.append(tech_alert)

            # Rule 4: Earnings surprise alert
            earnings_alert = self._evaluate_earnings_surprise(current_recommendation)
            if earnings_alert:
                alerts.append(earnings_alert)

            logger.info(f"{symbol} - Generated {len(alerts)} alerts")

        except Exception as e:
            logger.error(f"Error evaluating alerts: {e}")

        return alerts

    def _evaluate_score_change(self, current: Dict, previous: Dict) -> Optional[Dict]:
        """
        Evaluate score change alert

        Args:
            current: Current recommendation
            previous: Previous recommendation

        Returns:
            Alert dict if triggered, None otherwise
        """
        if not self.alert_rules['score_change']['enabled']:
            return None

        current_score = current.get('overall_score', 0)
        previous_score = previous.get('overall_score', 0)
        change = current_score - previous_score

        if not self._should_alert_score_change(previous_score, current_score):
            return None

        severity = self._classify_severity('score_change', change)

        direction = 'increased' if change > 0 else 'decreased'
        return {
            'symbol': current.get('symbol'),
            'type': 'score_change',
            'severity': severity,
            'message': f"Overall score {direction} from {previous_score:.1f} to {current_score:.1f} ({change:+.1f})",
            'details': {
                'old_score': previous_score,
                'new_score': current_score,
                'change': change
            },
            'timestamp': datetime.now()
        }

    def _should_alert_score_change(self, old_score: float, new_score: float) -> bool:
        """
        Determine if score change warrants an alert

        Args:
            old_score: Previous score
            new_score: Current score

        Returns:
            True if should alert
        """
        threshold = self.alert_rules['score_change']['threshold']
        change = abs(new_score - old_score)
        return change >= threshold

    def _evaluate_recommendation_change_alert(self, current: Dict, previous: Dict) -> Optional[Dict]:
        """
        Evaluate recommendation change alert

        Args:
            current: Current recommendation
            previous: Previous recommendation

        Returns:
            Alert dict if triggered, None otherwise
        """
        if not self.alert_rules['recommendation_change']['enabled']:
            return None

        current_rec = current.get('recommendation', '')
        previous_rec = previous.get('recommendation', '')

        if current_rec == previous_rec:
            return None

        return self._evaluate_recommendation_change(previous_rec, current_rec)

    def _evaluate_recommendation_change(self, old_rec: str, new_rec: str) -> Optional[Dict]:
        """
        Evaluate recommendation change

        Args:
            old_rec: Previous recommendation
            new_rec: Current recommendation

        Returns:
            Alert dict with appropriate severity
        """
        if old_rec == new_rec:
            return None

        # Map recommendations to numeric values for comparison
        rec_values = {'SELL': 1, 'HOLD': 2, 'BUY': 3}
        old_val = rec_values.get(old_rec, 2)
        new_val = rec_values.get(new_rec, 2)

        change_type = 'upgrade' if new_val > old_val else 'downgrade'

        # Downgrades are high severity, upgrades are medium
        severity = 'high' if change_type == 'downgrade' else 'medium'

        return {
            'symbol': None,  # Will be filled by caller
            'type': 'recommendation_change',
            'severity': severity,
            'message': f"Recommendation {change_type}d from {old_rec} to {new_rec}",
            'details': {
                'old_recommendation': old_rec,
                'new_recommendation': new_rec,
                'change_type': change_type
            },
            'timestamp': datetime.now()
        }

    def _evaluate_technical_breakdown(self, recommendation: Dict) -> Optional[Dict]:
        """
        Evaluate technical breakdown alert

        Args:
            recommendation: Current recommendation

        Returns:
            Alert dict if triggered, None otherwise
        """
        if not self.alert_rules['technical_breakdown']['enabled']:
            return None

        technical_score = recommendation.get('technical_score', 5.0)
        threshold = self.alert_rules['technical_breakdown']['technical_score_threshold']

        if technical_score >= threshold:
            return None

        # Check if price broke below support
        support_resistance = recommendation.get('support_resistance', {})
        if support_resistance:
            support_levels = support_resistance.get('support_levels', [])
            current_price = support_resistance.get('current_price', 0)

            if support_levels and current_price > 0:
                # Check if below all support levels
                below_support = all(current_price < level for level in support_levels)

                if below_support:
                    return {
                        'symbol': recommendation.get('symbol'),
                        'type': 'technical_breakdown',
                        'severity': 'high',
                        'message': f"Technical breakdown: Price below support, technical score {technical_score:.1f}",
                        'details': {
                            'technical_score': technical_score,
                            'support_levels': support_levels,
                            'current_price': current_price
                        },
                        'timestamp': datetime.now()
                    }

        # Just low technical score without support break
        if technical_score < 3.0:
            return {
                'symbol': recommendation.get('symbol'),
                'type': 'technical_breakdown',
                'severity': 'medium',
                'message': f"Weak technical indicators: score {technical_score:.1f}",
                'details': {
                    'technical_score': technical_score
                },
                'timestamp': datetime.now()
            }

        return None

    def _evaluate_earnings_surprise(self, recommendation: Dict) -> Optional[Dict]:
        """
        Evaluate earnings surprise alert

        Args:
            recommendation: Current recommendation

        Returns:
            Alert dict if triggered, None otherwise
        """
        if not self.alert_rules['earnings_surprise']['enabled']:
            return None

        quarterly_metrics = recommendation.get('quarterly_metrics', [])
        if not quarterly_metrics:
            return None

        # Check most recent quarter
        latest = quarterly_metrics[-1] if isinstance(quarterly_metrics, list) else quarterly_metrics

        if isinstance(latest, dict):
            actual_eps = latest.get('earnings_per_share')
            estimate_eps = latest.get('analyst_estimate')

            if actual_eps and estimate_eps and estimate_eps != 0:
                surprise_pct = ((actual_eps - estimate_eps) / abs(estimate_eps)) * 100

                threshold = self.alert_rules['earnings_surprise']['threshold_pct']

                if abs(surprise_pct) >= threshold:
                    beat_or_miss = 'beat' if surprise_pct > 0 else 'missed'
                    severity = 'medium' if surprise_pct > 0 else 'high'

                    return {
                        'symbol': recommendation.get('symbol'),
                        'type': 'earnings_surprise',
                        'severity': severity,
                        'message': f"Earnings {beat_or_miss} estimates by {abs(surprise_pct):.1f}%",
                        'details': {
                            'actual_eps': actual_eps,
                            'estimate_eps': estimate_eps,
                            'surprise_pct': surprise_pct
                        },
                        'timestamp': datetime.now()
                    }

        return None

    def _evaluate_8k_filing(self, filing_data: Dict) -> Optional[Dict]:
        """
        Evaluate 8-K filing alert

        Args:
            filing_data: 8-K filing information

        Returns:
            Alert dict if triggered, None otherwise
        """
        if not self.alert_rules['8k_filing']['enabled']:
            return None

        form_type = filing_data.get('form_type', '')
        if form_type != '8-K':
            return None

        items = filing_data.get('items', [])
        material_items = self.alert_rules['8k_filing']['material_items']

        # Check if any material items are present
        triggered_items = [item for item in items if item in material_items]

        if triggered_items:
            return {
                'symbol': filing_data.get('symbol'),
                'type': '8k_filing',
                'severity': 'high',
                'message': f"Material 8-K filing: {', '.join(triggered_items)}",
                'details': {
                    'filing_date': filing_data.get('filing_date'),
                    'items': triggered_items
                },
                'timestamp': datetime.now()
            }

        return None

    def _classify_severity(self, alert_type: str, value: float) -> str:
        """
        Classify alert severity

        Args:
            alert_type: Type of alert
            value: Value to classify (e.g., score change)

        Returns:
            Severity level: 'low', 'medium', or 'high'
        """
        if alert_type == 'score_change':
            abs_value = abs(value)
            if abs_value >= 2.5:
                return 'high'
            elif abs_value >= 1.5:
                return 'medium'
            else:
                return 'low'

        # Default severity
        return 'medium'

    def save_alert(self, alert: Dict) -> bool:
        """
        Save alert to database

        Args:
            alert: Alert dictionary

        Returns:
            True if saved successfully
        """
        if not self.db_manager:
            logger.warning("No database manager configured, alert not saved")
            return False

        try:
            from sqlalchemy import text

            query = text("""
                INSERT INTO alerts (symbol, alert_type, severity, message, details, created_at)
                VALUES (:symbol, :alert_type, :severity, :message, :details, :created_at)
            """)

            with self.db_manager.get_session() as session:
                session.execute(query, {
                    'symbol': alert.get('symbol'),
                    'alert_type': alert.get('type'),
                    'severity': alert.get('severity'),
                    'message': alert.get('message'),
                    'details': json.dumps(alert.get('details', {})),
                    'created_at': alert.get('timestamp', datetime.now())
                })
                session.commit()

            logger.info(f"Saved alert: {alert.get('type')} for {alert.get('symbol')}")
            return True

        except Exception as e:
            logger.error(f"Error saving alert: {e}")
            return False

    def get_active_alerts(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Get active alerts for a symbol

        Args:
            symbol: Stock symbol
            days: Number of days to look back

        Returns:
            List of active alerts
        """
        if not self.db_manager:
            return []

        try:
            from sqlalchemy import text

            cutoff_date = datetime.now() - timedelta(days=days)

            query = text("""
                SELECT symbol, alert_type, severity, message, details, created_at
                FROM alerts
                WHERE symbol = :symbol
                AND created_at >= :cutoff_date
                ORDER BY created_at DESC
            """)

            with self.db_manager.get_session() as session:
                results = session.execute(query, {
                    'symbol': symbol,
                    'cutoff_date': cutoff_date
                }).fetchall()

                alerts = []
                for row in results:
                    alerts.append({
                        'symbol': row[0],
                        'type': row[1],
                        'severity': row[2],
                        'message': row[3],
                        'details': row[4],
                        'timestamp': row[5]
                    })

                return alerts

        except Exception as e:
            logger.error(f"Error retrieving alerts: {e}")
            return []
