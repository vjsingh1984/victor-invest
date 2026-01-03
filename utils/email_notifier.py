"""
Email Notifier Module

Handles email notifications for investment alerts.
"""
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Email notifier for sending alert notifications"""

    def __init__(self, smtp_host: str, smtp_port: int, sender_email: str,
                 sender_password: str):
        """
        Initialize email notifier

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            sender_email: Sender email address
            sender_password: Sender email password
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password

    def format_alert_email(self, alerts: List[Dict], format: str = 'html') -> Dict:
        """
        Format alerts into email content

        Args:
            alerts: List of alert dictionaries
            format: Email format ('html' or 'text')

        Returns:
            Dictionary with 'subject' and 'body' keys
        """
        if not alerts:
            return {
                'subject': 'InvestiGator: No New Alerts',
                'body': 'No alerts to report.'
            }

        # Group alerts by severity
        grouped = self._group_alerts_by_severity(alerts)

        # Determine subject based on highest severity
        high_count = len(grouped.get('high', []))
        medium_count = len(grouped.get('medium', []))
        low_count = len(grouped.get('low', []))

        if high_count > 0:
            subject = f"⚠️ InvestiGator: {high_count} High Priority Alert{'s' if high_count != 1 else ''}"
        elif medium_count > 0:
            subject = f"📊 InvestiGator: {medium_count} Alert{'s' if medium_count != 1 else ''}"
        else:
            subject = f"ℹ️ InvestiGator: {low_count} Low Priority Alert{'s' if low_count != 1 else ''}"

        # Format body based on format
        if format == 'html':
            body = self._format_html_body(grouped, alerts)
        else:
            body = self._format_text_body(grouped, alerts)

        return {
            'subject': subject,
            'body': body
        }

    def _group_alerts_by_severity(self, alerts: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group alerts by severity level

        Args:
            alerts: List of alerts

        Returns:
            Dictionary with severity levels as keys
        """
        grouped = {'high': [], 'medium': [], 'low': []}

        for alert in alerts:
            severity = alert.get('severity', 'medium')
            if severity in grouped:
                grouped[severity].append(alert)

        return grouped

    def _format_html_body(self, grouped: Dict[str, List[Dict]], all_alerts: List[Dict]) -> str:
        """
        Format email body as HTML

        Args:
            grouped: Alerts grouped by severity
            all_alerts: All alerts

        Returns:
            HTML email body
        """
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                h2 { color: #333; }
                .alert { padding: 10px; margin: 10px 0; border-left: 4px solid #ccc; }
                .high { border-left-color: #dc3545; background-color: #f8d7da; }
                .medium { border-left-color: #ffc107; background-color: #fff3cd; }
                .low { border-left-color: #17a2b8; background-color: #d1ecf1; }
                .symbol { font-weight: bold; color: #007bff; }
                .timestamp { color: #666; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <h2>Investment Alert Summary</h2>
        """

        # Add high severity alerts first
        if grouped['high']:
            html += "<h3 style='color: #dc3545;'>🔴 High Priority Alerts</h3>"
            for alert in grouped['high']:
                html += self._format_alert_html(alert, 'high')

        # Medium severity
        if grouped['medium']:
            html += "<h3 style='color: #ffc107;'>🟡 Medium Priority Alerts</h3>"
            for alert in grouped['medium']:
                html += self._format_alert_html(alert, 'medium')

        # Low severity
        if grouped['low']:
            html += "<h3 style='color: #17a2b8;'>🔵 Low Priority Alerts</h3>"
            for alert in grouped['low']:
                html += self._format_alert_html(alert, 'low')

        html += f"""
            <hr>
            <p class='timestamp'>Alert generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><em>This is an automated notification from InvestiGator Investment Analysis System</em></p>
        </body>
        </html>
        """

        return html

    def _format_alert_html(self, alert: Dict, severity: str) -> str:
        """
        Format a single alert as HTML

        Args:
            alert: Alert dictionary
            severity: Severity level

        Returns:
            HTML snippet for the alert
        """
        symbol = alert.get('symbol', 'N/A')
        alert_type = alert.get('type', 'unknown').replace('_', ' ').title()
        message = alert.get('message', 'No message')
        timestamp = alert.get('timestamp', datetime.now())

        return f"""
        <div class='alert {severity}'>
            <strong class='symbol'>{symbol}</strong> - {alert_type}<br>
            {message}<br>
            <span class='timestamp'>{timestamp.strftime('%Y-%m-%d %H:%M')}</span>
        </div>
        """

    def _format_text_body(self, grouped: Dict[str, List[Dict]], all_alerts: List[Dict]) -> str:
        """
        Format email body as plain text

        Args:
            grouped: Alerts grouped by severity
            all_alerts: All alerts

        Returns:
            Plain text email body
        """
        lines = []
        lines.append("INVESTMENT ALERT SUMMARY")
        lines.append("=" * 50)
        lines.append("")

        # High severity
        if grouped['high']:
            lines.append("HIGH PRIORITY ALERTS:")
            lines.append("-" * 50)
            for alert in grouped['high']:
                lines.append(self._format_alert_text(alert))
            lines.append("")

        # Medium severity
        if grouped['medium']:
            lines.append("MEDIUM PRIORITY ALERTS:")
            lines.append("-" * 50)
            for alert in grouped['medium']:
                lines.append(self._format_alert_text(alert))
            lines.append("")

        # Low severity
        if grouped['low']:
            lines.append("LOW PRIORITY ALERTS:")
            lines.append("-" * 50)
            for alert in grouped['low']:
                lines.append(self._format_alert_text(alert))
            lines.append("")

        lines.append("=" * 50)
        lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("This is an automated notification from InvestiGator")

        return "\n".join(lines)

    def _format_alert_text(self, alert: Dict) -> str:
        """
        Format a single alert as plain text

        Args:
            alert: Alert dictionary

        Returns:
            Text snippet for the alert
        """
        symbol = alert.get('symbol', 'N/A')
        alert_type = alert.get('type', 'unknown').replace('_', ' ').title()
        message = alert.get('message', 'No message')
        timestamp = alert.get('timestamp', datetime.now())

        return f"""
[{symbol}] {alert_type}
{message}
Time: {timestamp.strftime('%Y-%m-%d %H:%M')}
"""

    def send_alert_email(self, recipient: str, alerts: List[Dict],
                        format: str = 'html') -> bool:
        """
        Send alert email to a recipient

        Args:
            recipient: Recipient email address
            alerts: List of alerts
            format: Email format ('html' or 'text')

        Returns:
            True if sent successfully
        """
        try:
            email_content = self.format_alert_email(alerts, format=format)

            msg = MIMEMultipart('alternative')
            msg['Subject'] = email_content['subject']
            msg['From'] = self.sender_email
            msg['To'] = recipient

            # Attach body
            if format == 'html':
                part = MIMEText(email_content['body'], 'html')
            else:
                part = MIMEText(email_content['body'], 'plain')

            msg.attach(part)

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            logger.info(f"Sent alert email to {recipient}")
            return True

        except Exception as e:
            logger.error(f"Error sending alert email to {recipient}: {e}")
            return False

    def send_batch_alerts(self, recipients: List[str], alerts: List[Dict],
                         format: str = 'html') -> Dict[str, bool]:
        """
        Send alerts to multiple recipients

        Args:
            recipients: List of recipient email addresses
            alerts: List of alerts
            format: Email format

        Returns:
            Dictionary mapping recipient to success status
        """
        results = {}

        for recipient in recipients:
            success = self.send_alert_email(recipient, alerts, format=format)
            results[recipient] = success

        successful = sum(1 for v in results.values() if v)
        logger.info(f"Sent batch alerts: {successful}/{len(recipients)} successful")

        return results
