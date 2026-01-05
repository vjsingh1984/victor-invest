"""
Credential Sanitizer - Detect and redact credentials in output.

Prevents accidental credential leakage by:
1. Detecting credentials in node/workflow output
2. Redacting sensitive values before logging/display
3. Alerting on potential credential exposure

Usage:
    from investigator.infrastructure.credential_sanitizer import (
        CredentialSanitizer,
        scan_for_credentials,
        redact_credentials,
    )

    # Scan text for potential credentials
    findings = scan_for_credentials(output_text)

    # Redact credentials from output
    safe_output = redact_credentials(output_text)

    # Use sanitizer for comprehensive scanning
    sanitizer = CredentialSanitizer()
    result = sanitizer.scan(data)
"""

import logging
import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple, Union

logger = logging.getLogger(__name__)


class CredentialPattern(Enum):
    """Types of credential patterns to detect."""
    API_KEY = "api_key"
    PASSWORD = "password"
    SECRET = "secret"
    TOKEN = "token"
    CONNECTION_STRING = "connection_string"
    PRIVATE_KEY = "private_key"
    AWS_KEY = "aws_key"
    DATABASE_URL = "database_url"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"


@dataclass
class CredentialFinding:
    """A detected credential in output."""
    pattern_type: CredentialPattern
    location: str  # Where found (field path, line number, etc.)
    matched_text: str  # The redacted version of what matched
    confidence: float  # 0.0 to 1.0
    severity: str  # "high", "medium", "low"
    context: str = ""  # Surrounding context (redacted)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type.value,
            "location": self.location,
            "matched_text": self.matched_text,
            "confidence": self.confidence,
            "severity": self.severity,
            "context": self.context,
        }


@dataclass
class ScanResult:
    """Result of credential scan."""
    has_credentials: bool
    findings: List[CredentialFinding]
    scanned_at: datetime
    redacted_output: Optional[Any] = None

    @property
    def high_severity_count(self) -> int:
        return len([f for f in self.findings if f.severity == "high"])

    @property
    def summary(self) -> str:
        if not self.has_credentials:
            return "No credentials detected"
        return f"Found {len(self.findings)} potential credential(s) ({self.high_severity_count} high severity)"


# Regex patterns for credential detection
CREDENTIAL_PATTERNS: Dict[CredentialPattern, List[Tuple[Pattern, float, str]]] = {
    CredentialPattern.API_KEY: [
        # Generic API key patterns
        (re.compile(r'["\']?api[_-]?key["\']?\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.I), 0.9, "high"),
        (re.compile(r'["\']?apikey["\']?\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.I), 0.9, "high"),
        # Anthropic API key
        (re.compile(r'sk-ant-[a-zA-Z0-9_\-]{40,}', re.I), 0.95, "high"),
        # OpenAI API key
        (re.compile(r'sk-[a-zA-Z0-9]{48,}'), 0.95, "high"),
        # Generic sk- pattern
        (re.compile(r'sk-[a-zA-Z0-9_\-]{20,}'), 0.8, "high"),
    ],
    CredentialPattern.PASSWORD: [
        (re.compile(r'["\']?password["\']?\s*[=:]\s*["\']?([^\s"\']{8,})["\']?', re.I), 0.85, "high"),
        (re.compile(r'["\']?passwd["\']?\s*[=:]\s*["\']?([^\s"\']{8,})["\']?', re.I), 0.85, "high"),
        (re.compile(r'["\']?pwd["\']?\s*[=:]\s*["\']?([^\s"\']{8,})["\']?', re.I), 0.7, "medium"),
    ],
    CredentialPattern.SECRET: [
        (re.compile(r'["\']?secret["\']?\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{16,})["\']?', re.I), 0.85, "high"),
        (re.compile(r'["\']?client[_-]?secret["\']?\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{16,})["\']?', re.I), 0.9, "high"),
    ],
    CredentialPattern.TOKEN: [
        (re.compile(r'["\']?token["\']?\s*[=:]\s*["\']?([a-zA-Z0-9_\-\.]{20,})["\']?', re.I), 0.8, "high"),
        (re.compile(r'["\']?access[_-]?token["\']?\s*[=:]\s*["\']?([a-zA-Z0-9_\-\.]{20,})["\']?', re.I), 0.9, "high"),
        (re.compile(r'["\']?refresh[_-]?token["\']?\s*[=:]\s*["\']?([a-zA-Z0-9_\-\.]{20,})["\']?', re.I), 0.9, "high"),
    ],
    CredentialPattern.CONNECTION_STRING: [
        # PostgreSQL connection string
        (re.compile(r'postgresql://[^:]+:([^@]+)@[^\s]+', re.I), 0.95, "high"),
        # MySQL connection string
        (re.compile(r'mysql://[^:]+:([^@]+)@[^\s]+', re.I), 0.95, "high"),
        # MongoDB connection string
        (re.compile(r'mongodb(\+srv)?://[^:]+:([^@]+)@[^\s]+', re.I), 0.95, "high"),
        # Redis connection string
        (re.compile(r'redis://[^:]+:([^@]+)@[^\s]+', re.I), 0.95, "high"),
    ],
    CredentialPattern.PRIVATE_KEY: [
        (re.compile(r'-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----', re.I), 0.99, "high"),
        (re.compile(r'-----BEGIN PGP PRIVATE KEY BLOCK-----', re.I), 0.99, "high"),
    ],
    CredentialPattern.AWS_KEY: [
        # AWS Access Key ID
        (re.compile(r'AKIA[0-9A-Z]{16}'), 0.95, "high"),
        # AWS Secret Access Key
        (re.compile(r'["\']?aws[_-]?secret[_-]?access[_-]?key["\']?\s*[=:]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?', re.I), 0.95, "high"),
    ],
    CredentialPattern.BEARER_TOKEN: [
        (re.compile(r'Bearer\s+([a-zA-Z0-9_\-\.]{20,})', re.I), 0.9, "high"),
        (re.compile(r'Authorization:\s*Bearer\s+([a-zA-Z0-9_\-\.]{20,})', re.I), 0.95, "high"),
    ],
    CredentialPattern.BASIC_AUTH: [
        (re.compile(r'Basic\s+([a-zA-Z0-9+/=]{20,})', re.I), 0.85, "high"),
        (re.compile(r'Authorization:\s*Basic\s+([a-zA-Z0-9+/=]{20,})', re.I), 0.9, "high"),
    ],
}

# Known safe patterns to exclude (reduce false positives)
SAFE_PATTERNS = [
    re.compile(r'password["\']?\s*[=:]\s*["\']?\*+["\']?', re.I),  # password = "****"
    re.compile(r'password["\']?\s*[=:]\s*["\']?<[^>]+>["\']?', re.I),  # password = "<redacted>"
    re.compile(r'api[_-]?key["\']?\s*[=:]\s*["\']?your[_-]?[a-z_]+["\']?', re.I),  # api_key = "your_api_key"
    re.compile(r'password["\']?\s*[=:]\s*["\']?None["\']?', re.I),  # password = None
    re.compile(r'token["\']?\s*[=:]\s*["\']?null["\']?', re.I),  # token = null
]


def _redact_value(value: str, show_chars: int = 4) -> str:
    """Redact a credential value, showing only first few chars.

    Args:
        value: Value to redact
        show_chars: Number of characters to show

    Returns:
        Redacted string like "sk-a***"
    """
    if len(value) <= show_chars:
        return "***"
    return value[:show_chars] + "***"


def _is_safe_pattern(text: str) -> bool:
    """Check if text matches a known safe pattern."""
    for pattern in SAFE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def scan_for_credentials(
    text: str,
    context_chars: int = 50,
) -> List[CredentialFinding]:
    """Scan text for potential credentials.

    Args:
        text: Text to scan
        context_chars: Number of characters of context to include

    Returns:
        List of credential findings
    """
    findings = []

    if not text or not isinstance(text, str):
        return findings

    # Check if it's a safe pattern first
    if _is_safe_pattern(text):
        return findings

    for pattern_type, patterns in CREDENTIAL_PATTERNS.items():
        for regex, confidence, severity in patterns:
            for match in regex.finditer(text):
                # Get the matched credential value
                if match.groups():
                    matched_value = match.group(1)
                else:
                    matched_value = match.group(0)

                # Skip very short matches (likely false positives)
                if len(matched_value) < 8:
                    continue

                # Get context
                start = max(0, match.start() - context_chars)
                end = min(len(text), match.end() + context_chars)
                context = text[start:end]

                # Redact the context too
                context = regex.sub(lambda m: _redact_value(m.group(0)), context)

                finding = CredentialFinding(
                    pattern_type=pattern_type,
                    location=f"char:{match.start()}-{match.end()}",
                    matched_text=_redact_value(matched_value),
                    confidence=confidence,
                    severity=severity,
                    context=context,
                )
                findings.append(finding)

    return findings


def redact_credentials(text: str) -> str:
    """Redact all detected credentials in text.

    Args:
        text: Text to redact

    Returns:
        Text with credentials redacted
    """
    if not text or not isinstance(text, str):
        return text

    result = text

    for pattern_type, patterns in CREDENTIAL_PATTERNS.items():
        for regex, confidence, severity in patterns:
            def redactor(match):
                if match.groups():
                    # Redact only the captured group
                    full = match.group(0)
                    captured = match.group(1)
                    return full.replace(captured, _redact_value(captured))
                return _redact_value(match.group(0))

            result = regex.sub(redactor, result)

    return result


class CredentialSanitizer:
    """Comprehensive credential scanner and sanitizer.

    Scans various data types (str, dict, list) for credentials
    and provides redacted versions.
    """

    def __init__(
        self,
        alert_callback: Optional[callable] = None,
        strict_mode: bool = False,
    ):
        """Initialize sanitizer.

        Args:
            alert_callback: Function to call when credentials detected
            strict_mode: If True, raise exception on credential detection
        """
        self._alert_callback = alert_callback
        self._strict_mode = strict_mode
        self._scan_history: List[ScanResult] = []

    def scan(
        self,
        data: Any,
        path: str = "root",
    ) -> ScanResult:
        """Scan data for credentials.

        Args:
            data: Data to scan (str, dict, list, or nested)
            path: Path prefix for location reporting

        Returns:
            ScanResult with findings and redacted output
        """
        findings = []
        redacted = self._scan_recursive(data, path, findings)

        result = ScanResult(
            has_credentials=len(findings) > 0,
            findings=findings,
            scanned_at=datetime.now(),
            redacted_output=redacted,
        )

        self._scan_history.append(result)

        # Alert if credentials found
        if result.has_credentials:
            logger.warning(f"[CREDENTIAL_LEAK] {result.summary}")

            if self._alert_callback:
                self._alert_callback(result)

            if self._strict_mode:
                raise CredentialLeakageError(
                    f"Credential leakage detected: {result.summary}"
                )

        return result

    def _scan_recursive(
        self,
        data: Any,
        path: str,
        findings: List[CredentialFinding],
    ) -> Any:
        """Recursively scan and redact data.

        Args:
            data: Data to scan
            path: Current path
            findings: List to append findings to

        Returns:
            Redacted version of data
        """
        if isinstance(data, str):
            found = scan_for_credentials(data)
            for f in found:
                f.location = f"{path}:{f.location}"
                findings.append(f)
            return redact_credentials(data)

        elif isinstance(data, dict):
            redacted = {}
            for key, value in data.items():
                new_path = f"{path}.{key}"

                # Check if key name suggests sensitive data
                if any(s in key.lower() for s in ["password", "secret", "token", "key", "credential"]):
                    if isinstance(value, str) and len(value) >= 8:
                        findings.append(CredentialFinding(
                            pattern_type=CredentialPattern.PASSWORD,
                            location=new_path,
                            matched_text=_redact_value(str(value)),
                            confidence=0.8,
                            severity="high",
                            context=f"{key}=***",
                        ))
                        redacted[key] = _redact_value(str(value))
                        continue

                redacted[key] = self._scan_recursive(value, new_path, findings)
            return redacted

        elif isinstance(data, list):
            return [
                self._scan_recursive(item, f"{path}[{i}]", findings)
                for i, item in enumerate(data)
            ]

        else:
            # For other types, convert to string and check
            str_val = str(data)
            if len(str_val) >= 20:  # Only check longer values
                found = scan_for_credentials(str_val)
                for f in found:
                    f.location = f"{path}:{f.location}"
                    findings.append(f)
            return data

    def get_scan_history(self, limit: int = 100) -> List[ScanResult]:
        """Get recent scan history.

        Args:
            limit: Maximum results to return

        Returns:
            List of scan results
        """
        return self._scan_history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get scanning statistics.

        Returns:
            Dict with scan statistics
        """
        total_scans = len(self._scan_history)
        scans_with_creds = len([s for s in self._scan_history if s.has_credentials])
        total_findings = sum(len(s.findings) for s in self._scan_history)

        by_pattern = {}
        by_severity = {"high": 0, "medium": 0, "low": 0}

        for scan in self._scan_history:
            for finding in scan.findings:
                by_pattern[finding.pattern_type.value] = by_pattern.get(
                    finding.pattern_type.value, 0
                ) + 1
                by_severity[finding.severity] += 1

        return {
            "total_scans": total_scans,
            "scans_with_credentials": scans_with_creds,
            "detection_rate": scans_with_creds / total_scans if total_scans else 0,
            "total_findings": total_findings,
            "by_pattern_type": by_pattern,
            "by_severity": by_severity,
        }


class CredentialLeakageError(Exception):
    """Raised when credential leakage is detected in strict mode."""
    pass


def create_output_sanitizer() -> CredentialSanitizer:
    """Create a sanitizer configured for workflow output scanning.

    Returns:
        Configured CredentialSanitizer
    """
    def alert_handler(result: ScanResult):
        logger.error(
            f"[SECURITY_ALERT] Potential credential leakage detected!\n"
            f"  Findings: {len(result.findings)}\n"
            f"  High severity: {result.high_severity_count}\n"
            f"  Details: {[f.to_dict() for f in result.findings[:3]]}"
        )

    return CredentialSanitizer(alert_callback=alert_handler)


__all__ = [
    "CredentialPattern",
    "CredentialFinding",
    "ScanResult",
    "scan_for_credentials",
    "redact_credentials",
    "CredentialSanitizer",
    "CredentialLeakageError",
    "create_output_sanitizer",
]
