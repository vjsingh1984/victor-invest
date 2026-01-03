#!/usr/bin/env python3
"""
InvestiGator - Event Analyzer for Vector Database
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Analyzes SEC filings and extracts meaningful financial events for semantic search
Processes 8-K forms, insider trading, and other event-driven disclosures
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum

from .vector_engine import VectorDocument

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of financial events we track"""

    EARNINGS_RELEASE = "earnings_release"
    MANAGEMENT_CHANGE = "management_change"
    ACQUISITION = "acquisition"
    MERGER = "merger"
    SPIN_OFF = "spin_off"
    BANKRUPTCY = "bankruptcy"
    MATERIAL_AGREEMENT = "material_agreement"
    INSIDER_TRADING = "insider_trading"
    DEBT_ISSUANCE = "debt_issuance"
    EQUITY_ISSUANCE = "equity_issuance"
    DIVIDEND_ANNOUNCEMENT = "dividend_announcement"
    SHARE_REPURCHASE = "share_repurchase"
    REGULATORY_ACTION = "regulatory_action"
    LITIGATION = "litigation"
    PRODUCT_LAUNCH = "product_launch"
    FACILITY_CLOSURE = "facility_closure"
    RESTRUCTURING = "restructuring"
    GUIDANCE_UPDATE = "guidance_update"
    OTHER = "other"


class EventSeverity(Enum):
    """Severity/importance levels for events"""

    CRITICAL = "critical"  # Major events like bankruptcies, large acquisitions
    HIGH = "high"  # Important events like earnings, management changes
    MEDIUM = "medium"  # Moderately important events
    LOW = "low"  # Minor events


@dataclass
class FinancialEvent:
    """Represents a specific financial event extracted from SEC filings"""

    # Event identification
    event_id: str
    event_type: EventType
    severity: EventSeverity

    # Content
    title: str
    description: str
    full_text: str

    # Company context
    symbol: str
    cik: str

    # Timing
    event_date: date
    filing_date: date
    reporting_period: Optional[str] = None

    # Source information
    form_type: str = "8-K"
    accession_number: Optional[str] = None
    section: Optional[str] = None

    # Analysis
    sentiment_score: Optional[float] = None  # -1 to 1 (negative to positive)
    market_impact: Optional[str] = None  # "positive", "negative", "neutral"
    affected_metrics: List[str] = field(default_factory=list)

    # Entities mentioned
    mentioned_companies: List[str] = field(default_factory=list)
    mentioned_people: List[str] = field(default_factory=list)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_vector_document(self) -> VectorDocument:
        """Convert to VectorDocument for storage in vector database"""
        # Combine title and description for better semantic search
        content = f"{self.title}\n\n{self.description}"
        if len(content) < 100 and self.full_text:
            content = self.full_text[:2000]  # Use full text if title/description too short

        doc_id = f"{self.symbol}_event_{self.event_id}"

        return VectorDocument(
            doc_id=doc_id,
            content=content,
            doc_type="event",
            symbol=self.symbol,
            fiscal_year=self.event_date.year,
            fiscal_period=f"Q{((self.event_date.month - 1) // 3) + 1}",
            form_type=self.form_type,
            topics=[self.event_type.value],
            sentiment_score=self.sentiment_score,
            importance_score=self._calculate_importance_score(),
            metadata={
                "event_type": self.event_type.value,
                "severity": self.severity.value,
                "event_date": self.event_date.isoformat(),
                "filing_date": self.filing_date.isoformat(),
                "market_impact": self.market_impact,
                "affected_metrics": self.affected_metrics,
                "mentioned_companies": self.mentioned_companies,
                "mentioned_people": self.mentioned_people,
                "section": self.section,
                "accession_number": self.accession_number,
                **self.metadata,
            },
        )

    def _calculate_importance_score(self) -> float:
        """Calculate importance score based on event characteristics"""
        base_scores = {
            EventSeverity.CRITICAL: 0.9,
            EventSeverity.HIGH: 0.7,
            EventSeverity.MEDIUM: 0.5,
            EventSeverity.LOW: 0.3,
        }

        score = base_scores.get(self.severity, 0.5)

        # Adjust based on event type
        if self.event_type in [EventType.EARNINGS_RELEASE, EventType.ACQUISITION, EventType.MERGER]:
            score += 0.1

        # Adjust based on sentiment
        if self.sentiment_score is not None:
            score += abs(self.sentiment_score) * 0.1  # More extreme sentiment = more important

        return min(1.0, score)


class EventAnalyzer:
    """Analyzes SEC filings and extracts financial events for vector storage"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Event detection patterns
        self.event_patterns = self._initialize_event_patterns()

        # Sentiment patterns
        self.sentiment_patterns = self._initialize_sentiment_patterns()

    def analyze_filing(
        self,
        filing_content: str,
        symbol: str,
        cik: str,
        form_type: str = "8-K",
        filing_date: date = None,
        accession_number: str = None,
    ) -> List[FinancialEvent]:
        """
        Analyze a SEC filing and extract financial events.

        Args:
            filing_content: Raw text content of the filing
            symbol: Stock ticker symbol
            cik: Central Index Key
            form_type: Type of SEC form (8-K, 10-Q, 10-K, etc.)
            filing_date: Date the filing was submitted
            accession_number: SEC accession number

        Returns:
            List of FinancialEvent objects
        """
        try:
            if not filing_content or not symbol:
                return []

            if filing_date is None:
                filing_date = date.today()

            # Clean and preprocess the filing content
            cleaned_content = self._clean_filing_content(filing_content)

            # Extract events based on form type
            if form_type == "8-K":
                events = self._analyze_8k_filing(cleaned_content, symbol, cik, filing_date, accession_number)
            elif form_type in ["10-Q", "10-K"]:
                events = self._analyze_periodic_filing(
                    cleaned_content, symbol, cik, filing_date, accession_number, form_type
                )
            else:
                events = self._analyze_generic_filing(
                    cleaned_content, symbol, cik, filing_date, accession_number, form_type
                )

            self.logger.info(f"Extracted {len(events)} events from {form_type} filing for {symbol}")
            return events

        except Exception as e:
            self.logger.error(f"Error analyzing filing for {symbol}: {e}")
            return []

    def _analyze_8k_filing(
        self, content: str, symbol: str, cik: str, filing_date: date, accession_number: str
    ) -> List[FinancialEvent]:
        """Analyze 8-K current report for specific events"""
        events = []

        # 8-K forms are structured with specific items
        # Parse by sections/items
        sections = self._extract_8k_sections(content)

        for section_num, section_content in sections.items():
            event_type = self._classify_8k_section(section_num, section_content)

            if event_type != EventType.OTHER:
                # Extract event details
                event = self._extract_event_from_section(
                    section_content, symbol, cik, filing_date, accession_number, event_type, section_num
                )
                if event:
                    events.append(event)

        # Also look for general patterns in the full text
        general_events = self._extract_events_by_patterns(content, symbol, cik, filing_date, accession_number, "8-K")
        events.extend(general_events)

        return events

    def _analyze_periodic_filing(
        self, content: str, symbol: str, cik: str, filing_date: date, accession_number: str, form_type: str
    ) -> List[FinancialEvent]:
        """Analyze 10-Q/10-K periodic reports for events"""
        events = []

        # Look for specific sections that typically contain events
        mda_section = self._extract_section(content, "Management's Discussion and Analysis")
        if mda_section:
            mda_events = self._extract_events_by_patterns(
                mda_section, symbol, cik, filing_date, accession_number, form_type, "MD&A"
            )
            events.extend(mda_events)

        # Look for subsequent events section
        subsequent_events = self._extract_section(content, "Subsequent Events")
        if subsequent_events:
            sub_events = self._extract_events_by_patterns(
                subsequent_events, symbol, cik, filing_date, accession_number, form_type, "Subsequent Events"
            )
            events.extend(sub_events)

        return events

    def _analyze_generic_filing(
        self, content: str, symbol: str, cik: str, filing_date: date, accession_number: str, form_type: str
    ) -> List[FinancialEvent]:
        """Analyze other types of filings for events"""
        return self._extract_events_by_patterns(content, symbol, cik, filing_date, accession_number, form_type)

    def _extract_events_by_patterns(
        self,
        content: str,
        symbol: str,
        cik: str,
        filing_date: date,
        accession_number: str,
        form_type: str,
        section: str = None,
    ) -> List[FinancialEvent]:
        """Extract events using pattern matching"""
        events = []

        for event_type, patterns in self.event_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)

                for match in matches:
                    # Extract surrounding context
                    start = max(0, match.start() - 500)
                    end = min(len(content), match.end() + 500)
                    context = content[start:end].strip()

                    # Create event
                    event_id = f"{symbol}_{event_type.value}_{filing_date}_{match.start()}"

                    event = FinancialEvent(
                        event_id=event_id,
                        event_type=event_type,
                        severity=self._determine_severity(event_type, context),
                        title=self._extract_title(match.group(), context),
                        description=self._extract_description(context),
                        full_text=context,
                        symbol=symbol,
                        cik=cik,
                        event_date=filing_date,  # Approximate - could be refined
                        filing_date=filing_date,
                        form_type=form_type,
                        accession_number=accession_number,
                        section=section,
                        sentiment_score=self._analyze_sentiment(context),
                        market_impact=self._determine_market_impact(event_type, context),
                        mentioned_companies=self._extract_mentioned_companies(context),
                        mentioned_people=self._extract_mentioned_people(context),
                    )

                    events.append(event)

        return events

    def _initialize_event_patterns(self) -> Dict[EventType, List[str]]:
        """Initialize regex patterns for different event types"""
        return {
            EventType.EARNINGS_RELEASE: [
                r"earnings\s+release",
                r"quarterly\s+results",
                r"reported\s+net\s+income",
                r"earnings\s+per\s+share",
            ],
            EventType.MANAGEMENT_CHANGE: [
                r"appointed.*(?:CEO|CFO|president|director)",
                r"resigned.*(?:CEO|CFO|president|director)",
                r"chief\s+executive\s+officer.*(?:appointed|resigned|retired)",
                r"board\s+of\s+directors.*(?:elected|appointed|resigned)",
            ],
            EventType.ACQUISITION: [
                r"acquire(?:d|s|ing).*(?:company|business|assets)",
                r"purchase(?:d|s|ing).*(?:company|business|assets)",
                r"acquisition\s+of",
                r"completed.*acquisition",
            ],
            EventType.MERGER: [
                r"merger\s+with",
                r"merge(?:d|s|ing)\s+with",
                r"combination\s+with",
                r"definitive\s+merger\s+agreement",
            ],
            EventType.MATERIAL_AGREEMENT: [
                r"material\s+agreement",
                r"definitive\s+agreement",
                r"entered\s+into.*agreement",
                r"contract.*(?:entered|signed|executed)",
            ],
            EventType.DEBT_ISSUANCE: [
                r"debt\s+offering",
                r"bond\s+issuance",
                r"credit\s+facility",
                r"term\s+loan",
                r"revolving\s+credit",
            ],
            EventType.EQUITY_ISSUANCE: [
                r"equity\s+offering",
                r"stock\s+offering",
                r"common\s+stock.*issued",
                r"share\s+issuance",
            ],
            EventType.DIVIDEND_ANNOUNCEMENT: [
                r"dividend.*(?:declared|announced|increased|decreased)",
                r"quarterly\s+dividend",
                r"special\s+dividend",
            ],
            EventType.SHARE_REPURCHASE: [
                r"share\s+repurchase",
                r"stock\s+buyback",
                r"repurchase\s+program",
                r"buy\s+back.*shares",
            ],
            EventType.RESTRUCTURING: [
                r"restructuring\s+plan",
                r"cost\s+reduction\s+program",
                r"workforce\s+reduction",
                r"facility\s+closure",
                r"reorganization",
            ],
            EventType.LITIGATION: [
                r"lawsuit",
                r"legal\s+proceedings",
                r"litigation",
                r"settlement.*(?:lawsuit|litigation|legal)",
            ],
            EventType.REGULATORY_ACTION: [
                r"SEC.*(?:investigation|action|settlement)",
                r"regulatory.*(?:investigation|action|approval)",
                r"FDA.*approval",
                r"antitrust.*(?:investigation|approval)",
            ],
        }

    def _initialize_sentiment_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for sentiment analysis"""
        return {
            "positive": [
                r"strong(?:er)?\s+(?:performance|growth|results)",
                r"exceed(?:ed|s)?\s+expectations",
                r"record\s+(?:revenue|earnings|performance)",
                r"successful(?:ly)?",
                r"improved?\s+(?:performance|results|margins)",
                r"growth\s+in\s+(?:revenue|earnings|market\s+share)",
            ],
            "negative": [
                r"weak(?:er)?\s+(?:performance|growth|results)",
                r"below\s+expectations",
                r"decline\s+in\s+(?:revenue|earnings|performance)",
                r"loss(?:es)?\s+(?:of|in|from)",
                r"restructuring\s+charges",
                r"impairment\s+charges?",
                r"investigation",
                r"lawsuit",
            ],
        }

    def _clean_filing_content(self, content: str) -> str:
        """Clean and preprocess filing content"""
        # Remove HTML tags
        content = re.sub(r"<[^>]+>", "", content)

        # Remove excessive whitespace
        content = re.sub(r"\s+", " ", content)

        # Remove table formatting artifacts
        content = re.sub(r"\|\s*\|", "", content)
        content = re.sub(r"-{3,}", "", content)

        return content.strip()

    def _extract_8k_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from 8-K filing by Item numbers"""
        sections = {}

        # Pattern to match Item sections
        item_pattern = r"Item\s+(\d+\.\d+)\s+([^\n]*)\n(.*?)(?=Item\s+\d+\.\d+|$)"
        matches = re.finditer(item_pattern, content, re.IGNORECASE | re.DOTALL)

        for match in matches:
            item_num = match.group(1)
            item_title = match.group(2).strip()
            item_content = match.group(3).strip()

            if item_content:  # Only include non-empty sections
                sections[item_num] = f"{item_title}\n\n{item_content}"

        return sections

    def _classify_8k_section(self, section_num: str, content: str) -> EventType:
        """Classify 8-K section by Item number"""
        # Standard 8-K Item mappings
        item_mappings = {
            "1.01": EventType.MATERIAL_AGREEMENT,
            "1.02": EventType.ACQUISITION,
            "2.01": EventType.ACQUISITION,
            "2.02": EventType.EQUITY_ISSUANCE,
            "2.03": EventType.MATERIAL_AGREEMENT,
            "2.06": EventType.DEBT_ISSUANCE,
            "5.02": EventType.MANAGEMENT_CHANGE,
            "5.03": EventType.MANAGEMENT_CHANGE,
            "7.01": EventType.REGULATORY_ACTION,
            "8.01": EventType.OTHER,  # Other Events
        }

        return item_mappings.get(section_num, EventType.OTHER)

    def _extract_event_from_section(
        self,
        content: str,
        symbol: str,
        cik: str,
        filing_date: date,
        accession_number: str,
        event_type: EventType,
        section_num: str,
    ) -> Optional[FinancialEvent]:
        """Extract specific event from 8-K section"""
        if not content.strip():
            return None

        # Extract title from first line or paragraph
        lines = content.split("\n")
        title = lines[0].strip() if lines else f"Item {section_num} Event"

        # Create description from first paragraph
        description = self._extract_description(content)

        event_id = f"{symbol}_{event_type.value}_{section_num}_{filing_date}"

        return FinancialEvent(
            event_id=event_id,
            event_type=event_type,
            severity=self._determine_severity(event_type, content),
            title=title,
            description=description,
            full_text=content,
            symbol=symbol,
            cik=cik,
            event_date=filing_date,
            filing_date=filing_date,
            form_type="8-K",
            accession_number=accession_number,
            section=f"Item {section_num}",
            sentiment_score=self._analyze_sentiment(content),
            market_impact=self._determine_market_impact(event_type, content),
            mentioned_companies=self._extract_mentioned_companies(content),
            mentioned_people=self._extract_mentioned_people(content),
        )

    def _extract_section(self, content: str, section_name: str) -> Optional[str]:
        """Extract a named section from filing content"""
        # Pattern to find section headers
        pattern = rf"{re.escape(section_name)}.*?\n(.*?)(?=\n[A-Z][^a-z]*\n|$)"
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)

        if match:
            return match.group(1).strip()

        return None

    def _determine_severity(self, event_type: EventType, content: str) -> EventSeverity:
        """Determine event severity based on type and content"""
        # Base severity by event type
        severity_map = {
            EventType.BANKRUPTCY: EventSeverity.CRITICAL,
            EventType.MERGER: EventSeverity.CRITICAL,
            EventType.ACQUISITION: EventSeverity.HIGH,
            EventType.MANAGEMENT_CHANGE: EventSeverity.HIGH,
            EventType.EARNINGS_RELEASE: EventSeverity.HIGH,
            EventType.DEBT_ISSUANCE: EventSeverity.MEDIUM,
            EventType.EQUITY_ISSUANCE: EventSeverity.MEDIUM,
            EventType.RESTRUCTURING: EventSeverity.MEDIUM,
            EventType.LITIGATION: EventSeverity.MEDIUM,
            EventType.DIVIDEND_ANNOUNCEMENT: EventSeverity.LOW,
        }

        base_severity = severity_map.get(event_type, EventSeverity.LOW)

        # Adjust based on content indicators
        if any(word in content.lower() for word in ["material", "significant", "major", "substantial"]):
            if base_severity == EventSeverity.LOW:
                return EventSeverity.MEDIUM
            elif base_severity == EventSeverity.MEDIUM:
                return EventSeverity.HIGH

        return base_severity

    def _extract_title(self, match_text: str, context: str) -> str:
        """Extract meaningful title for the event"""
        # Use the matched text as base, clean it up
        title = match_text.strip()

        # Capitalize properly
        title = " ".join(word.capitalize() for word in title.split())

        # Truncate if too long
        if len(title) > 100:
            title = title[:97] + "..."

        return title

    def _extract_description(self, content: str) -> str:
        """Extract description from content"""
        # Take first 2-3 sentences
        sentences = re.split(r"[.!?]+", content)

        description_sentences = []
        char_count = 0

        for sentence in sentences[:5]:  # Max 5 sentences
            sentence = sentence.strip()
            if sentence and char_count + len(sentence) < 500:
                description_sentences.append(sentence)
                char_count += len(sentence)
            else:
                break

        description = ". ".join(description_sentences)
        if description and not description.endswith("."):
            description += "."

        return description or content[:200] + "..." if len(content) > 200 else content

    def _analyze_sentiment(self, content: str) -> Optional[float]:
        """Analyze sentiment of event content"""
        positive_count = 0
        negative_count = 0

        content_lower = content.lower()

        # Count positive indicators
        for pattern in self.sentiment_patterns["positive"]:
            matches = re.findall(pattern, content_lower)
            positive_count += len(matches)

        # Count negative indicators
        for pattern in self.sentiment_patterns["negative"]:
            matches = re.findall(pattern, content_lower)
            negative_count += len(matches)

        total_indicators = positive_count + negative_count
        if total_indicators == 0:
            return None

        # Calculate score (-1 to 1)
        sentiment_score = (positive_count - negative_count) / total_indicators
        return max(-1.0, min(1.0, sentiment_score))

    def _determine_market_impact(self, event_type: EventType, content: str) -> Optional[str]:
        """Determine likely market impact"""
        sentiment = self._analyze_sentiment(content)

        # Base impact by event type
        if event_type in [EventType.BANKRUPTCY, EventType.LITIGATION]:
            return "negative"
        elif event_type in [EventType.ACQUISITION, EventType.DIVIDEND_ANNOUNCEMENT]:
            return "positive"

        # Use sentiment if available
        if sentiment is not None:
            if sentiment > 0.2:
                return "positive"
            elif sentiment < -0.2:
                return "negative"

        return "neutral"

    def _extract_mentioned_companies(self, content: str) -> List[str]:
        """Extract mentions of other companies"""
        # Look for company name patterns (simplified)
        company_patterns = [
            r"\b[A-Z][a-z]+\s+(?:Inc|Corp|Corporation|Company|Ltd|LLC)\b",
            r"\b[A-Z]+\s+(?:Inc|Corp|Corporation|Company|Ltd|LLC)\b",
        ]

        mentioned = set()
        for pattern in company_patterns:
            matches = re.findall(pattern, content)
            mentioned.update(matches)

        return list(mentioned)[:10]  # Limit to 10

    def _extract_mentioned_people(self, content: str) -> List[str]:
        """Extract mentions of people (executives, etc.)"""
        # Look for titles followed by names
        people_patterns = [
            r"(?:CEO|CFO|President|Chairman|Director)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+),?\s+(?:CEO|CFO|President|Chairman|Director)",
        ]

        mentioned = set()
        for pattern in people_patterns:
            matches = re.findall(pattern, content)
            mentioned.update(matches)

        return list(mentioned)[:10]  # Limit to 10
