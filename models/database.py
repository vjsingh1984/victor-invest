"""
Database Models for InvestiGator
SQLAlchemy models for the agentic architecture
"""

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Boolean,
    Text,
    JSON,
    ForeignKey,
    Index,
    UniqueConstraint,
    CheckConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum as PyEnum

Base = declarative_base()


class AnalysisStatus(PyEnum):
    """Analysis status enumeration"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(PyEnum):
    """Agent type enumeration"""

    SEC = "sec"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SYNTHESIS = "synthesis"
    SENTIMENT = "sentiment"


class Company(Base):
    """Company master data"""

    __tablename__ = "companies"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    cik = Column(String(10), unique=True, index=True)
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    exchange = Column(String(50))
    ipo_date = Column(DateTime)
    description = Column(Text)
    website = Column(String(255))
    employees = Column(Integer)
    headquarters = Column(String(255))
    metadata = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    # Relationships
    analyses = relationship("Analysis", back_populates="company", cascade="all, delete-orphan")
    financials = relationship("Financial", back_populates="company", cascade="all, delete-orphan")
    filings = relationship("SECFiling", back_populates="company", cascade="all, delete-orphan")

    __table_args__ = (Index("idx_company_sector_industry", "sector", "industry"),)


class Analysis(Base):
    """Analysis task record"""

    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True)
    task_id = Column(String(100), unique=True, nullable=False, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    mode = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default=AnalysisStatus.PENDING.value)
    priority = Column(Integer, default=3)
    requested_by = Column(String(100))

    # Timing
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)

    # Results
    overall_score = Column(Float)
    confidence = Column(Float)
    recommendation = Column(String(50))
    risk_score = Column(Float)

    # Full analysis data
    results = Column(JSON)
    errors = Column(JSON)
    metadata = Column(JSON)

    # Relationships
    company = relationship("Company", back_populates="analyses")
    agent_results = relationship("AgentResult", back_populates="analysis", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_analysis_status_created", "status", "created_at"),
        Index("idx_analysis_company_created", "company_id", "created_at"),
        CheckConstraint("priority >= 1 AND priority <= 5", name="check_priority_range"),
    )


class AgentResult(Base):
    """Individual agent execution results"""

    __tablename__ = "agent_results"

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    agent_type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)

    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)

    # Model info
    model_used = Column(String(100))
    tokens_used = Column(Integer)

    # Results
    result_data = Column(JSON)
    error_message = Column(Text)

    # Relationships
    analysis = relationship("Analysis", back_populates="agent_results")

    __table_args__ = (Index("idx_agent_result_analysis_type", "analysis_id", "agent_type"),)


class Financial(Base):
    """Financial statements and metrics"""

    __tablename__ = "financials"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    period_end = Column(DateTime, nullable=False)
    period_type = Column(String(20))  # annual, quarterly

    # Income Statement
    revenue = Column(Float)
    gross_profit = Column(Float)
    operating_income = Column(Float)
    net_income = Column(Float)
    eps = Column(Float)
    diluted_eps = Column(Float)

    # Balance Sheet
    total_assets = Column(Float)
    current_assets = Column(Float)
    total_liabilities = Column(Float)
    current_liabilities = Column(Float)
    total_equity = Column(Float)
    cash = Column(Float)
    total_debt = Column(Float)

    # Cash Flow
    operating_cash_flow = Column(Float)
    investing_cash_flow = Column(Float)
    financing_cash_flow = Column(Float)
    free_cash_flow = Column(Float)
    capex = Column(Float)

    # Ratios (calculated)
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    ps_ratio = Column(Float)
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    quick_ratio = Column(Float)
    gross_margin = Column(Float)
    operating_margin = Column(Float)
    net_margin = Column(Float)
    roe = Column(Float)
    roa = Column(Float)
    roic = Column(Float)

    # Additional data
    shares_outstanding = Column(Float)
    raw_data = Column(JSON)
    source = Column(String(50))
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    company = relationship("Company", back_populates="financials")

    __table_args__ = (
        UniqueConstraint("company_id", "period_end", "period_type", name="uq_company_period"),
        Index("idx_financial_company_period", "company_id", "period_end"),
    )


class SECFiling(Base):
    """SEC filing records"""

    __tablename__ = "sec_filings"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    filing_type = Column(String(20), nullable=False)  # 10-K, 10-Q, 8-K, etc.
    filing_date = Column(DateTime, nullable=False)
    period_end = Column(DateTime)
    accession_number = Column(String(50), unique=True)

    # URLs
    form_url = Column(String(500))
    xbrl_url = Column(String(500))

    # Extracted sections
    business_section = Column(Text)
    risk_factors = Column(Text)
    mda_section = Column(Text)
    financial_section = Column(Text)

    # Analysis results
    risks_identified = Column(JSON)
    key_metrics = Column(JSON)
    sentiment_scores = Column(JSON)

    # Metadata
    file_size = Column(Integer)
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    company = relationship("Company", back_populates="filings")

    __table_args__ = (
        Index("idx_filing_company_date", "company_id", "filing_date"),
        Index("idx_filing_type_date", "filing_type", "filing_date"),
    )


class MarketData(Base):
    """Historical market data"""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    date = Column(DateTime, nullable=False)

    # OHLCV data
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    adjusted_close = Column(Float)
    volume = Column(Integer)

    # Technical indicators (pre-calculated)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    bb_upper = Column(Float)
    bb_lower = Column(Float)

    # Additional metrics
    daily_return = Column(Float)
    volatility_20 = Column(Float)

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_symbol_date"),
        Index("idx_market_data_symbol_date", "symbol", "date"),
    )


class CacheEntry(Base):
    """Cache management table"""

    __tablename__ = "cache_entries"

    id = Column(Integer, primary_key=True)
    key = Column(String(255), unique=True, nullable=False, index=True)
    tier = Column(String(20), nullable=False)  # L1, L2, L3, L4
    size_bytes = Column(Integer)

    # Access patterns
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime)

    # Metadata
    data_type = Column(String(50))
    tags = Column(JSON)

    __table_args__ = (
        Index("idx_cache_tier_accessed", "tier", "last_accessed"),
        Index("idx_cache_expires", "expires_at"),
    )


class PeerGroup(Base):
    """Peer group definitions"""

    __tablename__ = "peer_groups"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    target_symbol = Column(String(10), nullable=False)
    peer_symbols = Column(JSON, nullable=False)  # List of peer symbols

    # Group characteristics
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap_range = Column(JSON)  # {"min": x, "max": y}

    # Analysis metadata
    last_analyzed = Column(DateTime)
    auto_update = Column(Boolean, default=True)
    update_frequency_days = Column(Integer, default=7)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    __table_args__ = (Index("idx_peer_group_target", "target_symbol"),)


class Alert(Base):
    """System alerts and notifications"""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)  # critical, warning, info
    source = Column(String(100))

    # Alert details
    title = Column(String(255), nullable=False)
    message = Column(Text)
    details = Column(JSON)

    # Status
    active = Column(Boolean, default=True)
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_alert_active_severity", "active", "severity"),
        Index("idx_alert_created", "created_at"),
    )


class Report(Base):
    """Generated reports"""

    __tablename__ = "reports"

    id = Column(Integer, primary_key=True)
    report_type = Column(String(50), nullable=False)  # individual, peer_comparison, portfolio
    name = Column(String(255), nullable=False)

    # Report data
    symbols = Column(JSON)  # List of symbols included
    analysis_ids = Column(JSON)  # List of analysis IDs used
    content = Column(JSON)  # Full report content

    # File references
    pdf_path = Column(String(500))
    html_path = Column(String(500))

    # Metadata
    generated_by = Column(String(100))
    generation_time_seconds = Column(Float)
    template_used = Column(String(100))

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (Index("idx_report_type_created", "report_type", "created_at"),)


class SystemMetric(Base):
    """System performance metrics"""

    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Resource usage
    cpu_percent = Column(Float)
    memory_percent = Column(Float)
    disk_usage_percent = Column(Float)

    # Application metrics
    active_agents = Column(Integer)
    queue_size = Column(Integer)
    cache_hit_rate = Column(Float)
    average_latency = Column(Float)

    # Throughput
    analyses_per_hour = Column(Float)
    success_rate = Column(Float)

    # Detailed metrics
    agent_metrics = Column(JSON)
    cache_metrics = Column(JSON)
    error_counts = Column(JSON)

    __table_args__ = (Index("idx_metric_timestamp", "timestamp"),)
