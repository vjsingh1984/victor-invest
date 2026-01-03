# Multi-stage Dockerfile for InvestiGator
# Stage 1: Build environment
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Stage 2: Runtime environment
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 investigator && \
    mkdir -p /app /data /logs && \
    chown -R investigator:investigator /app /data /logs

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=investigator:investigator . /app

# Create necessary directories
RUN mkdir -p \
    /data/llm_cache \
    /data/sec_cache \
    /data/technical_cache \
    /data/price_cache \
    /data/vector_db \
    /logs \
    /app/reports \
    && chown -R investigator:investigator /data /logs /app/reports

# Switch to non-root user
USER investigator

# Set environment variables
ENV PYTHONPATH=/app \
    INVESTIGATOR_DATA_DIR=/data \
    INVESTIGATOR_LOG_DIR=/logs \
    INVESTIGATOR_REPORT_DIR=/app/reports \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]