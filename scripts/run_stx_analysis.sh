#!/bin/bash
# STX Comprehensive Analysis Script
# Run this from the project root directory

set -e  # Exit on any error

echo "=== STX Comprehensive Analysis Setup ==="
echo ""

# Get project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"

echo "[1/6] Setting PYTHONPATH..."
echo "  Project Root: $PROJECT_ROOT"
echo "  PYTHONPATH=$PYTHONPATH"
echo ""

# Source database credentials
echo "[2/6] Loading database credentials..."
if [ -f ~/.ibkr_tradeapp.env ]; then
    source ~/.ibkr_tradeapp.env

    # Parse postgres URL: postgres://stockuser:Am1nt0r@dataserver1:5432/stock
    DB_URL="postgres://stockuser:Am1nt0r@dataserver1:5432/stock"

    # Use Python for robust URL parsing
    CREDS=$(python3 <<EOF
import re
url = "$DB_URL"
# Pattern: postgres://user:password@host:port/database
match = re.match(r'postgres://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', url)
if match:
    user, password, host, port, database = match.groups()
    print(f"{user}|{password}|{host}|{port}|{database}")
EOF
)

    STOCK_DB_USER=$(echo "$CREDS" | cut -d'|' -f1)
    STOCK_DB_PASSWORD=$(echo "$CREDS" | cut -d'|' -f2)
    STOCK_DB_HOST=$(echo "$CREDS" | cut -d'|' -f3)
    STOCK_DB_PORT=$(echo "$CREDS" | cut -d'|' -f4)
    STOCK_DB_NAME=$(echo "$CREDS" | cut -d'|' -f5)

    # Set stock database environment variables
    export STOCK_DB_USER="$STOCK_DB_USER"
    export STOCK_DB_PASSWORD="$STOCK_DB_PASSWORD"
    export STOCK_DB_HOST="$STOCK_DB_HOST"
    export STOCK_DB_PORT="$STOCK_DB_PORT"
    export STOCK_DB_NAME="$STOCK_DB_NAME"

    # Set SEC database variables (same host, different database)
    export SEC_DB_USER="$STOCK_DB_USER"
    export SEC_DB_PASSWORD="$STOCK_DB_PASSWORD"
    export SEC_DB_HOST="$STOCK_DB_HOST"
    export SEC_DB_PORT="$STOCK_DB_PORT"
    export SEC_DB_NAME="sec_database"

    # Legacy variables
    export DB_USER="$STOCK_DB_USER"
    export DB_PASSWORD="$STOCK_DB_PASSWORD"
    export DB_HOST="$STOCK_DB_HOST"
    export DB_PORT="$STOCK_DB_PORT"
    export DB_DATABASE="$STOCK_DB_NAME"

    echo "  ✓ Database credentials loaded"
    echo "    Stock DB: ${STOCK_DB_USER}@${STOCK_DB_HOST}:${STOCK_DB_PORT}/${STOCK_DB_NAME}"
    echo "    SEC DB: ${SEC_DB_USER}@${SEC_DB_HOST}:${SEC_DB_PORT}/${SEC_DB_NAME}"
    echo ""
else
    echo "  ✗ Error: ~/.ibkr_tradeapp.env not found"
    echo "  Please create it with database credentials"
    exit 1
fi

# Set FRED API key
export FRED_API_KEY="${TRADING__DATABASE__POSTGRES_URL:-6f80a1fbcf86c0a25a67a0a7e32b9de6}"

echo "[3/6] Checking Ollama Server..."
OLLAMA_HOST="192.168.1.20:11434"
OLLAMA_BASE_URL="http://192.168.1.20:11434"
export OLLAMA_HOST="$OLLAMA_HOST"
export OLLAMA_BASE_URL="$OLLAMA_BASE_URL"
echo "  Trying to connect to Ollama at $OLLAMA_BASE_URL..."

if curl -s http://$OLLAMA_HOST/api/tags > /dev/null 2>&1; then
    echo "  ✓ Ollama server is running"
    echo ""
    echo "  Available models:"
    curl -s http://$OLLAMA_HOST/api/tags 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'models' in data:
    for m in data['models']:
        print(f\"    - {m['name']}\")
" || echo "    (unable to list models)"
    echo ""

    # Check for GPT-OSS
    if curl -s http://$OLLAMA_HOST/api/tags 2>/dev/null | grep -q "gpt-oss"; then
        echo "  ✓ GPT-OSS model available"
    else
        echo "  ⚠ GPT-OSS model not found on server"
        echo "  Please run on Windows: ollama pull gpt-oss"
        echo "  Press Enter to continue anyway (or Ctrl+C to cancel)..."
        read
    fi
else
    echo "  ✗ Ollama server not running at http://$OLLAMA_HOST"
    echo ""
    echo "  Please ensure Ollama is running on Windows at 192.168.1.20:11434"
    echo ""
    echo "  Press Enter when ready (or Ctrl+C to cancel)..."
    read
fi

echo "[4/6] Configuring Analysis Parameters..."
echo "  Symbol: STX (Seagate Technology)"
echo "  Mode: comprehensive"
echo "  LLM: GPT-OSS at http://$OLLAMA_HOST"
echo "  Database: ${STOCK_DB_HOST} (stock) and ${SEC_DB_HOST} (SEC)"
echo ""

echo "[5/6] Running STX Comprehensive Analysis..."
cd "$PROJECT_ROOT"

# Run the analysis
investigator analyze single STX --mode comprehensive --detail verbose

echo ""
echo "[6/6] Analysis Complete!"
echo "  Check the output above for results and any issues"
