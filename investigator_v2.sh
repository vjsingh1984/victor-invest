#!/opt/homebrew/bin/bash
#
# InvestiGator v2 - Victor-Invest CLI with Multi-Server Support
#
# Main entrypoint: python -m victor_invest.cli (modern) or cli_orchestrator.py (legacy)
#
# This script provides the primary CLI interface using the victor_invest framework,
# which includes shared market data services for consistent behavior across:
# - rl_backtest.py (historical backtesting)
# - batch_analysis_runner.py (batch processing)
# - victor_invest CLI (single stock analysis)
#
# Key features:
# - Uses shared services for shares, prices, metadata, validation
# - Consistent split detection and normalization
# - SEC filing data as source of truth for shares outstanding
#

set -euo pipefail

# Error trap for debugging
trap 'echo "ERROR: Script failed at line $LINENO with exit code $?" >&2' ERR

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prevent Python from creating .pyc bytecode cache files
export PYTHONDONTWRITEBYTECODE=1

# Proactive bytecode cleanup on EVERY run to prevent stale code issues
cleanup_bytecode() {
    # Silent cleanup - only show errors if they occur
    find "${SCRIPT_DIR}/src" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${SCRIPT_DIR}/src" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "${SCRIPT_DIR}/utils" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${SCRIPT_DIR}/utils" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "${SCRIPT_DIR}/patterns" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${SCRIPT_DIR}/patterns" -type f -name "*.pyc" -delete 2>/dev/null || true
}

# Run bytecode cleanup on every execution to prevent stale code issues
cleanup_bytecode

# Default values
MODE="comprehensive"
OUTPUT_DIR="results"
SYMBOL=""
SYMBOLS=()
FORMAT="json"
DETAIL_LEVEL="standard"
LOG_LEVEL=""
GENERATE_REPORT=false
RISK_ASSESSMENT=false
WEEKLY_REPORT=false
PEER_GROUPS=false
PEER_SECTOR=""
PEER_INDUSTRY=""
PEER_RISK=false
SYNTHESIS_MODE=""
CACHE_OPERATION=""
TEST_OPERATION=""
SETUP_OPERATION=""
SYSTEM_STATS=false
FORCE_REFRESH=false
DEBUG=false
VERBOSE=false
BATCH_FILE=""

# RL Backtest & Training options
RL_BACKTEST=false
RL_BACKTEST_WORKFLOW=false
RL_TRAIN=false
RL_DEPLOY=false
RL_STATUS=false
RL_LOOKBACK=120
RL_LOOKBACK_RANGE=""
RL_INTERVAL="quarterly"
RL_PARALLEL=5
RL_ALL_SYMBOLS=false
RL_TOP_N=""

# Macro Data options
MACRO_SUMMARY=false
MACRO_BUFFETT=false
MACRO_CATEGORY=""
MACRO_INDICATORS=""
MACRO_TIME_SERIES=""
MACRO_LIST_CATEGORIES=false
MACRO_LOOKBACK_DAYS=1095
MACRO_LIMIT=1000
MACRO_JSON=false

# Insider Trading options
INSIDER_SENTIMENT=false
INSIDER_RECENT=false
INSIDER_CLUSTERS=false
INSIDER_KEY_INSIDERS=false
INSIDER_FETCH=false
INSIDER_DAYS=90
INSIDER_SIGNIFICANT_ONLY=false
INSIDER_JSON=false

# Treasury Data options
TREASURY_CURVE=false
TREASURY_SPREAD=false
TREASURY_REGIME=false
TREASURY_RECESSION=false
TREASURY_SUMMARY=false
TREASURY_HISTORY=false
TREASURY_DAYS=365
TREASURY_MATURITY="10y"
TREASURY_JSON=false

# Institutional Holdings options
INST_HOLDINGS=false
INST_TOP_HOLDERS=false
INST_CHANGES=false
INST_INSTITUTION=""
INST_SEARCH=""
INST_LIMIT=20
INST_QUARTERS=4
INST_QUARTER=""
INST_JSON=false

# Short Interest options
SHORT_CURRENT=false
SHORT_HISTORY=false
SHORT_VOLUME=false
SHORT_SQUEEZE=false
SHORT_MOST_SHORTED=false
SHORT_PERIODS=12
SHORT_DAYS=30
SHORT_LIMIT=20
SHORT_JSON=false

# Market Regime options
REGIME_SUMMARY=false
REGIME_CREDIT_CYCLE=false
REGIME_YIELD_CURVE=false
REGIME_RECESSION=false
REGIME_VOLATILITY=false
REGIME_RECOMMENDATIONS=false
REGIME_JSON=false

# Valuation Signals options
VAL_INTEGRATE=false
VAL_CREDIT_RISK=false
VAL_INSIDER=false
VAL_SHORT_INTEREST=false
VAL_MARKET_REGIME=false
VAL_BASE_FV=""
VAL_PRICE=""
VAL_JSON=false

# Print with color
log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $*"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $*"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $*"
}

# Usage information - aligned with investigator.sh
show_usage() {
    cat << EOF
${CYAN}╔════════════════════════════════════════════════════════════════════════════════╗${NC}
${CYAN}║${NC}                                                                                ${CYAN}║${NC}
${CYAN}║${NC}      ${GREEN}███${NC}${WHITE}╗${NC}   ${GREEN}███${NC}${WHITE}╗${NC} ${GREEN}██${NC}${WHITE}╗${NC} ${GREEN}███${NC}${WHITE}╗   ${NC}${GREEN}██${NC}${WHITE}╗${NC} ${GREEN}██${NC}${WHITE}╗${NC}   ${GREEN}███${NC}${WHITE}╗${NC}  ${GREEN}███████${NC}${WHITE}╗${NC}  ${GREEN}██████${NC}${WHITE}╗${NC}  ${GREEN}████████${NC}${WHITE}╗${NC}     ${CYAN}║${NC}
${CYAN}║${NC}      ${GREEN}████${NC}${WHITE}╗${NC} ${GREEN}████${NC}${WHITE}║${NC} ${GREEN}██${NC}${WHITE}║${NC} ${GREEN}████${NC}${WHITE}╗${NC}  ${GREEN}██${NC}${WHITE}║${NC} ${GREEN}██${NC}${WHITE}║${NC}  ${GREEN}████${NC}${WHITE}║${NC}  ${GREEN}██${NC}${WHITE}╔════╝${NC} ${GREEN}██${NC}${WHITE}╔═══${NC}${GREEN}██${NC}${WHITE}╗${NC} ╚══${GREEN}██${NC}${WHITE}╔══╝${NC}     ${CYAN}║${NC}
${CYAN}║${NC}      ${GREEN}██${NC}${WHITE}╔${NC}${GREEN}████${NC}${WHITE}╔${NC}${GREEN}██${NC}${WHITE}║${NC} ${GREEN}██${NC}${WHITE}║${NC} ${GREEN}██${NC}${WHITE}╔${NC}${GREEN}██${NC}${WHITE}╗${NC} ${GREEN}██${NC}${WHITE}║${NC} ${GREEN}██${NC}${WHITE}║${NC} ${GREEN}██${NC}${WHITE}╔${NC}${GREEN}██${NC}${WHITE}║${NC}  ${GREEN}█████${NC}${WHITE}╗${NC}   ${GREEN}██████${NC}${WHITE}╔╝${NC}    ${GREEN}██${NC}${WHITE}║${NC}        ${CYAN}║${NC}
${CYAN}║${NC}      ${GREEN}██${NC}${WHITE}║╚${NC}${GREEN}██${NC}${WHITE}╔╝${NC}${GREEN}██${NC}${WHITE}║${NC} ${GREEN}██${NC}${WHITE}║${NC} ${GREEN}██${NC}${WHITE}║╚${NC}${GREEN}██${NC}${WHITE}╗${NC}${GREEN}██${NC}${WHITE}║${NC} ${GREEN}██${NC}${WHITE}║${NC}${GREEN}██${NC}${WHITE}║╚${NC}${GREEN}██${NC}${WHITE}║${NC}  ${GREEN}██${NC}${WHITE}╔══╝${NC}   ${GREEN}██${NC}${WHITE}╔══${NC}${GREEN}██${NC}${WHITE}╗${NC}    ${GREEN}██${NC}${WHITE}║${NC}        ${CYAN}║${NC}
${CYAN}║${NC}      ${GREEN}██${NC}${WHITE}║${NC} ${WHITE}╚═╝${NC} ${GREEN}██${NC}${WHITE}║${NC} ${GREEN}██${NC}${WHITE}║${NC} ${GREEN}██${NC}${WHITE}║${NC} ${WHITE}╚${NC}${GREEN}████${NC}${WHITE}║${NC} ${WHITE}╚${NC}${GREEN}███${NC}${WHITE}╔${NC}${GREEN}███${NC}${WHITE}╔╝${NC}  ${GREEN}███████${NC}${WHITE}╗${NC} ${GREEN}██${NC}${WHITE}║${NC}  ${GREEN}██${NC}${WHITE}║${NC}    ${GREEN}██${NC}${WHITE}║${NC}        ${CYAN}║${NC}
${CYAN}║${NC}      ${WHITE}╚═╝${NC}     ${WHITE}╚═╝${NC} ${WHITE}╚═╝${NC} ${WHITE}╚═╝${NC}  ${WHITE}╚═══╝${NC}  ${WHITE}╚══╝${NC}${WHITE}╚══╝${NC}   ${WHITE}╚══════╝${NC} ${WHITE}╚═╝${NC}  ${WHITE}╚═╝${NC}    ${WHITE}╚═╝${NC}        ${CYAN}║${NC}
${CYAN}║${NC}                                                                                ${CYAN}║${NC}
${CYAN}║${NC}           ${WHITE}🐊 AI-Powered Investment Research Assistant 🤓${NC}                   ${CYAN}║${NC}
${CYAN}║${NC}           ${YELLOW}Multi-Server Ollama Pool • 3 Concurrent Tasks/Server${NC}           ${CYAN}║${NC}
${CYAN}╚════════════════════════════════════════════════════════════════════════════════╝${NC}

${YELLOW}USAGE:${NC}
    $0 [OPTIONS]

${YELLOW}STOCK ANALYSIS OPTIONS:${NC}
    --symbol SYMBOL              Analyze single stock (complete pipeline)
    --symbols SYMBOL1 SYMBOL2    Analyze multiple stocks (batch mode)
    --mode MODE                  Analysis mode: quick, standard, comprehensive (default: comprehensive)
    --report                     Generate PDF report (only works with --symbol, not --symbols)
    --weekly-report              Generate weekly portfolio report (legacy - uses investigator.sh)

${YELLOW}PEER GROUP ANALYSIS:${NC}
    --peer-groups-comprehensive  Generate comprehensive peer group report with all symbols
    --peer-groups-analysis       Alias for --peer-groups-comprehensive
    --peer-sector SECTOR         Target specific sector (financials, technology, healthcare)
    --peer-industry INDUSTRY     Target specific industry within sector
    --peer-risk-assessment       Include risk assessment in peer group analysis

${YELLOW}CACHE MANAGEMENT:${NC}
    --clean-cache               Clean cache for specified symbols (use with --symbol/--symbols)
    --clean-cache-all           Clean all caches completely
    --clean-cache-db            Clean database cache only
    --clean-cache-disk          Clean disk cache only
    --inspect-cache             Inspect cache contents
    --cache-sizes               Show cache sizes
    --force-refresh             Force refresh data (bypass cache, alias: --refresh)
    --refresh                   Alias for --force-refresh

${YELLOW}RL BACKTEST & TRAINING:${NC}
    --rl-backtest               Run RL backtesting on historical data (script mode)
    --rl-backtest-workflow      Run RL backtesting using victor workflow (StateGraph)
    --rl-train                  Train RL policy from backtest outcomes
    --rl-deploy                 Deploy trained RL policy
    --rl-status                 Show RL policy deployment status
    --lookback MONTHS           Single lookback period in months
    --lookback-range MONTHS     Generate lookbacks from 3mo to this value (e.g., 120 = 10 years)
    --interval TYPE             Interval: quarterly (40 points/10yr) or monthly (120 points/10yr)
    --parallel N                Number of parallel workers (default: 5)
    --all-symbols               Process all eligible symbols
    --top-n N                   Process top N symbols by market cap

${YELLOW}MACRO DATA (FRED):${NC}
    --macro-summary             Get comprehensive macro summary with all indicators
    --macro-buffett             Get Buffett Indicator (Market Cap / GDP ratio)
    --macro-category CATEGORY   Get indicators for category (growth, inflation, rates, etc.)
    --macro-indicators ID,ID    Get specific indicators by FRED series ID
    --macro-time-series ID      Get historical time series for indicator
    --macro-list-categories     List available categories and their indicators
    --macro-lookback-days DAYS  Days of historical data (default: 1095 = 3 years)
    --macro-limit N             Max data points for time series (default: 1000)
    --macro-json                Output raw JSON instead of formatted table

${YELLOW}INSIDER TRADING (SEC Form 4):${NC}
    --insider-sentiment SYMBOL  Get insider sentiment analysis (buy/sell ratio)
    --insider-recent SYMBOL     Get recent insider transactions
    --insider-clusters SYMBOL   Detect coordinated insider buying/selling
    --insider-key-insiders SYM  Get C-suite and director activity summary
    --insider-fetch SYMBOL      Fetch latest Form 4 filings from SEC EDGAR
    --insider-days DAYS         Analysis period in days (default: 90)
    --insider-significant-only  For --insider-recent: show only significant transactions
    --insider-json              Output raw JSON instead of formatted table

${YELLOW}TREASURY DATA & MARKET REGIME:${NC}
    --treasury-curve            Get current Treasury yield curve (1m to 30y)
    --treasury-spread           Get yield spread analysis (10Y-2Y, 10Y-3M)
    --treasury-regime           Get market regime from yield curve shape
    --treasury-recession        Get recession probability & economic phase
    --treasury-summary          Get comprehensive market regime summary
    --treasury-history          Get historical yield data
    --treasury-days DAYS        Days for history (default: 365)
    --treasury-maturity MAT     Maturity for history (default: 10y)
    --treasury-json             Output raw JSON instead of formatted table

${YELLOW}INSTITUTIONAL HOLDINGS (SEC 13F):${NC}
    --inst-holdings SYMBOL      Get institutional ownership summary
    --inst-top-holders SYMBOL   Get top institutional holders by value
    --inst-changes SYMBOL       Get ownership changes over quarters
    --inst-institution CIK      Get holdings for specific institution by CIK
    --inst-search QUERY         Search for institutions by name
    --inst-limit N              Number of results (default: 20)
    --inst-quarters N           Quarters for change analysis (default: 4)
    --inst-quarter QUARTER      Specific quarter (e.g., '2024-Q4')
    --inst-json                 Output raw JSON instead of formatted table

${YELLOW}SHORT INTEREST (FINRA):${NC}
    --short-current SYMBOL      Get current short interest
    --short-history SYMBOL      Get historical short interest
    --short-volume SYMBOL       Get daily short volume
    --short-squeeze SYMBOL      Calculate short squeeze risk
    --short-most-shorted        Get list of most shorted stocks
    --short-periods N           Periods for history (default: 12)
    --short-days N              Days for volume (default: 30)
    --short-limit N             Stocks for most-shorted (default: 20)
    --short-json                Output raw JSON instead of formatted table

${YELLOW}MARKET REGIME DETECTION:${NC}
    --regime-summary            Get comprehensive market regime summary
    --regime-credit-cycle       Get credit cycle phase analysis
    --regime-yield-curve        Get yield curve shape and signals
    --regime-recession          Get recession probability assessment
    --regime-volatility         Get volatility regime classification
    --regime-recommendations    Get sector allocation recommendations
    --regime-json               Output raw JSON instead of formatted table

${YELLOW}VALUATION SIGNAL INTEGRATION:${NC}
    --val-integrate SYMBOL      Full signal integration for adjusted fair value
    --val-credit-risk SYMBOL    Credit risk signal only (Altman Z, Beneish M, Piotroski F)
    --val-insider SYMBOL        Insider sentiment signal only
    --val-short-interest SYMBOL Short interest signal only
    --val-market-regime         Market regime adjustment (no symbol needed)
    --val-base-fv VALUE         Base fair value (required for --val-integrate)
    --val-price VALUE           Current price (required for --val-integrate)
    --val-json                  Output raw JSON instead of formatted table

${YELLOW}SYSTEM OPTIONS:${NC}
    --setup-system              Install dependencies and setup environment
    --setup-database            Setup database schema
    --setup-vectordb            Initialize vector database (RocksDB + FAISS)
    --test-system               Run system tests
    --test-cache                Run cache tests
    --run-tests                 Run all tests
    --system-stats              Show system statistics and metrics
    --debug                     Enable debug logging
    -v, --verbose               Pass verbose mode through to cli_orchestrator.py
    --help                      Show this help message

${YELLOW}OUTPUT OPTIONS:${NC}
    --output-dir DIR            Output directory (default: results)
    --format FORMAT             Output format: json, yaml, text (default: json)
    -d, --detail-level LEVEL    Output detail: minimal (summary), standard (investor-friendly, default), verbose (full)
    -l, --log-level LEVEL       Log level: DEBUG, INFO, WARNING, ERROR
    --batch-file FILE           Load symbols from file (one per line)

${YELLOW}EXAMPLES:${NC}
    ${GREEN}# Single stock analysis${NC}
    $0 --symbol AAPL
    $0 --symbol AAPL --mode quick --report
    $0 --symbol AAPL --force-refresh --report
    $0 --symbol AAPL --mode comprehensive --log-level DEBUG

    ${GREEN}# Output detail levels${NC}
    $0 --symbol AAPL --detail-level minimal          # Executive summary only
    $0 --symbol AAPL --detail-level standard         # Investor-friendly (default, 65% smaller)
    $0 --symbol AAPL --detail-level verbose          # Full analysis with all metadata

    ${GREEN}# Multiple stocks${NC}
    $0 --symbols AAPL GOOGL MSFT
    $0 --symbols AAPL GOOGL MSFT --mode standard
    $0 --symbols AAPL GOOGL MSFT --force-refresh
    $0 --symbols AAPL GOOGL MSFT --detail-level minimal

    ${GREEN}# Batch from file${NC}
    $0 --batch-file symbols.txt --mode comprehensive

    ${GREEN}# Reports and analysis (legacy)${NC}
    $0 --weekly-report
    $0 --peer-groups-comprehensive --peer-sector technology

    ${GREEN}# Cache management${NC}
    $0 --clean-cache --symbol AAPL
    $0 --clean-cache-all
    $0 --inspect-cache

    ${GREEN}# RL Backtest & Training (script mode - faster, batch-optimized)${NC}
    $0 --rl-backtest --all-symbols --lookback-range 120 --interval quarterly --parallel 5
    $0 --rl-backtest --top-n 100 --lookback-range 120 --interval quarterly   # 40 data points per symbol
    $0 --rl-backtest --symbols AAPL MSFT --lookback-range 60 --interval monthly  # 60 data points per symbol

    ${GREEN}# RL Backtest (workflow mode - StateGraph, shared services)${NC}
    $0 --rl-backtest-workflow --symbols AAPL MSFT --lookback-range 120 --parallel 3
    $0 --rl-backtest-workflow --all-symbols --lookback-range 60 --interval quarterly

    ${GREEN}# RL Training & Deployment${NC}
    $0 --rl-train                                # Train from outcomes
    $0 --rl-deploy                               # Deploy trained policy
    $0 --rl-status                               # Show policy status

    ${GREEN}# Macro Data (FRED)${NC}
    $0 --macro-summary                                   # All macro indicators + Buffett
    $0 --macro-buffett                                   # Market Cap / GDP ratio
    $0 --macro-category rates                            # Interest rates & yields
    $0 --macro-category inflation                        # CPI, PCE, breakeven rates
    $0 --macro-indicators DGS10,FEDFUNDS,VIXCLS          # Specific indicators
    $0 --macro-time-series DGS10 --macro-limit 365       # 10Y Treasury history
    $0 --macro-list-categories                           # Show all categories
    $0 --macro-summary --macro-json                      # Output as JSON

    ${GREEN}# Insider Trading (SEC Form 4)${NC}
    $0 --insider-sentiment AAPL                          # Get insider sentiment analysis
    $0 --insider-recent AAPL --insider-days 60           # Recent transactions (60 days)
    $0 --insider-clusters NVDA                           # Detect coordinated activity
    $0 --insider-key-insiders TSLA                       # C-suite/director summary
    $0 --insider-fetch MSFT                              # Fetch fresh Form 4 data
    $0 --insider-recent AAPL --insider-significant-only  # Only significant transactions
    $0 --insider-sentiment AAPL --insider-json           # Output as JSON

    ${GREEN}# Treasury Data & Market Regime${NC}
    $0 --treasury-curve                                  # Current yield curve
    $0 --treasury-spread                                 # Yield spread analysis
    $0 --treasury-regime                                 # Yield curve market regime
    $0 --treasury-recession                              # Recession probability
    $0 --treasury-summary                                # Complete market regime summary
    $0 --treasury-history --treasury-maturity 10y        # 10Y yield history
    $0 --treasury-curve --treasury-json                  # Output as JSON

    ${GREEN}# Institutional Holdings (SEC 13F)${NC}
    $0 --inst-holdings AAPL                              # Institutional ownership summary
    $0 --inst-top-holders AAPL --inst-limit 25           # Top 25 holders
    $0 --inst-changes NVDA --inst-quarters 8             # 8 quarters of changes
    $0 --inst-institution 0001067983                     # Berkshire holdings
    $0 --inst-search "vanguard"                          # Search for institutions
    $0 --inst-holdings AAPL --inst-json                  # Output as JSON

    ${GREEN}# Short Interest (FINRA)${NC}
    $0 --short-current GME                               # Current short interest
    $0 --short-history AMC --short-periods 12            # 12 periods history
    $0 --short-volume AAPL --short-days 30               # 30 days short volume
    $0 --short-squeeze GME                               # Squeeze risk assessment
    $0 --short-most-shorted --short-limit 25             # Top 25 most shorted
    $0 --short-current TSLA --short-json                 # Output as JSON

    ${GREEN}# Market Regime Detection${NC}
    $0 --regime-summary                                  # Comprehensive market regime
    $0 --regime-credit-cycle                             # Credit cycle phase analysis
    $0 --regime-yield-curve                              # Yield curve shape & signals
    $0 --regime-recession                                # Recession probability
    $0 --regime-volatility                               # Volatility regime
    $0 --regime-recommendations                          # Investment recommendations
    $0 --regime-summary --regime-json                    # Output as JSON

    ${GREEN}# Valuation Signal Integration${NC}
    $0 --val-integrate AAPL --val-base-fv 190 --val-price 185  # Full integration
    $0 --val-credit-risk AAPL                            # Credit risk signal only
    $0 --val-insider AAPL                                # Insider sentiment signal
    $0 --val-short-interest GME                          # Short interest signal
    $0 --val-market-regime                               # Market regime adjustment
    $0 --val-integrate AAPL --val-base-fv 190 --val-price 185 --val-json

    ${GREEN}# System operations${NC}
    $0 --test-system
    $0 --system-stats

${YELLOW}MULTI-SERVER ARCHITECTURE:${NC}
    ${GREEN}✓${NC} Automatically distributes work across available Ollama servers
    ${GREEN}✓${NC} Real-time VRAM monitoring via /api/ps
    ${GREEN}✓${NC} Intelligent model reuse (weights shared, KV cache per task)
    ${GREEN}✓${NC} Up to 3 concurrent tasks per server (6 total)

    ${WHITE}📊 Configured Servers:${NC}
    ${CYAN}┌─────────────────────────────────────────────────────────────┐${NC}
    ${CYAN}│${NC} ${BLUE}●${NC} http://localhost:11434                                ${CYAN}│${NC}
    ${CYAN}│${NC}   ${WHITE}├─${NC} 48GB usable VRAM                                  ${CYAN}│${NC}
    ${CYAN}│${NC}   ${WHITE}├─${NC} max_concurrent: 3                                 ${CYAN}│${NC}
    ${CYAN}│${NC}   ${WHITE}└─${NC} Metal GPU acceleration: enabled                   ${CYAN}│${NC}
    ${CYAN}├─────────────────────────────────────────────────────────────┤${NC}
    ${CYAN}│${NC} ${GREEN}●${NC} http://192.168.1.12:11434                            ${CYAN}│${NC}
    ${CYAN}│${NC}   ${WHITE}├─${NC} 36GB usable VRAM                                  ${CYAN}│${NC}
    ${CYAN}│${NC}   ${WHITE}├─${NC} max_concurrent: 3                                 ${CYAN}│${NC}
    ${CYAN}│${NC}   ${WHITE}└─${NC} Metal GPU acceleration: enabled                   ${CYAN}│${NC}
    ${CYAN}└─────────────────────────────────────────────────────────────┘${NC}

${CYAN}╚════════════════════════════════════════════════════════════════════════════════╝${NC}
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            ;;
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --symbols)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                SYMBOLS+=("$1")
                shift
            done
            ;;
        --batch-file)
            BATCH_FILE="$2"
            if [[ -f "$BATCH_FILE" ]]; then
                mapfile -t SYMBOLS < "$BATCH_FILE"
            else
                log_error "Batch file not found: $BATCH_FILE"
                exit 1
            fi
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        -d|--detail-level)
            DETAIL_LEVEL="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --report)
            GENERATE_REPORT=true
            shift
            ;;
        --risk-assessment)
            RISK_ASSESSMENT=true
            shift
            ;;
        --weekly-report)
            WEEKLY_REPORT=true
            shift
            ;;
        --synthesis-mode)
            SYNTHESIS_MODE="$2"
            shift 2
            ;;
        --peer-groups-comprehensive|--peer-groups-analysis)
            PEER_GROUPS=true
            shift
            ;;
        --peer-sector)
            PEER_SECTOR="$2"
            shift 2
            ;;
        --peer-industry)
            PEER_INDUSTRY="$2"
            shift 2
            ;;
        --peer-risk-assessment)
            PEER_RISK=true
            shift
            ;;
        --clean-cache)
            CACHE_OPERATION="--clean-cache"
            shift
            ;;
        --clean-cache-all)
            CACHE_OPERATION="--clean-cache-all"
            shift
            ;;
        --clean-cache-db)
            CACHE_OPERATION="--clean-cache-db"
            shift
            ;;
        --clean-cache-disk)
            CACHE_OPERATION="--clean-cache-disk"
            shift
            ;;
        --inspect-cache)
            CACHE_OPERATION="inspect-cache"
            shift
            ;;
        --cache-sizes)
            CACHE_OPERATION="cache-sizes"
            shift
            ;;
        --force-refresh|--refresh)
            FORCE_REFRESH=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --test-system)
            TEST_OPERATION="test-system"
            shift
            ;;
        --test-cache)
            TEST_OPERATION="test-cache"
            shift
            ;;
        --run-tests)
            TEST_OPERATION="run-tests"
            shift
            ;;
        --setup-database)
            SETUP_OPERATION="setup-database"
            shift
            ;;
        --setup-system)
            SETUP_OPERATION="setup-system"
            shift
            ;;
        --setup-vectordb)
            SETUP_OPERATION="setup-vectordb"
            shift
            ;;
        --system-stats)
            SYSTEM_STATS=true
            shift
            ;;
        --debug)
            DEBUG=true
            LOG_LEVEL="DEBUG"
            shift
            ;;
        --rl-backtest)
            RL_BACKTEST=true
            shift
            ;;
        --rl-backtest-workflow)
            RL_BACKTEST_WORKFLOW=true
            shift
            ;;
        --rl-train)
            RL_TRAIN=true
            shift
            ;;
        --rl-deploy)
            RL_DEPLOY=true
            shift
            ;;
        --rl-status)
            RL_STATUS=true
            shift
            ;;
        --lookback)
            RL_LOOKBACK="$2"
            shift 2
            ;;
        --lookback-range)
            RL_LOOKBACK_RANGE="$2"
            shift 2
            ;;
        --interval)
            RL_INTERVAL="$2"
            shift 2
            ;;
        --parallel)
            RL_PARALLEL="$2"
            shift 2
            ;;
        --all-symbols)
            RL_ALL_SYMBOLS=true
            shift
            ;;
        --top-n)
            RL_TOP_N="$2"
            shift 2
            ;;
        --macro-summary)
            MACRO_SUMMARY=true
            shift
            ;;
        --macro-buffett)
            MACRO_BUFFETT=true
            shift
            ;;
        --macro-category)
            MACRO_CATEGORY="$2"
            shift 2
            ;;
        --macro-indicators)
            MACRO_INDICATORS="$2"
            shift 2
            ;;
        --macro-time-series)
            MACRO_TIME_SERIES="$2"
            shift 2
            ;;
        --macro-list-categories)
            MACRO_LIST_CATEGORIES=true
            shift
            ;;
        --macro-lookback-days)
            MACRO_LOOKBACK_DAYS="$2"
            shift 2
            ;;
        --macro-limit)
            MACRO_LIMIT="$2"
            shift 2
            ;;
        --macro-json)
            MACRO_JSON=true
            shift
            ;;
        --insider-sentiment)
            INSIDER_SENTIMENT=true
            SYMBOL="$2"
            shift 2
            ;;
        --insider-recent)
            INSIDER_RECENT=true
            SYMBOL="$2"
            shift 2
            ;;
        --insider-clusters)
            INSIDER_CLUSTERS=true
            SYMBOL="$2"
            shift 2
            ;;
        --insider-key-insiders)
            INSIDER_KEY_INSIDERS=true
            SYMBOL="$2"
            shift 2
            ;;
        --insider-fetch)
            INSIDER_FETCH=true
            SYMBOL="$2"
            shift 2
            ;;
        --insider-days)
            INSIDER_DAYS="$2"
            shift 2
            ;;
        --insider-significant-only)
            INSIDER_SIGNIFICANT_ONLY=true
            shift
            ;;
        --insider-json)
            INSIDER_JSON=true
            shift
            ;;
        --treasury-curve)
            TREASURY_CURVE=true
            shift
            ;;
        --treasury-spread)
            TREASURY_SPREAD=true
            shift
            ;;
        --treasury-regime)
            TREASURY_REGIME=true
            shift
            ;;
        --treasury-recession)
            TREASURY_RECESSION=true
            shift
            ;;
        --treasury-summary)
            TREASURY_SUMMARY=true
            shift
            ;;
        --treasury-history)
            TREASURY_HISTORY=true
            shift
            ;;
        --treasury-days)
            TREASURY_DAYS="$2"
            shift 2
            ;;
        --treasury-maturity)
            TREASURY_MATURITY="$2"
            shift 2
            ;;
        --treasury-json)
            TREASURY_JSON=true
            shift
            ;;
        --inst-holdings)
            INST_HOLDINGS=true
            SYMBOL="$2"
            shift 2
            ;;
        --inst-top-holders)
            INST_TOP_HOLDERS=true
            SYMBOL="$2"
            shift 2
            ;;
        --inst-changes)
            INST_CHANGES=true
            SYMBOL="$2"
            shift 2
            ;;
        --inst-institution)
            INST_INSTITUTION="$2"
            shift 2
            ;;
        --inst-search)
            INST_SEARCH="$2"
            shift 2
            ;;
        --inst-limit)
            INST_LIMIT="$2"
            shift 2
            ;;
        --inst-quarters)
            INST_QUARTERS="$2"
            shift 2
            ;;
        --inst-quarter)
            INST_QUARTER="$2"
            shift 2
            ;;
        --inst-json)
            INST_JSON=true
            shift
            ;;
        --short-current)
            SHORT_CURRENT=true
            SYMBOL="$2"
            shift 2
            ;;
        --short-history)
            SHORT_HISTORY=true
            SYMBOL="$2"
            shift 2
            ;;
        --short-volume)
            SHORT_VOLUME=true
            SYMBOL="$2"
            shift 2
            ;;
        --short-squeeze)
            SHORT_SQUEEZE=true
            SYMBOL="$2"
            shift 2
            ;;
        --short-most-shorted)
            SHORT_MOST_SHORTED=true
            shift
            ;;
        --short-periods)
            SHORT_PERIODS="$2"
            shift 2
            ;;
        --short-days)
            SHORT_DAYS="$2"
            shift 2
            ;;
        --short-limit)
            SHORT_LIMIT="$2"
            shift 2
            ;;
        --short-json)
            SHORT_JSON=true
            shift
            ;;
        --regime-summary)
            REGIME_SUMMARY=true
            shift
            ;;
        --regime-credit-cycle)
            REGIME_CREDIT_CYCLE=true
            shift
            ;;
        --regime-yield-curve)
            REGIME_YIELD_CURVE=true
            shift
            ;;
        --regime-recession)
            REGIME_RECESSION=true
            shift
            ;;
        --regime-volatility)
            REGIME_VOLATILITY=true
            shift
            ;;
        --regime-recommendations)
            REGIME_RECOMMENDATIONS=true
            shift
            ;;
        --regime-json)
            REGIME_JSON=true
            shift
            ;;
        --val-integrate)
            VAL_INTEGRATE=true
            SYMBOL="$2"
            shift 2
            ;;
        --val-credit-risk)
            VAL_CREDIT_RISK=true
            SYMBOL="$2"
            shift 2
            ;;
        --val-insider)
            VAL_INSIDER=true
            SYMBOL="$2"
            shift 2
            ;;
        --val-short-interest)
            VAL_SHORT_INTEREST=true
            SYMBOL="$2"
            shift 2
            ;;
        --val-market-regime)
            VAL_MARKET_REGIME=true
            shift
            ;;
        --val-base-fv)
            VAL_BASE_FV="$2"
            shift 2
            ;;
        --val-price)
            VAL_PRICE="$2"
            shift 2
            ;;
        --val-json)
            VAL_JSON=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Set debug logging if requested
if [[ "$DEBUG" == "true" ]]; then
    set -x
    log_warning "Debug mode enabled"
fi

# Handle cache operations
if [[ -n "$CACHE_OPERATION" ]]; then
    log_info "Running cache operation: $CACHE_OPERATION"

    case "$CACHE_OPERATION" in
        --clean-cache)
            if [[ -n "$SYMBOL" ]]; then
                python3 -u cli_orchestrator.py clean-cache --symbol "$SYMBOL"
            elif [[ ${#SYMBOLS[@]} -gt 0 ]]; then
                for sym in "${SYMBOLS[@]}"; do
                    python3 -u cli_orchestrator.py clean-cache --symbol "$sym"
                done
            else
                python3 -u cli_orchestrator.py clean-cache
            fi
            ;;
        --clean-cache-all)
            python3 -u cli_orchestrator.py clean-cache --all
            ;;
        --clean-cache-db)
            if [[ -n "$SYMBOL" ]]; then
                python3 -u cli_orchestrator.py clean-cache --db --symbol "$SYMBOL"
            elif [[ ${#SYMBOLS[@]} -gt 0 ]]; then
                for sym in "${SYMBOLS[@]}"; do
                    python3 -u cli_orchestrator.py clean-cache --db --symbol "$sym"
                done
            else
                python3 -u cli_orchestrator.py clean-cache --db
            fi
            ;;
        --clean-cache-disk)
            if [[ -n "$SYMBOL" ]]; then
                python3 -u cli_orchestrator.py clean-cache --disk --symbol "$SYMBOL"
            elif [[ ${#SYMBOLS[@]} -gt 0 ]]; then
                for sym in "${SYMBOLS[@]}"; do
                    python3 -u cli_orchestrator.py clean-cache --disk --symbol "$sym"
                done
            else
                python3 -u cli_orchestrator.py clean-cache --disk
            fi
            ;;
        inspect-cache)
            python3 -u cli_orchestrator.py inspect-cache ${SYMBOL:+--symbol "$SYMBOL"}
            ;;
        cache-sizes)
            python3 -u cli_orchestrator.py cache-sizes
            ;;
    esac

    exit $?
fi

# Handle test operations
if [[ -n "$TEST_OPERATION" ]]; then
    log_info "Running test operation: $TEST_OPERATION"

    case "$TEST_OPERATION" in
        test-system)
            python3 -u cli_orchestrator.py test-system
            ;;
        test-cache)
            python3 -u cli_orchestrator.py test-system
            ;;
        run-tests)
            python3 -u cli_orchestrator.py run-tests
            ;;
    esac

    exit $?
fi

# Handle setup operations
if [[ -n "$SETUP_OPERATION" ]]; then
    log_info "Running setup operation: $SETUP_OPERATION"

    case "$SETUP_OPERATION" in
        setup-database)
            python3 -u cli_orchestrator.py setup-database
            ;;
        setup-system)
            python3 -u cli_orchestrator.py setup-system
            ;;
        setup-vectordb)
            log_warning "Vector DB setup not yet implemented in cli_orchestrator.py"
            log_warning "Falling back to investigator.sh for this operation"
            exec "$SCRIPT_DIR/investigator.sh" --setup-vectordb
            ;;
    esac

    exit $?
fi

# Handle system stats
if [[ "$SYSTEM_STATS" == "true" ]]; then
    log_info "Showing system statistics"
    python3 -u cli_orchestrator.py status
    python3 -u cli_orchestrator.py metrics
    exit $?
fi

# Handle macro data operations
if [[ "$MACRO_SUMMARY" == "true" ]] || [[ "$MACRO_BUFFETT" == "true" ]] || \
   [[ -n "$MACRO_CATEGORY" ]] || [[ -n "$MACRO_INDICATORS" ]] || \
   [[ -n "$MACRO_TIME_SERIES" ]] || [[ "$MACRO_LIST_CATEGORIES" == "true" ]]; then

    # Build macro data arguments
    MACRO_ARGS=()

    if [[ "$MACRO_SUMMARY" == "true" ]]; then
        MACRO_ARGS+=(--summary)
    elif [[ "$MACRO_BUFFETT" == "true" ]]; then
        MACRO_ARGS+=(--buffett)
    elif [[ -n "$MACRO_CATEGORY" ]]; then
        MACRO_ARGS+=(--category "$MACRO_CATEGORY")
    elif [[ -n "$MACRO_INDICATORS" ]]; then
        # Convert comma-separated to space-separated
        IFS=',' read -ra INDICATOR_ARRAY <<< "$MACRO_INDICATORS"
        MACRO_ARGS+=(--indicators "${INDICATOR_ARRAY[@]}")
    elif [[ -n "$MACRO_TIME_SERIES" ]]; then
        MACRO_ARGS+=(--time-series "$MACRO_TIME_SERIES")
    elif [[ "$MACRO_LIST_CATEGORIES" == "true" ]]; then
        MACRO_ARGS+=(--list-categories)
    fi

    MACRO_ARGS+=(--lookback-days "$MACRO_LOOKBACK_DAYS")
    MACRO_ARGS+=(--limit "$MACRO_LIMIT")
    [[ "$MACRO_JSON" == "true" ]] && MACRO_ARGS+=(--json)
    [[ "$VERBOSE" == "true" ]] && MACRO_ARGS+=(--verbose)

    log_info "Running macro data query..."
    PYTHONPATH=./src:. python3 scripts/macro_data_cli.py "${MACRO_ARGS[@]}"
    exit $?
fi

# Handle insider trading operations
if [[ "$INSIDER_SENTIMENT" == "true" ]] || [[ "$INSIDER_RECENT" == "true" ]] || \
   [[ "$INSIDER_CLUSTERS" == "true" ]] || [[ "$INSIDER_KEY_INSIDERS" == "true" ]] || \
   [[ "$INSIDER_FETCH" == "true" ]]; then

    if [[ -z "$SYMBOL" ]]; then
        log_error "Symbol is required for insider trading operations"
        exit 1
    fi

    # Build insider trading arguments
    INSIDER_ARGS=("$SYMBOL")

    if [[ "$INSIDER_SENTIMENT" == "true" ]]; then
        INSIDER_ARGS+=(--sentiment)
    elif [[ "$INSIDER_RECENT" == "true" ]]; then
        INSIDER_ARGS+=(--recent)
    elif [[ "$INSIDER_CLUSTERS" == "true" ]]; then
        INSIDER_ARGS+=(--clusters)
    elif [[ "$INSIDER_KEY_INSIDERS" == "true" ]]; then
        INSIDER_ARGS+=(--key-insiders)
    elif [[ "$INSIDER_FETCH" == "true" ]]; then
        INSIDER_ARGS+=(--fetch)
    fi

    INSIDER_ARGS+=(--days "$INSIDER_DAYS")
    [[ "$INSIDER_SIGNIFICANT_ONLY" == "true" ]] && INSIDER_ARGS+=(--significant-only)
    [[ "$INSIDER_JSON" == "true" ]] && INSIDER_ARGS+=(--json)

    log_info "Running insider trading analysis for ${SYMBOL}..."
    PYTHONPATH=./src:. python3 scripts/insider_trading_cli.py "${INSIDER_ARGS[@]}"
    exit $?
fi

# Handle treasury data operations
if [[ "$TREASURY_CURVE" == "true" ]] || [[ "$TREASURY_SPREAD" == "true" ]] || \
   [[ "$TREASURY_REGIME" == "true" ]] || [[ "$TREASURY_RECESSION" == "true" ]] || \
   [[ "$TREASURY_SUMMARY" == "true" ]] || [[ "$TREASURY_HISTORY" == "true" ]]; then

    # Build treasury arguments
    TREASURY_ARGS=()

    if [[ "$TREASURY_CURVE" == "true" ]]; then
        TREASURY_ARGS+=(--curve)
    elif [[ "$TREASURY_SPREAD" == "true" ]]; then
        TREASURY_ARGS+=(--spread)
    elif [[ "$TREASURY_REGIME" == "true" ]]; then
        TREASURY_ARGS+=(--regime)
    elif [[ "$TREASURY_RECESSION" == "true" ]]; then
        TREASURY_ARGS+=(--recession)
    elif [[ "$TREASURY_SUMMARY" == "true" ]]; then
        TREASURY_ARGS+=(--summary)
    elif [[ "$TREASURY_HISTORY" == "true" ]]; then
        TREASURY_ARGS+=(--history)
        TREASURY_ARGS+=(--maturity "$TREASURY_MATURITY")
        TREASURY_ARGS+=(--days "$TREASURY_DAYS")
    fi

    [[ "$TREASURY_JSON" == "true" ]] && TREASURY_ARGS+=(--json)

    log_info "Running treasury data query..."
    PYTHONPATH=./src:. python3 scripts/treasury_data_cli.py "${TREASURY_ARGS[@]}"
    exit $?
fi

# Handle institutional holdings operations
if [[ "$INST_HOLDINGS" == "true" ]] || [[ "$INST_TOP_HOLDERS" == "true" ]] || \
   [[ "$INST_CHANGES" == "true" ]] || [[ -n "$INST_INSTITUTION" ]] || \
   [[ -n "$INST_SEARCH" ]]; then

    # Build institutional holdings arguments
    INST_ARGS=()

    if [[ "$INST_HOLDINGS" == "true" ]]; then
        if [[ -z "$SYMBOL" ]]; then
            log_error "Symbol is required for --inst-holdings"
            exit 1
        fi
        INST_ARGS+=("$SYMBOL" --holdings)
    elif [[ "$INST_TOP_HOLDERS" == "true" ]]; then
        if [[ -z "$SYMBOL" ]]; then
            log_error "Symbol is required for --inst-top-holders"
            exit 1
        fi
        INST_ARGS+=("$SYMBOL" --top-holders)
    elif [[ "$INST_CHANGES" == "true" ]]; then
        if [[ -z "$SYMBOL" ]]; then
            log_error "Symbol is required for --inst-changes"
            exit 1
        fi
        INST_ARGS+=("$SYMBOL" --changes --quarters "$INST_QUARTERS")
    elif [[ -n "$INST_INSTITUTION" ]]; then
        INST_ARGS+=(--institution "$INST_INSTITUTION")
    elif [[ -n "$INST_SEARCH" ]]; then
        INST_ARGS+=(--search "$INST_SEARCH")
    fi

    INST_ARGS+=(--limit "$INST_LIMIT")
    [[ -n "$INST_QUARTER" ]] && INST_ARGS+=(--quarter "$INST_QUARTER")
    [[ "$INST_JSON" == "true" ]] && INST_ARGS+=(--json)

    log_info "Running institutional holdings query..."
    PYTHONPATH=./src:. python3 scripts/institutional_holdings_cli.py "${INST_ARGS[@]}"
    exit $?
fi

# Handle short interest operations
if [[ "$SHORT_CURRENT" == "true" ]] || [[ "$SHORT_HISTORY" == "true" ]] || \
   [[ "$SHORT_VOLUME" == "true" ]] || [[ "$SHORT_SQUEEZE" == "true" ]] || \
   [[ "$SHORT_MOST_SHORTED" == "true" ]]; then

    # Build short interest arguments
    SHORT_ARGS=()

    if [[ "$SHORT_CURRENT" == "true" ]]; then
        if [[ -z "$SYMBOL" ]]; then
            log_error "Symbol is required for --short-current"
            exit 1
        fi
        SHORT_ARGS+=("$SYMBOL" --current)
    elif [[ "$SHORT_HISTORY" == "true" ]]; then
        if [[ -z "$SYMBOL" ]]; then
            log_error "Symbol is required for --short-history"
            exit 1
        fi
        SHORT_ARGS+=("$SYMBOL" --history --periods "$SHORT_PERIODS")
    elif [[ "$SHORT_VOLUME" == "true" ]]; then
        if [[ -z "$SYMBOL" ]]; then
            log_error "Symbol is required for --short-volume"
            exit 1
        fi
        SHORT_ARGS+=("$SYMBOL" --volume --days "$SHORT_DAYS")
    elif [[ "$SHORT_SQUEEZE" == "true" ]]; then
        if [[ -z "$SYMBOL" ]]; then
            log_error "Symbol is required for --short-squeeze"
            exit 1
        fi
        SHORT_ARGS+=("$SYMBOL" --squeeze)
    elif [[ "$SHORT_MOST_SHORTED" == "true" ]]; then
        SHORT_ARGS+=(--most-shorted --limit "$SHORT_LIMIT")
    fi

    [[ "$SHORT_JSON" == "true" ]] && SHORT_ARGS+=(--json)

    log_info "Running short interest query..."
    PYTHONPATH=./src:. python3 scripts/short_interest_cli.py "${SHORT_ARGS[@]}"
    exit $?
fi

# Handle market regime operations
if [[ "$REGIME_SUMMARY" == "true" ]] || [[ "$REGIME_CREDIT_CYCLE" == "true" ]] || \
   [[ "$REGIME_YIELD_CURVE" == "true" ]] || [[ "$REGIME_RECESSION" == "true" ]] || \
   [[ "$REGIME_VOLATILITY" == "true" ]] || [[ "$REGIME_RECOMMENDATIONS" == "true" ]]; then

    # Build market regime arguments
    REGIME_ARGS=()

    if [[ "$REGIME_SUMMARY" == "true" ]]; then
        REGIME_ARGS+=(--summary)
    elif [[ "$REGIME_CREDIT_CYCLE" == "true" ]]; then
        REGIME_ARGS+=(--credit-cycle)
    elif [[ "$REGIME_YIELD_CURVE" == "true" ]]; then
        REGIME_ARGS+=(--yield-curve)
    elif [[ "$REGIME_RECESSION" == "true" ]]; then
        REGIME_ARGS+=(--recession)
    elif [[ "$REGIME_VOLATILITY" == "true" ]]; then
        REGIME_ARGS+=(--volatility)
    elif [[ "$REGIME_RECOMMENDATIONS" == "true" ]]; then
        REGIME_ARGS+=(--recommendations)
    fi

    [[ "$REGIME_JSON" == "true" ]] && REGIME_ARGS+=(--json)

    log_info "Running market regime analysis..."
    PYTHONPATH=./src:. python3 scripts/market_regime_cli.py "${REGIME_ARGS[@]}"
    exit $?
fi

# Handle valuation signal operations
if [[ "$VAL_INTEGRATE" == "true" ]] || [[ "$VAL_CREDIT_RISK" == "true" ]] || \
   [[ "$VAL_INSIDER" == "true" ]] || [[ "$VAL_SHORT_INTEREST" == "true" ]] || \
   [[ "$VAL_MARKET_REGIME" == "true" ]]; then

    # Build valuation signals arguments
    VAL_ARGS=()

    if [[ "$VAL_INTEGRATE" == "true" ]]; then
        if [[ -z "$SYMBOL" ]]; then
            log_error "Symbol is required for --val-integrate"
            exit 1
        fi
        if [[ -z "$VAL_BASE_FV" ]] || [[ -z "$VAL_PRICE" ]]; then
            log_error "--val-base-fv and --val-price are required for --val-integrate"
            exit 1
        fi
        VAL_ARGS+=(--integrate "$SYMBOL" --base-fv "$VAL_BASE_FV" --price "$VAL_PRICE")
    elif [[ "$VAL_CREDIT_RISK" == "true" ]]; then
        if [[ -z "$SYMBOL" ]]; then
            log_error "Symbol is required for --val-credit-risk"
            exit 1
        fi
        VAL_ARGS+=(--credit-risk "$SYMBOL")
    elif [[ "$VAL_INSIDER" == "true" ]]; then
        if [[ -z "$SYMBOL" ]]; then
            log_error "Symbol is required for --val-insider"
            exit 1
        fi
        VAL_ARGS+=(--insider "$SYMBOL")
    elif [[ "$VAL_SHORT_INTEREST" == "true" ]]; then
        if [[ -z "$SYMBOL" ]]; then
            log_error "Symbol is required for --val-short-interest"
            exit 1
        fi
        VAL_ARGS+=(--short-interest "$SYMBOL")
    elif [[ "$VAL_MARKET_REGIME" == "true" ]]; then
        VAL_ARGS+=(--market-regime)
    fi

    [[ "$VAL_JSON" == "true" ]] && VAL_ARGS+=(--json)

    log_info "Running valuation signal analysis..."
    PYTHONPATH=./src:. python3 scripts/valuation_signals_cli.py "${VAL_ARGS[@]}"
    exit $?
fi

# Handle RL operations
if [[ "$RL_STATUS" == "true" ]]; then
    log_info "Showing RL policy deployment status"
    PYTHONPATH=./src:. python3 scripts/rl_deploy.py --status
    exit $?
fi

if [[ "$RL_DEPLOY" == "true" ]]; then
    log_info "Deploying RL policy"
    PYTHONPATH=./src:. python3 scripts/rl_deploy.py
    exit $?
fi

if [[ "$RL_TRAIN" == "true" ]]; then
    log_info "Training RL policy from outcomes"
    PYTHONPATH=./src:. python3 scripts/rl_train.py --deploy
    exit $?
fi

if [[ "$RL_BACKTEST" == "true" ]]; then
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${WHITE}📊 RL BACKTEST INITIATED${NC}                                         ${CYAN}║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════════╣${NC}"
    if [[ -n "$RL_LOOKBACK_RANGE" ]]; then
        # Calculate data points based on interval
        if [[ "$RL_INTERVAL" == "monthly" ]]; then
            DATA_POINTS=$((RL_LOOKBACK_RANGE))
        else
            DATA_POINTS=$((RL_LOOKBACK_RANGE / 3))
        fi
        echo -e "${CYAN}║${NC}  ${YELLOW}Lookback Range:${NC}   ${GREEN}3 to ${RL_LOOKBACK_RANGE} months (${RL_INTERVAL})${NC}                 ${CYAN}║${NC}"
        echo -e "${CYAN}║${NC}  ${YELLOW}Data Points:${NC}      ${GREEN}${DATA_POINTS} per symbol${NC}                              ${CYAN}║${NC}"
    else
        echo -e "${CYAN}║${NC}  ${YELLOW}Lookback:${NC}         ${GREEN}${RL_LOOKBACK} months${NC}                                  ${CYAN}║${NC}"
    fi
    echo -e "${CYAN}║${NC}  ${YELLOW}Parallel:${NC}         ${GREEN}${RL_PARALLEL} workers${NC}                                 ${CYAN}║${NC}"
    [[ "$RL_ALL_SYMBOLS" == "true" ]] && echo -e "${CYAN}║${NC}  ${YELLOW}Symbols:${NC}          ${WHITE}All eligible symbols${NC}                        ${CYAN}║${NC}"
    [[ -n "$RL_TOP_N" ]] && echo -e "${CYAN}║${NC}  ${YELLOW}Top N:${NC}            ${WHITE}${RL_TOP_N} symbols by market cap${NC}                   ${CYAN}║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}  ${BLUE}⏳ Starting backtest (this may take a while)...${NC}                ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Build backtest arguments
    BACKTEST_ARGS=()
    if [[ -n "$RL_LOOKBACK_RANGE" ]]; then
        BACKTEST_ARGS+=(--lookback-range "$RL_LOOKBACK_RANGE")
        BACKTEST_ARGS+=(--interval "$RL_INTERVAL")
    else
        BACKTEST_ARGS+=(--lookback "$RL_LOOKBACK")
    fi
    BACKTEST_ARGS+=(--parallel "$RL_PARALLEL")
    [[ "$RL_ALL_SYMBOLS" == "true" ]] && BACKTEST_ARGS+=(--all-symbols)
    [[ -n "$RL_TOP_N" ]] && BACKTEST_ARGS+=(--top-n "$RL_TOP_N")
    # Pass specific symbols if provided
    if [[ ${#SYMBOLS[@]} -gt 0 ]]; then
        BACKTEST_ARGS+=(--symbols "${SYMBOLS[@]}")
    fi

    # Run backtest
    PYTHONPATH=./src:. python3 scripts/rl_backtest.py "${BACKTEST_ARGS[@]}"
    BACKTEST_EXIT=$?

    if [[ $BACKTEST_EXIT -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║${NC}  ${WHITE}✓ RL BACKTEST COMPLETED SUCCESSFULLY${NC}                            ${GREEN}║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}╔════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║${NC}  ${WHITE}✗ RL BACKTEST FAILED${NC}                                              ${RED}║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi

    exit $BACKTEST_EXIT
fi

# Handle RL backtest workflow (StateGraph-based)
if [[ "$RL_BACKTEST_WORKFLOW" == "true" ]]; then
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${WHITE}📊 RL BACKTEST WORKFLOW (StateGraph)${NC}                             ${CYAN}║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════════╣${NC}"
    if [[ -n "$RL_LOOKBACK_RANGE" ]]; then
        if [[ "$RL_INTERVAL" == "monthly" ]]; then
            DATA_POINTS=$((RL_LOOKBACK_RANGE))
        else
            DATA_POINTS=$((RL_LOOKBACK_RANGE / 3))
        fi
        echo -e "${CYAN}║${NC}  ${YELLOW}Lookback Range:${NC}   ${GREEN}3 to ${RL_LOOKBACK_RANGE} months (${RL_INTERVAL})${NC}                 ${CYAN}║${NC}"
        echo -e "${CYAN}║${NC}  ${YELLOW}Data Points:${NC}      ${GREEN}${DATA_POINTS} per symbol${NC}                              ${CYAN}║${NC}"
    else
        echo -e "${CYAN}║${NC}  ${YELLOW}Lookback:${NC}         ${GREEN}${RL_LOOKBACK} months${NC}                                  ${CYAN}║${NC}"
    fi
    echo -e "${CYAN}║${NC}  ${YELLOW}Parallel:${NC}         ${GREEN}${RL_PARALLEL} workers${NC}                                 ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}Mode:${NC}             ${WHITE}Victor Workflow (shared services)${NC}           ${CYAN}║${NC}"
    [[ "$RL_ALL_SYMBOLS" == "true" ]] && echo -e "${CYAN}║${NC}  ${YELLOW}Symbols:${NC}          ${WHITE}All eligible symbols${NC}                        ${CYAN}║${NC}"
    [[ -n "$RL_TOP_N" ]] && echo -e "${CYAN}║${NC}  ${YELLOW}Top N:${NC}            ${WHITE}${RL_TOP_N} symbols by market cap${NC}                   ${CYAN}║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}  ${BLUE}⏳ Starting workflow backtest...${NC}                                ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Build workflow arguments
    WORKFLOW_ARGS=()
    if [[ -n "$RL_LOOKBACK_RANGE" ]]; then
        WORKFLOW_ARGS+=(--max-lookback "$RL_LOOKBACK_RANGE")
        WORKFLOW_ARGS+=(--interval "$RL_INTERVAL")
    else
        WORKFLOW_ARGS+=(--max-lookback "$RL_LOOKBACK")
    fi
    WORKFLOW_ARGS+=(--parallel "$RL_PARALLEL")
    [[ "$RL_ALL_SYMBOLS" == "true" ]] && WORKFLOW_ARGS+=(--all-symbols)
    [[ -n "$RL_TOP_N" ]] && WORKFLOW_ARGS+=(--top-n "$RL_TOP_N")
    if [[ ${#SYMBOLS[@]} -gt 0 ]]; then
        WORKFLOW_ARGS+=(--symbols "${SYMBOLS[@]}")
    fi

    # Run workflow backtest
    PYTHONPATH=./src:. python3 scripts/rl_backtest_workflow.py "${WORKFLOW_ARGS[@]}"
    WORKFLOW_EXIT=$?

    if [[ $WORKFLOW_EXIT -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║${NC}  ${WHITE}✓ RL BACKTEST WORKFLOW COMPLETED${NC}                                ${GREEN}║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}╔════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║${NC}  ${WHITE}✗ RL BACKTEST WORKFLOW FAILED${NC}                                    ${RED}║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi

    exit $WORKFLOW_EXIT
fi

# Handle weekly report
if [[ "$WEEKLY_REPORT" == "true" ]]; then
    log_info "Generating weekly portfolio report"
    log_warning "Weekly reports not yet in cli_orchestrator.py"
    log_warning "Falling back to investigator.sh"
    exec "$SCRIPT_DIR/investigator.sh" --weekly-report
fi

# Handle peer group analysis
if [[ "$PEER_GROUPS" == "true" ]]; then
    log_info "Running peer group analysis"
    log_warning "Peer group analysis not yet in cli_orchestrator.py"
    log_warning "Falling back to investigator.sh"

    PEER_ARGS=("--peer-groups-comprehensive")
    [[ -n "$PEER_SECTOR" ]] && PEER_ARGS+=(--peer-sector "$PEER_SECTOR")
    [[ -n "$PEER_INDUSTRY" ]] && PEER_ARGS+=(--peer-industry "$PEER_INDUSTRY")
    [[ "$PEER_RISK" == "true" ]] && PEER_ARGS+=(--peer-risk-assessment)

    exec "$SCRIPT_DIR/investigator.sh" "${PEER_ARGS[@]}"
fi

# Build common arguments for analysis (returns via stdout)
build_analysis_args() {
    local args=()

    args+=(--mode "$MODE")
    [[ -n "$FORMAT" ]] && args+=(--format "$FORMAT")
    [[ -n "$DETAIL_LEVEL" ]] && args+=(--detail-level "$DETAIL_LEVEL")
    [[ "$GENERATE_REPORT" == "true" ]] && args+=(--report)
    [[ "$FORCE_REFRESH" == "true" ]] && args+=(--force-refresh)

    # Only print if there are args
    [[ ${#args[@]} -gt 0 ]] && printf '%s\n' "${args[@]}"
}

# Build global CLI options (returns via stdout)
build_global_args() {
    local args=()

    [[ -n "$LOG_LEVEL" ]] && args+=(--log-level "$LOG_LEVEL")
    [[ "$VERBOSE" == "true" ]] && args+=(--verbose)

    # Only print if there are args
    [[ ${#args[@]} -gt 0 ]] && printf '%s\n' "${args[@]}"
}

# Clear all caches for a symbol (file + database)
clear_symbol_caches() {
    local symbol=$1

    log_info "🧹 Clearing all caches for ${symbol}..."

    # Clear ALL file caches (comprehensive)
    rm -rf "data/llm_cache/${symbol}" 2>/dev/null
    rm -rf "data/sec_cache/${symbol}" 2>/dev/null
    rm -rf "data/sec_cache/facts/processed/${symbol}" 2>/dev/null
    rm -rf "data/sec_cache/quarterlymetrics/${symbol}" 2>/dev/null
    rm -rf "data/technical_cache/${symbol}" 2>/dev/null
    rm -rf "data/market_context_cache/${symbol}" 2>/dev/null
    rm -rf "data/vector_db/${symbol}" 2>/dev/null

    # Clear result/output files
    rm -f "results/${symbol}"*.json 2>/dev/null
    rm -f "reports/"*"${symbol}"*.pdf 2>/dev/null

    # Clear Python bytecode cache (.pyc files)
    log_info "Clearing Python bytecode cache..."
    find "${SCRIPT_DIR}/src" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${SCRIPT_DIR}/src" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "${SCRIPT_DIR}/utils" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${SCRIPT_DIR}/utils" -type f -name "*.pyc" -delete 2>/dev/null || true

    # Clear database caches (ALL tables including new 3-table architecture)
    if command -v psql &> /dev/null; then
        export PGPASSWORD=investigator
        psql -h ${DB_HOST:-localhost} -U investigator -d sec_database -c "
            DELETE FROM llm_responses WHERE symbol = '${symbol}';
            DELETE FROM sec_responses WHERE symbol = '${symbol}';
            DELETE FROM technical_indicators WHERE symbol = '${symbol}';
            DELETE FROM sec_companyfacts WHERE symbol = '${symbol}';
            DELETE FROM sec_companyfacts_raw WHERE symbol = '${symbol}';
            DELETE FROM sec_companyfacts_processed WHERE symbol = '${symbol}';
            DELETE FROM sec_companyfacts_metadata WHERE symbol = '${symbol}';
        " 2>/dev/null || true
    fi

    log_info "✅ All caches cleared for ${symbol}"
}

# Run single symbol analysis
if [[ -n "$SYMBOL" ]]; then
    # Clear caches if force refresh is requested
    if [[ "$FORCE_REFRESH" == "true" ]]; then
        clear_symbol_caches "$SYMBOL"
    fi

    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${WHITE}📈 STOCK ANALYSIS INITIATED${NC}                                     ${CYAN}║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}Symbol:${NC}           ${GREEN}${SYMBOL}${NC}                                       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}Mode:${NC}             ${WHITE}${MODE}${NC}                                 ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}Server Pool:${NC}      ${BLUE}2 servers × 3 concurrent = 6 total${NC}       ${CYAN}║${NC}"
    [[ "$GENERATE_REPORT" == "true" ]] && echo -e "${CYAN}║${NC}  ${YELLOW}PDF Report:${NC}       ${GREEN}✓ Enabled${NC}                                ${CYAN}║${NC}"
    [[ "$FORCE_REFRESH" == "true" ]] && echo -e "${CYAN}║${NC}  ${YELLOW}Cache:${NC}            ${YELLOW}⟳ Force Refresh${NC}                          ${CYAN}║${NC}"
    [[ -n "$LOG_LEVEL" ]] && echo -e "${CYAN}║${NC}  ${YELLOW}Log Level:${NC}        ${WHITE}${LOG_LEVEL}${NC}                                ${CYAN}║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}  ${BLUE}⏳ Initializing system (please wait 2-3 seconds)...${NC}          ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    printf "DEBUG: After banner (line %s)\n" "$LINENO"

    # Build global arguments
    printf "DEBUG: Calling build_global_args (line %s)\n" "$LINENO"
    mapfile -t GLOBAL_ARGS < <(build_global_args)
    printf "DEBUG: After build_global_args, got %d args (line %s)\n" "${#GLOBAL_ARGS[@]}" "$LINENO"

    # Build analysis arguments
    printf "DEBUG: Calling build_analysis_args (line %s)\n" "$LINENO"
    mapfile -t ANALYZE_ARGS < <(build_analysis_args)
    printf "DEBUG: After build_analysis_args, got %d args (line %s)\n" "${#ANALYZE_ARGS[@]}" "$LINENO"

    # Add output file
    ANALYZE_ARGS+=(--output "$OUTPUT_DIR/${SYMBOL}_$(date +%Y%m%d_%H%M%S).json")

    # Build output path
    OUTPUT_PATH="$OUTPUT_DIR/${SYMBOL}_$(date +%Y%m%d_%H%M%S).json"

    # Try victor_invest CLI first (uses shared market data services)
    # Fall back to cli_orchestrator.py if victor-core not installed
    if python3 -c "from victor.framework import Agent" 2>/dev/null; then
        # Victor-core available - use modern CLI
        CMD="python3 -m victor_invest.cli analyze $SYMBOL --mode $MODE --output $OUTPUT_DIR"
        echo "Using victor_invest CLI (shared services enabled)" >&2
    else
        # Fall back to cli_orchestrator.py
        CMD="python3 -u cli_orchestrator.py ${GLOBAL_ARGS[*]} analyze $SYMBOL ${ANALYZE_ARGS[*]}"
        echo "Using cli_orchestrator.py (victor-core not available)" >&2
    fi

    # Show command being executed (for debugging)
    printf "DEBUG: About to execute (line %s)\n" "$LINENO"
    echo "Executing: $CMD" >&2
    echo "" >&2

    # Run analysis
    if python3 -c "from victor.framework import Agent" 2>/dev/null; then
        python3 -m victor_invest.cli analyze "$SYMBOL" --mode "$MODE" --output "$OUTPUT_DIR" 2>&1
    else
        python3 -u cli_orchestrator.py "${GLOBAL_ARGS[@]}" analyze "$SYMBOL" "${ANALYZE_ARGS[@]}" 2>&1
    fi
    ANALYSIS_EXIT_CODE=$?

    echo "" >&2
    echo "Exit code: $ANALYSIS_EXIT_CODE" >&2

    if [[ $ANALYSIS_EXIT_CODE -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║${NC}  ${WHITE}✓ ANALYSIS COMPLETED SUCCESSFULLY${NC}                              ${GREEN}║${NC}"
        echo -e "${GREEN}╠════════════════════════════════════════════════════════════════════╣${NC}"
        echo -e "${GREEN}║${NC}  ${YELLOW}Symbol:${NC}    ${GREEN}${SYMBOL}${NC}                                                ${GREEN}║${NC}"
        echo -e "${GREEN}║${NC}  ${YELLOW}Status:${NC}    ${WHITE}Ready for investment analysis${NC}                     ${GREEN}║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}╔════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║${NC}  ${WHITE}✗ ANALYSIS FAILED${NC}                                              ${RED}║${NC}"
        echo -e "${RED}╠════════════════════════════════════════════════════════════════════╣${NC}"
        echo -e "${RED}║${NC}  ${YELLOW}Symbol:${NC}    ${RED}${SYMBOL}${NC}                                                ${RED}║${NC}"
        echo -e "${RED}║${NC}  ${YELLOW}Action:${NC}    ${WHITE}Check logs for details${NC}                             ${RED}║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        exit 1
    fi

# Run batch analysis
elif [[ ${#SYMBOLS[@]} -gt 0 ]]; then
    # Clear caches if force refresh is requested
    if [[ "$FORCE_REFRESH" == "true" ]]; then
        for sym in "${SYMBOLS[@]}"; do
            clear_symbol_caches "$sym"
        done
    fi

    # Warn if --report was requested (not supported for batch mode)
    if [[ "$GENERATE_REPORT" == "true" ]]; then
        log_warning "PDF report generation (--report) is only supported for single stock analysis"
        log_warning "Use --symbol AAPL --report for individual reports"
        echo ""
    fi

    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${WHITE}📊 BATCH ANALYSIS INITIATED${NC}                                     ${CYAN}║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}Symbols:${NC}          ${GREEN}${#SYMBOLS[@]} stocks${NC}                                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}Targets:${NC}          ${WHITE}${SYMBOLS[*]}${NC}                          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}Mode:${NC}             ${WHITE}${MODE}${NC}                                 ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}Processing:${NC}       ${BLUE}Parallel across multi-server pool${NC}        ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}Max Concurrent:${NC}   ${BLUE}Up to 6 simultaneous analyses${NC}            ${CYAN}║${NC}"
    [[ "$FORCE_REFRESH" == "true" ]] && echo -e "${CYAN}║${NC}  ${YELLOW}Cache:${NC}            ${YELLOW}⟳ Force Refresh${NC}                          ${CYAN}║${NC}"
    [[ -n "$LOG_LEVEL" ]] && echo -e "${CYAN}║${NC}  ${YELLOW}Log Level:${NC}        ${WHITE}${LOG_LEVEL}${NC}                                ${CYAN}║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}  ${BLUE}⏳ Initializing system (please wait 2-3 seconds)...${NC}          ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Build global arguments
    mapfile -t GLOBAL_ARGS < <(build_global_args)

    # Build batch arguments inline (batch doesn't use helper function)
    BATCH_ARGS=()
    BATCH_ARGS+=(--mode "$MODE")
    [[ -n "$DETAIL_LEVEL" ]] && BATCH_ARGS+=(--detail-level "$DETAIL_LEVEL")
    [[ "$FORCE_REFRESH" == "true" ]] && BATCH_ARGS+=(--force-refresh)
    BATCH_ARGS+=(--output-dir "$OUTPUT_DIR")

    # Run batch analysis using batch_analysis_runner.py
    # This uses the shared market data services for consistent behavior
    # Note: batch_analysis_runner.py integrates with victor_invest.workflows.run_analysis
    python3 -u scripts/batch_analysis_runner.py \
        --symbols "${SYMBOLS[@]}" \
        --mode "$MODE" \
        --output "$OUTPUT_DIR" \
        --batch-size 5 \
        --delay 2

    BATCH_EXIT_CODE=$?

    if [[ $BATCH_EXIT_CODE -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║${NC}  ${WHITE}✓ BATCH ANALYSIS COMPLETED SUCCESSFULLY${NC}                        ${GREEN}║${NC}"
        echo -e "${GREEN}╠════════════════════════════════════════════════════════════════════╣${NC}"
        echo -e "${GREEN}║${NC}  ${YELLOW}Processed:${NC}    ${GREEN}${#SYMBOLS[@]} symbols${NC}                                     ${GREEN}║${NC}"
        echo -e "${GREEN}║${NC}  ${YELLOW}Output:${NC}       ${WHITE}${OUTPUT_DIR}/${NC}                                   ${GREEN}║${NC}"
        echo -e "${GREEN}║${NC}  ${YELLOW}Status:${NC}       ${WHITE}All analyses complete${NC}                          ${GREEN}║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}╔════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║${NC}  ${WHITE}✗ BATCH ANALYSIS FAILED${NC}                                        ${RED}║${NC}"
        echo -e "${RED}╠════════════════════════════════════════════════════════════════════╣${NC}"
        echo -e "${RED}║${NC}  ${YELLOW}Symbols:${NC}      ${RED}${#SYMBOLS[@]} stocks${NC}                                      ${RED}║${NC}"
        echo -e "${RED}║${NC}  ${YELLOW}Action:${NC}       ${WHITE}Check logs for details${NC}                         ${RED}║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        exit 1
    fi

else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║${NC}  ${WHITE}✗ NO SYMBOL SPECIFIED${NC}                                          ${RED}║${NC}"
    echo -e "${RED}╠════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${RED}║${NC}  ${YELLOW}Error:${NC}        ${WHITE}Missing required symbol parameter${NC}              ${RED}║${NC}"
    echo -e "${RED}║${NC}  ${YELLOW}Usage:${NC}        ${WHITE}--symbol AAPL${NC}                                   ${RED}║${NC}"
    echo -e "${RED}║${NC}               ${WHITE}--symbols AAPL GOOGL MSFT${NC}                       ${RED}║${NC}"
    echo -e "${RED}║${NC}               ${WHITE}--batch-file symbols.txt${NC}                        ${RED}║${NC}"
    echo -e "${RED}║${NC}  ${YELLOW}Help:${NC}         ${WHITE}Run '$0 --help' for usage info${NC}              ${RED}║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  ${WHITE}🐊 InvestiGator v2 finished successfully! 🎉${NC}                    ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
