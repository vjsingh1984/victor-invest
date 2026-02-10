# STX Comprehensive Analysis Setup Script (PowerShell)
# This script configures the environment and runs the STX analysis

Write-Host "=== InvestiGator STX Analysis Setup ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Ollama Server Status
Write-Host "[1/5] Checking Ollama Server..." -ForegroundColor Yellow
$ollamaHost = "192.168.1.20:11434"
$env:OLLAMA_HOST = $ollamaHost

try {
    $response = Invoke-WebRequest -Uri "http://$ollamaHost/api/tags" -TimeoutSec 5 -ErrorAction Stop
    $data = $response.Content | ConvertFrom-Json
    $models = $data.models | ForEach-Object { $_.name }
    Write-Host "  ✓ Ollama server is running at http://$ollamaHost" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Available models:" -ForegroundColor Cyan
    $models | ForEach-Object { Write-Host "    - $_" -ForegroundColor White }
    Write-Host ""

    # Check if GPT-OSS is available
    if ($models -contains "gpt-oss") {
        Write-Host "  ✓ GPT-OSS model found" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ GPT-OSS model not found" -ForegroundColor Yellow
        Write-Host "  Please run: ollama pull gpt-oss" -ForegroundColor Yellow
        Write-Host "  Press Enter to continue anyway (or Ctrl+C to cancel)..." -ForegroundColor Yellow
        Read-Host
    }
} catch {
    Write-Host "  ✗ Ollama server is NOT running at http://$ollamaHost" -ForegroundColor Red
    Write-Host ""
    Write-Host "To start Ollama on Windows:" -ForegroundColor Yellow
    Write-Host "   ollama serve" -ForegroundColor White
    Write-Host ""
    Write-Host "To pull GPT-OSS model:" -ForegroundColor Yellow
    Write-Host "  ollama pull gpt-oss" -ForegroundColor White
    Write-Host ""
    Write-Host "Press Ctrl+C when Ollama is ready, then re-run this script." -ForegroundColor Red
    exit 1
}

# Step 2: Configure Environment Variables
Write-Host "[2/5] Configuring Environment Variables..." -ForegroundColor Yellow

# Source credentials from ibkr_tradeapp.env
$envPath = "$env:USERPROFILE\.ibkr_tradeapp.env"
if (Test-Path $envPath) {
    Write-Host "  Loading credentials from: $envPath" -ForegroundColor Cyan
    Get-Content $envPath | ForEach-Object {
        if ($_ -match '^TRADING__DATABASE__POSTGRES_URL=(.+)$') {
            $dbUrl = $Matches[1]
            # Parse URL: postgres://user:password@host:port/database
            if ($dbUrl -match 'postgres://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)') {
                $STOCK_DB_USER = $Matches[1]
                $STOCK_DB_PASSWORD = $Matches[2]
                $STOCK_DB_HOST = $Matches[3]
                $STOCK_DB_PORT = $Matches[4]
                $STOCK_DB_NAME = $Matches[5]

                # Set environment variables
                $env:STOCK_DB_USER = $STOCK_DB_USER
                $env:STOCK_DB_PASSWORD = $STOCK_DB_PASSWORD
                $env:STOCK_DB_HOST = $STOCK_DB_HOST
                $env:STOCK_DB_PORT = $STOCK_DB_PORT
                $env:STOCK_DB_NAME = $STOCK_DB_NAME
                $env:SEC_DB_USER = $STOCK_DB_USER
                $env:SEC_DB_PASSWORD = $STOCK_DB_PASSWORD
                $env:SEC_DB_HOST = $STOCK_DB_HOST
                $env:SEC_DB_PORT = $STOCK_DB_PORT
                $env:SEC_DB_NAME = "sec_database"

                # Legacy variables
                $env:DB_USER = $STOCK_DB_USER
                $env:DB_PASSWORD = $STOCK_DB_PASSWORD
                $env:DB_HOST = $STOCK_DB_HOST
                $env:DB_PORT = $STOCK_DB_PORT
                $env:DB_DATABASE = $STOCK_DB_NAME

                Write-Host "  ✓ Database credentials configured" -ForegroundColor Green
                Write-Host "    Stock: ${STOCK_DB_HOST}:${STOCK_DB_PORT}/${STOCK_DB_NAME}" -ForegroundColor Cyan
                Write-Host "    SEC: ${SEC_DB_HOST}:${SEC_DB_PORT}/${SEC_DB_NAME}" -ForegroundColor Cyan
            }
        }
    }
}

# Set FRED API key
$env:FRED_API_KEY = "6f80a1fbcf86c0a25a67a0a7e32b9de6"

# Step 3: Configure Ollama for GPT-OSS
Write-Host "[3/5] Configuring Ollama for GPT-OSS..." -ForegroundColor Yellow
Write-Host "  Model: GPT-OSS" -ForegroundColor Cyan
Write-Host "  Provider: Ollama" -ForegroundColor Cyan
Write-Host "  Context: finance, business analysis" -ForegroundColor Cyan
Write-Host "  Base URL: http://$ollamaHost" -ForegroundColor Cyan

# Step 4: Set PYTHONPATH
Write-Host "[4/5] Setting PYTHONPATH..." -ForegroundColor Yellow
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$srcPath = Join-Path $projectRoot "src"
$env:PYTHONPATH = "$srcPath;$projectRoot;$env:PYTHONPATH"
Write-Host "  PYTHONPATH configured" -ForegroundColor Green

# Step 5: Run Analysis
Write-Host "[5/5] Running STX Comprehensive Analysis..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Command: investigator analyze single STX --mode comprehensive" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
Set-Location $projectRoot

# Run the analysis
& investigator analyze single STX --mode comprehensive

Write-Host ""
Write-Host "=== Analysis Complete ===" -ForegroundColor Green
Write-Host "Results saved to: STX_analysis_results.json" -ForegroundColor Cyan
