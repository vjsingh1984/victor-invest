#!/bin/bash

# InvestiGator Development Environment Setup Script
# Sets up the complete agentic AI architecture environment

set -e

echo "==========================================="
echo "InvestiGator Agentic AI Setup"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc) -eq 1 ]]; then
            print_status "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.9+ required (found $PYTHON_VERSION)"
            exit 1
        fi
    else
        print_error "Python 3 not found"
        exit 1
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_status "Docker found"
    else
        print_warning "Docker not found - some features will be unavailable"
    fi
    
    # Check Ollama
    if command -v ollama &> /dev/null; then
        print_status "Ollama found"
    else
        print_warning "Ollama not found - installing..."
        install_ollama
    fi
}

# Install Ollama
install_ollama() {
    echo "Installing Ollama..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            curl -fsSL https://ollama.ai/install.sh | sh
        fi
    else
        print_error "Unsupported OS for automatic Ollama installation"
        echo "Please install Ollama manually from https://ollama.ai"
        exit 1
    fi
    
    print_status "Ollama installed"
}

# Setup Python environment
setup_python_env() {
    echo "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_status "Python dependencies installed"
    fi
    
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        print_status "Development dependencies installed"
    fi
}

# Pull Ollama models
setup_ollama_models() {
    echo "Setting up Ollama models..."
    
    # Start Ollama service if not running
    if ! pgrep -x "ollama" > /dev/null; then
        print_warning "Starting Ollama service..."
        ollama serve > /dev/null 2>&1 &
        sleep 5
    fi
    
    # List of required models
    MODELS=(
        "llama3.2:11b"
        "qwen2.5:14b"
        "phi3:mini"
    )
    
    for model in "${MODELS[@]}"; do
        echo "Pulling model: $model"
        if ollama list | grep -q "$model"; then
            print_status "Model $model already available"
        else
            ollama pull "$model"
            print_status "Model $model pulled successfully"
        fi
    done
}

# Setup Docker services
setup_docker_services() {
    echo "Setting up Docker services..."
    
    if command -v docker &> /dev/null; then
        # Create docker-compose.yml if it doesn't exist
        if [ ! -f "docker-compose.yml" ]; then
            create_docker_compose
        fi
        
        # Start services
        docker-compose up -d postgres redis
        
        # Wait for services to be ready
        echo "Waiting for services to be ready..."
        sleep 10
        
        # Check PostgreSQL
        if docker-compose exec -T postgres pg_isready -U investigator > /dev/null 2>&1; then
            print_status "PostgreSQL is ready"
        else
            print_warning "PostgreSQL might not be ready yet"
        fi
        
        # Check Redis
        if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
            print_status "Redis is ready"
        else
            print_warning "Redis might not be ready yet"
        fi
    else
        print_warning "Docker not available - skipping service setup"
        echo "You'll need to manually set up PostgreSQL and Redis"
    fi
}

# Create docker-compose.yml
create_docker_compose() {
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: investigator
      POSTGRES_PASSWORD: investment_pass
      POSTGRES_DB: investment_analysis
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  postgres_data:
  redis_data:
EOF
    print_status "docker-compose.yml created"
}

# Initialize database
setup_database() {
    echo "Setting up database..."
    
    # Run migrations if alembic is available
    if [ -f "migrations/alembic.ini" ]; then
        source venv/bin/activate
        
        # Initialize alembic if needed
        if [ ! -d "migrations/versions" ]; then
            cd migrations
            alembic init versions
            cd ..
        fi
        
        # Run migrations
        export DATABASE_URL="postgresql://investigator:investment_pass@localhost:5432/investment_analysis"
        alembic upgrade head
        
        print_status "Database migrations completed"
    else
        print_warning "Alembic not configured - skipping database migrations"
    fi
}

# Create configuration files
setup_config_files() {
    echo "Setting up configuration files..."
    
    # Create config.yaml if it doesn't exist
    if [ ! -f "config.yaml" ]; then
        cat > config.yaml << 'EOF'
ollama:
  base_url: http://localhost:11434
  timeout: 300
  max_retries: 3

cache:
  redis_url: redis://localhost:6379
  file_cache_path: data/cache
  db_url: postgresql+asyncpg://investigator:investment_pass@localhost:5432/investment_analysis
  ttl_default: 3600
  max_file_cache_gb: 10

orchestrator:
  max_concurrent_analyses: 5
  max_concurrent_agents: 10

api:
  host: 0.0.0.0
  port: 8000
  workers: 4

monitoring:
  export_interval: 60
  metrics_port: 9090
EOF
        print_status "config.yaml created"
    else
        print_status "config.yaml already exists"
    fi
    
    # Create .env file for environment variables
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
# InvestiGator Environment Variables
DATABASE_URL=postgresql://investigator:investment_pass@localhost:5432/investment_analysis
REDIS_URL=redis://localhost:6379
OLLAMA_BASE_URL=http://localhost:11434

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/investigator.log

# Cache settings
CACHE_TTL_DEFAULT=3600
CACHE_MAX_SIZE_GB=10

# Performance
MAX_WORKERS=4
MAX_CONCURRENT_AGENTS=10
EOF
        print_status ".env file created"
    else
        print_status ".env file already exists"
    fi
}

# Create directory structure
setup_directories() {
    echo "Creating directory structure..."
    
    DIRS=(
        "data/cache"
        "data/llm_cache"
        "logs"
        "metrics"
        "results"
        "reports"
    )
    
    for dir in "${DIRS[@]}"; do
        mkdir -p "$dir"
        print_status "Created $dir"
    done
}

# Run tests
run_tests() {
    echo "Running tests..."
    
    source venv/bin/activate
    
    # Run unit tests
    if python -m pytest tests/ -v --tb=short; then
        print_status "All tests passed"
    else
        print_warning "Some tests failed - please review"
    fi
}

# Create sample analysis script
create_sample_script() {
    cat > run_analysis.py << 'EOF'
#!/usr/bin/env python3
"""
Sample script to run InvestiGator analysis
"""

import asyncio
from agents.orchestrator import AgentOrchestrator, AnalysisMode
from utils.cache.cache_manager import CacheManager
from utils.monitoring import MetricsCollector

async def main():
    # Initialize components
    cache = CacheManager()
    await cache.initialize()
    
    metrics = MetricsCollector()
    await metrics.start()
    
    orchestrator = AgentOrchestrator(cache, metrics)
    await orchestrator.start()
    
    try:
        # Analyze a stock
        print("Analyzing AAPL...")
        task_id = await orchestrator.analyze(
            "AAPL",
            mode=AnalysisMode.COMPREHENSIVE
        )
        
        # Wait for results
        results = await orchestrator.get_results(task_id, wait=True)
        
        if results:
            print(f"Analysis complete!")
            print(f"Recommendation: {results.get('synthesis', {}).get('recommendation')}")
        else:
            print("Analysis failed or timed out")
            
    finally:
        # Cleanup
        await orchestrator.stop()
        await metrics.stop()
        await cache.close()

if __name__ == "__main__":
    asyncio.run(main())
EOF
    chmod +x run_analysis.py
    print_status "Sample analysis script created (run_analysis.py)"
}

# Main setup flow
main() {
    echo ""
    echo "Starting InvestiGator Agentic AI Setup..."
    echo ""
    
    check_prerequisites
    setup_directories
    setup_python_env
    setup_ollama_models
    setup_docker_services
    setup_config_files
    setup_database
    create_sample_script
    
    echo ""
    echo "==========================================="
    echo -e "${GREEN}Setup Complete!${NC}"
    echo "==========================================="
    echo ""
    echo "To get started:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run a test analysis: python run_analysis.py"
    echo "3. Start the API server: python main.py serve"
    echo "4. Check system status: python main.py status"
    echo ""
    echo "For more information, see README_AGENTIC.md"
    echo ""
    
    # Optional: run tests
    read -p "Would you like to run tests now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi
}

# Run main function
main