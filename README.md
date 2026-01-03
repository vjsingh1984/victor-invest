# InvestiGator

**AI-Powered Investment Analysis Platform**

InvestiGator is an intelligent investment research platform that combines SEC financial data analysis, technical indicators, and multi-agent AI synthesis to provide comprehensive stock evaluations.

## Features

- **Multi-Agent Analysis**: Coordinated AI agents for SEC filings, technical analysis, fundamental metrics, and market context
- **SEC Integration**: Direct access to EDGAR filings with automated financial statement extraction
- **Technical Analysis**: RSI, MACD, Bollinger Bands, moving averages, and custom indicators
- **Valuation Models**: DCF, P/E, P/S, EV/EBITDA, and Gordon Growth Model with dynamic weighting
- **Reinforcement Learning**: Adaptive model weighting based on historical prediction accuracy
- **Workflow Engine**: YAML-based workflow definitions for customizable analysis pipelines

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ (or SQLite for testing)
- 8GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/victor-invest.git
cd victor-invest

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Configure environment
cp config/.env.example .env
# Edit .env with your database credentials
```

### Database Setup

```bash
# For SQLite (testing/development)
python -m investigator.infrastructure.database.installer --sqlite investigator.db

# For PostgreSQL (production)
python -m investigator.infrastructure.database.installer --postgres postgresql://user:pass@host/db
```

### Running Analysis

```bash
# Quick analysis (technical + market context)
investigator analyze single AAPL --mode quick

# Standard analysis (includes SEC fundamentals)
investigator analyze single MSFT --mode standard

# Comprehensive analysis (full synthesis with peer comparison)
investigator analyze single GOOGL --mode comprehensive
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI / API Layer                          │
├─────────────────────────────────────────────────────────────────┤
│                     Workflow Orchestrator                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │   SEC    │  │Technical │  │Fundamental│  │    Synthesis    │ │
│  │  Agent   │  │  Agent   │  │  Agent    │  │      Agent      │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      Domain Services                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Valuation  │  │  RL Policy  │  │    Data Sources         │  │
│  │   Models    │  │   Engine    │  │    (SEC, Market, Macro) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Database   │  │    Cache    │  │    External APIs        │  │
│  │  (PG/SQLite)│  │   Manager   │  │    (SEC, FRED, Yahoo)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md)
- [Developer Guide](docs/DEVELOPER_GUIDE.adoc)
- [CLI Commands](docs/CLI_DATA_COMMANDS.md)
- [Valuation Models](docs/VALUATION_ASSUMPTIONS.md)
- [Agent Reference](docs/AGENTS.md)

## CLI Commands

```bash
# Analysis commands
investigator analyze single <SYMBOL> [--mode quick|standard|comprehensive]
investigator analyze batch <SYMBOLS...> [--parallel 4]

# Data commands
investigator data fetch <SYMBOL> --source <SOURCE>
investigator data status

# Cache management
investigator cache sizes
investigator cache clear [--symbol SYMBOL]

# System status
investigator system status
investigator system health
```

## Configuration

Configuration is managed through environment variables and `config.yaml`:

```yaml
# config.yaml
database:
  url: ${DATABASE_URL:-postgresql://localhost/investigator}
  pool_size: 5

llm:
  provider: ollama
  model: llama3.2

analysis:
  default_mode: standard
  quarters_to_analyze: 8
```

See [config/.env.example](config/.env.example) for all configuration options.

## Development

```bash
# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Lint
ruff check src/
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

For security concerns, please see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Disclaimer

InvestiGator is for informational and educational purposes only. It does not constitute financial advice, investment recommendations, or an offer to buy or sell securities. Always conduct your own research and consult with qualified financial advisors before making investment decisions.

---

Built with Python, SQLAlchemy, and a passion for data-driven investing.
