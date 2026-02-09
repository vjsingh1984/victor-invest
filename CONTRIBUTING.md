# Contributing to InvestiGator

Thank you for your interest in contributing to InvestiGator! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- PostgreSQL 14+ (optional, can use SQLite for development)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/victor-invest.git
   cd victor-invest
   ```

## Development Setup

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
# Install with dev dependencies
pip install -e ".[dev,viz,jupyter]"
```

### Configure Environment

```bash
cp config/.env.example .env
# Edit .env with your configuration
```

## Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **Black** for code formatting (line length: 120)
- Use **isort** for import sorting
- Use **type hints** on all public functions
- Maximum line length: 120 characters

### Code Formatting

```bash
# Format code
make format

# Or manually:
black src/ tests/
isort src/ tests/
```

### Linting

```bash
# Run linting
make lint

# Or manually:
flake8 src/ tests/
ruff check src/ tests/
```

### Type Checking

```bash
# Run type checker
make type-check

# Or manually:
mypy src/investigator/
```

### Naming Conventions

- **Modules/Files**: `snake_case`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

### Docstrings

Use Google style docstrings:

```python
def calculate_dcf(free_cash_flow: float, growth_rate: float) -> float:
    """Calculate Discounted Cash Flow valuation.

    Args:
        free_cash_flow: Projected free cash flow
        growth_rate: Expected growth rate (0.0-1.0)

    Returns:
        Present value of future cash flows

    Raises:
        ValueError: If growth rate is negative
    """
    pass
```

## Architecture Guidelines

### Clean Architecture

The project follows **Clean Architecture** principles:

```
src/investigator/
├── domain/           # Core business logic (no external deps)
├── application/      # Use case orchestration
├── infrastructure/   # External integrations
└── interfaces/       # CLI, API
```

**Rules:**
- Domain layer MUST NOT depend on infrastructure
- Use dependency injection for external services
- Prefer protocols/interfaces over concrete implementations

### Import Guidelines

```python
# ✅ GOOD - Clean architecture imports
from investigator.domain.services import ValuationService
from investigator.application import AnalysisService
from investigator.infrastructure.cache import CacheManager

# ❌ BAD - Old paths (being migrated)
from agents.fundamental_agent import FundamentalAgent
from utils.cache_manager import CacheManager
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/             # Fast, isolated tests
│   ├── domain/
│   ├── application/
│   └── infrastructure/
└── integration/      # Slower, end-to-end tests
```

### Running Tests

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/ -m unit

# With coverage
pytest --cov=investigator tests/
```

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_fast_calculation():
    pass

@pytest.mark.integration
def test_database_query():
    pass

@pytest.mark.slow
def test_full_pipeline():
    pass
```

## Pull Request Process

### Before Submitting

1. **Update tests** - Ensure all tests pass
2. **Add tests** - Add tests for new functionality
3. **Update docs** - Update relevant documentation
4. **Run checks** - Run `make pre-commit` to verify

### Creating a Pull Request

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "feat: add new valuation model"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a PR from your fork to the upstream repository

### Commit Message Format

Use conventional commit format:

```
<type>(<scope>): <subject>

<body>
```

**Types:** feat, fix, docs, style, refactor, test, chore

**Examples:**
```
feat(valuation): add PEG ratio valuation model

Fixes #123

- Implement PEG ratio calculation
- Add unit tests
- Update documentation
```

## Questions?

- Check [docs/](docs/) for detailed documentation
- See [docs/INDEX.md](docs/INDEX.md) for documentation index
- Open a GitHub issue for bugs or feature requests

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
