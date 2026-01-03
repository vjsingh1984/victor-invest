# Contributing to InvestiGator

Thank you for your interest in contributing to InvestiGator! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates.

When filing a bug report, include:

1. **Description**: Clear, concise description of the bug
2. **Steps to Reproduce**: Detailed steps to reproduce the behavior
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, database type
6. **Logs**: Relevant log output (sanitize any sensitive data)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

1. **Use Case**: Describe the problem you're trying to solve
2. **Proposed Solution**: Your suggested implementation
3. **Alternatives Considered**: Other approaches you've thought about
4. **Additional Context**: Any other relevant information

### Pull Requests

1. **Fork the Repository**: Create your own fork of the project
2. **Create a Branch**: Use a descriptive branch name
   ```bash
   git checkout -b feature/add-new-valuation-model
   git checkout -b fix/sec-parsing-error
   ```
3. **Make Changes**: Implement your changes following our coding standards
4. **Write Tests**: Add tests for new functionality
5. **Update Documentation**: Update relevant documentation
6. **Submit PR**: Create a pull request with a clear description

## Development Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ (or SQLite for testing)
- Git

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/victor-invest.git
cd victor-invest

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Create test database
python -m investigator.infrastructure.database.installer --sqlite test.db
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/investigator

# Run specific test file
pytest tests/test_valuation.py

# Run tests matching pattern
pytest -k "test_dcf"
```

## Coding Standards

### Style Guide

We follow PEP 8 with some modifications:

- **Line Length**: 100 characters max
- **Imports**: Use `isort` for import ordering
- **Formatting**: Use `black` for code formatting
- **Type Hints**: Required for all public functions

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check types
mypy src/

# Lint
ruff check src/
```

### Code Organization

```
src/investigator/
├── cli/              # Command-line interface
├── domain/           # Business logic
│   ├── agents/       # Analysis agents
│   ├── services/     # Domain services
│   └── models/       # Domain models
├── application/      # Application layer (orchestration)
├── infrastructure/   # External integrations
│   ├── database/     # Database access
│   └── external/     # External APIs
└── config/           # Configuration
```

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add Gordon Growth Model valuation

- Implement GGM calculator with configurable growth rates
- Add terminal value estimation
- Include sensitivity analysis
- Add unit tests for edge cases
```

**Prefixes**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Documentation

- Use docstrings for all public functions and classes
- Follow Google-style docstrings
- Update README and relevant docs for new features

```python
def calculate_fair_value(
    symbol: str,
    model: str = "dcf",
    **kwargs
) -> ValuationResult:
    """
    Calculate fair value for a stock using specified model.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
        model: Valuation model to use ("dcf", "pe", "ps", "ggm")
        **kwargs: Model-specific parameters

    Returns:
        ValuationResult containing fair value and confidence metrics

    Raises:
        ValueError: If symbol is invalid or data unavailable
        ModelError: If valuation calculation fails

    Example:
        >>> result = calculate_fair_value("AAPL", model="dcf")
        >>> print(f"Fair value: ${result.fair_value:.2f}")
    """
```

## Testing Guidelines

### Test Structure

```python
class TestDCFValuation:
    """Tests for DCF valuation model."""

    def test_basic_calculation(self):
        """Test basic DCF with standard inputs."""
        ...

    def test_negative_fcf_handling(self):
        """Test handling of negative free cash flow."""
        ...

    @pytest.mark.parametrize("growth_rate", [0.05, 0.10, 0.15])
    def test_growth_rate_sensitivity(self, growth_rate):
        """Test DCF sensitivity to growth rate changes."""
        ...
```

### Test Coverage

- Aim for 80%+ code coverage
- Focus on critical business logic
- Include edge cases and error conditions

## Review Process

1. **Automated Checks**: CI must pass (tests, linting, type checking)
2. **Code Review**: At least one maintainer approval required
3. **Documentation Review**: Ensure docs are updated
4. **Testing**: New features must include tests

## Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check docs/ directory for guides

## Recognition

Contributors are recognized in:
- Release notes
- CONTRIBUTORS.md (for significant contributions)
- Git commit history

Thank you for contributing to InvestiGator!
