# Contributing to Market Data ETL & Backtesting Engine

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, professional, and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Your environment (Python version, OS, etc.)
   - Relevant code snippets or error messages

### Suggesting Enhancements

1. Check if the enhancement has been suggested
2. Create a new issue describing:
   - The enhancement
   - Use cases
   - Potential implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes following our coding standards
4. Add or update tests as needed
5. Update documentation if needed
6. Ensure all tests pass: `pytest tests/ -v`
7. Run linters: `black .` and `flake8 .`
8. Commit your changes: `git commit -m 'Add amazing feature'`
9. Push to your fork: `git push origin feature/amazing-feature`
10. Create a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/Build-a-Market-Data-ETL-Strategy-Backtesting-Engine.git
cd Build-a-Market-Data-ETL-Strategy-Backtesting-Engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio pytest-cov black flake8 isort mypy

# Run tests
pytest tests/ -v
```

## Coding Standards

### Python Style
- Follow PEP 8
- Use Black for formatting: `black .`
- Use isort for imports: `isort .`
- Maximum line length: 127 characters
- Use type hints where appropriate

### Testing
- Write tests for new features
- Maintain or improve test coverage
- Use descriptive test names
- Include docstrings in test methods

### Documentation
- Update docstrings for public methods
- Follow Google style docstrings
- Update README.md if adding features
- Add usage examples for new functionality

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove)
- Reference issue numbers when applicable

## Areas for Contribution

### High Priority
- Additional strategy implementations
- Performance optimizations
- More comprehensive tests
- Documentation improvements

### Medium Priority
- Additional data sources/adapters
- Advanced portfolio management features
- Risk management tools
- More visualization options

### Nice to Have
- Web dashboard
- Real-time monitoring
- Cloud deployment scripts
- Jupyter notebook examples

## Questions?

Open an issue with the "question" label or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
