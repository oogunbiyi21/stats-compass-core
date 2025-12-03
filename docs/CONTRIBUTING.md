# Contributing to stats-compass-core

Thank you for considering contributing to stats-compass-core! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/oogunbiyi21/stats-compass-core/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and package versions
   - Minimal code example

### Suggesting Enhancements

1. Check if the enhancement has been suggested
2. Create an issue describing:
   - The problem it solves
   - Proposed solution
   - Examples of usage
   - Any potential drawbacks

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes following our coding standards
4. Write or update tests
5. Update documentation
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Poetry for dependency management

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/stats-compass-core.git
cd stats-compass-core

# Install dependencies
poetry install --with dev

# Activate virtual environment
poetry shell
```

## Creating a New Tool

### Tool Structure

Each tool should follow this structure:

```python
"""
One-line description of the tool.

More detailed description if needed.
"""
import pandas as pd
from pydantic import BaseModel, Field
from stats_compass_core.registry import registry


class ToolNameInput(BaseModel):
    """Input schema for tool_name."""
    
    param1: str = Field(
        description="Description of what param1 does"
    )
    param2: int = Field(
        default=10,
        ge=0,
        description="Description of param2 with constraints"
    )


@registry.register(
    category="category_name",
    input_schema=ToolNameInput,
    description="Brief description for tool listing"
)
def tool_name(df: pd.DataFrame, params: ToolNameInput) -> pd.DataFrame:
    """
    Detailed description of what the tool does.
    
    Args:
        df: Input DataFrame
        params: Tool parameters
    
    Returns:
        Transformed DataFrame
    
    Raises:
        ValueError: When this error occurs
        TypeError: When this error occurs
    """
    # Validate inputs
    if some_condition:
        raise ValueError("Descriptive error message")
    
    # Process data (never modify df in place)
    result = df.copy()  # or create new DataFrame
    # ... transformations ...
    
    return result
```

### Design Principles

1. **Pure Functions**
   - No side effects
   - Don't modify input DataFrames
   - Always return new objects
   - Deterministic behavior

2. **Type Safety**
   - Use complete type hints
   - Define Pydantic schemas for inputs
   - Use Python 3.11+ type syntax (`list[str]` not `List[str]`)

3. **Validation**
   - Validate all inputs with Pydantic
   - Check DataFrame columns exist
   - Validate data types
   - Raise descriptive exceptions

4. **Documentation**
   - Comprehensive docstrings
   - Document all parameters
   - Document return values
   - Document all possible exceptions

5. **Error Handling**
   - Raise typed exceptions (ValueError, TypeError, KeyError)
   - Provide clear, actionable error messages
   - Never swallow exceptions

### Tool Categories

- **cleaning/**: Data cleaning operations (removing NA, deduplication, etc.)
- **transforms/**: Data transformations (grouping, pivoting, reshaping)
- **eda/**: Exploratory data analysis (statistics, correlations, distributions)
- **ml/**: Machine learning operations (training, prediction, evaluation)
- **plots/**: Visualization tools (charts, graphs, plots)

## Testing

### Writing Tests

Create a test file in `tests/` matching your tool:

```python
import pandas as pd
import pytest
from stats_compass_core.category.tool_name import tool_name, ToolNameInput


class TestToolName:
    """Test suite for tool_name."""
    
    def test_basic_functionality(self):
        """Test basic tool operation."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        params = ToolNameInput(param1='value')
        
        result = tool_name(df, params)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        # More specific assertions
    
    def test_input_validation(self):
        """Test that invalid inputs raise errors."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="expected error message"):
            params = ToolNameInput(param1='invalid')
            tool_name(df, params)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty DataFrame
        df = pd.DataFrame()
        # DataFrame with one row
        # DataFrame with special values (inf, -inf, etc.)
        # etc.
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=stats_compass_core --cov-report=html

# Run specific test file
poetry run pytest tests/test_tool_name.py

# Run specific test
poetry run pytest tests/test_tool_name.py::TestToolName::test_basic_functionality
```

## Code Quality

### Formatting

We use Black for code formatting:

```bash
# Format all code
poetry run black stats_compass_core tests

# Check formatting without changes
poetry run black --check stats_compass_core tests
```

### Linting

We use Ruff for linting:

```bash
# Lint code
poetry run ruff check stats_compass_core tests

# Auto-fix issues
poetry run ruff check --fix stats_compass_core tests
```

### Type Checking

We use mypy for static type checking:

```bash
# Type check code
poetry run mypy stats_compass_core

# Type check specific file
poetry run mypy stats_compass_core/category/tool_name.py
```

### Pre-commit Checklist

Before committing:

1. ✅ Format code with Black
2. ✅ Lint with Ruff
3. ✅ Type check with mypy
4. ✅ Run tests with pytest
5. ✅ Ensure coverage is maintained
6. ✅ Update documentation

## Documentation

### Docstring Style

We use Google-style docstrings:

```python
def function(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    More detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When this happens
        TypeError: When this happens
    
    Examples:
        >>> function("test", 10)
        True
    """
```

### Updating Documentation

When adding a tool:

1. Update README.md with tool description
2. Add entry to docs/API.md
3. Include usage examples
4. Update any relevant guides

## Commit Messages

Use clear, descriptive commit messages:

```
Add histogram tool for data visualization

- Implement histogram function with matplotlib
- Add HistogramInput schema for validation
- Include tests for basic functionality and edge cases
- Update API documentation
```

Format:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description with bullet points

## Pull Request Process

1. **Before submitting:**
   - Run all tests and quality checks
   - Update documentation
   - Rebase on latest main branch

2. **PR Description:**
   - Describe what changes were made
   - Explain why the changes are needed
   - Link related issues
   - Include examples if applicable

3. **Review Process:**
   - Address reviewer feedback
   - Keep discussion focused and respectful
   - Make requested changes in new commits

4. **After Approval:**
   - Squash commits if requested
   - Maintainer will merge

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag
4. Build package: `poetry build`
5. Publish to PyPI: `poetry publish`

## Questions?

- Check existing [documentation](../README.md)
- Search [issues](https://github.com/oogunbiyi21/stats-compass-core/issues)
- Ask in discussions
- Create a new issue

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to stats-compass-core!
