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

- Python 3.12 or higher
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

## Architecture Overview

stats-compass-core uses an **MCP-compatible stateful architecture**:

- **DataFrameState**: Server-side state manager that stores DataFrames and trained models
- **Pydantic Models**: All inputs and outputs are strongly typed
- **JSON Serialization**: All tool returns are JSON-serializable for protocol transport
- **Registry Pattern**: Tools are registered by category for discovery

This architecture enables stats-compass-core to work with the Model Context Protocol (MCP) where tools can't receive raw DataFrames or model objects directly.

## Creating a New Tool

### Tool Structure

Each tool must follow this structure:

```python
"""
One-line description of the tool.

More detailed description if needed.
"""
import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState
from stats_compass_core.results import (
    DataFrameQueryResult,  # or appropriate result type
)


class ToolNameInput(BaseModel):
    """Input schema for tool_name."""
    
    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame. Uses active DataFrame if not specified."
    )
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
def tool_name(state: DataFrameState, params: ToolNameInput) -> DataFrameQueryResult:
    """
    Detailed description of what the tool does.
    
    Args:
        state: DataFrameState containing the DataFrames
        params: Tool parameters
    
    Returns:
        DataFrameQueryResult with operation results
    
    Raises:
        ValueError: When parameters are invalid
        KeyError: When DataFrame not found
    """
    # Get DataFrame from state
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state._active_dataframe
    
    # Validate inputs
    if params.param1 not in df.columns:
        raise ValueError(f"Column '{params.param1}' not found in DataFrame")
    
    # Process data
    result = do_something(df, params.param1)
    
    # Return JSON-serializable result
    return DataFrameQueryResult(
        data={"result_key": result},
        row_count=len(df),
        dataframe_name=source_name,
    )
```

### Design Principles

1. **Stateful Pattern**
   - Always accept `state: DataFrameState` as first parameter
   - Always accept `params: InputSchema` as second parameter
   - Never modify state unless the tool is meant to (like `load_csv`)
   - Use `state.get_dataframe()` to access DataFrames

2. **JSON-Serializable Returns**
   - All returns must be Pydantic models
   - Use existing result types from `results.py` when possible
   - Charts return base64-encoded PNGs in `ChartResult`
   - Never return raw DataFrames, numpy arrays, or model objects

3. **Type Safety**
   - Use complete type hints
   - Define Pydantic schemas for all inputs
   - Use Python 3.12+ type syntax (`list[str]` not `List[str]`)

4. **Validation**
   - Validate all inputs with Pydantic Field constraints
   - Check DataFrame columns exist before using them
   - Validate data types are compatible
   - Raise descriptive exceptions

5. **Documentation**
   - Comprehensive docstrings
   - Document all parameters
   - Document return values
   - Document all possible exceptions

6. **Error Handling**
   - Raise typed exceptions (ValueError, TypeError, KeyError)
   - Provide clear, actionable error messages
   - Never swallow exceptions

### Result Types

Choose the appropriate result type for your tool:

| Result Type | Use Case |
|-------------|----------|
| `DataFrameInfo` | When returning info about a stored DataFrame |
| `DataFrameQueryResult` | For read operations returning data/stats |
| `DataFrameMutationResult` | For operations that modify a DataFrame |
| `ChartResult` | For visualization tools |
| `ModelTrainingResult` | For ML training tools |
| `HypothesisTestResult` | For statistical tests |
| `ClassificationEvaluationResult` | For classification model evaluation |
| `RegressionEvaluationResult` | For regression model evaluation |

If none fit, add a new result type to `stats_compass_core/results.py`.

### Tool Categories

- **data/**: Data loading and inspection (CSV loading, schema inspection)
- **cleaning/**: Data cleaning operations (removing NA, deduplication, imputation)
- **transforms/**: Data transformations (grouping, pivoting, filtering)
- **eda/**: Exploratory data analysis (statistics, correlations, hypothesis tests)
- **ml/**: Machine learning operations (training, prediction, evaluation)
- **plots/**: Visualization tools (histograms, scatter plots, line charts)

### Adding a Result Model

If you need a new result type:

```python
# In stats_compass_core/results.py

class NewToolResult(BaseModel):
    """Result from new_tool operation."""
    
    primary_output: float = Field(description="Main result value")
    secondary_data: dict[str, Any] = Field(description="Additional data")
    dataframe_name: str = Field(description="Source DataFrame name")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "primary_output": 0.95,
                "secondary_data": {"key": "value"},
                "dataframe_name": "my_data",
            }
        }
    )
```

## Testing

### Writing Tests

Create a test file in `tests/` matching your tool:

```python
import pandas as pd
import pytest

from stats_compass_core.state import DataFrameState
from stats_compass_core.category.tool_name import tool_name, ToolNameInput


class TestToolName:
    """Test suite for tool_name."""
    
    @pytest.fixture
    def state_with_data(self):
        """Create DataFrameState with test data."""
        state = DataFrameState()
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'category': ['x', 'y', 'x', 'y', 'x'],
        })
        state.set_dataframe("test_data", df)
        return state
    
    def test_basic_functionality(self, state_with_data):
        """Test basic tool operation."""
        params = ToolNameInput(param1='A')
        
        result = tool_name(state_with_data, params)
        
        assert result.dataframe_name == "test_data"
        assert result.row_count == 5
    
    def test_explicit_dataframe_name(self, state_with_data):
        """Test specifying dataframe_name explicitly."""
        params = ToolNameInput(
            dataframe_name="test_data",
            param1='B'
        )
        
        result = tool_name(state_with_data, params)
        
        assert result.dataframe_name == "test_data"
    
    def test_column_not_found(self, state_with_data):
        """Test error when column doesn't exist."""
        params = ToolNameInput(param1='nonexistent')
        
        with pytest.raises(ValueError, match="not found"):
            tool_name(state_with_data, params)
    
    def test_result_is_json_serializable(self, state_with_data):
        """Test that result can be JSON serialized."""
        params = ToolNameInput(param1='A')
        
        result = tool_name(state_with_data, params)
        
        # Should not raise
        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
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

# Run with verbose output
poetry run pytest -v
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
def function(state: DataFrameState, params: FunctionInput) -> FunctionResult:
    """
    Brief description of function.
    
    More detailed description if needed.
    
    Args:
        state: DataFrameState containing the DataFrames
        params: Function parameters
    
    Returns:
        FunctionResult with computed values
    
    Raises:
        ValueError: When parameters are invalid
        KeyError: When DataFrame not found
    
    Examples:
        >>> state = DataFrameState()
        >>> state.set_dataframe("data", pd.DataFrame({'A': [1,2,3]}))
        >>> params = FunctionInput(column='A')
        >>> result = function(state, params)
        >>> result.value
        2.0
    """
```

### Updating Documentation

When adding a tool:

1. Update README.md with tool description in the appropriate category
2. Add entry to docs/API.md with input schema and result type
3. Include usage examples in docs/EXAMPLES.md
4. Update any relevant guides

## Commit Messages

Use clear, descriptive commit messages:

```
Add histogram tool for data visualization

- Implement histogram function with DataFrameState
- Add HistogramInput schema for validation
- Return ChartResult with base64 PNG
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

## Common Patterns

### Tool That Creates New DataFrame

```python
@registry.register(...)
def transform_tool(state: DataFrameState, params: TransformInput) -> DataFrameQueryResult:
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state._active_dataframe
    
    # Create new DataFrame
    result_df = df.copy()
    # ... transformations ...
    
    # Save to state with new name (set_dataframe returns the name string)
    output_name = params.output_name or f"{source_name}_transformed"
    stored_name = state.set_dataframe(result_df, name=output_name, operation="transform")
    
    return DataFrameQueryResult(
        data={"created": output_name},
        row_count=len(result_df),
        dataframe_name=stored_name,
    )
```

### Tool That Stores a Model

```python
@registry.register(...)
def train_tool(state: DataFrameState, params: TrainInput) -> ModelTrainingResult:
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state._active_dataframe
    
    # Train model
    model = SomeModel()
    model.fit(X, y)
    
    # Store in state
    model_id = state.store_model(model, source_name, "model_type")
    
    return ModelTrainingResult(
        model_id=model_id,
        model_type="SomeModel",
        # ... other metrics ...
    )
```

### Tool That Uses a Stored Model

```python
@registry.register(...)
def evaluate_tool(state: DataFrameState, params: EvaluateInput) -> EvaluationResult:
    # Get model from state
    model = state.get_model(params.model_id)
    if model is None:
        raise ValueError(f"Model '{params.model_id}' not found")
    
    # Get data
    df = state.get_dataframe(params.dataframe_name)
    
    # Use model
    predictions = model.predict(X)
    
    return EvaluationResult(...)
```

## Questions?

- Check existing [documentation](../README.md)
- Search [issues](https://github.com/oogunbiyi21/stats-compass-core/issues)
- Ask in discussions
- Create a new issue

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to stats-compass-core!
