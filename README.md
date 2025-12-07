# stats-compass-core

A clean, production-ready toolkit of deterministic pandas-based data tools.

## Overview

**stats-compass-core** is a lightweight Python package that provides a curated collection of atomic, side-effect-free tools for working with tabular data. Each tool is a pure function that operates on pandas DataFrames and returns typed results.

### Key Features

- 🧹 **Clean Architecture**: Organized into logical categories (cleaning, transforms, eda, ml, plots)
- 🔒 **Type-Safe**: Complete type hints with Pydantic schemas for input validation
- 🎯 **Deterministic**: Pure functions with no side effects
- 📦 **Lightweight**: Minimal dependencies (pandas, pydantic, numpy)
- 🔌 **Extensible**: Auto-discovering registry for easy tool addition
- 📝 **Well-Documented**: Comprehensive docstrings and usage examples

## Installation

### Basic Installation

```bash
pip install stats-compass-core
```

### With Optional Dependencies

For machine learning tools:
```bash
pip install stats-compass-core[ml]
```

For plotting tools:
```bash
pip install stats-compass-core[plots]
```

For all optional features:
```bash
pip install stats-compass-core[all]
```

### For Development

```bash
git clone https://github.com/oogunbiyi21/stats-compass-core.git
cd stats-compass-core
poetry install
```

## Quick Start

```python
import pandas as pd
from stats_compass_core import registry
from stats_compass_core.cleaning.dropna import drop_na, DropNAInput
from stats_compass_core.eda.describe import describe, DescribeInput

# Create sample data
df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [5, None, 7, 8],
    'C': [9, 10, 11, 12]
})

# Use cleaning tool
params = DropNAInput(axis=0, how='any')
clean_df = drop_na(df, params)

# Use EDA tool
desc_params = DescribeInput(percentiles=[0.25, 0.5, 0.75])
stats = describe(clean_df, desc_params)
print(stats)
```

## Available Tools

### Cleaning Tools (`stats_compass_core.cleaning`)

- **drop_na**: Remove rows or columns with missing values
- **dedupe**: Remove duplicate rows

### Transform Tools (`stats_compass_core.transforms`)

- **groupby_aggregate**: Group data and apply aggregation functions
- **pivot**: Reshape data from long to wide format

### EDA Tools (`stats_compass_core.eda`)

- **describe**: Generate descriptive statistics
- **correlations**: Compute pairwise correlations

### ML Tools (`stats_compass_core.ml`) *[requires ml extra]*

- **train_classifier**: Train classification models (logistic regression, random forest, gradient boosting)
- **train_regressor**: Train regression models (linear regression, random forest, gradient boosting)

### Plotting Tools (`stats_compass_core.plots`) *[requires plots extra]*

- **histogram**: Create histogram visualizations
- **lineplot**: Create line plot visualizations

## Usage Examples

### Data Cleaning

```python
from stats_compass_core.cleaning.dedupe import dedupe, DedupeInput

# Remove duplicates
params = DedupeInput(
    subset=['column1', 'column2'],
    keep='first',
    ignore_index=True
)
deduplicated_df = dedupe(df, params)
```

### Data Transformation

```python
from stats_compass_core.transforms.groupby_aggregate import (
    groupby_aggregate,
    GroupByAggregateInput
)

# Group and aggregate
params = GroupByAggregateInput(
    by=['category'],
    agg_func={'sales': 'sum', 'quantity': 'mean'},
    as_index=False
)
aggregated_df = groupby_aggregate(df, params)
```

### Exploratory Data Analysis

```python
from stats_compass_core.eda.correlations import correlations, CorrelationsInput

# Compute correlations
params = CorrelationsInput(
    method='pearson',
    numeric_only=True
)
corr_matrix = correlations(df, params)
```

### Machine Learning

```python
from stats_compass_core.ml.train_classifier import (
    train_classifier,
    TrainClassifierInput
)

# Train a classifier
params = TrainClassifierInput(
    target_column='label',
    feature_columns=['feature1', 'feature2', 'feature3'],
    model_type='random_forest',
    test_size=0.2,
    random_state=42
)
result = train_classifier(df, params)
print(f"Training score: {result.train_score:.3f}")
```

### Plotting

```python
from stats_compass_core.plots.histogram import histogram, HistogramInput

# Create histogram
params = HistogramInput(
    column='sales',
    bins=20,
    title='Sales Distribution',
    figsize=(12, 6)
)
result = histogram(df, params)
result.figure.savefig('histogram.png')
```

## Tool Registry

The package includes a registry that automatically discovers and manages all tools:

```python
from stats_compass_core import registry

# List all tools
all_tools = registry.list_tools()
for tool in all_tools:
    print(f"{tool.category}.{tool.name}: {tool.description}")

# List tools by category
cleaning_tools = registry.list_tools(category='cleaning')

# Get a specific tool
tool_func = registry.get_tool('cleaning', 'drop_na')
```

## Contributing

We welcome contributions! Here's how to create a new tool:

### 1. Choose a Category

Place your tool in the appropriate folder:
- `cleaning/` - Data cleaning operations
- `transforms/` - Data transformations
- `eda/` - Exploratory data analysis
- `ml/` - Machine learning operations
- `plots/` - Visualization tools

### 2. Create a New Tool File

Each tool should be in its own file (e.g., `my_tool.py`):

```python
"""
Tool description goes here.
"""
import pandas as pd
from pydantic import BaseModel, Field
from stats_compass_core.registry import registry


class MyToolInput(BaseModel):
    """Input schema for my_tool."""
    
    param1: str = Field(description="Description of param1")
    param2: int = Field(default=10, ge=0, description="Description of param2")


@registry.register(
    category="category_name",
    input_schema=MyToolInput,
    description="Short description of what the tool does"
)
def my_tool(df: pd.DataFrame, params: MyToolInput) -> pd.DataFrame:
    """
    Detailed description of the tool.
    
    Args:
        df: Input DataFrame
        params: Tool parameters
    
    Returns:
        Transformed DataFrame
    
    Raises:
        ValueError: Description of when this error occurs
    """
    # Your implementation here
    return result_df
```

### 3. Design Principles

- **Pure Functions**: No side effects, no global state mutation
- **Type Hints**: Use complete type annotations everywhere
- **Pydantic Schemas**: Define input validation schemas
- **Return New Objects**: Never modify input DataFrames in place
- **Descriptive Errors**: Raise typed exceptions with clear messages
- **Comprehensive Docstrings**: Document args, returns, and raises

### 4. Write Tests

Create a test file in `tests/` (e.g., `test_my_tool.py`):

```python
import pandas as pd
import pytest
from stats_compass_core.category.my_tool import my_tool, MyToolInput


def test_my_tool_basic():
    df = pd.DataFrame({'A': [1, 2, 3]})
    params = MyToolInput(param1='value')
    result = my_tool(df, params)
    assert isinstance(result, pd.DataFrame)
    # Add more assertions


def test_my_tool_validation():
    df = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(ValueError):
        params = MyToolInput(param1='invalid')
        my_tool(df, params)
```

### 5. Run Tests and Linting

```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=stats_compass_core

# Format code
poetry run black .

# Lint code
poetry run ruff check .

# Type check
poetry run mypy stats_compass_core
```

### 6. Submit a Pull Request

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-tool`)
3. Commit your changes (`git commit -m 'Add my_tool'`)
4. Push to the branch (`git push origin feature/my-tool`)
5. Open a Pull Request

## Development

### Project Structure

```
stats-compass-core/
├── stats_compass_core/
│   ├── __init__.py
│   ├── registry.py
│   ├── cleaning/
│   │   ├── __init__.py
│   │   ├── dropna.py
│   │   └── dedupe.py
│   ├── transforms/
│   │   ├── __init__.py
│   │   ├── groupby_aggregate.py
│   │   └── pivot.py
│   ├── eda/
│   │   ├── __init__.py
│   │   ├── describe.py
│   │   └── correlations.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── train_classifier.py
│   │   └── train_regressor.py
│   └── plots/
│       ├── __init__.py
│       ├── histogram.py
│       └── lineplot.py
├── tests/
├── docs/
├── pyproject.toml
├── README.md
└── LICENSE
```

### Running Tests

```bash
poetry run pytest
```

### Code Quality

```bash
# Format
poetry run black stats_compass_core tests

# Lint
poetry run ruff check stats_compass_core tests

# Type check
poetry run mypy stats_compass_core
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- **Repository**: https://github.com/oogunbiyi21/stats-compass-core
- **Issues**: https://github.com/oogunbiyi21/stats-compass-core/issues
- **Documentation**: Coming soon

## Roadmap

- [ ] Additional cleaning tools (outlier detection, type conversion)
- [ ] More transformation tools (melting, merging, joining)
- [ ] Advanced EDA tools (distribution testing, feature importance)
- [ ] Time series analysis tools
- [ ] More visualization options
- [ ] Performance optimization tools
- [ ] Data validation tools
- [ ] Export/import utilities

## Support

For questions, issues, or contributions, please:

1. Check existing [issues](https://github.com/oogunbiyi21/stats-compass-core/issues)
2. Create a new issue with detailed information
3. Join our discussions

---

Made with ❤️ for the data science community
