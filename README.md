<div align="center">
  <img src="./assets/logo/logo1.png" alt="Stats Compass Logo" width="200"/>
  
  # stats-compass-core
  
  **Stateful pandas toolkit for AI agents.** 50+ data tools via MCP.

  [![PyPI version](https://badge.fury.io/py/stats-compass-core.svg)](https://badge.fury.io/py/stats-compass-core)
  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

## What is this?

A Python library that turns pandas operations into JSON-serializable tools for LLM agents. Unlike raw pandas, it manages server-side state. DataFrames and trained models persist across tool calls.

**Looking for the MCP server?** See [stats-compass-mcp](https://github.com/oogunbiyi21/stats-compass-mcp).

## Quick Start

```bash
pip install stats-compass-core[all]
```

```python
from stats_compass_core import DataFrameState, registry
import pandas as pd

# Initialize state (one per session)
state = DataFrameState()

# Load data
df = pd.read_csv("data.csv")
state.set_dataframe(df, name="my_data", operation="load")

# Call tools via registry
result = registry.invoke("eda", "describe", state, {"dataframe_name": "my_data"})
print(result.model_dump_json())  # JSON-serializable output

# Run complete workflows
result = registry.invoke("workflows", "run_classification", state, {
    "target_column": "churn",
    "feature_columns": ["age", "tenure", "balance"],
    "config": {"model_type": "random_forest", "generate_plots": True}
})
```

## What's Included

| Category | Tools | Description |
|----------|-------|-------------|
| **Data** | `load_csv`, `get_schema`, `list_dataframes` | Load and inspect data |
| **Cleaning** | `drop_na`, `impute`, `dedupe`, `handle_outliers` | Clean messy data |
| **Transforms** | `filter`, `groupby`, `pivot`, `encode`, `scale` | Reshape and transform |
| **EDA** | `describe`, `correlations`, `hypothesis_test` | Statistical analysis |
| **Plots** | `histogram`, `scatter`, `bar`, `roc_curve` | Visualizations (base64 PNG) |
| **ML** | `train_*`, `evaluate`, `predict`, `cross_validate` | Machine learning |
| **Workflows** | `run_preprocessing`, `run_classification`, `run_regression` | Multi-step pipelines |

See [docs/TOOLS.md](docs/TOOLS.md) for the complete list of 50+ tools.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     stats-compass-core                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   DataFrameState                        │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │
│  │  │ DataFrames  │  │   Models    │  │   History   │      │    │
│  │  │ (by name)   │  │  (by ID)    │  │  (lineage)  │      │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                  │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│  ┌──────────────────┐ ┌──────────────┐ ┌───────────────────┐   │
│  │ Workflow Tools   │ │  Sub-Tools   │ │  Result Models    │   │
│  │  (orchestrate)   │ │  (atomic)    │ │  (JSON-safe)      │   │
│  └──────────────────┘ └──────────────┘ └───────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key concepts:**

1. **DataFrameState** - Stores DataFrames and models server-side
2. **Registry** - Discovers and invokes tools by category/name  
3. **Result Models** - All tools return Pydantic models (JSON-serializable)
4. **Workflows** - High-level tools that chain sub-tools together

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design docs.

## Installation Options

| Use Case | Command |
|----------|---------|
| Core only (data, cleaning, EDA) | `pip install stats-compass-core` |
| With ML (scikit-learn) | `pip install stats-compass-core[ml]` |
| With plotting (matplotlib) | `pip install stats-compass-core[plots]` |
| With time series (statsmodels) | `pip install stats-compass-core[timeseries]` |
| Everything | `pip install stats-compass-core[all]` |

## Documentation

- [**TOOLS.md**](docs/TOOLS.md) - Complete tool reference (50+ tools)
- [**EXAMPLES.md**](docs/EXAMPLES.md) - Detailed usage examples  
- [**API.md**](docs/API.md) - Core classes and methods
- [**ARCHITECTURE.md**](ARCHITECTURE.md) - Design decisions

## Development

```bash
git clone https://github.com/oogunbiyi21/stats-compass-core.git
cd stats-compass-core
poetry install --with dev
poetry run pytest
```

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for contribution guidelines.

## License

MIT
