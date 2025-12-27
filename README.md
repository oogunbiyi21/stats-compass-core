<div align="center">
  <img src="./assets/logo/logo1.png" alt="Stats Compass Logo" width="200"/>
  
  <h1>stats-compass-core</h1>
  
  <p>A stateful, MCP-compatible toolkit of pandas-based data tools for AI-powered data analysis.</p>

  [![PyPI version](https://badge.fury.io/py/stats-compass-core.svg)](https://badge.fury.io/py/stats-compass-core)
  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

> ⚠️ **Status: Early developer release (v0.1)**  
> Optimized for Claude Desktop. VS Code Copilot support is beta.  
> Gemini and GPT tool calling may be inconsistent.

## Overview

**stats-compass-core** is a Python package that provides a curated collection of data tools designed for use with LLM agents via the Model Context Protocol (MCP). Unlike traditional pandas libraries, this package manages server-side state, allowing AI agents to work with DataFrames across multiple tool invocations without passing raw data over the wire.

Key features:
- **Workflow Tools**: Single-call solutions for common multi-step tasks (preprocessing, classification, time series forecasting)
- **Sub-Tool Functions**: 50+ atomic operations for fine-grained control
- **Stateful Design**: Server-side state management for DataFrames and trained models
- **JSON-Serializable**: All results are Pydantic models that serialize to JSON
- **MCP-Compatible**: Designed for Model Context Protocol integration

This is the **core library** containing the business logic, state management, and tool definitions. If you are looking for the MCP server to use with Claude or other clients, please see [stats-compass-mcp](https://github.com/oogunbiyi21/stats-compass-mcp).

## ✅ Supported Clients

Stats Compass is designed and tested for official Model Context Protocol (MCP) integrations.

- **VS Code Copilot Chat**: Fully supported via native MCP integration.
- **Claude Desktop**: Fully supported.

> **Note:** Third-party extensions such as **Roo Code** are **not supported** due to incompatible JSON Schema validation logic that conflicts with the official spec.

## 🚀 Quick Start

### 1. Install
```bash
pip install stats-compass-core[all]
```

### 2. Usage in Python

```python
from stats_compass_core import DataFrameState, registry
import pandas as pd

# Initialize state
state = DataFrameState()

# Load data
df = pd.read_csv("data.csv")
state.set_dataframe(df, name="my_data", operation="load")

# Invoke tools
result = registry.invoke("eda", "describe", state, {"dataframe_name": "my_data"})
print(result.statistics)
```

### Key Features

- 🎯 **Workflow Tools**: One-call solutions for preprocessing, classification, regression, EDA, and time series forecasting
- 🔄 **Stateful Design**: Server-side `DataFrameState` manages multiple DataFrames and trained models
- 📦 **MCP-Compatible**: All tools return JSON-serializable Pydantic models
- 🧹 **Clean Architecture**: Organized into logical categories (data, cleaning, transforms, eda, ml, plots, workflows)
- 🔒 **Type-Safe**: Complete type hints with Pydantic schemas for input validation
- 🎯 **Memory-Managed**: Configurable memory limits prevent runaway state growth
- 📊 **Base64 Charts**: Visualization tools return PNG images as base64 strings
- 🤖 **Model Storage**: Trained ML models stored by ID for later use
- ⚡ **50+ Sub-Tools**: Fine-grained atomic operations for precise control

## 📂 Data Loading Guide

**Crucial:** Stats Compass tools operate on local files. When using this library via an MCP server (like `stats-compass-mcp`), the server runs locally on your machine. It cannot see files you drag-and-drop into a chat window. You must tell it where your files are on your disk.

### How to load your own data

1.  **Find your file**:
    Use the `list_files` tool to explore directories.
    
2.  **Load the file**:
    Use `load_csv` or `load_excel` with the correct **absolute path**.

### Why does drag-and-drop not work?
When you drag a file into a chat interface, it stays in the cloud sandbox. Stats Compass tools run on your local computer. To bridge this gap, you must point the tools to the actual file path on your hard drive.

### Saving your work

You can save your processed data and trained models back to your local disk.

- **Save Data**: Use `save_csv` to save a DataFrame to a CSV file.
  > "Save the cleaned dataframe to ~/Documents/cleaned_data.csv"

- **Save Models**: Use `save_model` to save a trained model (using joblib).
  > "Save the regression model to ~/models/price_predictor.joblib"

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
│  │ Workflow Tools   │ │  Sub-Tools   │ │  Category Tools   │   │
│  │  (orchestrate)   │ │  (atomic)    │ │  (dispatch)       │   │
│  └────────┬─────────┘ └──────┬───────┘ └─────────┬─────────┘   │
│           │                  │                   │              │
│           └──────────────────┼───────────────────┘              │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Pydantic Result Models                     │    │
│  │         (WorkflowResult, ChartResult, etc.)             │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Four-Tier Tool Architecture

**Tier 1: Workflow Tools** - High-level orchestration (6 tools)
- `run_preprocessing`, `run_classification`, `run_regression`, `run_eda_report`, `run_timeseries_forecast`
- Single-call solutions for common multi-step tasks
- Return `WorkflowResult` with step-by-step execution details

**Tier 2: Category Tools (Optional)** - Dynamic dispatchers (~12 tools)
- `describe_cleaning`, `execute_cleaning`, `describe_eda`, `execute_eda`, etc.
- Reduce tool count for LLM clients with limits
- Used by MCP clients struggling with 50+ tools (Gemini, GPT)
- Not needed for Claude Desktop or VS Code

**Tier 3: Sub-Tool Functions** - Atomic operations (50+ tools)
- `load_csv`, `drop_na`, `describe`, `train_random_forest_classifier`, etc.
- Each does one thing well
- Backward compatible with existing code

**Tier 4: DataFrameState** - Shared memory layer
- Multiple named DataFrames
- Trained model storage by ID
- Memory limits and cleanup

### Three-Layer Stack

1. **stats-compass-core** (this package) - Stateful Python tools
   - Manages DataFrames and models server-side
   - Returns JSON-serializable Pydantic results
   - Pure data operations, no UI or orchestration

2. **stats-compass-mcp** (separate package) - MCP Server
   - Exposes tools via Model Context Protocol
   - Handles JSON transport to/from LLM agents
   - **Not part of this repository**

3. **stats-compass-app** (separate package) - SaaS Application
   - Web UI for human interaction
   - Multi-tool pipelines and workflows
   - **Not part of this repository**

### Registry & Tool Discovery Flow

The `registry` module is the central nervous system for tool management. Here's how it works:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STARTUP / INITIALIZATION                         │
├─────────────────────────────────────────────────────────────────────────┤
│  1. App calls registry.auto_discover()                                  │
│  2. Registry walks category folders (data/, cleaning/, transforms/...)  │
│  3. Each module is imported via importlib.import_module()               │
│  4. @registry.register decorators fire, populating _tools dict          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TOOL INVOCATION                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  1. MCP server receives request: {"tool": "cleaning.drop_na", ...}      │
│  2. Calls registry.invoke("cleaning", "drop_na", state, params)         │
│  3. Registry validates params against Pydantic input_schema             │
│  4. Registry calls tool function with (state, validated_params)         │
│  5. Tool returns Pydantic result model (JSON-serializable)              │
│  6. MCP server sends result.model_dump_json() back to LLM               │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key files:**
- `registry.py` - Tool registration and invocation
- `state.py` - DataFrameState for server-side data management  
- `results.py` - Pydantic result types for JSON serialization

## Installation

### Basic Installation (Core Only)

```bash
pip install stats-compass-core
```

This installs the core functionality: data loading, cleaning, transforms, and EDA tools. Dependencies: pandas, numpy, scipy, pydantic.

### With Optional Features

```bash
# For machine learning tools (scikit-learn)
pip install stats-compass-core[ml]

# For plotting tools (matplotlib, seaborn)
pip install stats-compass-core[plots]

# For time series / ARIMA tools (statsmodels)
pip install stats-compass-core[timeseries]

# For everything
pip install stats-compass-core[all]
```

### For Development

```bash
git clone https://github.com/oogunbiyi21/stats-compass-core.git
cd stats-compass-core
poetry install --with dev  # Installs all deps including optional ones
```

### Installation Matrix

| Use Case | Install Command |
|----------|----------------|
| Core only (data, cleaning, EDA) | `pip install stats-compass-core` |
| With ML tools | `pip install stats-compass-core[ml]` |
| With plotting | `pip install stats-compass-core[plots]` |
| With time series | `pip install stats-compass-core[timeseries]` |
| Everything | `pip install stats-compass-core[all]` |

## Quick Start

### Workflow Example (Recommended)

The fastest way to accomplish common tasks:

```python
from stats_compass_core import DataFrameState, registry
import pandas as pd

# Initialize state
state = DataFrameState()

# Load data
df = pd.read_csv("sales_data.csv")
state.set_dataframe(df, name="sales", operation="load")

# Run complete preprocessing in one call
result = registry.invoke("workflows", "run_preprocessing", state, {
    "dataframe_name": "sales",
    "config": {
        "date_cleaning": {
            "date_column": "order_date",
            "fill_method": "ffill",
            "infer_frequency": True
        },
        "imputation": {"strategy": "median"},
        "outliers": {"method": "iqr", "action": "cap"},
        "dedupe": True
    }
})

# Check execution details
print(f"Status: {result.status}")
print(f"Duration: {result.total_duration_ms}ms")
print(f"Steps completed: {len([s for s in result.steps if s.status == 'success'])}")
print(f"Final DataFrame: {result.artifacts.final_dataframe}")

# Use the cleaned data
cleaned_df = state.get_dataframe(result.artifacts.final_dataframe)
```

### Sub-Tool Usage Pattern (Fine-Grained Control)

For precise control over individual operations:

All tools follow the same pattern:
1. Create a `DataFrameState` instance (once per session)
2. Load data into state
3. Call tools with `(state, params)` signature
4. Tools return JSON-serializable result objects

```python
import pandas as pd
from stats_compass_core import DataFrameState, registry

# 1. Create state manager (one per session)
state = DataFrameState(memory_limit_mb=500)

# 2. Load data into state
df = pd.read_csv("sales_data.csv")
state.set_dataframe(df, name="sales", operation="load_csv")

# 3. Call tools via registry
result = registry.invoke("eda", "describe", state, {})
print(result.model_dump_json())  # JSON-serializable output

# 4. Chain operations
result = registry.invoke("transforms", "groupby_aggregate", state, {
    "by": ["region"],
    "aggregations": [
        {"column": "revenue", "functions": ["sum"]},
        {"column": "quantity", "functions": ["mean"]}
    ]
})
# Result DataFrame saved to state automatically
print(f"New DataFrame: {result.dataframe_name}")
```

### Direct Tool Usage

You can also import and call tools directly:

```python
from stats_compass_core import DataFrameState
from stats_compass_core.eda.describe import describe, DescribeInput
from stats_compass_core.cleaning.dropna import drop_na, DropNAInput

# Create state and load data
state = DataFrameState()
state.set_dataframe(my_dataframe, name="data", operation="manual")

# Call tool with typed params
params = DescribeInput(percentiles=[0.25, 0.5, 0.75])
result = describe(state, params)

# Result is a Pydantic model
print(result.statistics)  # dict of column stats
print(result.dataframe_name)  # "data"
```

## Core Concepts

### DataFrameState

The `DataFrameState` class manages all server-side data:

```python
from stats_compass_core import DataFrameState

state = DataFrameState(memory_limit_mb=500)

# Store DataFrames (multiple allowed)
state.set_dataframe(df1, name="raw_data", operation="load_csv")
state.set_dataframe(df2, name="cleaned", operation="drop_na")

# Retrieve DataFrames
df = state.get_dataframe("raw_data")
df = state.get_dataframe()  # Gets active DataFrame

# Check what's stored
print(state.list_dataframes())          # [DataFrameInfo(...), ...]
print(state.get_active_dataframe_name())  # 'cleaned' (most recent)

# Store trained models
model_id = state.store_model(
    model=trained_model,
    model_type="random_forest_classifier", 
    target_column="churn",
    feature_columns=["age", "tenure", "balance"],
    source_dataframe="training_data"
)

# Retrieve models
model = state.get_model(model_id)
info = state.get_model_info(model_id)
```

### Result Types

All tools return Pydantic models that serialize to JSON:

| Result Type | Used By | Key Fields |
|-------------|---------|------------|
| `DataFrameLoadResult` | data loading tools | `dataframe_name`, `shape`, `columns` |
| `DataFrameMutationResult` | cleaning tools | `rows_before`, `rows_after`, `rows_affected` |
| `DataFrameQueryResult` | transform tools | `data`, `shape`, `dataframe_name` |
| `DescribeResult` | describe | `statistics`, `columns_analyzed` |
| `CorrelationsResult` | correlations | `correlations`, `method` |
| `ChartResult` | all plot tools | `image_base64`, `chart_type` |
| `ModelTrainingResult` | ML training | `model_id`, `metrics`, `feature_columns` |
| `HypothesisTestResult` | statistical tests | `statistic`, `p_value`, `significant_at_05` |

### Registry

The registry provides tool discovery and invocation:

```python
from stats_compass_core import registry

# List all tools
for key, metadata in registry._tools.items():
    print(f"{key}: {metadata.description}")

# Invoke a tool (handles param validation)
result = registry.invoke(
    category="cleaning",
    tool_name="drop_na",
    state=state,
    params={"how": "any", "axis": 0}
)
```

## Available Tools

### Workflow Tools (`stats_compass_core.workflows`) *[Recommended]*

High-level orchestration tools that execute complete multi-step pipelines in a single call:

| Workflow | Description | Use Case |
|----------|-------------|----------|
| `run_preprocessing` | Complete data cleaning pipeline | Clean messy data for analysis/ML |
| `run_classification` | Train + evaluate classification model | Predict categories (churn, sentiment, etc.) |
| `run_regression` | Train + evaluate regression model | Predict continuous values (price, sales, etc.) |
| `run_eda_report` | Comprehensive exploratory analysis | Understand dataset characteristics |
| `run_timeseries_forecast` | ARIMA forecasting with validation | Predict future values from time series |

**Example:**
```python
# Single call does: analyze → clean dates → impute → handle outliers → dedupe
result = registry.invoke("workflows", "run_preprocessing", state, {
    "config": {
        "date_cleaning": {"date_column": "Date", "fill_method": "ffill"},
        "imputation": {"strategy": "median"},
        "outliers": {"method": "iqr", "action": "cap"}
    }
})
```

Returns `WorkflowResult` with:
- `steps`: Step-by-step execution details
- `artifacts`: Created DataFrames, models, charts
- `status`: "success" | "partial_failure" | "failed"
- `suggestion`: Recovery hints if failed

---

### Data Tools (`stats_compass_core.data`)

| Tool | Description | Returns |
|------|-------------|---------|
| `load_csv` | Load CSV file into state | `DataFrameLoadResult` |
| `get_schema` | Get DataFrame column types and stats | `SchemaResult` |
| `get_sample` | Get sample rows from DataFrame | `SampleResult` |
| `list_dataframes` | List all DataFrames in state | `DataFrameListResult` |

### Cleaning Tools (`stats_compass_core.cleaning`)

| Tool | Description | Returns |
|------|-------------|---------|
| `drop_na` | Remove rows/columns with missing values | `DataFrameMutationResult` |
| `dedupe` | Remove duplicate rows | `DataFrameMutationResult` |
| `apply_imputation` | Fill missing values (mean/median/mode/constant) | `DataFrameMutationResult` |
| `handle_outliers` | Handle outliers (cap/remove/winsorize/log/IQR) | `OutlierHandlingResult` |

### Transform Tools (`stats_compass_core.transforms`)

| Tool | Description | Returns |
|------|-------------|---------|
| `groupby_aggregate` | Group and aggregate data | `DataFrameQueryResult` |
| `pivot` | Reshape long to wide format | `DataFrameQueryResult` |
| `filter_dataframe` | Filter with pandas query syntax | `DataFrameQueryResult` |
| `bin_rare_categories` | Bin rare categories into 'Other' | `BinRareCategoriesResult` |
| `mean_target_encoding` | Target encoding for categoricals *[requires ml]* | `MeanTargetEncodingResult` |

### EDA Tools (`stats_compass_core.eda`)

| Tool | Description | Returns |
|------|-------------|---------|
| `describe` | Descriptive statistics | `DescribeResult` |
| `correlations` | Correlation matrix | `CorrelationsResult` |
| `t_test` | Two-sample t-test | `HypothesisTestResult` |
| `z_test` | Two-sample z-test | `HypothesisTestResult` |
| `chi_square_independence` | Chi-square test for independence | `HypothesisTestResult` |
| `chi_square_goodness_of_fit` | Chi-square goodness-of-fit test | `HypothesisTestResult` |
| `analyze_missing_data` | Analyze missing data patterns | `MissingDataAnalysisResult` |
| `detect_outliers` | Detect outliers using IQR/Z-score | `OutlierDetectionResult` |
| `data_quality_report` | Comprehensive data quality report | `DataQualityReportResult` |

### ML Tools (`stats_compass_core.ml`) *[requires ml extra]*

| Tool | Description | Returns |
|------|-------------|---------|
| `train_linear_regression` | Train linear regression | `ModelTrainingResult` |
| `train_logistic_regression` | Train logistic regression | `ModelTrainingResult` |
| `train_random_forest_classifier` | Train RF classifier | `ModelTrainingResult` |
| `train_random_forest_regressor` | Train RF regressor | `ModelTrainingResult` |
| `train_gradient_boosting_classifier` | Train GB classifier | `ModelTrainingResult` |
| `train_gradient_boosting_regressor` | Train GB regressor | `ModelTrainingResult` |
| `evaluate_classification_model` | Evaluate classifier | `ClassificationEvaluationResult` |
| `evaluate_regression_model` | Evaluate regressor | `RegressionEvaluationResult` |

### Plotting Tools (`stats_compass_core.plots`) *[requires plots extra]*

| Tool | Description | Returns |
|------|-------------|---------|
| `histogram` | Histogram of numeric column | `ChartResult` |
| `lineplot` | Line plot of time series | `ChartResult` |
| `bar_chart` | Bar chart of category counts | `ChartResult` |
| `scatter_plot` | Scatter plot of two columns | `ChartResult` |
| `feature_importance` | Feature importance from model | `ChartResult` |
| `roc_curve_plot` | ROC curve for classification model | `ChartResult` |
| `precision_recall_curve_plot` | Precision-recall curve | `ChartResult` |

### Time Series Tools (`stats_compass_core.ml`) *[requires timeseries extra]*

| Tool | Description | Returns |
|------|-------------|---------|
| `fit_arima` | Fit ARIMA(p,d,q) model | `ARIMAResult` |
| `forecast_arima` | Generate forecasts (supports natural language periods) | `ARIMAForecastResult` |
| `find_optimal_arima` | Grid search for best ARIMA parameters | `ARIMAParameterSearchResult` |
| `check_stationarity` | ADF/KPSS stationarity tests | `StationarityTestResult` |
| `infer_frequency` | Infer time series frequency | `InferFrequencyResult` |

## Usage Examples

### Complete Workflow Example

```python
import pandas as pd
from stats_compass_core import DataFrameState, registry

# Initialize state
state = DataFrameState()

# Load data
df = pd.DataFrame({
    "region": ["North", "South", "North", "South", "East"],
    "product": ["A", "A", "B", "B", "A"],
    "revenue": [100, 150, 200, None, 120],
    "quantity": [10, 15, 20, 12, 11]
})
state.set_dataframe(df, name="sales", operation="manual_load")

# Step 1: Check schema
result = registry.invoke("data", "get_schema", state, {})
print(f"Columns: {[c['name'] for c in result.columns]}")

# Step 2: Handle missing values
result = registry.invoke("cleaning", "apply_imputation", state, {
    "strategy": "mean",
    "columns": ["revenue"]
})
print(f"Filled {result.rows_affected} values")

# Step 3: Aggregate by region
result = registry.invoke("transforms", "groupby_aggregate", state, {
    "by": ["region"],
    "aggregations": [
        {"column": "revenue", "functions": ["sum"]},
        {"column": "quantity", "functions": ["mean"]}
    ],
    "save_as": "regional_summary"
})
print(f"Created: {result.dataframe_name}")

# Step 4: Describe the summary
result = registry.invoke("eda", "describe", state, {
    "dataframe_name": "regional_summary"
})
print(result.model_dump_json(indent=2))

# Step 5: Create visualization
result = registry.invoke("plots", "bar_chart", state, {
    "dataframe_name": "regional_summary",
    "column": "region"
})
# result.image_base64 contains PNG image
```

### Workflow Examples (Complete Pipelines)

#### Preprocessing + Classification Pipeline

```python
from stats_compass_core import DataFrameState, registry
import pandas as pd

state = DataFrameState()

# Load raw data
df = pd.read_csv("customer_churn.csv")
state.set_dataframe(df, name="raw_data", operation="load")

# Step 1: Clean the data
preprocessing_result = registry.invoke("workflows", "run_preprocessing", state, {
    "dataframe_name": "raw_data",
    "config": {
        "imputation": {"strategy": "mean"},
        "outliers": {"method": "iqr", "action": "cap"},
        "dedupe": True
    },
    "save_as": "cleaned_data"
})

print(f"Preprocessing: {preprocessing_result.status}")
print(f"Steps: {len(preprocessing_result.steps)}")
print(f"Cleaned DataFrame: {preprocessing_result.artifacts.final_dataframe}")

# Step 2: Train classification model
classification_result = registry.invoke("workflows", "run_classification", state, {
    "dataframe_name": "cleaned_data",
    "target_column": "churn",
    "feature_columns": ["age", "tenure", "balance", "num_products"],
    "config": {
        "model_type": "random_forest",
        "test_size": 0.2,
        "generate_plots": True,
        "plots": ["confusion_matrix", "roc", "feature_importance"]
    }
})

print(f"\nModel ID: {classification_result.artifacts.models_created[0]}")
print(f"Charts generated: {len(classification_result.artifacts.charts)}")
for step in classification_result.steps:
    if step.status == "success":
        print(f"  ✓ {step.step_name}")
```

#### Time Series Forecasting with Date Cleaning

```python
from stats_compass_core import DataFrameState, registry

state = DataFrameState()

# Load time series data with missing dates
df = pd.read_csv("stock_prices.csv")  # Has gaps in date sequence
state.set_dataframe(df, name="stock_data", operation="load")

# Clean dates first (optional but recommended)
preprocessing_result = registry.invoke("workflows", "run_preprocessing", state, {
    "dataframe_name": "stock_data",
    "config": {
        "date_cleaning": {
            "date_column": "Date",
            "fill_method": "ffill",
            "infer_frequency": True,
            "create_missing_dates": False
        }
    },
    "save_as": "stock_data_clean"
})

# Run time series forecast
forecast_result = registry.invoke("workflows", "run_timeseries_forecast", state, {
    "dataframe_name": "stock_data_clean",
    "date_column": "Date",
    "target_column": "Close",
    "config": {
        "forecast_periods": 30,
        "auto_find_params": True,
        "check_stationarity": True,
        "generate_forecast_plot": True
    }
})

print(f"ARIMA model: {forecast_result.artifacts.models_created[0]}")
print(f"Forecast status: {forecast_result.status}")
for step in forecast_result.steps:
    print(f"  {step.step_name}: {step.status}")
```

#### EDA Report Generation

```python
# Generate comprehensive EDA report
eda_result = registry.invoke("workflows", "run_eda_report", state, {
    "dataframe_name": "my_data",
    "config": {
        "include_describe": True,
        "include_correlations": True,
        "include_missing_analysis": True,
        "include_quality_report": True,
        "generate_histograms": True,
        "generate_bar_charts": True,
        "max_categorical_cardinality": 20
    }
})

# Access results
for step in eda_result.steps:
    if step.result:
        print(f"{step.step_name}: {step.summary}")

# Save charts
import base64
for i, chart in enumerate(eda_result.artifacts.charts):
    image_bytes = base64.b64decode(chart.image_base64)
    with open(f"chart_{i}_{chart.chart_type}.png", "wb") as f:
        f.write(image_bytes)
```

### Working with Charts

```python
import base64
from stats_compass_core import DataFrameState, registry

state = DataFrameState()
state.set_dataframe(my_df, name="data", operation="load")

# Create histogram
result = registry.invoke("plots", "histogram", state, {
    "column": "price",
    "bins": 20,
    "title": "Price Distribution"
})

# Decode and save the image
image_bytes = base64.b64decode(result.image_base64)
with open("histogram.png", "wb") as f:
    f.write(image_bytes)

# Or use in web response
# return Response(content=image_bytes, media_type="image/png")
```

### Training and Using Models

```python
from stats_compass_core import DataFrameState, registry

state = DataFrameState()
state.set_dataframe(training_df, name="training", operation="load")

# Train model
result = registry.invoke("ml", "train_random_forest_classifier", state, {
    "target_column": "churn",
    "feature_columns": ["age", "tenure", "balance", "num_products"],
    "test_size": 0.2
})

print(f"Model ID: {result.model_id}")
print(f"Accuracy: {result.metrics['accuracy']:.3f}")
print(f"Features: {result.feature_columns}")

# Model is stored in state for later use
model = state.get_model(result.model_id)

# Visualize feature importance
chart_result = registry.invoke("plots", "feature_importance", state, {
    "model_id": result.model_id,
    "top_n": 10
})
```

## Design Principles

### 1. Stateful, Not Pure

Unlike traditional pandas libraries, tools mutate shared state:

```python
# Tools operate on state, not raw DataFrames
result = drop_na(state, params)  # ✓ Correct
result = drop_na(df, params)     # ✗ Old pattern
```

### 2. JSON-Serializable Returns

All returns must be Pydantic models:

```python
# Returns JSON-serializable result
result = describe(state, params)
json_str = result.model_dump_json()  # Always works

# NOT raw DataFrames or matplotlib figures
```

### 3. Transform Tools Save to State

Transform operations create new named DataFrames:

```python
result = registry.invoke("transforms", "groupby_aggregate", state, {
    "by": ["region"],
    "aggregations": [{"column": "sales", "functions": ["sum"]}],
    "save_as": "regional_totals"  # Optional custom name
})
# New DataFrame now available as state.get_dataframe("regional_totals")
```

### 4. Models Stored by ID

Trained models aren't returned directly - they're stored:

```python
result = train_random_forest_classifier(state, params)
# result.model_id = "random_forest_classifier_churn_20241207_143022"
# Use state.get_model(result.model_id) to retrieve
```

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed contribution guidelines.

### Quick Start for Contributors

1. Fork and clone the repository
2. Install dependencies: `poetry install`
3. Create a new tool following the pattern in existing tools
4. Write tests in `tests/`
5. Submit a pull request

### Tool Signature Pattern

All tools must follow this signature:

```python
from stats_compass_core.state import DataFrameState
from stats_compass_core.results import SomeResult
from stats_compass_core.registry import registry

class MyToolInput(BaseModel):
    dataframe_name: str | None = Field(default=None)
    # ... other params

@registry.register(category="category", input_schema=MyToolInput, description="...")
def my_tool(state: DataFrameState, params: MyToolInput) -> SomeResult:
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()
    
    # ... do work ...
    
    return SomeResult(...)
```

## License

MIT License - see [LICENSE](LICENSE) for details.
