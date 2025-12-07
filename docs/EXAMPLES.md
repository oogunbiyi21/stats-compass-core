# Usage Examples

This document provides detailed examples of how to use stats-compass-core tools with the stateful architecture.

## Table of Contents

- [Getting Started](#getting-started)
- [Data Loading](#data-loading)
- [Data Cleaning](#data-cleaning)
- [Data Transformation](#data-transformation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning](#machine-learning)
- [Data Visualization](#data-visualization)
- [Complete Workflows](#complete-workflows)

## Getting Started

### Initialize State

Every session starts with creating a `DataFrameState` instance:

```python
from stats_compass_core import DataFrameState, registry

# Create state manager (one per session/user)
state = DataFrameState(memory_limit_mb=500)  # 500MB limit
```

### Two Ways to Call Tools

**Method 1: Via Registry (Recommended for MCP)**

```python
result = registry.invoke("eda", "describe", state, {
    "percentiles": [0.25, 0.5, 0.75]
})
```

**Method 2: Direct Import**

```python
from stats_compass_core.eda.describe import describe, DescribeInput

params = DescribeInput(percentiles=[0.25, 0.5, 0.75])
result = describe(state, params)
```

Both return the same JSON-serializable result.

---

## Data Loading

### Loading CSV Files

```python
from stats_compass_core import DataFrameState, registry

state = DataFrameState()

# Load from file
result = registry.invoke("data", "load_csv", state, {
    "file_path": "/path/to/sales_data.csv",
    "name": "sales"  # Optional custom name
})

print(f"Loaded: {result.dataframe_name}")
print(f"Shape: {result.shape}")
print(f"Columns: {result.columns}")
```

### Loading from pandas DataFrame

```python
import pandas as pd

# Create or load DataFrame externally
df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=100),
    "region": ["North", "South", "East", "West"] * 25,
    "revenue": [100 + i * 2 for i in range(100)],
    "quantity": [10 + i % 5 for i in range(100)]
})

# Add to state
state.set_dataframe(df, name="sales", operation="pandas_import")

# Verify
print(f"Active DataFrame: {state._active_dataframe}")
print(f"DataFrames: {list(state._dataframes.keys())}")
```

### Inspecting Data

```python
# Get schema (column types, nulls, etc.)
result = registry.invoke("data", "get_schema", state, {})
for col in result.columns:
    print(f"{col['name']}: {col['dtype']} ({col['missing']} missing)")

# Get sample rows
result = registry.invoke("data", "get_sample", state, {"n": 5})
print(result.data)  # List of record dicts

# List all DataFrames in state
result = registry.invoke("data", "list_dataframes", state, {})
for df_info in result.dataframes:
    print(f"{df_info['name']}: {df_info['shape']}")
```

---

## Data Cleaning

### Removing Missing Values

```python
import pandas as pd
from stats_compass_core import DataFrameState, registry

# Setup with messy data
state = DataFrameState()
df = pd.DataFrame({
    "name": ["Alice", "Bob", None, "David"],
    "age": [25, None, 35, 40],
    "city": ["NYC", "LA", "Chicago", None]
})
state.set_dataframe(df, name="users", operation="load")

# Drop rows with ANY missing values
result = registry.invoke("cleaning", "drop_na", state, {
    "axis": 0,
    "how": "any"
})
print(f"Rows: {result.rows_before} -> {result.rows_after}")
# Output: Rows: 4 -> 1

# Alternative: Drop only if ALL values missing
state.set_dataframe(df, name="users", operation="reset")  # Reset
result = registry.invoke("cleaning", "drop_na", state, {
    "how": "all"
})
# Keeps all rows (none are completely empty)

# Drop based on specific columns only
state.set_dataframe(df, name="users", operation="reset")
result = registry.invoke("cleaning", "drop_na", state, {
    "subset": ["name", "age"]
})
print(f"Rows: {result.rows_before} -> {result.rows_after}")
# Output: Rows: 4 -> 2 (David kept, city=None is OK)
```

### Filling Missing Values (Imputation)

```python
# Setup
state = DataFrameState()
df = pd.DataFrame({
    "product": ["A", "B", "C", "D", "E"],
    "price": [100, None, 150, None, 200],
    "category": ["X", "X", None, "Y", "Y"]
})
state.set_dataframe(df, name="products", operation="load")

# Fill numeric columns with mean
result = registry.invoke("cleaning", "apply_imputation", state, {
    "strategy": "mean",
    "columns": ["price"]
})
print(f"Filled {result.rows_affected} values")
print(f"Details: {result.details}")
# Details show which columns were filled and how many values

# Fill with median
state.set_dataframe(df, name="products", operation="reset")
result = registry.invoke("cleaning", "apply_imputation", state, {
    "strategy": "median",
    "columns": ["price"]
})

# Fill with most frequent value (mode)
state.set_dataframe(df, name="products", operation="reset")
result = registry.invoke("cleaning", "apply_imputation", state, {
    "strategy": "most_frequent",
    "columns": ["category"]
})

# Fill with constant
state.set_dataframe(df, name="products", operation="reset")
result = registry.invoke("cleaning", "apply_imputation", state, {
    "strategy": "constant",
    "columns": ["price"],
    "fill_value": 0
})
```

### Removing Duplicates

```python
state = DataFrameState()
df = pd.DataFrame({
    "id": [1, 2, 2, 3, 3],
    "name": ["Alice", "Bob", "Bob", "Charlie", "Charlie"],
    "score": [90, 85, 85, 92, 95]
})
state.set_dataframe(df, name="scores", operation="load")

# Remove exact duplicates, keep first
result = registry.invoke("cleaning", "dedupe", state, {
    "keep": "first"
})
print(f"Removed {result.rows_before - result.rows_after} duplicates")

# Dedupe based on specific columns
state.set_dataframe(df, name="scores", operation="reset")
result = registry.invoke("cleaning", "dedupe", state, {
    "subset": ["id", "name"],
    "keep": "last"  # Keep last occurrence
})
```

---

## Data Transformation

### Group By and Aggregate

```python
state = DataFrameState()
df = pd.DataFrame({
    "region": ["North", "South", "North", "South", "North"],
    "product": ["A", "A", "B", "B", "A"],
    "sales": [100, 150, 200, 120, 110],
    "units": [10, 15, 20, 12, 11]
})
state.set_dataframe(df, name="sales", operation="load")

# Simple aggregation by region
result = registry.invoke("transforms", "groupby_aggregate", state, {
    "by": ["region"],
    "agg_func": {"sales": "sum", "units": "mean"}
})
print(f"Created: {result.dataframe_name}")  # Auto-generated name
print(f"Shape: {result.shape}")
print(f"Preview: {result.data}")

# Multiple aggregations per column
result = registry.invoke("transforms", "groupby_aggregate", state, {
    "by": ["region", "product"],
    "agg_func": {
        "sales": ["sum", "mean", "count"],
        "units": "sum"
    },
    "save_as": "region_product_summary"
})

# Now both DataFrames are in state
print(list(state._dataframes.keys()))
# ['sales', 'sales_grouped_by_region', 'region_product_summary']
```

### Pivot Tables

```python
state = DataFrameState()
df = pd.DataFrame({
    "date": ["2024-01", "2024-01", "2024-02", "2024-02"],
    "product": ["A", "B", "A", "B"],
    "sales": [100, 150, 120, 180]
})
state.set_dataframe(df, name="monthly", operation="load")

# Pivot: dates as rows, products as columns
result = registry.invoke("transforms", "pivot", state, {
    "index": "date",
    "columns": "product",
    "values": "sales",
    "aggfunc": "sum",
    "fill_value": 0,
    "save_as": "sales_pivot"
})
print(f"Pivoted shape: {result.shape}")
```

### Filtering Data

```python
state = DataFrameState()
df = pd.DataFrame({
    "customer": ["A", "B", "C", "D", "E"],
    "revenue": [1000, 500, 2000, 300, 1500],
    "region": ["North", "South", "North", "South", "East"]
})
state.set_dataframe(df, name="customers", operation="load")

# Filter high-value customers
result = registry.invoke("transforms", "filter_dataframe", state, {
    "query": "revenue > 800",
    "save_as": "high_value"
})
print(f"Found {result.shape[0]} high-value customers")

# Complex filter
result = registry.invoke("transforms", "filter_dataframe", state, {
    "query": "revenue > 500 and region == 'North'",
    "save_as": "north_premium"
})

# Filter with limit
result = registry.invoke("transforms", "filter_dataframe", state, {
    "query": "revenue > 0",
    "limit": 3,  # Top 3 only
    "save_as": "top3"
})
```

---

## Exploratory Data Analysis

### Descriptive Statistics

```python
state = DataFrameState()
df = pd.DataFrame({
    "age": [25, 30, 35, 40, 45, 50, 55, 60],
    "salary": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
    "department": ["IT", "HR", "IT", "Finance", "IT", "HR", "Finance", "IT"]
})
state.set_dataframe(df, name="employees", operation="load")

# Basic statistics
result = registry.invoke("eda", "describe", state, {})
print(f"Columns analyzed: {result.columns_analyzed}")
for col, stats in result.statistics.items():
    print(f"\n{col}:")
    for stat, value in stats.items():
        print(f"  {stat}: {value}")

# Custom percentiles
result = registry.invoke("eda", "describe", state, {
    "percentiles": [0.1, 0.25, 0.5, 0.75, 0.9]
})

# Include all columns (including non-numeric)
result = registry.invoke("eda", "describe", state, {
    "include": "all"
})
```

### Correlation Analysis

```python
state = DataFrameState()
df = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5],
    "feature2": [2, 4, 6, 8, 10],  # Perfectly correlated
    "feature3": [5, 4, 3, 2, 1],   # Negatively correlated
    "noise": [0.5, 0.8, 0.3, 0.9, 0.2]  # Random
})
state.set_dataframe(df, name="features", operation="load")

# Pearson correlation
result = registry.invoke("eda", "correlations", state, {
    "method": "pearson"
})
print("Correlation matrix:")
for col, corrs in result.correlations.items():
    print(f"{col}: {corrs}")

# Spearman (rank-based)
result = registry.invoke("eda", "correlations", state, {
    "method": "spearman"
})
```

### Hypothesis Testing

```python
import numpy as np

state = DataFrameState()
np.random.seed(42)
df = pd.DataFrame({
    "control": np.random.normal(100, 15, 50),
    "treatment": np.random.normal(110, 15, 50)  # Slightly higher mean
})
state.set_dataframe(df, name="experiment", operation="load")

# T-test
result = registry.invoke("eda", "t_test", state, {
    "column_a": "control",
    "column_b": "treatment",
    "alternative": "two-sided"
})
print(f"Test type: {result.test_type}")
print(f"T-statistic: {result.statistic:.3f}")
print(f"P-value: {result.p_value:.4f}")
print(f"Significant at α=0.05: {result.significant_at_05}")
print(f"Details: {result.details}")

# Welch's t-test (unequal variances)
result = registry.invoke("eda", "t_test", state, {
    "column_a": "control",
    "column_b": "treatment",
    "equal_var": False
})

# Z-test (when population std is known)
result = registry.invoke("eda", "z_test", state, {
    "column_a": "control",
    "column_b": "treatment",
    "population_std_a": 15,
    "population_std_b": 15
})
```

---

## Machine Learning

### Training Classification Models

```python
import numpy as np

state = DataFrameState()
np.random.seed(42)

# Create classification dataset
n_samples = 200
df = pd.DataFrame({
    "age": np.random.randint(20, 60, n_samples),
    "income": np.random.randint(30000, 150000, n_samples),
    "tenure": np.random.randint(1, 20, n_samples),
    "churn": np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
})
state.set_dataframe(df, name="customers", operation="load")

# Train Random Forest Classifier
result = registry.invoke("ml", "train_random_forest_classifier", state, {
    "target_column": "churn",
    "feature_columns": ["age", "income", "tenure"],
    "test_size": 0.2,
    "n_estimators": 100,
    "max_depth": 5
})

print(f"Model ID: {result.model_id}")
print(f"Accuracy: {result.metrics['accuracy']:.3f}")
print(f"Features: {result.feature_columns}")
print(f"Train/Test split: {result.n_train_samples}/{result.n_test_samples}")

# Model is stored in state - retrieve it
model = state.get_model(result.model_id)
```

### Training Regression Models

```python
state = DataFrameState()
np.random.seed(42)

# Create regression dataset
df = pd.DataFrame({
    "sqft": np.random.randint(500, 3000, 100),
    "bedrooms": np.random.randint(1, 6, 100),
    "age": np.random.randint(0, 50, 100),
    "price": None  # Will compute
})
# Simulate price
df["price"] = df["sqft"] * 200 + df["bedrooms"] * 50000 - df["age"] * 2000 + np.random.normal(0, 10000, 100)
state.set_dataframe(df, name="housing", operation="load")

# Train Linear Regression
result = registry.invoke("ml", "train_linear_regression", state, {
    "target_column": "price",
    "feature_columns": ["sqft", "bedrooms", "age"],
    "test_size": 0.2
})
print(f"R² Score: {result.metrics['r2_score']:.3f}")

# Train Gradient Boosting Regressor
result = registry.invoke("ml", "train_gradient_boosting_regressor", state, {
    "target_column": "price",
    "n_estimators": 100,
    "learning_rate": 0.1
})
print(f"Model ID: {result.model_id}")
```

### Evaluating Models

```python
# After training, add predictions to a test DataFrame
model = state.get_model(result.model_id)
test_df = state.get_dataframe("housing").copy()
test_df["predicted_price"] = model.predict(test_df[["sqft", "bedrooms", "age"]])
state.set_dataframe(test_df, name="housing_with_preds", operation="add_predictions")

# Evaluate regression model
result = registry.invoke("ml", "evaluate_regression_model", state, {
    "dataframe_name": "housing_with_preds",
    "target_column": "price",
    "prediction_column": "predicted_price"
})
print(f"RMSE: {result.rmse:.2f}")
print(f"MAE: {result.mae:.2f}")
print(f"R²: {result.r2:.3f}")
```

---

## Data Visualization

All plot tools return `ChartResult` with base64-encoded PNG images.

### Histograms

```python
import base64

state = DataFrameState()
df = pd.DataFrame({"prices": np.random.lognormal(4, 0.5, 1000)})
state.set_dataframe(df, name="prices", operation="load")

result = registry.invoke("plots", "histogram", state, {
    "column": "prices",
    "bins": 30,
    "title": "Price Distribution"
})

# Save the image
image_bytes = base64.b64decode(result.image_base64)
with open("histogram.png", "wb") as f:
    f.write(image_bytes)

print(f"Chart type: {result.chart_type}")
print(f"Title: {result.title}")
```

### Line Plots

```python
state = DataFrameState()
df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=30),
    "revenue": np.cumsum(np.random.normal(1000, 200, 30))
})
state.set_dataframe(df, name="daily", operation="load")

result = registry.invoke("plots", "lineplot", state, {
    "x_column": "date",
    "y_column": "revenue",
    "title": "Cumulative Revenue",
    "marker": "o"
})
```

### Bar Charts

```python
state = DataFrameState()
df = pd.DataFrame({
    "category": ["A", "B", "A", "C", "B", "A", "D", "B", "C", "A"]
})
state.set_dataframe(df, name="categories", operation="load")

result = registry.invoke("plots", "bar_chart", state, {
    "column": "category",
    "top_n": 5,
    "orientation": "horizontal",
    "title": "Category Distribution"
})
```

### Scatter Plots

```python
state = DataFrameState()
df = pd.DataFrame({
    "x": np.random.randn(100),
    "y": np.random.randn(100),
    "group": np.random.choice(["A", "B", "C"], 100)
})
state.set_dataframe(df, name="points", operation="load")

result = registry.invoke("plots", "scatter_plot", state, {
    "x": "x",
    "y": "y",
    "hue": "group",  # Color by group
    "alpha": 0.7,
    "title": "Grouped Scatter Plot"
})
```

### Feature Importance

```python
# After training a model
result = registry.invoke("plots", "feature_importance", state, {
    "model_id": model_result.model_id,  # From previous training
    "top_n": 10,
    "orientation": "horizontal",
    "title": "Top 10 Features"
})
```

---

## Complete Workflows

### Sales Analysis Pipeline

```python
import pandas as pd
import numpy as np
from stats_compass_core import DataFrameState, registry

# Initialize
state = DataFrameState()

# 1. Load data
df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=365),
    "region": np.random.choice(["North", "South", "East", "West"], 365),
    "product": np.random.choice(["A", "B", "C"], 365),
    "revenue": np.random.lognormal(7, 0.5, 365),
    "quantity": np.random.randint(1, 100, 365)
})
# Add some nulls
df.loc[np.random.choice(365, 20), "revenue"] = None
state.set_dataframe(df, name="sales", operation="load")

# 2. Check for issues
schema_result = registry.invoke("data", "get_schema", state, {})
print("Schema:")
for col in schema_result.columns:
    print(f"  {col['name']}: {col['dtype']}, {col['missing']} missing")

# 3. Clean data
clean_result = registry.invoke("cleaning", "apply_imputation", state, {
    "strategy": "median",
    "columns": ["revenue"]
})
print(f"\nCleaned: filled {clean_result.rows_affected} missing values")

# 4. Aggregate by region
agg_result = registry.invoke("transforms", "groupby_aggregate", state, {
    "by": ["region"],
    "agg_func": {"revenue": ["sum", "mean"], "quantity": "sum"},
    "save_as": "regional_summary"
})
print(f"\nCreated regional summary: {agg_result.dataframe_name}")

# 5. Describe the summary
desc_result = registry.invoke("eda", "describe", state, {
    "dataframe_name": "regional_summary"
})
print("\nRegional statistics:")
print(desc_result.model_dump_json(indent=2))

# 6. Create visualizations
bar_result = registry.invoke("plots", "bar_chart", state, {
    "dataframe_name": "regional_summary",
    "column": "region",
    "title": "Transactions by Region"
})

# 7. Save chart
import base64
with open("regional_bar.png", "wb") as f:
    f.write(base64.b64decode(bar_result.image_base64))
print("\nSaved chart to regional_bar.png")

# Summary of state
print("\n" + "="*50)
print("DataFrames in state:", list(state._dataframes.keys()))
```

### Churn Prediction Pipeline

```python
import pandas as pd
import numpy as np
from stats_compass_core import DataFrameState, registry

state = DataFrameState()

# 1. Load customer data
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "customer_id": range(n),
    "age": np.random.randint(18, 70, n),
    "tenure_months": np.random.randint(1, 72, n),
    "monthly_charges": np.random.uniform(20, 100, n),
    "total_charges": None,  # Will compute
    "num_support_calls": np.random.poisson(2, n),
    "churned": np.random.choice([0, 1], n, p=[0.73, 0.27])
})
df["total_charges"] = df["monthly_charges"] * df["tenure_months"] * np.random.uniform(0.9, 1.1, n)
state.set_dataframe(df, name="customers", operation="load")

# 2. EDA
print("=== Exploratory Data Analysis ===")
desc = registry.invoke("eda", "describe", state, {})
print(f"Analyzed columns: {desc.columns_analyzed}")

corr = registry.invoke("eda", "correlations", state, {})
print(f"\nCorrelations with churn:")
churn_corrs = corr.correlations.get("churned", {})
for col, val in sorted(churn_corrs.items(), key=lambda x: abs(x[1]), reverse=True):
    if col != "churned":
        print(f"  {col}: {val:.3f}")

# 3. Train model
print("\n=== Model Training ===")
model_result = registry.invoke("ml", "train_random_forest_classifier", state, {
    "target_column": "churned",
    "feature_columns": ["age", "tenure_months", "monthly_charges", "total_charges", "num_support_calls"],
    "test_size": 0.2,
    "n_estimators": 100
})
print(f"Model: {model_result.model_id}")
print(f"Accuracy: {model_result.metrics['accuracy']:.3f}")

# 4. Feature importance visualization
print("\n=== Feature Importance ===")
fi_result = registry.invoke("plots", "feature_importance", state, {
    "model_id": model_result.model_id,
    "top_n": 5,
    "orientation": "horizontal"
})

import base64
with open("churn_feature_importance.png", "wb") as f:
    f.write(base64.b64decode(fi_result.image_base64))
print("Saved feature importance chart")

# 5. Show all artifacts
print("\n=== Session Summary ===")
print(f"DataFrames: {list(state._dataframes.keys())}")
print(f"Models: {list(state._models.keys())}")
```

---

## Best Practices

### 1. Always Check State

```python
# Before operations, verify state
print(f"Active: {state._active_dataframe}")
print(f"Available: {list(state._dataframes.keys())}")
```

### 2. Use Descriptive Names

```python
# Good: Descriptive names
registry.invoke("transforms", "groupby_aggregate", state, {
    ...,
    "save_as": "monthly_sales_by_region"
})

# Avoid: Auto-generated names (harder to track)
```

### 3. Handle JSON Serialization

```python
# All results serialize to JSON
result = registry.invoke("eda", "describe", state, {})

# As JSON string
json_str = result.model_dump_json()

# As Python dict
data = result.model_dump()
```

### 4. Save Charts Properly

```python
import base64

result = registry.invoke("plots", "histogram", state, {...})

# Decode base64 to bytes
image_bytes = base64.b64decode(result.image_base64)

# Save to file
with open("chart.png", "wb") as f:
    f.write(image_bytes)

# Or send in HTTP response
# return Response(content=image_bytes, media_type="image/png")
```

### 5. Chain Operations Cleanly

```python
# Each operation builds on the last
state = DataFrameState()
state.set_dataframe(raw_df, name="raw", operation="load")

registry.invoke("cleaning", "drop_na", state, {"dataframe_name": "raw"})
registry.invoke("cleaning", "dedupe", state, {})  # Uses active (raw)
registry.invoke("transforms", "groupby_aggregate", state, {
    "by": ["category"],
    "agg_func": {"value": "sum"},
    "save_as": "summary"
})
registry.invoke("eda", "describe", state, {"dataframe_name": "summary"})
```

---

For detailed API documentation, see [API.md](API.md).
