# Usage Examples

This document provides detailed examples of how to use stats-compass-core tools.

## Table of Contents

- [Data Cleaning](#data-cleaning)
- [Data Transformation](#data-transformation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning](#machine-learning)
- [Data Visualization](#data-visualization)
- [Using the Registry](#using-the-registry)

## Data Cleaning

### Removing Missing Values

```python
import pandas as pd
from stats_compass_core.cleaning.dropna import drop_na, DropNAInput

# Create sample data with missing values
df = pd.DataFrame({
    'name': ['Alice', 'Bob', None, 'David'],
    'age': [25, None, 35, 40],
    'city': ['NYC', 'LA', 'Chicago', 'Boston']
})

# Drop rows with any missing values
params = DropNAInput(axis=0, how='any')
clean_df = drop_na(df, params)
print(clean_df)
# Output: Only rows with complete data

# Drop rows only if ALL values are missing
params = DropNAInput(axis=0, how='all')
mostly_clean = drop_na(df, params)

# Keep rows with at least 2 non-NA values
params = DropNAInput(axis=0, thresh=2)
thresh_df = drop_na(df, params)

# Consider only specific columns when dropping
params = DropNAInput(axis=0, how='any', subset=['name', 'age'])
subset_clean = drop_na(df, params)
```

### Removing Duplicates

```python
from stats_compass_core.cleaning.dedupe import dedupe, DedupeInput

# Create data with duplicates
df = pd.DataFrame({
    'id': [1, 2, 2, 3, 3],
    'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'Charlie'],
    'score': [90, 85, 85, 92, 95]
})

# Remove exact duplicates, keeping first occurrence
params = DedupeInput(keep='first')
unique_df = dedupe(df, params)

# Remove duplicates based on specific columns only
params = DedupeInput(subset=['id', 'name'], keep='last')
dedupe_df = dedupe(df, params)

# Drop all duplicates (keep none)
params = DedupeInput(keep='False')
no_dupes = dedupe(df, params)
```

## Data Transformation

### Group By and Aggregate

```python
from stats_compass_core.transforms.groupby_aggregate import (
    groupby_aggregate,
    GroupByAggregateInput
)

# Sales data
df = pd.DataFrame({
    'region': ['North', 'South', 'North', 'South', 'North'],
    'product': ['A', 'A', 'B', 'B', 'A'],
    'sales': [100, 150, 200, 120, 110],
    'units': [10, 15, 20, 12, 11]
})

# Simple aggregation
params = GroupByAggregateInput(
    by=['region'],
    agg_func={'sales': 'sum', 'units': 'sum'},
    as_index=False
)
regional_totals = groupby_aggregate(df, params)
print(regional_totals)

# Multiple aggregation functions
params = GroupByAggregateInput(
    by=['region', 'product'],
    agg_func={
        'sales': ['sum', 'mean', 'count'],
        'units': 'sum'
    }
)
detailed_stats = groupby_aggregate(df, params)
print(detailed_stats)
```

### Pivot Tables

```python
from stats_compass_core.transforms.pivot import pivot, PivotInput

# Time series data
df = pd.DataFrame({
    'date': ['2024-01', '2024-01', '2024-02', '2024-02'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 150, 120, 180],
    'quantity': [10, 15, 12, 18]
})

# Pivot sales by date and product
params = PivotInput(
    index='date',
    columns='product',
    values='sales',
    aggfunc='sum'
)
pivot_df = pivot(df, params)
print(pivot_df)

# Pivot with multiple value columns
params = PivotInput(
    index='date',
    columns='product',
    values=['sales', 'quantity'],
    aggfunc='mean',
    fill_value=0
)
multi_pivot = pivot(df, params)
```

## Exploratory Data Analysis

### Descriptive Statistics

```python
from stats_compass_core.eda.describe import describe, DescribeInput

# Numerical data
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50],
    'salary': [50000, 60000, 70000, 80000, 90000, 100000],
    'experience': [2, 5, 8, 12, 15, 20]
})

# Basic statistics
params = DescribeInput()
stats = describe(df, params)
print(stats)

# Custom percentiles
params = DescribeInput(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
detailed_stats = describe(df, params)

# Include all column types
mixed_df = pd.DataFrame({
    'numeric': [1, 2, 3, 4, 5],
    'category': ['A', 'B', 'A', 'B', 'C'],
    'text': ['foo', 'bar', 'baz', 'qux', 'quux']
})

params = DescribeInput(include='all')
all_stats = describe(mixed_df, params)
```

### Correlation Analysis

```python
from stats_compass_core.eda.correlations import correlations, CorrelationsInput

# Financial data
df = pd.DataFrame({
    'stock_a': [100, 102, 101, 105, 107],
    'stock_b': [50, 52, 51, 54, 56],
    'stock_c': [200, 198, 195, 190, 185],
    'volume': [1000, 1100, 1050, 1200, 1150]
})

# Pearson correlation
params = CorrelationsInput(method='pearson')
corr_matrix = correlations(df, params)
print(corr_matrix)

# Spearman correlation (rank-based)
params = CorrelationsInput(method='spearman', min_periods=3)
rank_corr = correlations(df, params)

# Find highly correlated pairs
high_corr = corr_matrix[abs(corr_matrix) > 0.8]
```

## Machine Learning

### Classification

```python
from stats_compass_core.ml.train_classifier import (
    train_classifier,
    TrainClassifierInput
)
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=200, n_features=4, random_state=42)
df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
df['target'] = y

# Train a random forest classifier
params = TrainClassifierInput(
    target_column='target',
    model_type='random_forest',
    test_size=0.2,
    random_state=42
)
result = train_classifier(df, params)

print(f"Model: {result.model}")
print(f"Features used: {result.feature_columns}")
print(f"Training score: {result.train_score:.3f}")
print(f"Training samples: {result.n_samples}")

# Use the trained model for predictions
new_data = pd.DataFrame({
    'feature1': [0.5],
    'feature2': [1.2],
    'feature3': [-0.3],
    'feature4': [0.8]
})
prediction = result.model.predict(new_data)
print(f"Prediction: {prediction}")
```

### Regression

```python
from stats_compass_core.ml.train_regressor import (
    train_regressor,
    TrainRegressorInput
)
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=200, n_features=3, noise=0.1, random_state=42)
df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
df['target'] = y

# Train a gradient boosting regressor
params = TrainRegressorInput(
    target_column='target',
    model_type='gradient_boosting',
    test_size=0.2,
    random_state=42
)
result = train_regressor(df, params)

print(f"Training R² score: {result.train_score:.3f}")

# Make predictions
new_data = pd.DataFrame({
    'feature1': [0.5],
    'feature2': [1.2],
    'feature3': [-0.3]
})
prediction = result.model.predict(new_data)
print(f"Predicted value: {prediction[0]:.2f}")
```

## Data Visualization

### Histograms

```python
from stats_compass_core.plots.histogram import histogram, HistogramInput
import matplotlib.pyplot as plt

# Age distribution data
df = pd.DataFrame({
    'age': [25, 27, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60]
})

# Create histogram
params = HistogramInput(
    column='age',
    bins=5,
    title='Age Distribution',
    xlabel='Age',
    ylabel='Frequency',
    figsize=(10, 6)
)
result = histogram(df, params)
result.figure.savefig('age_histogram.png')
plt.close()

print(f"Plotted column: {result.column}")
print(f"Number of bins: {result.bins}")
```

### Line Plots

```python
from stats_compass_core.plots.lineplot import lineplot, LineplotInput

# Time series data
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10),
    'temperature': [32, 35, 33, 36, 38, 40, 39, 37, 35, 33],
    'humidity': [60, 58, 62, 59, 55, 53, 56, 61, 63, 65]
})

# Plot temperature over time
params = LineplotInput(
    x_column='date',
    y_column='temperature',
    title='Temperature Trend',
    xlabel='Date',
    ylabel='Temperature (°F)',
    marker='o',
    figsize=(12, 6)
)
result = lineplot(df, params)
result.figure.savefig('temperature_trend.png')
plt.close()

# Plot with index as x-axis
params = LineplotInput(
    y_column='humidity',
    title='Humidity Over Time'
)
result = lineplot(df, params)
```

## Using the Registry

### Discovering Tools

```python
from stats_compass_core import registry

# List all available categories
categories = registry.get_categories()
print("Categories:", categories)

# List all tools
all_tools = registry.list_tools()
for tool in all_tools:
    print(f"{tool.category}.{tool.name}: {tool.description}")

# List tools in a specific category
cleaning_tools = registry.list_tools(category='cleaning')
print("\nCleaning tools:")
for tool in cleaning_tools:
    print(f"  - {tool.name}: {tool.description}")
```

### Getting and Using Tools

```python
from stats_compass_core import registry
import pandas as pd

# Get a tool by category and name
drop_na_func = registry.get_tool('cleaning', 'drop_na')

# Use the tool (you'll need to construct params manually)
from stats_compass_core.cleaning.dropna import DropNAInput

df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [5, None, 7, 8]
})

params = DropNAInput(axis=0, how='any')
result = drop_na_func(df, params)
print(result)
```

## Chaining Operations

```python
import pandas as pd
from stats_compass_core.cleaning.dropna import drop_na, DropNAInput
from stats_compass_core.cleaning.dedupe import dedupe, DedupeInput
from stats_compass_core.transforms.groupby_aggregate import (
    groupby_aggregate,
    GroupByAggregateInput
)

# Start with messy data
df = pd.DataFrame({
    'category': ['A', 'A', 'B', None, 'B', 'A'],
    'value': [10, 10, 20, 30, None, 15]
})

# Clean: remove rows with missing values
clean_params = DropNAInput(axis=0, how='any')
df_clean = drop_na(df, clean_params)

# Clean: remove duplicates
dedupe_params = DedupeInput(keep='first')
df_unique = dedupe(df_clean, dedupe_params)

# Transform: aggregate by category
agg_params = GroupByAggregateInput(
    by=['category'],
    agg_func={'value': ['sum', 'mean', 'count']},
    as_index=False
)
df_aggregated = groupby_aggregate(df_unique, agg_params)

print("Final result:")
print(df_aggregated)
```

## Best Practices

### 1. Always Validate Inputs

```python
from pydantic import ValidationError
from stats_compass_core.cleaning.dropna import DropNAInput

try:
    # This will fail validation
    params = DropNAInput(axis=3)  # axis must be 0 or 1
except ValidationError as e:
    print(f"Validation error: {e}")
```

### 2. Handle Exceptions

```python
from stats_compass_core.eda.correlations import correlations, CorrelationsInput

df = pd.DataFrame({'text_col': ['a', 'b', 'c']})

try:
    params = CorrelationsInput(method='pearson', numeric_only=True)
    result = correlations(df, params)
except ValueError as e:
    print(f"Error: {e}")
    # Handle the case where no numeric columns exist
```

### 3. Use Type Hints

```python
import pandas as pd
from stats_compass_core.cleaning.dropna import drop_na, DropNAInput

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values."""
    params = DropNAInput(axis=0, how='any')
    return drop_na(df, params)
```

### 4. Test Your Pipelines

```python
import pandas as pd
from stats_compass_core.cleaning.dropna import drop_na, DropNAInput

# Original data
df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, 6]})

# Apply transformation
params = DropNAInput(axis=0, how='any')
result = drop_na(df, params)

# Verify expectations
assert len(result) == 2, "Should have 2 rows after dropping NAs"
assert result['A'].isna().sum() == 0, "No NAs should remain"
```

---

For more examples and detailed API documentation, see [API.md](API.md).
