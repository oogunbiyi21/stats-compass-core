# API Reference

## Core Classes

### DataFrameState

The central state manager for MCP sessions. Stores DataFrames, trained models, and operation history.

```python
from stats_compass_core import DataFrameState

state = DataFrameState(memory_limit_mb=500)
```

#### Constructor

```python
DataFrameState(memory_limit_mb: float = 500.0)
```

**Parameters:**
- `memory_limit_mb`: Maximum total memory for all DataFrames (default 500MB)

#### DataFrame Methods

##### `set_dataframe(df, name, operation, set_active=True)`

Store a DataFrame in state.

```python
stored_name = state.set_dataframe(df, name="sales", operation="load_csv")
```

**Parameters:**
- `df` (pd.DataFrame): DataFrame to store
- `name` (str): Unique name for the DataFrame
- `operation` (str): Description of operation that created it (for lineage)
- `set_active` (bool): Whether to make this the active DataFrame (default: True)

**Returns:** `str` - The name of the stored DataFrame

##### `get_dataframe(name=None)`

Retrieve a DataFrame from state.

```python
df = state.get_dataframe("sales")  # By name
df = state.get_dataframe()          # Get active DataFrame
```

**Parameters:**
- `name` (str, optional): Name of DataFrame. If None, returns active DataFrame.

**Returns:** `pd.DataFrame`

**Raises:** `KeyError` if DataFrame not found

##### `list_dataframes()`

List all stored DataFrames with metadata.

```python
info_list = state.list_dataframes()
for info in info_list:
    print(f"{info.name}: {info.shape}, {info.memory_mb:.1f}MB")
```

**Returns:** `list[DataFrameInfo]`

##### `delete_dataframe(name)`

Remove a DataFrame from state.

```python
state.delete_dataframe("temp_data")
```

**Parameters:**
- `name` (str): Name of DataFrame to delete

**Raises:** `KeyError` if DataFrame not found

#### Model Methods

##### `store_model(model, model_type, target_column, feature_columns, source_dataframe)`

Store a trained model in state.

```python
model_id = state.store_model(
    model=trained_rf,
    model_type="random_forest_classifier",
    target_column="churn",
    feature_columns=["age", "balance"],
    source_dataframe="training_data"
)
```

**Parameters:**
- `model` (object): Trained sklearn model
- `model_type` (str): Type identifier (e.g., "random_forest_classifier")
- `target_column` (str): Name of target column used for training
- `feature_columns` (list[str]): Feature column names
- `source_dataframe` (str): Name of DataFrame used for training

**Returns:** `str` - Unique model ID (format: `{model_type}_{target}_{timestamp}`)

##### `get_model(model_id)`

Retrieve a trained model.

```python
model = state.get_model("random_forest_classifier_churn_20241207_143022")
predictions = model.predict(X_new)
```

**Parameters:**
- `model_id` (str): Model identifier

**Returns:** Trained model object

**Raises:** `KeyError` if model not found

##### `get_model_info(model_id)`

Get metadata about a stored model.

```python
info = state.get_model_info(model_id)
print(f"Type: {info.model_type}")
print(f"Features: {info.feature_columns}")
```

**Returns:** `ModelInfo`

#### Properties

- `_active_dataframe` (str | None): Name of currently active DataFrame
- `_dataframes` (dict[str, pd.DataFrame]): All stored DataFrames
- `_models` (dict[str, object]): All stored models
- `_history` (list[HistoryEntry]): Operation history

---

## Result Models

All tools return Pydantic models. Import from `stats_compass_core.results`.

### DataFrameLoadResult

Returned by data loading tools.

```python
class DataFrameLoadResult(BaseModel):
    dataframe_name: str      # Name assigned to loaded DataFrame
    shape: tuple[int, int]   # (rows, columns)
    columns: list[str]       # Column names
    dtypes: dict[str, str]   # Column data types
    message: str             # Human-readable summary
```

### DataFrameMutationResult

Returned by cleaning tools that modify DataFrames in-place.

```python
class DataFrameMutationResult(BaseModel):
    rows_before: int         # Row count before operation
    rows_after: int          # Row count after operation
    rows_affected: int       # Rows changed/removed
    dataframe_name: str      # Name of mutated DataFrame
    operation: str           # Description of operation
    details: dict[str, Any]  # Operation-specific details
```

### DataFrameQueryResult

Returned by transform tools that create new DataFrames.

```python
class DataFrameQueryResult(BaseModel):
    data: dict[str, Any]      # {"records": [...], "truncated": bool}
    shape: tuple[int, int]    # Shape of result
    columns: list[str]        # Column names
    dataframe_name: str       # Name of NEW DataFrame in state
    source_dataframe: str     # Name of source DataFrame
```

### DescribeResult

Returned by `describe` tool.

```python
class DescribeResult(BaseModel):
    statistics: dict[str, dict[str, Any]]  # Stats per column
    dataframe_name: str                     # Source DataFrame
    columns_analyzed: list[str]             # Columns included
    include_types: list[str] | None         # Data types included
```

### CorrelationsResult

Returned by `correlations` tool.

```python
class CorrelationsResult(BaseModel):
    correlations: dict[str, dict[str, float]]  # Correlation matrix
    method: str                                 # pearson/spearman/kendall
    dataframe_name: str                         # Source DataFrame
    columns: list[str]                          # Columns included
    high_correlations: list[dict] | None        # Pairs above threshold
```

### ChartResult

Returned by all plotting tools.

```python
class ChartResult(BaseModel):
    image_base64: str         # Base64-encoded PNG image
    image_format: str         # Always "png"
    title: str                # Chart title
    chart_type: str           # histogram/lineplot/bar_chart/etc.
    dataframe_name: str       # Source DataFrame
    parameters: dict[str, Any]  # Parameters used
```

### ModelTrainingResult

Returned by ML training tools.

```python
class ModelTrainingResult(BaseModel):
    model_id: str                  # ID for retrieving model from state
    model_type: str                # Type of model trained
    target_column: str             # Target variable
    feature_columns: list[str]     # Features used
    metrics: dict[str, float]      # Training metrics (accuracy, r2, etc.)
    n_train_samples: int           # Training set size
    n_test_samples: int            # Test set size
    source_dataframe: str          # Training data source
```

### HypothesisTestResult

Returned by statistical test tools.

```python
class HypothesisTestResult(BaseModel):
    test_type: str           # "t-test (Student)", "z-test", etc.
    statistic: float         # Test statistic
    p_value: float           # P-value
    alternative: str         # two-sided/less/greater
    n_a: int                 # Sample size group A
    n_b: int                 # Sample size group B
    significant_at_05: bool  # p < 0.05
    significant_at_01: bool  # p < 0.01
    dataframe_name: str      # Source DataFrame
    details: dict[str, Any]  # Additional details
```

### ClassificationEvaluationResult

Returned by `evaluate_classification_model`.

```python
class ClassificationEvaluationResult(BaseModel):
    accuracy: float                    # Accuracy score
    precision: float                   # Precision
    recall: float                      # Recall
    f1: float                          # F1 score
    confusion_matrix: list[list[int]]  # Confusion matrix
    labels: list[Any]                  # Class labels
    n_samples: int                     # Samples evaluated
    average: str                       # Averaging method
    dataframe_name: str                # Source DataFrame
    target_column: str                 # True labels column
    prediction_column: str             # Predictions column
```

### RegressionEvaluationResult

Returned by `evaluate_regression_model`.

```python
class RegressionEvaluationResult(BaseModel):
    rmse: float              # Root Mean Squared Error
    mae: float               # Mean Absolute Error
    r2: float                # R-squared
    n_samples: int           # Samples evaluated
    dataframe_name: str      # Source DataFrame
    target_column: str       # True values column
    prediction_column: str   # Predictions column
```

---

## Registry

The tool registry provides discovery and invocation.

```python
from stats_compass_core import registry
```

### Methods

#### `invoke(category, tool_name, state, params)`

Invoke a tool with automatic parameter validation.

```python
result = registry.invoke(
    category="eda",
    tool_name="describe",
    state=state,
    params={"percentiles": [0.25, 0.5, 0.75]}
)
```

**Parameters:**
- `category` (str): Tool category
- `tool_name` (str): Tool name
- `state` (DataFrameState): State instance
- `params` (dict): Tool parameters (will be validated)

**Returns:** Tool result (Pydantic model)

**Raises:** `KeyError` if tool not found, `ValidationError` if params invalid

#### `get_tool(category, name)`

Get a tool function directly.

```python
describe_func = registry.get_tool("eda", "describe")
```

**Returns:** Callable or None

#### `list_tools(category=None)`

List registered tools.

```python
# All tools
all_tools = registry.list_tools()

# By category
eda_tools = registry.list_tools(category="eda")
```

**Returns:** List of `ToolMetadata`

#### `get_categories()`

Get available categories.

```python
categories = registry.get_categories()
# ['data', 'cleaning', 'transforms', 'eda', 'ml', 'plots']
```

---

## Tools by Category

### Data Tools

#### `load_csv`

Load a CSV file into state.

```python
result = registry.invoke("data", "load_csv", state, {
    "file_path": "/path/to/data.csv",
    "name": "my_data"  # Optional, defaults to filename
})
```

**Parameters:**
- `file_path` (str): Path to CSV file
- `name` (str, optional): Name for DataFrame in state

**Returns:** `DataFrameLoadResult`

#### `get_schema`

Get schema information for a DataFrame.

```python
result = registry.invoke("data", "get_schema", state, {
    "dataframe_name": "sales"  # Optional, uses active
})
```

**Returns:** `SchemaResult` with column types, null counts, unique counts

#### `get_sample`

Get sample rows from a DataFrame.

```python
result = registry.invoke("data", "get_sample", state, {
    "n": 10,
    "dataframe_name": "sales"
})
```

**Returns:** `SampleResult` with sample data as records

#### `list_dataframes`

List all DataFrames in state.

```python
result = registry.invoke("data", "list_dataframes", state, {})
```

**Returns:** `DataFrameListResult`

---

### Cleaning Tools

#### `drop_na`

Remove rows or columns with missing values.

```python
result = registry.invoke("cleaning", "drop_na", state, {
    "axis": 0,           # 0=rows, 1=columns
    "how": "any",        # "any" or "all"
    "subset": ["col1"],  # Optional: only consider these columns
    "thresh": 2          # Optional: keep rows with >= 2 non-NA
})
```

**Returns:** `DataFrameMutationResult`

#### `dedupe`

Remove duplicate rows.

```python
result = registry.invoke("cleaning", "dedupe", state, {
    "subset": ["id", "date"],  # Optional: columns to consider
    "keep": "first"            # "first", "last", or "False"
})
```

**Returns:** `DataFrameMutationResult`

#### `apply_imputation`

Fill missing values with imputation strategy.

```python
result = registry.invoke("cleaning", "apply_imputation", state, {
    "strategy": "mean",       # mean/median/most_frequent/constant
    "columns": ["revenue"],   # Optional: specific columns
    "fill_value": 0           # Required if strategy="constant"
})
```

**Returns:** `DataFrameMutationResult`

---

### Transform Tools

#### `groupby_aggregate`

Group by columns and aggregate.

```python
result = registry.invoke("transforms", "groupby_aggregate", state, {
    "by": ["region", "product"],
    "agg_func": {
        "revenue": "sum",
        "quantity": ["mean", "count"]
    },
    "as_index": False,
    "save_as": "aggregated"  # Optional: custom name for result
})
```

**Returns:** `DataFrameQueryResult` (new DataFrame saved to state)

#### `pivot`

Reshape from long to wide format.

```python
result = registry.invoke("transforms", "pivot", state, {
    "index": "date",
    "columns": "product",
    "values": "sales",
    "aggfunc": "sum",
    "fill_value": 0,
    "save_as": "pivoted"
})
```

**Returns:** `DataFrameQueryResult` (new DataFrame saved to state)

#### `filter_dataframe`

Filter rows using pandas query syntax.

```python
result = registry.invoke("transforms", "filter_dataframe", state, {
    "query": "revenue > 1000 and region == 'North'",
    "limit": 100,      # Optional: max rows
    "save_as": "filtered"
})
```

**Returns:** `DataFrameQueryResult` (new DataFrame saved to state)

---

### EDA Tools

#### `describe`

Generate descriptive statistics.

```python
result = registry.invoke("eda", "describe", state, {
    "percentiles": [0.1, 0.25, 0.5, 0.75, 0.9],
    "include": "all"  # Optional: include non-numeric
})
```

**Returns:** `DescribeResult`

#### `correlations`

Compute correlation matrix.

```python
result = registry.invoke("eda", "correlations", state, {
    "method": "pearson",    # pearson/spearman/kendall
    "min_periods": 10,      # Optional: min observations
    "numeric_only": True
})
```

**Returns:** `CorrelationsResult`

#### `t_test`

Two-sample t-test.

```python
result = registry.invoke("eda", "t_test", state, {
    "column_a": "group1_scores",
    "column_b": "group2_scores",
    "alternative": "two-sided",  # two-sided/less/greater
    "equal_var": True            # True=Student, False=Welch
})
```

**Returns:** `HypothesisTestResult`

#### `z_test`

Two-sample z-test.

```python
result = registry.invoke("eda", "z_test", state, {
    "column_a": "sample1",
    "column_b": "sample2",
    "population_std_a": 10.0,  # Optional: known pop std
    "population_std_b": 12.0,
    "alternative": "two-sided"
})
```

**Returns:** `HypothesisTestResult`

---

### ML Tools

*Requires `[ml]` extra*

#### `train_linear_regression`

```python
result = registry.invoke("ml", "train_linear_regression", state, {
    "target_column": "price",
    "feature_columns": ["sqft", "bedrooms", "bathrooms"],
    "test_size": 0.2
})
```

**Returns:** `ModelTrainingResult`

#### `train_logistic_regression`

```python
result = registry.invoke("ml", "train_logistic_regression", state, {
    "target_column": "churn",
    "feature_columns": ["age", "tenure", "balance"],
    "test_size": 0.2
})
```

**Returns:** `ModelTrainingResult`

#### `train_random_forest_classifier`

```python
result = registry.invoke("ml", "train_random_forest_classifier", state, {
    "target_column": "category",
    "feature_columns": ["f1", "f2", "f3"],
    "n_estimators": 100,
    "max_depth": 10,
    "test_size": 0.2
})
```

**Returns:** `ModelTrainingResult`

#### `train_random_forest_regressor`

```python
result = registry.invoke("ml", "train_random_forest_regressor", state, {
    "target_column": "price",
    "feature_columns": ["f1", "f2"],
    "n_estimators": 100,
    "test_size": 0.2
})
```

**Returns:** `ModelTrainingResult`

#### `train_gradient_boosting_classifier`

```python
result = registry.invoke("ml", "train_gradient_boosting_classifier", state, {
    "target_column": "label",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3
})
```

**Returns:** `ModelTrainingResult`

#### `train_gradient_boosting_regressor`

```python
result = registry.invoke("ml", "train_gradient_boosting_regressor", state, {
    "target_column": "value",
    "n_estimators": 100,
    "learning_rate": 0.1
})
```

**Returns:** `ModelTrainingResult`

#### `evaluate_classification_model`

Evaluate classifier predictions.

```python
result = registry.invoke("ml", "evaluate_classification_model", state, {
    "target_column": "actual",
    "prediction_column": "predicted",
    "average": "weighted"  # micro/macro/weighted/binary
})
```

**Returns:** `ClassificationEvaluationResult`

#### `evaluate_regression_model`

Evaluate regressor predictions.

```python
result = registry.invoke("ml", "evaluate_regression_model", state, {
    "target_column": "actual",
    "prediction_column": "predicted"
})
```

**Returns:** `RegressionEvaluationResult`

---

### Plotting Tools

*Requires `[plots]` extra*

All plotting tools return `ChartResult` with base64-encoded PNG.

#### `histogram`

```python
result = registry.invoke("plots", "histogram", state, {
    "column": "price",
    "bins": 30,
    "title": "Price Distribution",
    "figsize": [10, 6]
})
# result.image_base64 contains PNG
```

#### `lineplot`

```python
result = registry.invoke("plots", "lineplot", state, {
    "x_column": "date",
    "y_column": "revenue",
    "title": "Revenue Over Time",
    "marker": "o"
})
```

#### `bar_chart`

```python
result = registry.invoke("plots", "bar_chart", state, {
    "column": "category",
    "top_n": 10,
    "orientation": "horizontal",  # vertical/horizontal
    "title": "Top Categories"
})
```

#### `scatter_plot`

```python
result = registry.invoke("plots", "scatter_plot", state, {
    "x": "age",
    "y": "income",
    "hue": "segment",  # Optional: color by category
    "alpha": 0.7
})
```

#### `feature_importance`

```python
result = registry.invoke("plots", "feature_importance", state, {
    "model_id": "random_forest_classifier_churn_20241207",
    "top_n": 15,
    "orientation": "horizontal"
})
```

---

## Error Handling

All tools raise typed exceptions:

```python
from stats_compass_core import DataFrameState, registry

state = DataFrameState()

# KeyError: DataFrame not found
try:
    df = state.get_dataframe("nonexistent")
except KeyError as e:
    print(f"DataFrame not found: {e}")

# ValueError: Invalid parameters
try:
    result = registry.invoke("cleaning", "drop_na", state, {
        "axis": 5  # Invalid
    })
except ValueError as e:
    print(f"Invalid params: {e}")

# ValidationError: Pydantic validation failed
from pydantic import ValidationError
try:
    result = registry.invoke("eda", "describe", state, {
        "percentiles": "invalid"  # Should be list
    })
except ValidationError as e:
    print(f"Validation failed: {e}")
```
