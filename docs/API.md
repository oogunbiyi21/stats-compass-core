# API Reference

## Registry

### `ToolRegistry`

The central registry for managing all tools in stats-compass-core.

#### Methods

##### `register(category, name=None, input_schema=None, description="")`

Decorator for registering a tool function.

**Parameters:**
- `category` (str): Tool category ('cleaning', 'transforms', 'eda', 'ml', 'plots')
- `name` (str, optional): Tool name (defaults to function name)
- `input_schema` (type[BaseModel], optional): Pydantic schema for input validation
- `description` (str, optional): Tool description

**Returns:** Decorated function

##### `get_tool(category, name)`

Get a specific tool function by category and name.

**Parameters:**
- `category` (str): Tool category
- `name` (str): Tool name

**Returns:** Callable or None

##### `list_tools(category=None)`

List all registered tools, optionally filtered by category.

**Parameters:**
- `category` (str, optional): Category to filter by

**Returns:** List of ToolMetadata objects

##### `get_categories()`

Get list of available tool categories.

**Returns:** List of category names

##### `auto_discover()`

Automatically discover and import all tool modules.

## Cleaning Tools

### `drop_na(df, params)`

Drop rows or columns with missing values from a DataFrame.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `params` (DropNAInput): Parameters for dropping NA values
  - `axis` (int): 0 for rows, 1 for columns (default: 0)
  - `how` (str): 'any' or 'all' (default: 'any')
  - `thresh` (int, optional): Minimum number of non-NA values
  - `subset` (list[str], optional): Column labels to consider

**Returns:** pd.DataFrame - New DataFrame with NA values removed

**Raises:**
- `ValueError`: If subset columns don't exist in DataFrame

### `dedupe(df, params)`

Remove duplicate rows from a DataFrame.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `params` (DedupeInput): Parameters for deduplication
  - `subset` (list[str], optional): Column labels for identifying duplicates
  - `keep` (str): 'first', 'last', or 'False' (default: 'first')
  - `ignore_index` (bool): Reset index in result (default: False)

**Returns:** pd.DataFrame - New DataFrame with duplicates removed

**Raises:**
- `ValueError`: If subset columns don't exist in DataFrame

## Transform Tools

### `groupby_aggregate(df, params)`

Group DataFrame by specified columns and apply aggregation functions.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `params` (GroupByAggregateInput): Parameters for groupby and aggregation
  - `by` (list[str]): Column labels to group by
  - `agg_func` (dict[str, str | list[str]]): Mapping of columns to aggregation functions
  - `as_index` (bool): Use group keys as index (default: True)

**Returns:** pd.DataFrame - New aggregated DataFrame

**Raises:**
- `ValueError`: If group-by or aggregation columns don't exist
- `TypeError`: If aggregation function is not supported

### `pivot(df, params)`

Pivot a DataFrame from long to wide format.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `params` (PivotInput): Parameters for pivoting
  - `index` (str | list[str]): Column(s) to use as row index
  - `columns` (str | list[str]): Column(s) to use as column headers
  - `values` (str | list[str], optional): Column(s) for values
  - `aggfunc` (str): Aggregation function (default: 'mean')
  - `fill_value` (float, optional): Value to replace missing values

**Returns:** pd.DataFrame - New pivoted DataFrame

**Raises:**
- `ValueError`: If specified columns don't exist
- `KeyError`: If pivot creates duplicate entries without aggregation

## EDA Tools

### `describe(df, params)`

Generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset's distribution.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `params` (DescribeInput): Parameters for describe operation
  - `percentiles` (list[float], optional): Percentiles to include (0-1)
  - `include` (str | list[str], optional): Data types to include
  - `exclude` (str | list[str], optional): Data types to exclude

**Returns:** pd.DataFrame - DataFrame containing descriptive statistics

**Raises:**
- `ValueError`: If percentiles are out of range or incompatible types specified

### `correlations(df, params)`

Compute pairwise correlation of columns, excluding NA/null values.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `params` (CorrelationsInput): Parameters for correlation computation
  - `method` (str): 'pearson', 'kendall', or 'spearman' (default: 'pearson')
  - `min_periods` (int, optional): Minimum observations per pair
  - `numeric_only` (bool): Include only numeric columns (default: True)

**Returns:** pd.DataFrame - DataFrame containing correlation matrix

**Raises:**
- `ValueError`: If no numeric columns available or computation fails

## ML Tools

*Requires installation with `[ml]` extra*

### `train_classifier(df, params)`

Train a classification model on DataFrame data.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame with features and target
- `params` (TrainClassifierInput): Parameters for model training
  - `target_column` (str): Name of the target column
  - `feature_columns` (list[str], optional): List of feature columns
  - `model_type` (str): 'logistic_regression', 'random_forest', or 'gradient_boosting' (default: 'random_forest')
  - `test_size` (float): Fraction for testing (default: 0.2)
  - `random_state` (int, optional): Random seed (default: 42)

**Returns:** TrainedClassifierResult
- `model` (object): Trained scikit-learn model
- `feature_columns` (list[str]): List of feature columns used
- `target_column` (str): Target column name
- `train_score` (float): Training accuracy score
- `n_samples` (int): Number of training samples

**Raises:**
- `ImportError`: If scikit-learn is not installed
- `ValueError`: If columns don't exist or data is insufficient

### `train_regressor(df, params)`

Train a regression model on DataFrame data.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame with features and target
- `params` (TrainRegressorInput): Parameters for model training
  - `target_column` (str): Name of the target column
  - `feature_columns` (list[str], optional): List of feature columns
  - `model_type` (str): 'linear_regression', 'random_forest', or 'gradient_boosting' (default: 'random_forest')
  - `test_size` (float): Fraction for testing (default: 0.2)
  - `random_state` (int, optional): Random seed (default: 42)

**Returns:** TrainedRegressorResult
- `model` (object): Trained scikit-learn model
- `feature_columns` (list[str]): List of feature columns used
- `target_column` (str): Target column name
- `train_score` (float): Training R² score
- `n_samples` (int): Number of training samples

**Raises:**
- `ImportError`: If scikit-learn is not installed
- `ValueError`: If columns don't exist or data is insufficient

## Plotting Tools

*Requires installation with `[plots]` extra*

### `histogram(df, params)`

Create a histogram plot from a DataFrame column.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `params` (HistogramInput): Parameters for histogram creation
  - `column` (str): Name of the column to plot
  - `bins` (int): Number of bins (default: 30)
  - `title` (str, optional): Plot title
  - `xlabel` (str, optional): X-axis label
  - `ylabel` (str): Y-axis label (default: 'Frequency')
  - `figsize` (tuple[float, float]): Figure size (default: (10, 6))

**Returns:** HistogramResult
- `figure` (object): Matplotlib figure object
- `column` (str): Column name plotted
- `bins` (int): Number of bins used

**Raises:**
- `ImportError`: If matplotlib is not installed
- `ValueError`: If column doesn't exist or is not numeric

### `lineplot(df, params)`

Create a line plot from DataFrame columns.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `params` (LineplotInput): Parameters for line plot creation
  - `x_column` (str, optional): Name of column for x-axis (uses index if None)
  - `y_column` (str): Name of column for y-axis
  - `title` (str, optional): Plot title
  - `xlabel` (str, optional): X-axis label
  - `ylabel` (str, optional): Y-axis label
  - `figsize` (tuple[float, float]): Figure size (default: (10, 6))
  - `marker` (str, optional): Marker style (e.g., 'o', 's', '^')

**Returns:** LineplotResult
- `figure` (object): Matplotlib figure object
- `x_column` (str | None): X-axis column name
- `y_column` (str): Y-axis column name

**Raises:**
- `ImportError`: If matplotlib is not installed
- `ValueError`: If specified columns don't exist
