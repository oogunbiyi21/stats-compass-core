# Available Tools

stats-compass-core provides 50+ tools organized into categories.

## Workflow Tools (`stats_compass_core.workflows`)

High-level orchestration tools that execute complete multi-step pipelines in a single call.

| Workflow | Description | Use Case |
|----------|-------------|----------|
| `run_preprocessing` | Complete data cleaning pipeline | Clean messy data for analysis/ML |
| `run_classification` | Train + evaluate classification model | Predict categories (churn, sentiment, etc.) |
| `run_regression` | Train + evaluate regression model | Predict continuous values (price, sales, etc.) |
| `run_eda_report` | Comprehensive exploratory analysis | Understand dataset characteristics |
| `run_timeseries_forecast` | ARIMA forecasting with validation | Predict future values from time series |

## Data Tools (`stats_compass_core.data`)

| Tool | Description | Returns |
|------|-------------|---------|
| `load_csv` | Load CSV file into state | `DataFrameLoadResult` |
| `load_excel` | Load Excel file into state | `DataFrameLoadResult` |
| `get_schema` | Get DataFrame column types and stats | `SchemaResult` |
| `get_sample` | Get sample rows from DataFrame | `SampleResult` |
| `list_dataframes` | List all DataFrames in state | `DataFrameListResult` |
| `save_csv` | Save DataFrame to CSV file | `SaveResult` |
| `save_model` | Save trained model to disk | `SaveResult` |

## Cleaning Tools (`stats_compass_core.cleaning`)

| Tool | Description | Returns |
|------|-------------|---------|
| `drop_na` | Remove rows/columns with missing values | `DataFrameMutationResult` |
| `dedupe` | Remove duplicate rows | `DataFrameMutationResult` |
| `apply_imputation` | Fill missing values (mean/median/mode/constant) | `DataFrameMutationResult` |
| `handle_outliers` | Handle outliers (cap/remove/winsorize/log/IQR) | `OutlierHandlingResult` |

## Transform Tools (`stats_compass_core.transforms`)

| Tool | Description | Returns |
|------|-------------|---------|
| `groupby_aggregate` | Group and aggregate data | `DataFrameQueryResult` |
| `pivot` | Reshape long to wide format | `DataFrameQueryResult` |
| `filter_dataframe` | Filter with pandas query syntax | `DataFrameQueryResult` |
| `add_column` | Add calculated column | `DataFrameMutationResult` |
| `drop_columns` | Drop columns | `DataFrameMutationResult` |
| `rename_columns` | Rename columns | `DataFrameMutationResult` |
| `merge` | Merge two DataFrames | `DataFrameQueryResult` |
| `concat` | Concatenate DataFrames | `DataFrameQueryResult` |
| `bin_rare_categories` | Bin rare categories into 'Other' | `BinRareCategoriesResult` |
| `mean_target_encoding` | Target encoding for categoricals | `MeanTargetEncodingResult` |
| `encode` | One-hot or label encoding | `DataFrameMutationResult` |
| `scale` | Standard or MinMax scaling | `DataFrameMutationResult` |

## EDA Tools (`stats_compass_core.eda`)

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

## ML Tools (`stats_compass_core.ml`)

*Requires `pip install stats-compass-core[ml]`*

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
| `predict` | Make predictions with stored model | `PredictionResult` |
| `cross_validate` | Cross-validation scoring | `CrossValidationResult` |

## Plotting Tools (`stats_compass_core.plots`)

*Requires `pip install stats-compass-core[plots]`*

| Tool | Description | Returns |
|------|-------------|---------|
| `histogram` | Histogram of numeric column | `ChartResult` |
| `lineplot` | Line plot of time series | `ChartResult` |
| `bar_chart` | Bar chart of category counts | `ChartResult` |
| `scatter_plot` | Scatter plot of two columns | `ChartResult` |
| `box_plot` | Box plot for distributions | `ChartResult` |
| `heatmap` | Correlation heatmap | `ChartResult` |
| `feature_importance` | Feature importance from model | `ChartResult` |
| `roc_curve_plot` | ROC curve for classification model | `ChartResult` |
| `precision_recall_curve_plot` | Precision-recall curve | `ChartResult` |
| `confusion_matrix_plot` | Confusion matrix visualization | `ChartResult` |

## Time Series Tools (`stats_compass_core.ml`)

*Requires `pip install stats-compass-core[timeseries]`*

| Tool | Description | Returns |
|------|-------------|---------|
| `fit_arima` | Fit ARIMA(p,d,q) model | `ARIMAResult` |
| `forecast_arima` | Generate forecasts | `ARIMAForecastResult` |
| `find_optimal_arima` | Grid search for best ARIMA parameters | `ARIMAParameterSearchResult` |
| `check_stationarity` | ADF/KPSS stationarity tests | `StationarityTestResult` |
| `infer_frequency` | Infer time series frequency | `InferFrequencyResult` |

## Result Types

All tools return Pydantic models that serialize to JSON:

| Result Type | Used By | Key Fields |
|-------------|---------|------------|
| `DataFrameLoadResult` | data loading tools | `dataframe_name`, `shape`, `columns` |
| `DataFrameMutationResult` | cleaning/transform tools | `rows_before`, `rows_after`, `rows_affected` |
| `DataFrameQueryResult` | transform tools | `data`, `shape`, `dataframe_name` |
| `DescribeResult` | describe | `statistics`, `columns_analyzed` |
| `CorrelationsResult` | correlations | `correlations`, `method` |
| `ChartResult` | all plot tools | `image_base64`, `chart_type` |
| `ModelTrainingResult` | ML training | `model_id`, `metrics`, `feature_columns` |
| `HypothesisTestResult` | statistical tests | `statistic`, `p_value`, `significant_at_05` |
| `WorkflowResult` | workflows | `steps`, `artifacts`, `status`, `suggestion` |
