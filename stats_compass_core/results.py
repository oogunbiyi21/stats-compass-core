"""
Base result models for MCP-compatible tool returns.

All tool results must be JSON-serializable. These Pydantic models
provide the common return types for different categories of tools.
"""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


def dataframe_to_json_safe_records(df: pd.DataFrame, max_rows: int | None = None) -> list[dict[str, Any]]:
    """
    Convert a DataFrame to a list of JSON-safe dicts.
    
    Handles NaN, NaT, Inf values by converting them to None.
    Converts numpy types to Python native types.
    Converts Timestamps to ISO format strings.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        List of dicts suitable for JSON serialization
    """
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    records = df.to_dict(orient="records")

    # Normalize each record to be JSON-safe
    safe_records = []
    for record in records:
        safe_record = {}
        for key, value in record.items():
            safe_record[key] = _normalize_value(value)
        safe_records.append(safe_record)

    return safe_records


def _normalize_value(value: Any) -> Any:
    """Convert a single value to JSON-safe format."""
    # Handle None/NaN/NaT
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if pd.isna(value):
        return None

    # Handle numpy scalar types
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_normalize_value(v) for v in value.tolist()]

    # Handle pandas Timestamp/Timedelta
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)

    # Handle datetime types
    if hasattr(value, 'isoformat'):
        return value.isoformat()

    return value


class DataFrameMutationResult(BaseModel):
    """Result for tools that modify a DataFrame (drop_na, dedupe, etc.)."""

    success: bool = Field(description="Whether the operation succeeded")
    operation: str = Field(description="Name of the operation performed")
    rows_before: int = Field(description="Number of rows before the operation")
    rows_after: int = Field(description="Number of rows after the operation")
    rows_affected: int = Field(description="Number of rows changed/removed")
    message: str = Field(description="Human-readable summary of the operation")
    dataframe_name: str = Field(description="Name of the DataFrame in state")
    columns_affected: list[str] | None = Field(
        default=None,
        description="Columns that were affected by the operation"
    )


class DataFrameQueryResult(BaseModel):
    """Result for tools that query/aggregate data without modifying the source."""

    data: dict[str, Any] = Field(description="Query result data as dict")
    shape: tuple[int, int] = Field(description="Shape of the result (rows, cols)")
    columns: list[str] = Field(description="Column names in the result")
    dataframe_name: str | None = Field(
        default=None,
        description="Name of saved DataFrame if result was stored"
    )
    source_dataframe: str = Field(
        default="active",
        description="Name of the source DataFrame"
    )


class DataFrameSchemaResult(BaseModel):
    """Result for tools that return DataFrame schema/metadata."""

    dataframe_name: str = Field(description="Name of the DataFrame")
    shape: tuple[int, int] = Field(description="Shape (rows, cols)")
    columns: list[dict[str, Any]] = Field(
        description="Column info: name, dtype, null_count, sample_values"
    )
    memory_usage_bytes: int = Field(description="Memory usage in bytes")
    index_info: dict[str, Any] | None = Field(
        default=None,
        description="Information about the DataFrame index"
    )


class DataFrameSampleResult(BaseModel):
    """Result for tools that return sample rows."""

    dataframe_name: str = Field(description="Name of the DataFrame")
    data: list[dict[str, Any]] = Field(description="Sample rows as list of dicts")
    total_rows: int = Field(description="Total rows in the DataFrame")
    sample_size: int = Field(description="Number of rows in this sample")
    columns: list[str] = Field(description="Column names")


class DataFrameLoadResult(BaseModel):
    """Result for data loading tools."""

    success: bool = Field(description="Whether the load succeeded")
    dataframe_name: str = Field(description="Name assigned to the DataFrame")
    source: str = Field(description="Source path or identifier")
    shape: tuple[int, int] = Field(description="Shape of loaded data (rows, cols)")
    columns: list[str] = Field(description="Column names")
    dtypes: dict[str, str] = Field(description="Column data types")
    message: str = Field(description="Human-readable summary")


class FileListResult(BaseModel):
    """Result for file listing tools."""

    directory: str = Field(description="Directory that was scanned")
    files: list[str] = Field(description="List of file paths found")
    count: int = Field(description="Number of files found")
    message: str = Field(description="Human-readable summary")


class DescribeResult(BaseModel):
    """Result for descriptive statistics tools."""

    statistics: dict[str, dict[str, Any]] = Field(
        description="Statistics per column: count, mean, std, min, max, etc."
    )
    dataframe_name: str = Field(description="Name of the analyzed DataFrame")
    columns_analyzed: list[str] = Field(description="Columns included in analysis")
    include_types: list[str] | None = Field(
        default=None,
        description="Data types included in analysis"
    )


class CorrelationsResult(BaseModel):
    """Result for correlation analysis tools."""

    correlations: dict[str, dict[str, float]] = Field(
        description="Correlation matrix as nested dict"
    )
    method: str = Field(description="Correlation method used (pearson, spearman, etc.)")
    dataframe_name: str = Field(description="Name of the analyzed DataFrame")
    columns: list[str] = Field(description="Columns included in correlation")
    high_correlations: list[dict[str, Any]] | None = Field(
        default=None,
        description="Pairs with correlation above threshold"
    )


class ChartResult(BaseModel):
    """Result for visualization tools."""

    image_base64: str | None = Field(default=None, description="Base64-encoded PNG image")
    image_format: str = Field(default="png", description="Image format")
    title: str = Field(description="Chart title")
    chart_type: str = Field(description="Type of chart (histogram, line, scatter, etc.)")
    dataframe_name: str = Field(description="Name of the source DataFrame")
    data: dict[str, Any] | None = Field(default=None, description="Raw data for interactive charts")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional chart metadata (axes, columns used, etc.)"
    )


class ModelTrainingResult(BaseModel):
    """Result for ML training tools."""

    model_id: str = Field(description="Unique identifier for the trained model")
    model_type: str = Field(
        description="Type of model (linear_regression, random_forest, etc.)"
    )
    target_column: str = Field(description="Target/label column name")
    feature_columns: list[str] = Field(description="Feature column names")

    # Training metrics
    metrics: dict[str, float] = Field(
        description="Training metrics (r2, mse, accuracy, f1, etc.)"
    )

    # Optional detailed info
    feature_importances: dict[str, float] | None = Field(
        default=None,
        description="Feature importance scores (for tree-based models)"
    )
    coefficients: dict[str, float] | None = Field(
        default=None,
        description="Model coefficients (for linear models)"
    )
    intercept: float | None = Field(
        default=None,
        description="Model intercept (for linear models)"
    )

    # Data info
    train_size: int = Field(description="Number of training samples")
    test_size: int | None = Field(
        default=None,
        description="Number of test samples if split was used"
    )
    dataframe_name: str = Field(description="Name of the source DataFrame")

    # Predictions DataFrame info (for regression and classification)
    predictions_dataframe: str | None = Field(
        default=None,
        description="Name of DataFrame containing predictions (original columns + predictions)"
    )
    prediction_column: str | None = Field(
        default=None,
        description="Name of the prediction column in predictions_dataframe"
    )
    probability_columns: list[str] | None = Field(
        default=None,
        description="Names of probability columns for classification (one per class)"
    )
    class_labels: list[Any] | None = Field(
        default=None,
        description="Class labels for classification models"
    )

    # Hyperparameters
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Model hyperparameters used"
    )


class ModelPredictionResult(BaseModel):
    """Result for model prediction tools."""

    model_id: str = Field(description="ID of the model used for prediction")
    predictions: list[Any] = Field(description="Predicted values")
    prediction_count: int = Field(description="Number of predictions made")
    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame if predictions were added as column"
    )
    prediction_column: str | None = Field(
        default=None,
        description="Name of the prediction column if added"
    )


class HypothesisTestResult(BaseModel):
    """Result for hypothesis tests (t-test, z-test, etc.)."""

    test_type: str = Field(description="Type of test performed (t-test, z-test, etc.)")
    statistic: float = Field(description="Test statistic value")
    p_value: float = Field(description="P-value for the test")
    alternative: str = Field(description="Alternative hypothesis (two-sided, less, greater)")
    n_a: int = Field(description="Sample size for group A")
    n_b: int = Field(description="Sample size for group B")
    significant_at_05: bool = Field(description="Whether result is significant at alpha=0.05")
    significant_at_01: bool = Field(description="Whether result is significant at alpha=0.01")
    dataframe_name: str = Field(description="Name of the analyzed DataFrame")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional test-specific details"
    )


class ClassificationEvaluationResult(BaseModel):
    """Result for classification model evaluation."""

    accuracy: float = Field(description="Accuracy score")
    precision: float = Field(description="Precision score")
    recall: float = Field(description="Recall score")
    f1: float = Field(description="F1 score")
    confusion_matrix: list[list[int]] = Field(description="Confusion matrix as nested list")
    labels: list[Any] = Field(description="Class labels")
    n_samples: int = Field(description="Number of samples evaluated")
    average: str = Field(description="Averaging method used")
    dataframe_name: str = Field(description="Name of the DataFrame evaluated")
    target_column: str = Field(description="Name of the target column")
    prediction_column: str = Field(description="Name of the prediction column")


class RegressionEvaluationResult(BaseModel):
    """Result for regression model evaluation."""

    rmse: float = Field(description="Root Mean Squared Error")
    mae: float = Field(description="Mean Absolute Error")
    r2: float = Field(description="R-squared (coefficient of determination)")
    n_samples: int = Field(description="Number of samples evaluated")
    dataframe_name: str = Field(description="Name of the DataFrame evaluated")
    target_column: str = Field(description="Name of the target column")
    prediction_column: str = Field(description="Name of the prediction column")


class ModelListResult(BaseModel):
    """Result for listing available models."""

    models: list[dict[str, Any]] = Field(
        description="List of available models with metadata"
    )
    total_count: int = Field(description="Total number of models")


class DataFrameListResult(BaseModel):
    """Result for listing available DataFrames."""

    dataframes: list[dict[str, Any]] = Field(
        description="List of DataFrames with metadata (name, shape, memory)"
    )
    active_dataframe: str | None = Field(
        default=None,
        description="Name of the currently active DataFrame"
    )
    total_count: int = Field(description="Total number of DataFrames")
    total_memory_bytes: int = Field(description="Total memory usage")


class ChiSquareResult(BaseModel):
    """Result for chi-square tests (independence and goodness-of-fit)."""

    test_type: str = Field(description="Type of chi-square test (independence or goodness_of_fit)")
    chi2_statistic: float = Field(description="Chi-square statistic")
    p_value: float = Field(description="P-value for the test")
    degrees_of_freedom: int = Field(description="Degrees of freedom")
    n_samples: int = Field(description="Total sample size")

    # Effect size (Cramér's V for independence test)
    effect_size: float | None = Field(
        default=None,
        description="Effect size (Cramér's V for independence test)"
    )
    effect_interpretation: str | None = Field(
        default=None,
        description="Interpretation of effect size (negligible, small, medium, large)"
    )

    # Significance
    significant_at_05: bool = Field(description="Whether result is significant at alpha=0.05")
    significant_at_01: bool = Field(description="Whether result is significant at alpha=0.01")

    # Tables
    observed_frequencies: dict[str, Any] = Field(
        description="Observed frequency table as nested dict"
    )
    expected_frequencies: dict[str, Any] = Field(
        description="Expected frequency table as nested dict"
    )

    # Metadata
    dataframe_name: str = Field(description="Name of the analyzed DataFrame")
    column1: str = Field(description="First column (row variable for independence)")
    column2: str | None = Field(
        default=None,
        description="Second column (column variable for independence test)"
    )

    # Warnings
    low_expected_count: int = Field(
        default=0,
        description="Number of cells with expected frequency < 5"
    )
    low_expected_warning: str | None = Field(
        default=None,
        description="Warning about cells with low expected frequency"
    )


class OutlierHandlingResult(BaseModel):
    """Result for outlier handling operations."""

    success: bool = Field(description="Whether the operation succeeded")
    method: str = Field(description="Outlier handling method used")
    column: str = Field(description="Column that was processed")

    # Statistics
    rows_before: int = Field(description="Number of rows before operation")
    rows_after: int = Field(description="Number of rows after operation")
    values_affected: int = Field(description="Number of values affected/capped/removed")
    percentage_affected: float = Field(description="Percentage of values affected")

    # Thresholds used
    lower_threshold: float | None = Field(default=None, description="Lower threshold used")
    upper_threshold: float | None = Field(default=None, description="Upper threshold used")

    # Result column
    result_column: str = Field(description="Column containing the result (same or new column)")
    dataframe_name: str = Field(description="Name of the DataFrame in state")

    # Statistics before/after
    stats_before: dict[str, float] = Field(
        default_factory=dict,
        description="Statistics before operation (min, max, mean, std)"
    )
    stats_after: dict[str, float] = Field(
        default_factory=dict,
        description="Statistics after operation (min, max, mean, std)"
    )

    message: str = Field(description="Human-readable summary of the operation")


class DataQualityResult(BaseModel):
    """Result for data quality analysis tools."""

    dataframe_name: str = Field(description="Name of the analyzed DataFrame")
    total_rows: int = Field(description="Total number of rows")
    total_columns: int = Field(description="Total number of columns")

    # Missing data summary
    missing_summary: dict[str, Any] = Field(
        description="Missing data analysis: columns_with_missing, percentages, patterns"
    )

    # Outlier summary (if analyzed)
    outlier_summary: dict[str, Any] | None = Field(
        default=None,
        description="Outlier analysis by column: counts, percentages, bounds"
    )

    # Recommendations
    recommendations: list[str] = Field(
        default_factory=list,
        description="List of data quality improvement recommendations"
    )

    # Quality score
    quality_score: float | None = Field(
        default=None,
        description="Overall data quality score (0-100)"
    )


class ClassificationCurveResult(BaseModel):
    """Result for ROC and PR curve tools."""

    curve_type: str = Field(description="Type of curve (roc or precision_recall)")

    # Curve data
    x_values: list[float] = Field(description="X-axis values (FPR for ROC, Recall for PR)")
    y_values: list[float] = Field(description="Y-axis values (TPR for ROC, Precision for PR)")
    thresholds: list[float] | None = Field(
        default=None,
        description="Classification thresholds at each point"
    )

    # Summary metrics
    auc_score: float = Field(description="Area under the curve")

    # Chart
    image_base64: str | None = Field(default=None, description="Base64-encoded PNG image of the curve")

    # Metadata
    model_id: str = Field(description="ID of the model being evaluated")
    dataframe_name: str = Field(description="Name of the source DataFrame")
    target_column: str = Field(description="Name of the target column")

    # Interpretation
    interpretation: str = Field(description="Human-readable interpretation of the curve")


class ARIMAResult(BaseModel):
    """Result for ARIMA model fitting."""

    success: bool = Field(description="Whether model fitting succeeded")
    operation: str = Field(default="arima_fit", description="Operation performed")

    # Model parameters
    order: tuple[int, int, int] = Field(description="ARIMA order (p, d, q)")
    seasonal_order: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Seasonal ARIMA order (P, D, Q, m) if seasonal"
    )

    # Model diagnostics
    aic: float | None = Field(default=None, description="Akaike Information Criterion")
    bic: float | None = Field(default=None, description="Bayesian Information Criterion")

    # Fitted values summary
    n_observations: int = Field(description="Number of observations used")

    # Model storage
    model_id: str = Field(description="ID of the stored model in state")
    dataframe_name: str = Field(description="Name of the source DataFrame")
    target_column: str = Field(description="Name of the time series column")

    # Summary statistics
    residual_std: float | None = Field(
        default=None,
        description="Standard deviation of residuals"
    )

    # Interpretation
    message: str = Field(description="Human-readable model summary")


class ARIMAForecastResult(BaseModel):
    """Result for ARIMA forecasting."""

    success: bool = Field(description="Whether forecasting succeeded")
    operation: str = Field(default="arima_forecast", description="Operation performed")

    # Forecast data
    forecast_values: list[float] = Field(description="Point forecasts")
    forecast_index: list[str] = Field(description="Index/dates for forecast periods")

    # Confidence intervals (optional)
    lower_ci: list[float] | None = Field(
        default=None,
        description="Lower confidence interval bounds"
    )
    upper_ci: list[float] | None = Field(
        default=None,
        description="Upper confidence interval bounds"
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for intervals"
    )

    # Metadata
    n_periods: int = Field(description="Number of periods forecasted")
    model_id: str = Field(description="ID of the ARIMA model used")

    # Chart (optional)
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded PNG image of the forecast plot"
    )

    # Interpretation
    message: str = Field(description="Human-readable forecast summary")


class ARIMAParameterSearchResult(BaseModel):
    """Result for automatic ARIMA parameter search."""

    success: bool = Field(description="Whether parameter search succeeded")
    operation: str = Field(default="arima_parameter_search", description="Operation performed")

    # Best parameters
    best_order: tuple[int, int, int] = Field(description="Best ARIMA order (p, d, q)")
    best_seasonal_order: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Best seasonal order (P, D, Q, m) if seasonal"
    )
    best_aic: float = Field(description="AIC of the best model")

    # Search summary
    models_evaluated: int = Field(description="Number of models evaluated")
    search_time_seconds: float = Field(description="Time taken for search in seconds")

    # Top models comparison
    top_models: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top N models with their parameters and AIC scores"
    )

    # Metadata
    dataframe_name: str = Field(description="Name of the source DataFrame")
    target_column: str = Field(description="Name of the time series column")

    # Interpretation
    message: str = Field(description="Human-readable search summary")


class MeanTargetEncodingResult(BaseModel):
    """Result for mean target encoding of categorical columns."""

    success: bool = Field(description="Whether encoding succeeded")
    operation: str = Field(default="mean_target_encoding", description="Operation performed")

    # DataFrame info
    dataframe_name: str = Field(description="Name of the modified DataFrame")
    source_dataframe: str = Field(description="Name of the source DataFrame")
    rows_affected: int = Field(description="Number of rows in the DataFrame")

    # Encoding details
    encoded_columns: list[str] = Field(
        description="Names of newly created encoded columns"
    )
    original_columns: list[str] = Field(
        description="Original categorical column names that were encoded"
    )
    target_column: str = Field(description="Target column used for encoding")

    # Column mapping (original -> encoded)
    column_mapping: dict[str, str | list[str]] = Field(
        description="Mapping from original column names to encoded column names"
    )

    # Encoding statistics per column
    encoding_stats: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Statistics for each encoded column (unique categories, range of encoded values)"
    )

    # Encoder storage
    encoder_id: str = Field(
        description="ID of the stored encoder for applying to new data"
    )

    # Parameters used
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for encoding (cv, smooth, target_type)"
    )

    # Interpretation
    message: str = Field(description="Human-readable encoding summary")


class BinRareCategoriesResult(BaseModel):
    """Result for binning rare categories."""

    success: bool = Field(description="Whether binning succeeded")
    operation: str = Field(default="bin_rare_categories", description="Operation performed")

    # DataFrame info
    dataframe_name: str = Field(description="Name of the modified DataFrame")
    source_dataframe: str = Field(description="Name of the source DataFrame")
    rows_affected: int = Field(description="Number of rows in the DataFrame")

    # Binning details
    columns_processed: list[str] = Field(
        description="Columns that were processed"
    )
    columns_modified: list[str] = Field(
        description="Columns that actually had categories binned"
    )

    # Per-column binning info
    binning_details: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Details per column: categories_before, categories_after, binned_categories, rows_affected"
    )

    # Mapping for applying to new data
    category_mapping: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description="Mapping from original categories to binned categories per column"
    )

    # Parameters used
    threshold: float = Field(
        description="Frequency threshold used for determining rare categories"
    )
    bin_label: str = Field(
        default="Other",
        description="Label used for binned categories"
    )

    # Interpretation
    message: str = Field(description="Human-readable binning summary")


class OperationError(BaseModel):
    """Result for failed operations."""

    success: bool = Field(default=False, description="Always False for errors")
    error_type: str = Field(description="Type of error")
    error_message: str = Field(description="Human-readable error message")
    operation: str = Field(description="Operation that failed")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error details"
    )
