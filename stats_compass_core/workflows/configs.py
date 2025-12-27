"""
Configuration schemas for workflow tools.

These models allow users to customize workflow behavior while
providing sensible defaults for common use cases.
"""

from typing import Literal

from pydantic import BaseModel, Field


# =============================================================================
# Preprocessing Config
# =============================================================================

class ImputationConfig(BaseModel):
    """Configuration for missing value imputation."""

    strategy: Literal["mean", "median", "mode", "constant", "drop"] = Field(
        default="mean",
        description="Imputation strategy for numeric columns"
    )
    constant_value: str | int | float | None = Field(
        default=None,
        description="Value to use when strategy='constant'"
    )
    columns: list[str] | None = Field(
        default=None,
        description="Specific columns to impute. If None, applies to all with missing values."
    )


class OutlierConfig(BaseModel):
    """Configuration for outlier handling."""

    method: Literal["iqr", "zscore", "none"] = Field(
        default="iqr",
        description="Outlier detection method"
    )
    action: Literal["cap", "remove", "flag"] = Field(
        default="cap",
        description="How to handle detected outliers"
    )
    threshold: float = Field(
        default=1.5,
        description="IQR multiplier or z-score threshold"
    )
    columns: list[str] | None = Field(
        default=None,
        description="Specific numeric columns to check. If None, checks all numeric."
    )


class FeatureEngineeringConfig(BaseModel):
    """
    Configuration for feature engineering before model training.
    
    Applied in order: bin_rare_categories -> detect categoricals -> target_encode
    """

    encode_categoricals: bool = Field(
        default=True,
        description="Apply target encoding to categorical columns"
    )
    categorical_columns: list[str] | None = Field(
        default=None,
        description="Columns to encode. If None, auto-detects object/category dtype columns."
    )
    bin_rare_categories: bool = Field(
        default=True,
        description="Bin rare categories before encoding (recommended for high-cardinality columns)"
    )
    rare_threshold: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Frequency threshold for binning rare categories (0-1)"
    )
    bin_label: str = Field(
        default="Other",
        description="Label for binned rare categories"
    )


class DateCleaningConfig(BaseModel):
    """Configuration for date/time column cleaning."""

    date_column: str = Field(
        description="Name of the date/time column to clean"
    )
    fill_method: Literal["ffill", "bfill", "interpolate", "drop"] = Field(
        default="ffill",
        description="Method to handle missing dates: forward fill, backward fill, interpolate, or drop rows"
    )
    infer_frequency: bool = Field(
        default=True,
        description="Whether to automatically detect and fill missing date gaps in the sequence"
    )
    create_missing_dates: bool = Field(
        default=False,
        description="Whether to create rows for missing dates in the sequence (useful for time series)"
    )


class PreprocessingConfig(BaseModel):
    """
    Configuration for the run_preprocessing workflow.
    
    All fields are optional - defaults provide sensible behavior.
    """

    imputation: ImputationConfig | None = Field(
        default=None,
        description="Missing value handling config. Uses defaults if None."
    )
    outliers: OutlierConfig | None = Field(
        default=None,
        description="Outlier handling config. Uses defaults if None."
    )
    date_cleaning: DateCleaningConfig | None = Field(
        default=None,
        description="Date/time column cleaning config. Uses defaults if None."
    )
    dedupe: bool = Field(
        default=True,
        description="Whether to remove duplicate rows"
    )
    drop_columns: list[str] | None = Field(
        default=None,
        description="Columns to drop before processing"
    )
    encode_categoricals: bool = Field(
        default=False,
        description="Whether to apply target encoding to categorical columns"
    )
    target_column: str | None = Field(
        default=None,
        description="Target column for encoding (required if encode_categoricals=True)"
    )


# =============================================================================
# EDA Config
# =============================================================================

class EDAConfig(BaseModel):
    """
    Configuration for the run_eda_report workflow.
    
    Controls which analyses and visualizations to include.
    """

    include_describe: bool = Field(
        default=True,
        description="Include descriptive statistics"
    )
    include_correlations: bool = Field(
        default=True,
        description="Include correlation matrix"
    )
    include_missing_analysis: bool = Field(
        default=True,
        description="Include missing data analysis"
    )
    include_quality_report: bool = Field(
        default=True,
        description="Include data quality report"
    )
    generate_histograms: bool = Field(
        default=True,
        description="Generate histograms for numeric columns"
    )
    generate_bar_charts: bool = Field(
        default=True,
        description="Generate bar charts for categorical columns"
    )
    max_categorical_cardinality: int = Field(
        default=20,
        description="Skip bar charts for categoricals with more unique values than this"
    )
    correlation_method: Literal["pearson", "spearman", "kendall"] = Field(
        default="pearson",
        description="Correlation calculation method"
    )
    max_histograms: int = Field(
        default=10,
        description="Maximum number of histogram charts to generate"
    )
    max_bar_charts: int = Field(
        default=10,
        description="Maximum number of bar charts to generate"
    )


# =============================================================================
# Classification Config
# =============================================================================

class ClassificationConfig(BaseModel):
    """
    Configuration for the run_classification workflow.
    """

    model_type: Literal["random_forest", "gradient_boosting", "logistic"] = Field(
        default="random_forest",
        description="Type of classification model to train"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Fraction of data to use for testing"
    )
    random_state: int | None = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    hyperparameters: dict | None = Field(
        default=None,
        description="Model-specific hyperparameters to override defaults"
    )
    feature_engineering: FeatureEngineeringConfig | None = Field(
        default_factory=FeatureEngineeringConfig,
        description="Feature engineering config. Set to None to skip."
    )
    generate_plots: bool = Field(
        default=True,
        description="Whether to generate evaluation plots"
    )
    plots: list[str] = Field(
        default=["confusion_matrix", "roc", "precision_recall", "feature_importance"],
        description="Which plots to generate"
    )
    save_model: bool = Field(
        default=False,
        description="Whether to save the trained model to disk"
    )
    model_save_path: str | None = Field(
        default=None,
        description="Path to save model (required if save_model=True)"
    )


# =============================================================================
# Regression Config
# =============================================================================

class RegressionConfig(BaseModel):
    """
    Configuration for the run_regression workflow.
    """

    model_type: Literal["random_forest", "gradient_boosting", "linear"] = Field(
        default="random_forest",
        description="Type of regression model to train"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Fraction of data to use for testing"
    )
    random_state: int | None = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    hyperparameters: dict | None = Field(
        default=None,
        description="Model-specific hyperparameters to override defaults"
    )
    feature_engineering: FeatureEngineeringConfig | None = Field(
        default_factory=FeatureEngineeringConfig,
        description="Feature engineering config. Set to None to skip."
    )
    generate_plots: bool = Field(
        default=True,
        description="Whether to generate evaluation plots"
    )
    plots: list[str] = Field(
        default=["residuals", "predicted_vs_actual", "feature_importance"],
        description="Which plots to generate"
    )
    save_model: bool = Field(
        default=False,
        description="Whether to save the trained model to disk"
    )
    model_save_path: str | None = Field(
        default=None,
        description="Path to save model (required if save_model=True)"
    )


# =============================================================================
# Time Series Config
# =============================================================================

class TimeSeriesConfig(BaseModel):
    """
    Configuration for the run_timeseries_forecast workflow.
    """

    date_column: str = Field(
        description="Name of the date/time column"
    )
    target_column: str = Field(
        description="Name of the column to forecast"
    )
    forecast_periods: int | str = Field(
        default=30,
        description="Number of periods to forecast, or natural language like '1 month'"
    )
    auto_find_params: bool = Field(
        default=True,
        description="Use grid search to find optimal ARIMA parameters"
    )
    arima_order: tuple[int, int, int] | None = Field(
        default=None,
        description="Manual ARIMA (p, d, q) order. Ignored if auto_find_params=True."
    )
    seasonal_order: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Seasonal ARIMA (P, D, Q, s) order for SARIMA models"
    )
    check_stationarity: bool = Field(
        default=True,
        description="Run stationarity tests before fitting"
    )
    validate_dates: bool = Field(
        default=True,
        description="Validate that date column has no missing values before fitting. Raises error if missing dates found."
    )
    handle_missing_dates: Literal["error", "ffill", "bfill", "drop"] = Field(
        default="error",
        description="How to handle missing dates: 'error' stops execution, 'ffill' forward fills, 'drop' removes rows"
    )
    generate_forecast_plot: bool = Field(
        default=True,
        description="Generate a plot of the forecast"
    )


# =============================================================================
# Model Comparison Config
# =============================================================================

class CompareModelsConfig(BaseModel):
    """
    Configuration for the compare_models workflow.
    """

    task_type: Literal["classification", "regression"] = Field(
        description="Type of ML task"
    )
    models: list[str] | None = Field(
        default=None,
        description="List of model types to compare. If None, uses all available for task type."
    )
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Fraction of data to use for testing"
    )
    random_state: int | None = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    metrics: list[str] | None = Field(
        default=None,
        description="Metrics to include in comparison. If None, uses all relevant metrics."
    )
