"""
ARIMA time series modeling tools.

Requires the [timeseries] extra: pip install stats-compass-core[timeseries]
"""

import base64
import io
import itertools
import os
import time
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import (
    ARIMAForecastResult,
    ARIMAParameterSearchResult,
    ARIMAResult,
    OperationError,
)
from stats_compass_core.state import DataFrameState

# Check for optional dependencies
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, kpss

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Input Schemas
# ---------------------------------------------------------------------------


class FitARIMAInput(StrictToolInput):
    """Input parameters for fitting an ARIMA model."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of the DataFrame to use. If not provided, uses the active DataFrame.",
    )
    target_column: str = Field(
        description="Name of the column containing the time series values"
    )
    date_column: str | None = Field(
        default=None,
        description="Name of the date/time column for index. If not provided, uses row index.",
    )
    p: int = Field(default=1, ge=0, le=10, description="AR order (autoregressive)")
    d: int = Field(default=1, ge=0, le=3, description="Differencing order")
    q: int = Field(default=1, ge=0, le=10, description="MA order (moving average)")
    seasonal: bool = Field(default=False, description="Whether to fit a seasonal model")
    seasonal_order: list[int] | None = Field(
        default=None,
        description="Seasonal order (P, D, Q, m) where m is the seasonal period. Must be a list of 4 integers.",
    )
    model_name: str | None = Field(
        default=None, description="Custom name for storing the model"
    )
    save_path: str | None = Field(
        default=None, description="Path to save the trained model (e.g., 'model.joblib')"
    )


class ForecastARIMAInput(StrictToolInput):
    """Input parameters for ARIMA forecasting."""

    model_id: str = Field(description="ID of the fitted ARIMA model to use")
    n_periods: int | None = Field(
        default=None,
        ge=1,
        le=365,
        description="Number of periods to forecast. Use this OR (forecast_number + forecast_unit).",
    )
    forecast_number: int | None = Field(
        default=None,
        ge=1,
        description="Number of time units to forecast (e.g., 30 for '30 days'). Use with forecast_unit.",
    )
    forecast_unit: Literal["days", "weeks", "months", "quarters", "years"] | None = Field(
        default=None,
        description="Time unit for forecast. Use with forecast_number.",
    )
    confidence_level: float = Field(
        default=0.95, ge=0.5, le=0.99, description="Confidence level for intervals"
    )
    include_plot: bool = Field(
        default=True, description="Whether to generate a forecast plot"
    )


class FindOptimalARIMAInput(StrictToolInput):
    """Input parameters for automatic ARIMA parameter search."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of the DataFrame to use. If not provided, uses the active DataFrame.",
    )
    target_column: str = Field(
        description="Name of the column containing the time series values"
    )
    date_column: str | None = Field(
        default=None,
        description="Name of the date/time column for index. If not provided, uses row index.",
    )
    max_p: int = Field(default=3, ge=0, le=5, description="Maximum AR order to try")
    max_d: int = Field(default=2, ge=0, le=2, description="Maximum differencing order")
    max_q: int = Field(default=3, ge=0, le=5, description="Maximum MA order to try")
    criterion: Literal["aic", "bic"] = Field(
        default="aic", description="Information criterion for model selection"
    )
    top_n: int = Field(
        default=5, ge=1, le=10, description="Number of top models to return"
    )


class StationarityTestInput(StrictToolInput):
    """Input parameters for stationarity testing."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of the DataFrame to use. If not provided, uses the active DataFrame.",
    )
    target_column: str = Field(
        description="Name of the column containing the time series values"
    )
    test_type: Literal["adf", "kpss", "both"] = Field(
        default="both",
        description="Type of stationarity test: 'adf' (Augmented Dickey-Fuller), 'kpss', or 'both'",
    )


class InferFrequencyInput(StrictToolInput):
    """Input parameters for time series frequency inference."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of the DataFrame to use. If not provided, uses the active DataFrame.",
    )
    date_column: str = Field(
        description="Name of the date/time column to analyze"
    )


class InferFrequencyResult(BaseModel):
    """Result for time series frequency inference."""

    success: bool = Field(description="Whether the inference succeeded")
    operation: str = Field(default="infer_frequency", description="Operation performed")

    # Frequency info
    frequency_description: str = Field(
        description="Human-readable description of the frequency (e.g., 'daily', 'weekly')"
    )
    frequency_timedelta: str = Field(
        description="Timedelta string representation (e.g., '1 days', '7 days')"
    )
    frequency_days: float = Field(
        description="Frequency in days (e.g., 1.0 for daily, 7.0 for weekly)"
    )

    # Data info
    n_observations: int = Field(description="Number of observations in the time series")
    date_range: str = Field(description="Date range of the time series")

    # Conversion examples
    conversion_examples: dict[str, int] = Field(
        description="Examples of period conversions (e.g., {'30 days': 30, '3 months': 90})"
    )

    # Message
    message: str = Field(description="Human-readable summary")


# ---------------------------------------------------------------------------
# Result Models for Stationarity Tests
# ---------------------------------------------------------------------------


class StationarityTestResult(BaseModel):
    """Result for a single stationarity test (ADF or KPSS)."""

    test_type: str = Field(description="Type of test performed ('adf' or 'kpss')")
    test_statistic: float = Field(description="Test statistic value")
    p_value: float = Field(description="P-value of the test")
    critical_values: dict[str, float] = Field(
        description="Critical values at different significance levels"
    )
    is_stationary: bool = Field(
        description="Whether the series is stationary according to this test"
    )
    n_lags: int | None = Field(default=None, description="Number of lags used")
    interpretation: str = Field(description="Human-readable interpretation of results")


class StationarityResult(BaseModel):
    """
    Combined result for stationarity testing.
    
    This unified result type provides a consistent interface for accessing
    stationarity test results, whether one or both tests were run.
    Useful for ARIMA preprocessing and other time series applications.
    """

    success: bool = Field(description="Whether the operation succeeded")
    operation: str = Field(default="check_stationarity", description="Operation performed")

    # Individual test results (None if not requested)
    adf_result: StationarityTestResult | None = Field(
        default=None, description="ADF test result (null hypothesis: non-stationary)"
    )
    kpss_result: StationarityTestResult | None = Field(
        default=None, description="KPSS test result (null hypothesis: stationary)"
    )

    # Overall assessment
    is_stationary: bool = Field(
        description="Overall stationarity assessment based on test(s) performed"
    )
    recommendation: str = Field(
        description="Recommendation for differencing based on test results"
    )

    # Metadata
    target_column: str = Field(description="Column that was tested")
    n_observations: int = Field(description="Number of observations in the series")
    message: str = Field(description="Human-readable summary")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _check_dependencies() -> OperationError | None:
    """Check if required dependencies are available."""
    if not STATSMODELS_AVAILABLE:
        return OperationError(
            error_type="ImportError",
            error_message="statsmodels is required for ARIMA modeling. "
            "Install with: pip install stats-compass-core[timeseries]",
            operation="arima",
            details={"missing_package": "statsmodels"},
        )
    return None


def _prepare_series(
    state: DataFrameState,
    dataframe_name: str | None,
    target_column: str,
    date_column: str | None,
) -> tuple[pd.Series, str, str] | OperationError:
    """Prepare time series data for ARIMA modeling."""
    # Get DataFrame
    try:
        df = state.get_dataframe(dataframe_name)
        if not dataframe_name:
            dataframe_name = state.get_active_dataframe_name()
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg:
            return OperationError(
                error_type="DataFrameNotFound",
                error_message=error_msg,
                operation="arima",
                details={"dataframe_name": dataframe_name},
            )
        else:
            return OperationError(
                error_type="NoActiveDataFrame",
                error_message=error_msg,
                operation="arima",
                details={},
            )

    # Validate target column
    if target_column not in df.columns:
        return OperationError(
            error_type="ColumnNotFound",
            error_message=f"Column '{target_column}' not found in DataFrame",
            operation="arima",
            details={"column": target_column, "available": list(df.columns)},
        )

    # Prepare series
    series = df[target_column].copy()

    # Set index if date column provided
    if date_column:
        if date_column not in df.columns:
            return OperationError(
                error_type="ColumnNotFound",
                error_message=f"Date column '{date_column}' not found",
                operation="arima",
                details={"column": date_column},
            )
        datetime_index = pd.to_datetime(df[date_column])
        # Infer and set frequency to avoid statsmodels warnings
        inferred_freq = pd.infer_freq(datetime_index)
        if inferred_freq:
            series.index = pd.DatetimeIndex(datetime_index, freq=inferred_freq)
        else:
            series.index = pd.DatetimeIndex(datetime_index)

    # Drop NaN values
    series = series.dropna()

    if len(series) < 10:
        return OperationError(
            error_type="InsufficientData",
            error_message=f"Need at least 10 observations, got {len(series)}",
            operation="arima",
            details={"n_observations": len(series)},
        )

    return series, dataframe_name, target_column


def _create_forecast_plot(
    historical: pd.Series,
    forecast: pd.Series,
    lower_ci: pd.Series | None,
    upper_ci: pd.Series | None,
    title: str,
) -> str | None:
    """Create a forecast plot and return as base64 string."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical data
    ax.plot(historical.index, historical.values, label="Historical", color="blue")

    # Plot forecast
    ax.plot(forecast.index, forecast.values, label="Forecast", color="red", linestyle="--")

    # Plot confidence interval
    if lower_ci is not None and upper_ci is not None:
        ax.fill_between(
            forecast.index,
            lower_ci.values,
            upper_ci.values,
            alpha=0.3,
            color="red",
            label="Confidence Interval",
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels if datetime
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return image_base64


def _infer_time_frequency(time_index: pd.DatetimeIndex | pd.Index) -> pd.Timedelta:
    """
    Infer the frequency of a time series as a Timedelta.
    
    Uses the MEDIAN time difference between consecutive observations,
    which is robust to:
    - Missing data points
    - Occasional gaps (weekends, holidays)
    - Irregular but mostly-consistent spacing
    
    Args:
        time_index: DatetimeIndex or Index of the time series
        
    Returns:
        pd.Timedelta representing the typical time between observations
    """
    if len(time_index) < 2:
        return pd.Timedelta(days=1)  # Default fallback

    # Convert to DatetimeIndex if not already
    if not isinstance(time_index, pd.DatetimeIndex):
        try:
            time_index = pd.DatetimeIndex(time_index)
        except (TypeError, ValueError):
            return pd.Timedelta(days=1)  # Can't infer, use default

    # Calculate all time differences
    time_diffs = time_index.to_series().diff().dropna()

    if len(time_diffs) == 0:
        return pd.Timedelta(days=1)

    # Use median - more robust than mean or mode
    return time_diffs.median()


def _convert_forecast_period_to_steps(
    time_index: pd.DatetimeIndex | pd.Index,
    forecast_number: int,
    forecast_unit: str,
) -> int:
    """
    Convert a human-readable forecast period (e.g., "30 days", "6 months")
    into the number of data points to forecast.
    
    Args:
        time_index: DatetimeIndex of the time series
        forecast_number: Number of time units (e.g., 30 for "30 days")
        forecast_unit: Unit of time ('days', 'weeks', 'months', 'quarters', 'years')
        
    Returns:
        Number of forecast steps to generate
    """
    # Get the actual data frequency
    data_freq = _infer_time_frequency(time_index)

    # Convert user's request to timedelta
    unit_mapping = {
        "days": pd.Timedelta(days=forecast_number),
        "weeks": pd.Timedelta(weeks=forecast_number),
        "months": pd.Timedelta(days=forecast_number * 30),  # Approximate
        "quarters": pd.Timedelta(days=forecast_number * 91),  # Approximate
        "years": pd.Timedelta(days=forecast_number * 365),  # Approximate
    }

    requested_period = unit_mapping.get(forecast_unit.lower())
    if requested_period is None:
        # Invalid unit, just return the number as-is
        return forecast_number

    # Calculate steps: how many data_freq periods fit in requested_period?
    steps = int(round(requested_period / data_freq))

    return max(1, min(steps, 365))  # Bound between 1 and 365


def _describe_frequency(freq: pd.Timedelta) -> str:
    """
    Describe a frequency timedelta in human-readable terms.
    
    Args:
        freq: Timedelta representing the data frequency
        
    Returns:
        Human-readable description (e.g., "daily", "weekly", "monthly")
    """
    days = freq.days

    if days == 0:
        hours = freq.seconds // 3600
        if hours <= 1:
            return "hourly"
        elif hours <= 12:
            return f"every {hours} hours"
        else:
            return "sub-daily"
    elif days == 1:
        return "daily"
    elif 5 <= days <= 8:
        return "weekly"
    elif 13 <= days <= 16:
        return "bi-weekly"
    elif 28 <= days <= 32:
        return "monthly"
    elif 85 <= days <= 95:
        return "quarterly"
    elif 360 <= days <= 370:
        return "yearly"
    else:
        return f"every {days} days"


# ---------------------------------------------------------------------------
# Tool Functions
# ---------------------------------------------------------------------------


@registry.register(
    category="ml",
    input_schema=FitARIMAInput,
    description="Fit an ARIMA model to time series data for forecasting",
)
def fit_arima(
    state: DataFrameState, params: FitARIMAInput
) -> ARIMAResult | OperationError:
    """
    Fit an ARIMA model to time series data.

    ARIMA (AutoRegressive Integrated Moving Average) models are used for
    time series forecasting. The model is specified by three parameters:
    - p: Order of autoregressive terms
    - d: Degree of differencing
    - q: Order of moving average terms

    Args:
        state: DataFrameState containing the data
        params: FitARIMAInput with model configuration

    Returns:
        ARIMAResult with model diagnostics and storage info
    """
    # Check dependencies
    error = _check_dependencies()
    if error:
        return error

    # Prepare data
    result = _prepare_series(
        state, params.dataframe_name, params.target_column, params.date_column
    )
    if isinstance(result, OperationError):
        return result

    series, df_name, target_col = result

    # Prepare ARIMA order
    order = (params.p, params.d, params.q)
    seasonal_order = None

    if params.seasonal and params.seasonal_order:
        # Convert list to tuple if needed
        if isinstance(params.seasonal_order, list):
            seasonal_order = tuple(params.seasonal_order)
        else:
            seasonal_order = params.seasonal_order

    try:
        # Fit ARIMA model (suppress convergence warnings)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')

            if seasonal_order:
                model = ARIMA(series, order=order, seasonal_order=seasonal_order)
            else:
                model = ARIMA(series, order=order)

            fitted_model = model.fit()

        # Store model in state
        model_name = params.model_name or f"arima_{params.p}_{params.d}_{params.q}"
        model_id = state.store_model(
            model=fitted_model,
            model_type="arima",
            target_column=target_col,
            feature_columns=[],  # ARIMA doesn't use feature columns
            source_dataframe=df_name,
            custom_name=model_name,
        )

        # Calculate residual std
        residual_std = float(np.std(fitted_model.resid))

        # Save model to disk if requested
        if params.save_path:
            filepath = os.path.expanduser(params.save_path)
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            joblib.dump(fitted_model, filepath)

        # Create summary message
        if seasonal_order:
            msg = (
                f"ARIMA({params.p},{params.d},{params.q})x{seasonal_order} model fitted successfully. "
                f"AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}"
            )
        else:
            msg = (
                f"ARIMA({params.p},{params.d},{params.q}) model fitted successfully. "
                f"AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}"
            )

        return ARIMAResult(
            success=True,
            order=order,
            seasonal_order=seasonal_order,
            aic=float(fitted_model.aic),
            bic=float(fitted_model.bic),
            n_observations=len(series),
            model_id=model_id,
            dataframe_name=df_name,
            target_column=target_col,
            residual_std=residual_std,
            message=msg,
        )

    except Exception as e:
        return OperationError(
            error_type="ARIMAFitError",
            error_message=f"Failed to fit ARIMA model: {str(e)}",
            operation="fit_arima",
            details={"order": order, "error": str(e)},
        )


@registry.register(
    category="ml",
    input_schema=ForecastARIMAInput,
    description="Generate forecasts using a fitted ARIMA model",
)
def forecast_arima(
    state: DataFrameState, params: ForecastARIMAInput
) -> ARIMAForecastResult | OperationError:
    """
    Generate forecasts using a fitted ARIMA model.

    Supports two ways to specify forecast horizon:
    1. n_periods: Direct number of periods to forecast
    2. forecast_number + forecast_unit: Natural language (e.g., "30 days", "6 months")
       - Automatically calculates steps based on data frequency

    Args:
        state: DataFrameState containing the fitted model
        params: ForecastARIMAInput with forecast configuration

    Returns:
        ARIMAForecastResult with forecast values and optional plot
    """
    # Check dependencies
    error = _check_dependencies()
    if error:
        return error

    # Get fitted model
    model = state.get_model(params.model_id)
    if model is None:
        return OperationError(
            error_type="ModelNotFound",
            error_message=f"Model '{params.model_id}' not found. Fit a model first.",
            operation="forecast_arima",
            details={"model_id": params.model_id},
        )

    try:
        # Determine number of periods to forecast
        n_periods: int
        freq_description = ""

        if params.forecast_number is not None and params.forecast_unit is not None:
            # Use natural language period specification
            time_index = model.model._index
            n_periods = _convert_forecast_period_to_steps(
                time_index, params.forecast_number, params.forecast_unit
            )
            inferred_freq = _infer_time_frequency(time_index)
            freq_description = f" (data frequency: {_describe_frequency(inferred_freq)})"
        elif params.n_periods is not None:
            # Use direct n_periods
            n_periods = params.n_periods
        else:
            # Default to 10 periods
            n_periods = 10

        # Generate forecast
        forecast_result = model.get_forecast(steps=n_periods)
        forecast_mean = forecast_result.predicted_mean

        # Get confidence intervals
        alpha = 1 - params.confidence_level
        conf_int = forecast_result.conf_int(alpha=alpha)

        # Prepare output
        forecast_values = forecast_mean.tolist()
        forecast_index = [str(idx) for idx in forecast_mean.index]

        lower_ci = conf_int.iloc[:, 0].tolist()
        upper_ci = conf_int.iloc[:, 1].tolist()

        # Generate plot if requested
        image_base64 = None
        if params.include_plot:
            # Get historical data from model
            endog = model.model.endog
            # Flatten if 2D
            if hasattr(endog, 'ndim') and endog.ndim > 1:
                endog = endog.flatten()
            historical = pd.Series(endog, index=model.model._index)
            image_base64 = _create_forecast_plot(
                historical=historical,
                forecast=forecast_mean,
                lower_ci=conf_int.iloc[:, 0],
                upper_ci=conf_int.iloc[:, 1],
                title=f"ARIMA Forecast ({n_periods} periods)",
            )

        # Create summary message
        if params.forecast_number is not None and params.forecast_unit is not None:
            msg = (
                f"Generated {n_periods}-period forecast for {params.forecast_number} {params.forecast_unit}{freq_description}. "
                f"Forecast range: {forecast_values[0]:.2f} to {forecast_values[-1]:.2f}"
            )
        else:
            msg = (
                f"Generated {n_periods}-period forecast. "
                f"Forecast range: {forecast_values[0]:.2f} to {forecast_values[-1]:.2f}"
            )

        return ARIMAForecastResult(
            success=True,
            forecast_values=forecast_values,
            forecast_index=forecast_index,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            confidence_level=params.confidence_level,
            n_periods=n_periods,
            model_id=params.model_id,
            image_base64=image_base64,
            message=msg,
        )

    except Exception as e:
        return OperationError(
            error_type="ForecastError",
            error_message=f"Failed to generate forecast: {str(e)}",
            operation="forecast_arima",
            details={"error": str(e)},
        )


@registry.register(
    category="ml",
    input_schema=FindOptimalARIMAInput,
    description="Automatically find optimal ARIMA parameters using grid search",
)
def find_optimal_arima(
    state: DataFrameState, params: FindOptimalARIMAInput
) -> ARIMAParameterSearchResult | OperationError:
    """
    Automatically find optimal ARIMA parameters using grid search.

    This function evaluates multiple ARIMA models with different (p, d, q)
    combinations and selects the best one based on AIC or BIC.

    Note: This can take several minutes for large datasets or wide search ranges.

    Args:
        state: DataFrameState containing the data
        params: FindOptimalARIMAInput with search configuration

    Returns:
        ARIMAParameterSearchResult with best parameters and top models
    """
    # Check dependencies
    error = _check_dependencies()
    if error:
        return error

    # Prepare data
    result = _prepare_series(
        state, params.dataframe_name, params.target_column, params.date_column
    )
    if isinstance(result, OperationError):
        return result

    series, df_name, target_col = result

    start_time = time.time()

    # Generate parameter combinations
    p_range = range(0, params.max_p + 1)
    d_range = range(0, params.max_d + 1)
    q_range = range(0, params.max_q + 1)

    combinations = list(itertools.product(p_range, d_range, q_range))
    results_list: list[dict[str, Any]] = []

    # Suppress warnings during grid search
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')

        for p, d, q in combinations:
            try:
                model = ARIMA(series, order=(p, d, q))
                fitted = model.fit()

                score = fitted.aic if params.criterion == "aic" else fitted.bic

                results_list.append(
                    {
                        "order": (p, d, q),
                        "aic": float(fitted.aic),
                        "bic": float(fitted.bic),
                        "score": float(score),
                    }
                )
            except Exception:
                # Skip models that fail to converge
                continue

    search_time = time.time() - start_time

    if not results_list:
        return OperationError(
            error_type="NoValidModels",
            error_message="No valid ARIMA models found. Try adjusting parameters.",
            operation="find_optimal_arima",
            details={},
        )

    # Sort by criterion
    results_list.sort(key=lambda x: x["score"])

    # Get best model
    best = results_list[0]
    best_order = best["order"]

    # Get top N models
    top_models = results_list[: params.top_n]

    # Create summary message
    msg = (
        f"Evaluated {len(results_list)} models in {search_time:.1f}s. "
        f"Best model: ARIMA{best_order} with {params.criterion.upper()}={best['score']:.2f}"
    )

    return ARIMAParameterSearchResult(
        success=True,
        best_order=best_order,
        best_seasonal_order=None,
        best_aic=best["aic"],
        models_evaluated=len(results_list),
        search_time_seconds=search_time,
        top_models=top_models,
        dataframe_name=df_name,
        target_column=target_col,
        message=msg,
    )


@registry.register(
    category="ml",
    input_schema=StationarityTestInput,
    description="Test if a time series is stationary using ADF and/or KPSS tests",
)
def check_stationarity(
    state: DataFrameState, params: StationarityTestInput
) -> StationarityResult | OperationError:
    """
    Test if a time series is stationary using ADF and/or KPSS tests.

    Stationarity is important for ARIMA modeling and other time series analysis.
    A stationary series has constant mean, variance, and autocorrelation over time.

    - ADF test: Null hypothesis is that the series has a unit root (non-stationary)
    - KPSS test: Null hypothesis is that the series is stationary

    Args:
        state: DataFrameState containing the data
        params: StationarityTestInput with test configuration

    Returns:
        StationarityResult with .adf_result and/or .kpss_result attributes
    """
    # Check dependencies
    error = _check_dependencies()
    if error:
        return error

    # Prepare data
    result = _prepare_series(
        state, params.dataframe_name, params.target_column, None
    )
    if isinstance(result, OperationError):
        return result

    series, _, target_col = result

    adf_test_result: StationarityTestResult | None = None
    kpss_test_result: StationarityTestResult | None = None

    # ADF Test
    if params.test_type in ("adf", "both"):
        try:
            adf_result = adfuller(series, autolag="AIC")
            adf_statistic = float(adf_result[0])
            adf_pvalue = float(adf_result[1])
            adf_lags = int(adf_result[2])
            adf_critical = {k: float(v) for k, v in adf_result[4].items()}

            # Series is stationary if p-value < 0.05 (reject null of unit root)
            adf_is_stationary = adf_pvalue < 0.05

            if adf_is_stationary:
                interp = (
                    f"ADF test statistic: {adf_statistic:.4f} (p-value: {adf_pvalue:.4f}). "
                    f"The series IS stationary (p < 0.05). No differencing needed."
                )
            else:
                interp = (
                    f"ADF test statistic: {adf_statistic:.4f} (p-value: {adf_pvalue:.4f}). "
                    f"The series is NOT stationary (p >= 0.05). Consider differencing (d >= 1)."
                )

            adf_test_result = StationarityTestResult(
                test_type="adf",
                test_statistic=adf_statistic,
                p_value=adf_pvalue,
                critical_values=adf_critical,
                is_stationary=adf_is_stationary,
                n_lags=adf_lags,
                interpretation=interp,
            )
        except Exception as e:
            return OperationError(
                error_type="TestError",
                error_message=f"ADF test failed: {str(e)}",
                operation="check_stationarity",
                details={"error": str(e)},
            )

    # KPSS Test
    if params.test_type in ("kpss", "both"):
        try:
            kpss_result = kpss(series, regression="c", nlags="auto")
            kpss_statistic = float(kpss_result[0])
            kpss_pvalue = float(kpss_result[1])
            kpss_lags = int(kpss_result[2])
            kpss_critical = {k: float(v) for k, v in kpss_result[3].items()}

            # Series is stationary if p-value > 0.05 (fail to reject null of stationarity)
            kpss_is_stationary = kpss_pvalue > 0.05

            if kpss_is_stationary:
                interp = (
                    f"KPSS test statistic: {kpss_statistic:.4f} (p-value: {kpss_pvalue:.4f}). "
                    f"The series IS stationary (p > 0.05). No differencing needed."
                )
            else:
                interp = (
                    f"KPSS test statistic: {kpss_statistic:.4f} (p-value: {kpss_pvalue:.4f}). "
                    f"The series is NOT stationary (p <= 0.05). Consider differencing."
                )

            kpss_test_result = StationarityTestResult(
                test_type="kpss",
                test_statistic=kpss_statistic,
                p_value=kpss_pvalue,
                critical_values=kpss_critical,
                is_stationary=kpss_is_stationary,
                n_lags=kpss_lags,
                interpretation=interp,
            )
        except Exception as e:
            return OperationError(
                error_type="TestError",
                error_message=f"KPSS test failed: {str(e)}",
                operation="check_stationarity",
                details={"error": str(e)},
            )

    # Determine overall stationarity and recommendation
    if adf_test_result and kpss_test_result:
        # Both tests run - use consensus
        if adf_test_result.is_stationary and kpss_test_result.is_stationary:
            overall_stationary = True
            recommendation = "Series is stationary. Use d=0 for ARIMA."
        elif not adf_test_result.is_stationary and not kpss_test_result.is_stationary:
            overall_stationary = False
            recommendation = "Series is non-stationary. Use d=1 or d=2 for ARIMA."
        elif adf_test_result.is_stationary and not kpss_test_result.is_stationary:
            overall_stationary = False
            recommendation = "Tests disagree (trend-stationary). Consider d=1 for ARIMA."
        else:  # ADF says non-stationary, KPSS says stationary
            overall_stationary = False
            recommendation = "Tests disagree (difference-stationary). Consider d=1 for ARIMA."
        msg = f"ADF: {'stationary' if adf_test_result.is_stationary else 'non-stationary'}, KPSS: {'stationary' if kpss_test_result.is_stationary else 'non-stationary'}. {recommendation}"
    elif adf_test_result:
        overall_stationary = adf_test_result.is_stationary
        recommendation = "No differencing needed." if overall_stationary else "Consider d=1 for ARIMA."
        msg = f"ADF test: {'stationary' if overall_stationary else 'non-stationary'}. {recommendation}"
    else:  # kpss_test_result
        overall_stationary = kpss_test_result.is_stationary
        recommendation = "No differencing needed." if overall_stationary else "Consider d=1 for ARIMA."
        msg = f"KPSS test: {'stationary' if overall_stationary else 'non-stationary'}. {recommendation}"

    return StationarityResult(
        success=True,
        adf_result=adf_test_result,
        kpss_result=kpss_test_result,
        is_stationary=overall_stationary,
        recommendation=recommendation,
        target_column=target_col,
        n_observations=len(series),
        message=msg,
    )


@registry.register(
    category="ml",
    input_schema=InferFrequencyInput,
    description="Infer the frequency of a time series (daily, weekly, monthly, etc.)",
)
def infer_frequency(
    state: DataFrameState, params: InferFrequencyInput
) -> InferFrequencyResult | OperationError:
    """
    Infer the frequency of a time series.

    This function analyzes the date column to determine the typical
    interval between observations (daily, weekly, monthly, etc.).
    It's useful for:
    - Understanding your time series data structure
    - Determining how many forecast steps correspond to a time period
    - Verifying data is properly ordered and spaced

    Args:
        state: DataFrameState containing the data
        params: InferFrequencyInput with column specification

    Returns:
        InferFrequencyResult with frequency details and conversion examples
    """
    # Get DataFrame
    try:
        df = state.get_dataframe(params.dataframe_name)
    except ValueError as e:
        return OperationError(
            error_type="DataFrameNotFound",
            error_message=str(e),
            operation="infer_frequency",
            details={"dataframe_name": params.dataframe_name},
        )

    # Validate date column
    if params.date_column not in df.columns:
        return OperationError(
            error_type="ColumnNotFound",
            error_message=f"Date column '{params.date_column}' not found",
            operation="infer_frequency",
            details={"column": params.date_column, "available": list(df.columns)},
        )

    # Convert to datetime index
    try:
        time_index = pd.DatetimeIndex(pd.to_datetime(df[params.date_column]))
    except Exception as e:
        return OperationError(
            error_type="DateConversionError",
            error_message=f"Could not convert '{params.date_column}' to datetime: {str(e)}",
            operation="infer_frequency",
            details={"column": params.date_column, "error": str(e)},
        )

    if len(time_index) < 2:
        return OperationError(
            error_type="InsufficientData",
            error_message="Need at least 2 observations to infer frequency",
            operation="infer_frequency",
            details={"n_observations": len(time_index)},
        )

    # Infer frequency
    freq = _infer_time_frequency(time_index)
    freq_description = _describe_frequency(freq)
    freq_days = freq.total_seconds() / (24 * 3600)

    # Calculate conversion examples
    conversion_examples = {
        "7 days": _convert_forecast_period_to_steps(time_index, 7, "days"),
        "30 days": _convert_forecast_period_to_steps(time_index, 30, "days"),
        "3 months": _convert_forecast_period_to_steps(time_index, 3, "months"),
        "6 months": _convert_forecast_period_to_steps(time_index, 6, "months"),
        "1 year": _convert_forecast_period_to_steps(time_index, 1, "years"),
    }

    # Date range
    date_range = f"{time_index.min().strftime('%Y-%m-%d')} to {time_index.max().strftime('%Y-%m-%d')}"

    # Build message
    msg = (
        f"Time series frequency: {freq_description}. "
        f"Date range: {date_range} ({len(time_index)} observations). "
        f"To forecast 30 days, use {conversion_examples['30 days']} periods."
    )

    return InferFrequencyResult(
        success=True,
        frequency_description=freq_description,
        frequency_timedelta=str(freq),
        frequency_days=freq_days,
        n_observations=len(time_index),
        date_range=date_range,
        conversion_examples=conversion_examples,
        message=msg,
    )
