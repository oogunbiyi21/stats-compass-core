"""
ARIMA time series modeling tools.

Requires the [timeseries] extra: pip install stats-compass-core[timeseries]
"""

import base64
import io
import itertools
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

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


class FitARIMAInput(BaseModel):
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
    seasonal_order: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Seasonal order (P, D, Q, m) where m is the seasonal period",
    )
    model_name: str | None = Field(
        default=None, description="Custom name for storing the model"
    )


class ForecastARIMAInput(BaseModel):
    """Input parameters for ARIMA forecasting."""

    model_id: str = Field(description="ID of the fitted ARIMA model to use")
    n_periods: int = Field(
        default=10, ge=1, le=365, description="Number of periods to forecast"
    )
    confidence_level: float = Field(
        default=0.95, ge=0.5, le=0.99, description="Confidence level for intervals"
    )
    include_plot: bool = Field(
        default=True, description="Whether to generate a forecast plot"
    )


class FindOptimalARIMAInput(BaseModel):
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


class StationarityTestInput(BaseModel):
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


# ---------------------------------------------------------------------------
# Result Models for Stationarity Tests
# ---------------------------------------------------------------------------


class StationarityTestResult(BaseModel):
    """Result for stationarity tests."""

    success: bool = Field(description="Whether the test succeeded")
    operation: str = Field(default="stationarity_test", description="Operation performed")

    # Test results
    test_type: str = Field(description="Type of test performed")
    test_statistic: float = Field(description="Test statistic value")
    p_value: float = Field(description="P-value of the test")
    critical_values: dict[str, float] = Field(
        description="Critical values at different significance levels"
    )
    is_stationary: bool = Field(
        description="Whether the series is stationary according to the test"
    )

    # Additional info
    n_lags: int | None = Field(default=None, description="Number of lags used")

    # Interpretation
    interpretation: str = Field(description="Human-readable interpretation of results")


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
        if dataframe_name:
            df = state.get_dataframe(dataframe_name)
        else:
            df = state.get_active_dataframe()
            dataframe_name = state._active_df_name
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
        series.index = pd.to_datetime(df[date_column])

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


# ---------------------------------------------------------------------------
# Tool Functions
# ---------------------------------------------------------------------------


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
        seasonal_order = params.seasonal_order

    try:
        # Fit ARIMA model
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


def forecast_arima(
    state: DataFrameState, params: ForecastARIMAInput
) -> ARIMAForecastResult | OperationError:
    """
    Generate forecasts using a fitted ARIMA model.

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
        # Generate forecast
        forecast_result = model.get_forecast(steps=params.n_periods)
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
                title=f"ARIMA Forecast ({params.n_periods} periods)",
            )

        # Create summary message
        msg = (
            f"Generated {params.n_periods}-period forecast. "
            f"Forecast range: {forecast_values[0]:.2f} to {forecast_values[-1]:.2f}"
        )

        return ARIMAForecastResult(
            success=True,
            forecast_values=forecast_values,
            forecast_index=forecast_index,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            confidence_level=params.confidence_level,
            n_periods=params.n_periods,
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


def check_stationarity(
    state: DataFrameState, params: StationarityTestInput
) -> StationarityTestResult | list[StationarityTestResult] | OperationError:
    """
    Test if a time series is stationary using ADF and/or KPSS tests.

    Stationarity is important for ARIMA modeling. A stationary series has
    constant mean, variance, and autocorrelation over time.

    - ADF test: Null hypothesis is that the series has a unit root (non-stationary)
    - KPSS test: Null hypothesis is that the series is stationary

    Args:
        state: DataFrameState containing the data
        params: StationarityTestInput with test configuration

    Returns:
        StationarityTestResult or list of results if both tests requested
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

    series, _, _ = result

    results: list[StationarityTestResult] = []

    # ADF Test
    if params.test_type in ("adf", "both"):
        try:
            adf_result = adfuller(series, autolag="AIC")
            adf_statistic = float(adf_result[0])
            adf_pvalue = float(adf_result[1])
            adf_lags = int(adf_result[2])
            adf_critical = {k: float(v) for k, v in adf_result[4].items()}

            # Series is stationary if p-value < 0.05 (reject null of unit root)
            is_stationary = adf_pvalue < 0.05

            if is_stationary:
                interp = (
                    f"ADF test statistic: {adf_statistic:.4f} (p-value: {adf_pvalue:.4f}). "
                    f"The series IS stationary (p < 0.05). No differencing needed."
                )
            else:
                interp = (
                    f"ADF test statistic: {adf_statistic:.4f} (p-value: {adf_pvalue:.4f}). "
                    f"The series is NOT stationary (p >= 0.05). Consider differencing (d >= 1)."
                )

            results.append(
                StationarityTestResult(
                    success=True,
                    test_type="adf",
                    test_statistic=adf_statistic,
                    p_value=adf_pvalue,
                    critical_values=adf_critical,
                    is_stationary=is_stationary,
                    n_lags=adf_lags,
                    interpretation=interp,
                )
            )
        except Exception as e:
            return OperationError(
                error_type="TestError",
                error_message=f"ADF test failed: {str(e)}",
                operation="test_stationarity",
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
            is_stationary = kpss_pvalue > 0.05

            if is_stationary:
                interp = (
                    f"KPSS test statistic: {kpss_statistic:.4f} (p-value: {kpss_pvalue:.4f}). "
                    f"The series IS stationary (p > 0.05). No differencing needed."
                )
            else:
                interp = (
                    f"KPSS test statistic: {kpss_statistic:.4f} (p-value: {kpss_pvalue:.4f}). "
                    f"The series is NOT stationary (p <= 0.05). Consider differencing."
                )

            results.append(
                StationarityTestResult(
                    success=True,
                    test_type="kpss",
                    test_statistic=kpss_statistic,
                    p_value=kpss_pvalue,
                    critical_values=kpss_critical,
                    is_stationary=is_stationary,
                    n_lags=kpss_lags,
                    interpretation=interp,
                )
            )
        except Exception as e:
            return OperationError(
                error_type="TestError",
                error_message=f"KPSS test failed: {str(e)}",
                operation="test_stationarity",
                details={"error": str(e)},
            )

    # Return single result or list
    if len(results) == 1:
        return results[0]
    return results
