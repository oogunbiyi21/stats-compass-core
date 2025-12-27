"""
Time series analysis and forecasting tools.

This module provides tools for time series analysis:

ARIMA:
- fit_arima: Fit ARIMA model to time series data
- forecast_arima: Generate forecasts using fitted ARIMA model
- find_optimal_arima: Automatically find optimal ARIMA parameters
- check_stationarity: Test if a time series is stationary (ADF, KPSS)
- infer_frequency: Infer the frequency of a time series

Note: Requires statsmodels. Install with: pip install stats-compass-core[timeseries]
"""

# Re-export all tools
from stats_compass_core.ml.timeseries.arima import (
    # Functions
    fit_arima,
    forecast_arima,
    find_optimal_arima,
    check_stationarity,
    infer_frequency,
    # Input schemas
    FitARIMAInput,
    ForecastARIMAInput,
    FindOptimalARIMAInput,
    StationarityTestInput,
    InferFrequencyInput,
    # Result types
    StationarityResult,
    StationarityTestResult,
)

__all__ = [
    # Functions
    "fit_arima",
    "forecast_arima",
    "find_optimal_arima",
    "check_stationarity",
    "infer_frequency",
    # Input schemas
    "FitARIMAInput",
    "ForecastARIMAInput",
    "FindOptimalARIMAInput",
    "StationarityTestInput",
    "InferFrequencyInput",
    # Result types
    "StationarityResult",
    "StationarityTestResult",
]
