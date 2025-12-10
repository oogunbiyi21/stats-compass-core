"""
Machine learning tools.

This module provides atomic ML training tools following the single-responsibility
principle. Each algorithm is exposed as a separate tool for better MCP compatibility
and easier testing/debugging.

Available tools (one per algorithm):
- train_logistic_regression: Train logistic regression classifier
- train_random_forest_classifier: Train random forest classifier
- train_gradient_boosting_classifier: Train gradient boosting classifier
- train_linear_regression: Train linear regression model
- train_random_forest_regressor: Train random forest regressor
- train_gradient_boosting_regressor: Train gradient boosting regressor

Time Series tools:
- fit_arima: Fit ARIMA model to time series data
- forecast_arima: Generate forecasts using fitted ARIMA model
- find_optimal_arima: Automatically find optimal ARIMA parameters
- check_stationarity: Test if a time series is stationary
- infer_frequency: Infer the frequency of a time series

Legacy files (deprecated, kept for backward compatibility):
- _deprecated_train_classifier.py: Old multi-algorithm classifier
- _deprecated_train_regressor.py: Old multi-algorithm regressor

Note: These tools are automatically discovered by the registry.
Import scikit-learn separately with: pip install stats-compass-core[ml]
Import statsmodels separately with: pip install stats-compass-core[timeseries]
"""

# ARIMA tools - available when statsmodels is installed
try:
    from stats_compass_core.ml.arima import (
        check_stationarity,
        find_optimal_arima,
        fit_arima,
        forecast_arima,
        infer_frequency,
        # Result types for type hints
        StationarityResult,
        StationarityTestResult,
    )

    __all__ = [
        "fit_arima",
        "forecast_arima",
        "find_optimal_arima",
        "check_stationarity",
        "infer_frequency",
        "StationarityResult",
        "StationarityTestResult",
    ]
except ImportError:
    # statsmodels not available
    __all__ = []
