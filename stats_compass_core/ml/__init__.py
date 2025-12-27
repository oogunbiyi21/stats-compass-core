"""
Machine learning tools.

This module provides atomic ML training tools following the single-responsibility
principle. Each algorithm is exposed as a separate tool for better MCP compatibility
and easier testing/debugging.

Submodules:
- supervised: Classification and regression tools (requires scikit-learn)
- timeseries: ARIMA and time series analysis tools (requires statsmodels)

Supervised Learning (stats_compass_core.ml.supervised):
- train_logistic_regression: Train logistic regression classifier
- train_random_forest_classifier: Train random forest classifier
- train_gradient_boosting_classifier: Train gradient boosting classifier
- train_linear_regression: Train linear regression model
- train_random_forest_regressor: Train random forest regressor
- train_gradient_boosting_regressor: Train gradient boosting regressor
- evaluate_classification_model: Evaluate classifier performance
- evaluate_regression_model: Evaluate regressor performance

Time Series (stats_compass_core.ml.timeseries):
- fit_arima: Fit ARIMA model to time series data
- forecast_arima: Generate forecasts using fitted ARIMA model
- find_optimal_arima: Automatically find optimal ARIMA parameters
- check_stationarity: Test if a time series is stationary
- infer_frequency: Infer the frequency of a time series

Utility:
- save_model: Save a trained model to disk

Note: These tools are automatically discovered by the registry.
Install extras: pip install stats-compass-core[ml,timeseries]
"""

# Re-export timeseries tools for backward compatibility
try:
    from stats_compass_core.ml.timeseries import (
        check_stationarity,
        find_optimal_arima,
        fit_arima,
        forecast_arima,
        infer_frequency,
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
