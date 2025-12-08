"""
Tests for ARIMA time series tools.

Tests cover:
- fit_arima: Fit ARIMA models
- forecast_arima: Generate forecasts
- find_optimal_arima: Automatic parameter search
- check_stationarity: ADF and KPSS tests
"""

import numpy as np
import pandas as pd
import pytest

from stats_compass_core.ml.arima import (
    FitARIMAInput,
    ForecastARIMAInput,
    FindOptimalARIMAInput,
    StationarityTestInput,
    fit_arima,
    forecast_arima,
    find_optimal_arima,
    check_stationarity,
)
from stats_compass_core.results import (
    ARIMAResult,
    ARIMAForecastResult,
    ARIMAParameterSearchResult,
    OperationError,
)
from stats_compass_core.state import DataFrameState


def create_state_with_timeseries(n: int = 100, trend: bool = False) -> DataFrameState:
    """Create a DataFrameState with time series data."""
    state = DataFrameState()
    np.random.seed(42)
    
    # Create date index
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    
    # Create a time series with some autocorrelation
    values = np.zeros(n)
    values[0] = 100
    for i in range(1, n):
        # AR(1) process with some noise
        values[i] = 0.7 * values[i - 1] + np.random.normal(0, 5)
        if trend:
            values[i] += 0.1 * i  # Add trend
    
    df = pd.DataFrame({"date": dates, "value": values, "other": np.random.randn(n)})
    state.set_dataframe(df, name="timeseries", operation="test")
    return state


class TestFitARIMA:
    """Tests for fit_arima function."""

    def test_fit_arima_basic(self) -> None:
        """Test basic ARIMA model fitting."""
        state = create_state_with_timeseries()
        params = FitARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            p=1,
            d=0,
            q=0,
        )
        
        result = fit_arima(state, params)
        
        assert isinstance(result, ARIMAResult)
        assert result.success is True
        assert result.order == (1, 0, 0)
        assert result.aic is not None
        assert result.bic is not None
        assert result.n_observations == 100
        assert result.model_id is not None
        assert "fitted successfully" in result.message

    def test_fit_arima_with_differencing(self) -> None:
        """Test ARIMA model with differencing."""
        state = create_state_with_timeseries(trend=True)
        params = FitARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            p=1,
            d=1,
            q=1,
        )
        
        result = fit_arima(state, params)
        
        assert isinstance(result, ARIMAResult)
        assert result.success is True
        assert result.order == (1, 1, 1)

    def test_fit_arima_with_date_column(self) -> None:
        """Test ARIMA with date column for index."""
        state = create_state_with_timeseries()
        params = FitARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            date_column="date",
            p=1,
            d=0,
            q=0,
        )
        
        result = fit_arima(state, params)
        
        assert isinstance(result, ARIMAResult)
        assert result.success is True

    def test_fit_arima_custom_model_name(self) -> None:
        """Test ARIMA with custom model name."""
        state = create_state_with_timeseries()
        params = FitARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            p=1,
            d=0,
            q=0,
            model_name="my_arima_model",
        )
        
        result = fit_arima(state, params)
        
        assert isinstance(result, ARIMAResult)
        assert "my_arima_model" in result.model_id

    def test_fit_arima_missing_column(self) -> None:
        """Test error when column doesn't exist."""
        state = create_state_with_timeseries()
        params = FitARIMAInput(
            dataframe_name="timeseries",
            target_column="nonexistent",
            p=1,
            d=0,
            q=0,
        )
        
        result = fit_arima(state, params)
        
        assert isinstance(result, OperationError)
        assert result.error_type == "ColumnNotFound"

    def test_fit_arima_missing_dataframe(self) -> None:
        """Test error when dataframe doesn't exist."""
        state = DataFrameState()
        params = FitARIMAInput(
            dataframe_name="nonexistent",
            target_column="value",
            p=1,
            d=0,
            q=0,
        )
        
        result = fit_arima(state, params)
        
        assert isinstance(result, OperationError)
        assert result.error_type == "DataFrameNotFound"

    def test_fit_arima_insufficient_data(self) -> None:
        """Test error with insufficient data points."""
        state = DataFrameState()
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})  # Only 5 points
        state.set_dataframe(df, name="small", operation="test")
        
        params = FitARIMAInput(
            dataframe_name="small",
            target_column="value",
            p=1,
            d=0,
            q=0,
        )
        
        result = fit_arima(state, params)
        
        assert isinstance(result, OperationError)
        assert result.error_type == "InsufficientData"


class TestForecastARIMA:
    """Tests for forecast_arima function."""

    def test_forecast_basic(self) -> None:
        """Test basic ARIMA forecasting."""
        state = create_state_with_timeseries()
        
        # First fit a model
        fit_params = FitARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            p=1,
            d=0,
            q=0,
        )
        fit_result = fit_arima(state, fit_params)
        assert isinstance(fit_result, ARIMAResult)
        
        # Then forecast
        forecast_params = ForecastARIMAInput(
            model_id=fit_result.model_id,
            n_periods=10,
            include_plot=False,
        )
        
        result = forecast_arima(state, forecast_params)
        
        assert isinstance(result, ARIMAForecastResult)
        assert result.success is True
        assert len(result.forecast_values) == 10
        assert len(result.forecast_index) == 10
        assert result.lower_ci is not None
        assert result.upper_ci is not None
        assert len(result.lower_ci) == 10
        assert result.n_periods == 10

    def test_forecast_with_plot(self) -> None:
        """Test ARIMA forecasting with plot generation."""
        state = create_state_with_timeseries()
        
        # First fit a model
        fit_params = FitARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            p=1,
            d=0,
            q=0,
        )
        fit_result = fit_arima(state, fit_params)
        assert isinstance(fit_result, ARIMAResult)
        
        # Then forecast with plot
        forecast_params = ForecastARIMAInput(
            model_id=fit_result.model_id,
            n_periods=10,
            include_plot=True,
        )
        
        result = forecast_arima(state, forecast_params)
        
        assert isinstance(result, ARIMAForecastResult)
        assert result.image_base64 is not None
        assert len(result.image_base64) > 0

    def test_forecast_custom_confidence(self) -> None:
        """Test ARIMA forecasting with custom confidence level."""
        state = create_state_with_timeseries()
        
        # First fit a model
        fit_params = FitARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            p=1,
            d=0,
            q=0,
        )
        fit_result = fit_arima(state, fit_params)
        assert isinstance(fit_result, ARIMAResult)
        
        # Forecast with 90% confidence
        forecast_params = ForecastARIMAInput(
            model_id=fit_result.model_id,
            n_periods=5,
            confidence_level=0.90,
            include_plot=False,
        )
        
        result = forecast_arima(state, forecast_params)
        
        assert isinstance(result, ARIMAForecastResult)
        assert result.confidence_level == 0.90

    def test_forecast_model_not_found(self) -> None:
        """Test error when model doesn't exist."""
        state = DataFrameState()
        
        params = ForecastARIMAInput(
            model_id="nonexistent_model",
            n_periods=10,
        )
        
        result = forecast_arima(state, params)
        
        assert isinstance(result, OperationError)
        assert result.error_type == "ModelNotFound"


class TestFindOptimalARIMA:
    """Tests for find_optimal_arima function."""

    def test_find_optimal_basic(self) -> None:
        """Test basic automatic parameter search."""
        state = create_state_with_timeseries(n=50)  # Smaller dataset for speed
        params = FindOptimalARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            max_p=2,
            max_d=1,
            max_q=2,
            criterion="aic",
            top_n=3,
        )
        
        result = find_optimal_arima(state, params)
        
        assert isinstance(result, ARIMAParameterSearchResult)
        assert result.success is True
        assert result.best_order is not None
        assert len(result.best_order) == 3
        assert result.models_evaluated > 0
        assert result.search_time_seconds > 0
        assert len(result.top_models) <= 3

    def test_find_optimal_bic_criterion(self) -> None:
        """Test parameter search using BIC criterion."""
        state = create_state_with_timeseries(n=50)
        params = FindOptimalARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            max_p=1,
            max_d=1,
            max_q=1,
            criterion="bic",
        )
        
        result = find_optimal_arima(state, params)
        
        assert isinstance(result, ARIMAParameterSearchResult)
        assert result.success is True
        # BIC should be in the message
        assert "BIC" in result.message

    def test_find_optimal_top_models(self) -> None:
        """Test that top_models are properly sorted."""
        state = create_state_with_timeseries(n=50)
        params = FindOptimalARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            max_p=2,
            max_d=1,
            max_q=2,
            top_n=5,
        )
        
        result = find_optimal_arima(state, params)
        
        assert isinstance(result, ARIMAParameterSearchResult)
        # Verify top models are sorted by AIC (ascending)
        if len(result.top_models) > 1:
            scores = [m["score"] for m in result.top_models]
            assert scores == sorted(scores)


class TestCheckStationarity:
    """Tests for check_stationarity function."""

    def test_stationarity_adf(self) -> None:
        """Test ADF stationarity test."""
        state = create_state_with_timeseries()
        params = StationarityTestInput(
            dataframe_name="timeseries",
            target_column="value",
            test_type="adf",
        )
        
        result = check_stationarity(state, params)
        
        # Should return single result for single test
        from stats_compass_core.ml.arima import StationarityTestResult
        assert isinstance(result, StationarityTestResult)
        assert result.success is True
        assert result.test_type == "adf"
        assert result.test_statistic is not None
        assert result.p_value is not None
        assert result.critical_values is not None
        assert isinstance(result.is_stationary, bool)

    def test_stationarity_kpss(self) -> None:
        """Test KPSS stationarity test."""
        state = create_state_with_timeseries()
        params = StationarityTestInput(
            dataframe_name="timeseries",
            target_column="value",
            test_type="kpss",
        )
        
        result = check_stationarity(state, params)
        
        from stats_compass_core.ml.arima import StationarityTestResult
        assert isinstance(result, StationarityTestResult)
        assert result.success is True
        assert result.test_type == "kpss"

    def test_stationarity_both(self) -> None:
        """Test both ADF and KPSS tests."""
        state = create_state_with_timeseries()
        params = StationarityTestInput(
            dataframe_name="timeseries",
            target_column="value",
            test_type="both",
        )
        
        result = check_stationarity(state, params)
        
        # Should return list with two results
        assert isinstance(result, list)
        assert len(result) == 2
        test_types = {r.test_type for r in result}
        assert test_types == {"adf", "kpss"}

    def test_stationarity_nonstationary_series(self) -> None:
        """Test stationarity on a clearly non-stationary series (random walk)."""
        state = DataFrameState()
        np.random.seed(42)
        
        # Random walk (non-stationary)
        n = 200
        values = np.cumsum(np.random.randn(n))
        df = pd.DataFrame({"value": values})
        state.set_dataframe(df, name="random_walk", operation="test")
        
        params = StationarityTestInput(
            dataframe_name="random_walk",
            target_column="value",
            test_type="adf",
        )
        
        result = check_stationarity(state, params)
        
        from stats_compass_core.ml.arima import StationarityTestResult
        assert isinstance(result, StationarityTestResult)
        # Random walk should typically be non-stationary
        # (though not guaranteed with small samples)
        assert result.interpretation is not None

    def test_stationarity_missing_column(self) -> None:
        """Test error when column doesn't exist."""
        state = create_state_with_timeseries()
        params = StationarityTestInput(
            dataframe_name="timeseries",
            target_column="nonexistent",
            test_type="adf",
        )
        
        result = check_stationarity(state, params)
        
        assert isinstance(result, OperationError)
        assert result.error_type == "ColumnNotFound"


class TestARIMAIntegration:
    """Integration tests for complete ARIMA workflow."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: fit -> forecast."""
        state = create_state_with_timeseries(n=100)
        
        # 1. Test stationarity
        stat_params = StationarityTestInput(
            dataframe_name="timeseries",
            target_column="value",
            test_type="adf",
        )
        stat_result = check_stationarity(state, stat_params)
        from stats_compass_core.ml.arima import StationarityTestResult
        assert isinstance(stat_result, StationarityTestResult)
        
        # 2. Fit ARIMA model
        fit_params = FitARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            p=1,
            d=1 if not stat_result.is_stationary else 0,
            q=1,
        )
        fit_result = fit_arima(state, fit_params)
        assert isinstance(fit_result, ARIMAResult)
        
        # 3. Generate forecast
        forecast_params = ForecastARIMAInput(
            model_id=fit_result.model_id,
            n_periods=7,
            include_plot=True,
        )
        forecast_result = forecast_arima(state, forecast_params)
        assert isinstance(forecast_result, ARIMAForecastResult)
        assert forecast_result.success is True
        assert len(forecast_result.forecast_values) == 7

    def test_parameter_search_and_fit(self) -> None:
        """Test parameter search followed by fitting best model."""
        state = create_state_with_timeseries(n=50)
        
        # 1. Find optimal parameters
        search_params = FindOptimalARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            max_p=2,
            max_d=1,
            max_q=2,
        )
        search_result = find_optimal_arima(state, search_params)
        assert isinstance(search_result, ARIMAParameterSearchResult)
        
        # 2. Fit with best parameters
        best_p, best_d, best_q = search_result.best_order
        fit_params = FitARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            p=best_p,
            d=best_d,
            q=best_q,
        )
        fit_result = fit_arima(state, fit_params)
        assert isinstance(fit_result, ARIMAResult)
        assert fit_result.order == search_result.best_order


# =============================================================================
# Test Frequency Inference
# =============================================================================


class TestInferFrequency:
    """Tests for infer_frequency function."""

    def test_infer_daily_frequency(self) -> None:
        """Test inferring daily frequency from data."""
        from stats_compass_core.ml.arima import infer_frequency, InferFrequencyInput, InferFrequencyResult
        
        state = create_state_with_timeseries(n=100)
        
        params = InferFrequencyInput(
            dataframe_name="timeseries",
            date_column="date",
        )
        
        result = infer_frequency(state, params)
        
        assert isinstance(result, InferFrequencyResult)
        assert result.success is True
        assert result.frequency_description == "daily"
        assert result.frequency_days == pytest.approx(1.0, rel=0.1)
        assert result.n_observations == 100
        assert "30 days" in result.conversion_examples
        assert result.conversion_examples["30 days"] == 30

    def test_infer_weekly_frequency(self) -> None:
        """Test inferring weekly frequency from data."""
        from stats_compass_core.ml.arima import infer_frequency, InferFrequencyInput, InferFrequencyResult
        
        # Create weekly data
        state = DataFrameState()
        dates = pd.date_range(start="2023-01-01", periods=52, freq="W")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.randn(52).cumsum(),
        })
        state.set_dataframe(df, name="weekly", operation="test")
        
        params = InferFrequencyInput(
            dataframe_name="weekly",
            date_column="date",
        )
        
        result = infer_frequency(state, params)
        
        assert isinstance(result, InferFrequencyResult)
        assert result.success is True
        assert result.frequency_description == "weekly"
        assert result.frequency_days == pytest.approx(7.0, rel=0.1)
        # 30 days with weekly data should be ~4 steps
        assert result.conversion_examples["30 days"] in [4, 5]

    def test_infer_frequency_invalid_column(self) -> None:
        """Test error handling for invalid date column."""
        from stats_compass_core.ml.arima import infer_frequency, InferFrequencyInput
        from stats_compass_core.results import OperationError
        
        state = create_state_with_timeseries()
        
        params = InferFrequencyInput(
            dataframe_name="timeseries",
            date_column="nonexistent",
        )
        
        result = infer_frequency(state, params)
        
        assert isinstance(result, OperationError)
        assert result.error_type == "ColumnNotFound"


class TestForecastWithNaturalLanguage:
    """Tests for forecast_arima with natural language period specification."""

    def test_forecast_30_days_daily_data(self) -> None:
        """Test forecasting 30 days on daily data."""
        state = create_state_with_timeseries(n=100)
        
        # Fit model
        fit_params = FitARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            date_column="date",
            p=1,
            d=0,
            q=0,
        )
        fit_result = fit_arima(state, fit_params)
        assert isinstance(fit_result, ARIMAResult)
        
        # Forecast 30 days using natural language
        forecast_params = ForecastARIMAInput(
            model_id=fit_result.model_id,
            forecast_number=30,
            forecast_unit="days",
            include_plot=False,
        )
        
        result = forecast_arima(state, forecast_params)
        
        assert isinstance(result, ARIMAForecastResult)
        assert result.success is True
        # 30 days on daily data = 30 periods
        assert result.n_periods == 30
        assert "30 days" in result.message

    def test_forecast_3_months_daily_data(self) -> None:
        """Test forecasting 3 months on daily data."""
        state = create_state_with_timeseries(n=100)
        
        # Fit model with date column for proper frequency inference
        fit_params = FitARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            date_column="date",
            p=1,
            d=0,
            q=0,
        )
        fit_result = fit_arima(state, fit_params)
        assert isinstance(fit_result, ARIMAResult)
        
        # Forecast 3 months
        forecast_params = ForecastARIMAInput(
            model_id=fit_result.model_id,
            forecast_number=3,
            forecast_unit="months",
            include_plot=False,
        )
        
        result = forecast_arima(state, forecast_params)
        
        assert isinstance(result, ARIMAForecastResult)
        assert result.success is True
        # 3 months ≈ 90 days on daily data
        assert result.n_periods == pytest.approx(90, abs=5)
        assert "3 months" in result.message

    def test_forecast_defaults_to_10_when_no_period_specified(self) -> None:
        """Test that forecast defaults to 10 periods when neither n_periods nor natural language specified."""
        state = create_state_with_timeseries(n=100)
        
        # Fit model
        fit_params = FitARIMAInput(
            dataframe_name="timeseries",
            target_column="value",
            p=1,
            d=0,
            q=0,
        )
        fit_result = fit_arima(state, fit_params)
        assert isinstance(fit_result, ARIMAResult)
        
        # Forecast without specifying any period
        forecast_params = ForecastARIMAInput(
            model_id=fit_result.model_id,
            include_plot=False,
        )
        
        result = forecast_arima(state, forecast_params)
        
        assert isinstance(result, ARIMAForecastResult)
        assert result.success is True
        assert result.n_periods == 10
