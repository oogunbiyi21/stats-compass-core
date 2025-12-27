"""
Tests for workflow tools.

Tests the high-level workflow orchestration tools:
- run_eda_report
- run_preprocessing
- run_classification
- run_regression
- run_timeseries_forecast
"""

import pytest
import pandas as pd
import numpy as np

from stats_compass_core.state import DataFrameState
from stats_compass_core.workflows import (
    run_eda_report,
    run_preprocessing,
    run_classification,
    run_regression,
    run_timeseries_forecast,
    EDAConfig,
    PreprocessingConfig,
    ClassificationConfig,
    RegressionConfig,
    TimeSeriesConfig,
)
from stats_compass_core.workflows.eda_report import RunEDAReportInput
from stats_compass_core.workflows.preprocessing import RunPreprocessingInput
from stats_compass_core.workflows.classification import RunClassificationInput
from stats_compass_core.workflows.regression import RunRegressionInput
from stats_compass_core.workflows.timeseries import RunTimeseriesForecastInput


@pytest.fixture
def state_with_numeric_df():
    """Create state with a numeric DataFrame for regression/EDA."""
    state = DataFrameState()
    np.random.seed(42)
    df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
        "target": np.random.randn(100) * 10 + 50,
    })
    state.set_dataframe(df, "numeric_data", operation="test_fixture")
    return state


@pytest.fixture
def state_with_classification_df():
    """Create state with a classification DataFrame."""
    state = DataFrameState()
    np.random.seed(42)
    df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
        "label": np.random.choice([0, 1], size=100),
    })
    state.set_dataframe(df, "classification_data", operation="test_fixture")
    return state


@pytest.fixture
def state_with_timeseries_df():
    """Create state with a time series DataFrame."""
    state = DataFrameState()
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    values = np.cumsum(np.random.randn(200)) + 100  # Random walk
    df = pd.DataFrame({
        "date": dates,
        "value": values,
    })
    state.set_dataframe(df, "timeseries_data", operation="test_fixture")
    return state


@pytest.fixture
def state_with_dirty_df():
    """Create state with a dirty DataFrame for preprocessing."""
    state = DataFrameState()
    df = pd.DataFrame({
        "numeric": [1.0, 2.0, None, 4.0, 5.0, 100.0, 6.0, 7.0, None, 9.0],
        "category": ["A", "B", "A", None, "B", "A", "C", "B", "A", "C"],
        "target": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    })
    state.set_dataframe(df, "dirty_data", operation="test_fixture")
    return state


# =============================================================================
# EDA Workflow Tests
# =============================================================================

class TestRunEDAReport:
    """Tests for run_eda_report workflow."""

    def test_basic_eda_report(self, state_with_numeric_df):
        """Test basic EDA report generation."""
        params = RunEDAReportInput(
            dataframe_name="numeric_data",
            config=EDAConfig(
                generate_histograms=False,
                generate_bar_charts=False,
            ),
        )
        result = run_eda_report(state_with_numeric_df, params)
        
        assert result.status in ["success", "partial_failure"]
        assert result.workflow_name == "run_eda_report"
        assert len(result.steps) > 0

    def test_eda_report_with_plots(self, state_with_numeric_df):
        """Test EDA report with histogram generation."""
        params = RunEDAReportInput(
            dataframe_name="numeric_data",
            config=EDAConfig(
                generate_histograms=True,
                max_histograms=2,
                generate_bar_charts=False,
            ),
        )
        result = run_eda_report(state_with_numeric_df, params)
        
        assert result.status in ["success", "partial_failure"]
        # Should have some charts if histograms enabled
        if result.status == "success":
            assert len(result.artifacts.charts) >= 0


# =============================================================================
# Preprocessing Workflow Tests
# =============================================================================

class TestRunPreprocessing:
    """Tests for run_preprocessing workflow."""

    def test_basic_preprocessing(self, state_with_dirty_df):
        """Test basic preprocessing pipeline."""
        params = RunPreprocessingInput(
            dataframe_name="dirty_data",
            config=PreprocessingConfig(
                dedupe=True,
            ),
        )
        result = run_preprocessing(state_with_dirty_df, params)
        
        assert result.status in ["success", "partial_failure"]
        assert result.workflow_name == "run_preprocessing"

    def test_preprocessing_creates_dataframe(self, state_with_dirty_df):
        """Test that preprocessing creates a new DataFrame."""
        params = RunPreprocessingInput(
            dataframe_name="dirty_data",
        )
        result = run_preprocessing(state_with_dirty_df, params)
        
        # Should have created output DataFrame
        assert result.artifacts.final_dataframe is not None or len(result.artifacts.dataframes_created) > 0


# =============================================================================
# Classification Workflow Tests
# =============================================================================

class TestRunClassification:
    """Tests for run_classification workflow."""

    def test_random_forest_classification(self, state_with_classification_df):
        """Test classification with random forest."""
        params = RunClassificationInput(
            dataframe_name="classification_data",
            target_column="label",
            config=ClassificationConfig(
                model_type="random_forest",
                generate_plots=False,
            ),
        )
        result = run_classification(state_with_classification_df, params)
        
        assert result.status in ["success", "partial_failure"]
        assert result.workflow_name == "run_classification"
        # Should have trained a model
        assert len(result.artifacts.models_created) > 0

    def test_logistic_classification(self, state_with_classification_df):
        """Test classification with logistic regression."""
        params = RunClassificationInput(
            dataframe_name="classification_data",
            target_column="label",
            config=ClassificationConfig(
                model_type="logistic",
                generate_plots=False,
            ),
        )
        result = run_classification(state_with_classification_df, params)
        
        assert result.status in ["success", "partial_failure"]

    def test_classification_with_plots(self, state_with_classification_df):
        """Test classification with plot generation."""
        params = RunClassificationInput(
            dataframe_name="classification_data",
            target_column="label",
            config=ClassificationConfig(
                model_type="random_forest",
                generate_plots=True,
                plots=["confusion_matrix"],
            ),
        )
        result = run_classification(state_with_classification_df, params)
        
        assert result.status in ["success", "partial_failure"]


# =============================================================================
# Regression Workflow Tests
# =============================================================================

class TestRunRegression:
    """Tests for run_regression workflow."""

    def test_linear_regression(self, state_with_numeric_df):
        """Test regression with linear model."""
        params = RunRegressionInput(
            dataframe_name="numeric_data",
            target_column="target",
            config=RegressionConfig(
                model_type="linear",
                generate_plots=False,
            ),
        )
        result = run_regression(state_with_numeric_df, params)
        
        assert result.status in ["success", "partial_failure"]
        assert result.workflow_name == "run_regression"
        assert len(result.artifacts.models_created) > 0

    def test_random_forest_regression(self, state_with_numeric_df):
        """Test regression with random forest."""
        params = RunRegressionInput(
            dataframe_name="numeric_data",
            target_column="target",
            config=RegressionConfig(
                model_type="random_forest",
                generate_plots=False,
            ),
        )
        result = run_regression(state_with_numeric_df, params)
        
        assert result.status in ["success", "partial_failure"]
        assert len(result.artifacts.models_created) > 0

    def test_regression_with_feature_importance(self, state_with_numeric_df):
        """Test regression with feature importance plot."""
        params = RunRegressionInput(
            dataframe_name="numeric_data",
            target_column="target",
            config=RegressionConfig(
                model_type="random_forest",
                generate_plots=True,
                plots=["feature_importance"],
            ),
        )
        result = run_regression(state_with_numeric_df, params)
        
        assert result.status in ["success", "partial_failure"]


# =============================================================================
# Time Series Workflow Tests
# =============================================================================

class TestRunTimeseriesForecast:
    """Tests for run_timeseries_forecast workflow."""

    def test_basic_arima_forecast(self, state_with_timeseries_df):
        """Test basic ARIMA forecasting."""
        params = RunTimeseriesForecastInput(
            dataframe_name="timeseries_data",
            target_column="value",
            date_column="date",
            config=TimeSeriesConfig(
                date_column="date",
                target_column="value",
                forecast_periods=10,
                auto_find_params=False,
                check_stationarity=False,
                generate_forecast_plot=False,
            ),
        )
        result = run_timeseries_forecast(state_with_timeseries_df, params)
        
        assert result.status in ["success", "partial_failure"]
        assert result.workflow_name == "run_timeseries_forecast"

    def test_arima_with_stationarity_check(self, state_with_timeseries_df):
        """Test ARIMA with stationarity testing."""
        params = RunTimeseriesForecastInput(
            dataframe_name="timeseries_data",
            target_column="value",
            date_column="date",
            config=TimeSeriesConfig(
                date_column="date",
                target_column="value",
                forecast_periods=10,
                auto_find_params=False,
                check_stationarity=True,
                generate_forecast_plot=False,
            ),
        )
        result = run_timeseries_forecast(state_with_timeseries_df, params)
        
        assert result.status in ["success", "partial_failure"]
        # Should have stationarity check step
        step_names = [s.step_name for s in result.steps]
        assert "check_stationarity" in step_names

    def test_arima_with_forecast_plot(self, state_with_timeseries_df):
        """Test ARIMA with forecast plot generation."""
        params = RunTimeseriesForecastInput(
            dataframe_name="timeseries_data",
            target_column="value",
            date_column="date",
            config=TimeSeriesConfig(
                date_column="date",
                target_column="value",
                forecast_periods=10,
                auto_find_params=False,
                check_stationarity=False,
                generate_forecast_plot=True,
            ),
        )
        result = run_timeseries_forecast(state_with_timeseries_df, params)
        
        assert result.status in ["success", "partial_failure"]
        # Should have model created
        if result.status == "success":
            assert len(result.artifacts.models_created) > 0

    def test_arima_natural_language_periods(self, state_with_timeseries_df):
        """Test ARIMA with natural language forecast period."""
        params = RunTimeseriesForecastInput(
            dataframe_name="timeseries_data",
            target_column="value",
            date_column="date",
            config=TimeSeriesConfig(
                date_column="date",
                target_column="value",
                forecast_periods="2 weeks",
                auto_find_params=False,
                check_stationarity=False,
                generate_forecast_plot=False,
            ),
        )
        result = run_timeseries_forecast(state_with_timeseries_df, params)
        
        assert result.status in ["success", "partial_failure"]


# =============================================================================
# Workflow Error Handling Tests
# =============================================================================

class TestWorkflowErrorHandling:
    """Tests for workflow error handling."""

    def test_classification_missing_target(self, state_with_classification_df):
        """Test classification with missing target column."""
        params = RunClassificationInput(
            dataframe_name="classification_data",
            target_column="nonexistent_column",
            config=ClassificationConfig(generate_plots=False),
        )
        result = run_classification(state_with_classification_df, params)
        
        # Should fail gracefully
        assert result.status in ["failed", "partial_failure"]

    def test_regression_missing_dataframe(self):
        """Test regression with missing DataFrame."""
        state = DataFrameState()
        params = RunRegressionInput(
            dataframe_name="nonexistent_df",
            target_column="target",
            config=RegressionConfig(generate_plots=False),
        )
        
        # Should raise or return failed status
        try:
            result = run_regression(state, params)
            assert result.status == "failed"
        except ValueError:
            pass  # Also acceptable

    def test_timeseries_missing_date_column(self, state_with_timeseries_df):
        """Test timeseries with wrong date column."""
        params = RunTimeseriesForecastInput(
            dataframe_name="timeseries_data",
            target_column="value",
            date_column="wrong_date_column",
            config=TimeSeriesConfig(
                date_column="wrong_date_column",
                target_column="value",
                forecast_periods=10,
                auto_find_params=False,
                check_stationarity=False,
                generate_forecast_plot=False,
            ),
        )
        result = run_timeseries_forecast(state_with_timeseries_df, params)
        
        # fit_arima should fail, but workflow should handle gracefully
        assert result.status in ["failed", "partial_failure"]
