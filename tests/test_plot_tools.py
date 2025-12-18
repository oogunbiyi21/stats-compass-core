"""Tests for plot tools."""

import base64

import pandas as pd
import pytest

from stats_compass_core.plots.bar_chart import BarChartInput, bar_chart
from stats_compass_core.plots.histogram import HistogramInput, histogram
from stats_compass_core.plots.scatter_plot import ScatterPlotInput, scatter_plot
from stats_compass_core.state import DataFrameState


class TestHistogram:
    """Tests for histogram tool."""

    @pytest.fixture
    def state_with_data(self):
        """Create state with test DataFrame."""
        state = DataFrameState()
        df = pd.DataFrame({
            "values": [1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
            "category": ["A", "A", "B", "B", "C", "C", "C", "D", "D", "D"],
        })
        state.set_dataframe(df, name="test_data", operation="test")
        return state

    def test_histogram_basic(self, state_with_data):
        """Test basic histogram creation."""
        params = HistogramInput(column="values")

        result = histogram(state_with_data, params)

        assert result.chart_type == "histogram"
        assert result.image_format == "png"
        assert len(result.image_base64) > 0

    def test_histogram_valid_base64(self, state_with_data):
        """Test that image is valid base64."""
        params = HistogramInput(column="values")

        result = histogram(state_with_data, params)

        # Should decode without error
        decoded = base64.b64decode(result.image_base64)
        # PNG magic bytes
        assert decoded[:4] == b"\x89PNG"

    def test_histogram_json_serializable(self, state_with_data):
        """Test that result is JSON serializable."""
        params = HistogramInput(column="values")

        result = histogram(state_with_data, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)


class TestBarChart:
    """Tests for bar_chart tool."""

    @pytest.fixture
    def state_with_data(self):
        """Create state with categorical data."""
        state = DataFrameState()
        df = pd.DataFrame({
            "category": ["A", "A", "A", "B", "B", "C"],
        })
        state.set_dataframe(df, name="test_data", operation="test")
        return state

    def test_bar_chart_basic(self, state_with_data):
        """Test basic bar chart creation."""
        params = BarChartInput(column="category")

        result = bar_chart(state_with_data, params)

        assert result.chart_type == "bar_chart"
        assert result.image_format == "png"

    def test_bar_chart_metadata_field(self, state_with_data):
        """Test that metadata field is used, not parameters."""
        params = BarChartInput(column="category")

        result = bar_chart(state_with_data, params)

        assert hasattr(result, "metadata")
        assert "column" in result.metadata
        assert result.metadata["column"] == "category"

    def test_bar_chart_json_serializable(self, state_with_data):
        """Test that result is JSON serializable."""
        params = BarChartInput(column="category")

        result = bar_chart(state_with_data, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)


class TestScatterPlot:
    """Tests for scatter_plot tool."""

    @pytest.fixture
    def state_with_data(self):
        """Create state with numeric data."""
        state = DataFrameState()
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "group": ["A", "A", "B", "B", "B"],
        })
        state.set_dataframe(df, name="test_data", operation="test")
        return state

    def test_scatter_basic(self, state_with_data):
        """Test basic scatter plot creation."""
        params = ScatterPlotInput(x="x", y="y")

        result = scatter_plot(state_with_data, params)

        assert result.chart_type == "scatter_plot"
        assert result.image_format == "png"

    def test_scatter_with_hue(self, state_with_data):
        """Test scatter plot with color grouping."""
        params = ScatterPlotInput(x="x", y="y", hue="group")

        result = scatter_plot(state_with_data, params)

        assert result.metadata["hue"] == "group"

    def test_scatter_metadata_field(self, state_with_data):
        """Test that metadata field is used, not parameters."""
        params = ScatterPlotInput(x="x", y="y")

        result = scatter_plot(state_with_data, params)

        assert hasattr(result, "metadata")
        assert "x" in result.metadata
        assert "y" in result.metadata

    def test_scatter_json_serializable(self, state_with_data):
        """Test that result is JSON serializable."""
        params = ScatterPlotInput(x="x", y="y")

        result = scatter_plot(state_with_data, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)


# Import numpy for classification curve tests
import numpy as np


def make_state_with_df(df: pd.DataFrame, name: str = "test") -> DataFrameState:
    """Helper to create state with a DataFrame."""
    state = DataFrameState()
    state.set_dataframe(df, name=name, operation="test_setup")
    return state


class TestClassificationCurves:
    """Tests for ROC and PR curve tools."""

    def test_roc_curve_basic(self):
        """Test ROC curve generation."""
        from stats_compass_core.plots.classification_curves import (
            ROCCurveInput,
            roc_curve_plot,
        )

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "y_true": np.random.randint(0, 2, n),
            "y_prob": np.random.random(n),
        })

        state = make_state_with_df(df)

        params = ROCCurveInput(
            dataframe_name="test",
            true_column="y_true",
            prob_column="y_prob",
            model_id="test_model",
        )

        result = roc_curve_plot(state, params)

        assert result.curve_type == "roc"
        assert len(result.x_values) > 0
        assert len(result.y_values) > 0
        assert 0 <= result.auc_score <= 1
        assert result.image_base64  # Should have base64 image
        assert result.model_id == "test_model"

    def test_roc_curve_perfect_classifier(self):
        """Test ROC curve with perfect classifier."""
        from stats_compass_core.plots.classification_curves import (
            ROCCurveInput,
            roc_curve_plot,
        )

        df = pd.DataFrame({
            "y_true": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "y_prob": [0.1, 0.2, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
        })

        state = make_state_with_df(df)

        params = ROCCurveInput(
            dataframe_name="test",
            true_column="y_true",
            prob_column="y_prob",
        )

        result = roc_curve_plot(state, params)

        # Perfect classifier should have AUC = 1.0
        assert result.auc_score == 1.0

    def test_precision_recall_curve_basic(self):
        """Test PR curve generation."""
        from stats_compass_core.plots.classification_curves import (
            PrecisionRecallCurveInput,
            precision_recall_curve_plot,
        )

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "y_true": np.random.randint(0, 2, n),
            "y_prob": np.random.random(n),
        })

        state = make_state_with_df(df)

        params = PrecisionRecallCurveInput(
            dataframe_name="test",
            true_column="y_true",
            prob_column="y_prob",
        )

        result = precision_recall_curve_plot(state, params)

        assert result.curve_type == "precision_recall"
        assert len(result.x_values) > 0
        assert len(result.y_values) > 0
        assert 0 <= result.auc_score <= 1
        assert result.image_base64

    def test_curve_missing_column(self):
        """Test curve tools raise error for missing columns."""
        from stats_compass_core.plots.classification_curves import (
            ROCCurveInput,
            roc_curve_plot,
        )

        df = pd.DataFrame({"y_true": [0, 1, 0, 1]})
        state = make_state_with_df(df)

        params = ROCCurveInput(
            dataframe_name="test",
            true_column="y_true",
            prob_column="nonexistent",
        )

        with pytest.raises(ValueError, match="not found"):
            roc_curve_plot(state, params)


class TestConfusionMatrix:
    """Tests for confusion_matrix_plot tool."""

    def test_confusion_matrix_basic(self):
        """Test basic confusion matrix generation."""
        from stats_compass_core.plots.confusion_matrix import (
            ConfusionMatrixInput,
            confusion_matrix_plot,
        )

        df = pd.DataFrame({
            "y_true": [0, 0, 0, 1, 1, 1, 0, 1],
            "y_pred": [0, 0, 1, 1, 1, 0, 0, 1],
        })
        state = make_state_with_df(df)

        params = ConfusionMatrixInput(
            dataframe_name="test",
            true_column="y_true",
            pred_column="y_pred",
        )

        result = confusion_matrix_plot(state, params)

        assert result.chart_type == "confusion_matrix"
        assert result.image_base64 is not None
        assert len(result.image_base64) > 0

    def test_confusion_matrix_multiclass(self):
        """Test confusion matrix with multiple classes."""
        from stats_compass_core.plots.confusion_matrix import (
            ConfusionMatrixInput,
            confusion_matrix_plot,
        )

        df = pd.DataFrame({
            "y_true": [0, 0, 1, 1, 2, 2, 0, 1, 2],
            "y_pred": [0, 1, 1, 1, 2, 0, 0, 1, 2],
        })
        state = make_state_with_df(df)

        params = ConfusionMatrixInput(
            dataframe_name="test",
            true_column="y_true",
            pred_column="y_pred",
        )

        result = confusion_matrix_plot(state, params)

        # Should have metadata with metrics for 3 classes
        assert result.metadata is not None
        assert result.metadata.get("n_classes") == 3

    def test_confusion_matrix_with_normalization(self):
        """Test confusion matrix with normalization."""
        from stats_compass_core.plots.confusion_matrix import (
            ConfusionMatrixInput,
            confusion_matrix_plot,
        )

        df = pd.DataFrame({
            "y_true": [0, 0, 0, 1, 1, 1],
            "y_pred": [0, 0, 1, 1, 1, 0],
        })
        state = make_state_with_df(df)

        params = ConfusionMatrixInput(
            dataframe_name="test",
            true_column="y_true",
            pred_column="y_pred",
            normalize="true",  # Normalize by true labels
        )

        result = confusion_matrix_plot(state, params)

        assert result.chart_type == "confusion_matrix"
        assert result.image_base64 is not None

    def test_confusion_matrix_json_format(self):
        """Test confusion matrix with JSON output format."""
        from stats_compass_core.plots.confusion_matrix import (
            ConfusionMatrixInput,
            confusion_matrix_plot,
        )

        df = pd.DataFrame({
            "y_true": [0, 0, 1, 1],
            "y_pred": [0, 1, 1, 1],
        })
        state = make_state_with_df(df)

        params = ConfusionMatrixInput(
            dataframe_name="test",
            true_column="y_true",
            pred_column="y_pred",
            format="json",
        )

        result = confusion_matrix_plot(state, params)

        # JSON format should have data with matrix
        assert result.data is not None
        assert "confusion_matrix" in result.data
        # No image in JSON mode
        assert result.image_base64 is None

    def test_confusion_matrix_json_serializable(self):
        """Test that confusion matrix result is JSON serializable."""
        from stats_compass_core.plots.confusion_matrix import (
            ConfusionMatrixInput,
            confusion_matrix_plot,
        )

        df = pd.DataFrame({
            "y_true": [0, 0, 1, 1],
            "y_pred": [0, 1, 1, 1],
        })
        state = make_state_with_df(df)

        params = ConfusionMatrixInput(
            dataframe_name="test",
            true_column="y_true",
            pred_column="y_pred",
        )

        result = confusion_matrix_plot(state, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)
