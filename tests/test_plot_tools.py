"""Tests for plot tools."""

import pandas as pd
import pytest
import base64

from stats_compass_core.state import DataFrameState
from stats_compass_core.plots.histogram import histogram, HistogramInput
from stats_compass_core.plots.bar_chart import bar_chart, BarChartInput
from stats_compass_core.plots.scatter_plot import scatter_plot, ScatterPlotInput


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
