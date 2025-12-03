"""Tests for the describe EDA tool."""

import pandas as pd
import pytest

from stats_compass_core.eda.describe import DescribeInput, describe


class TestDescribe:
    """Test suite for describe tool."""

    def test_basic_describe(self) -> None:
        """Test basic describe functionality."""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})

        params = DescribeInput()
        result = describe(df, params)

        assert isinstance(result, pd.DataFrame)
        assert "A" in result.columns
        assert "B" in result.columns
        assert "mean" in result.index
        assert "std" in result.index
        assert result.loc["mean", "A"] == 3.0
        assert result.loc["mean", "B"] == 30.0

    def test_describe_with_percentiles(self) -> None:
        """Test describe with custom percentiles."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        params = DescribeInput(percentiles=[0.1, 0.5, 0.9])
        result = describe(df, params)

        assert "10%" in result.index
        assert "50%" in result.index
        assert "90%" in result.index

    def test_describe_include_all(self) -> None:
        """Test describe with include='all'."""
        df = pd.DataFrame(
            {
                "numeric": [1, 2, 3],
                "string": ["a", "b", "c"],
                "category": pd.Categorical(["x", "y", "z"]),
            }
        )

        params = DescribeInput(include="all")
        result = describe(df, params)

        assert "numeric" in result.columns
        assert "string" in result.columns
        assert "category" in result.columns

    def test_describe_include_numeric_only(self) -> None:
        """Test describe with numeric columns only."""
        df = pd.DataFrame(
            {"numeric1": [1, 2, 3], "numeric2": [4, 5, 6], "string": ["a", "b", "c"]}
        )

        params = DescribeInput(include="number")
        result = describe(df, params)

        assert "numeric1" in result.columns
        assert "numeric2" in result.columns
        assert "string" not in result.columns

    def test_describe_exclude_numeric(self) -> None:
        """Test describe excluding numeric columns."""
        df = pd.DataFrame({"numeric": [1, 2, 3], "string": ["a", "b", "c"]})

        params = DescribeInput(exclude="number")
        result = describe(df, params)

        assert "numeric" not in result.columns
        assert "string" in result.columns

    def test_describe_invalid_percentile(self) -> None:
        """Test error with invalid percentile values."""
        df = pd.DataFrame({"A": [1, 2, 3]})

        # Percentile > 1
        params = DescribeInput(percentiles=[0.5, 1.5])
        with pytest.raises(ValueError, match="Percentiles must be between 0 and 1"):
            describe(df, params)

        # Percentile < 0
        params = DescribeInput(percentiles=[-0.1, 0.5])
        with pytest.raises(ValueError, match="Percentiles must be between 0 and 1"):
            describe(df, params)

    def test_describe_empty_dataframe(self) -> None:
        """Test describe with empty DataFrame."""
        df = pd.DataFrame(columns=["A", "B"])

        params = DescribeInput()
        result = describe(df, params)

        assert isinstance(result, pd.DataFrame)

    def test_describe_with_na_values(self) -> None:
        """Test describe with NA values."""
        df = pd.DataFrame({"A": [1, 2, None, 4, 5], "B": [10, None, 30, 40, 50]})

        params = DescribeInput()
        result = describe(df, params)

        assert isinstance(result, pd.DataFrame)
        # describe() automatically excludes NA values
        assert result.loc["count", "A"] == 4.0
        assert result.loc["count", "B"] == 4.0

    def test_describe_mixed_types(self) -> None:
        """Test describe with mixed data types."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )

        # Default behavior (numeric only)
        params = DescribeInput()
        result = describe(df, params)

        assert "int_col" in result.columns
        assert "float_col" in result.columns
        # bool is treated as numeric by pandas
        assert "str_col" not in result.columns
