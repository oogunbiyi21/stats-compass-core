"""Tests for EDA tools (chi-square, data quality, etc.)."""

import numpy as np
import pandas as pd
import pytest

from stats_compass_core.state import DataFrameState


def make_state_with_df(df: pd.DataFrame, name: str = "test") -> DataFrameState:
    """Helper to create state with a DataFrame."""
    state = DataFrameState()
    state.set_dataframe(df, name=name, operation="test_setup")
    return state


class TestChiSquareTests:
    """Tests for chi-square statistical tests."""

    def test_chi_square_independence_basic(self):
        """Test chi-square independence test with basic categorical data."""
        from stats_compass_core.eda.chi_square_tests import (
            chi_square_independence,
            ChiSquareIndependenceInput,
        )

        # Create sample data with association
        np.random.seed(42)
        df = pd.DataFrame({
            "gender": np.random.choice(["M", "F"], size=200),
            "preference": np.random.choice(["A", "B", "C"], size=200),
        })

        state = make_state_with_df(df)

        params = ChiSquareIndependenceInput(
            dataframe_name="test",
            column1="gender",
            column2="preference",
        )

        result = chi_square_independence(state, params)

        assert result.test_type == "independence"
        assert result.chi2_statistic >= 0
        assert 0 <= result.p_value <= 1
        assert result.degrees_of_freedom > 0
        assert result.n_samples == 200
        assert result.effect_size is not None
        assert result.effect_interpretation in ["negligible", "small", "medium", "large"]
        assert isinstance(result.observed_frequencies, dict)
        assert isinstance(result.expected_frequencies, dict)

    def test_chi_square_independence_significant(self):
        """Test chi-square independence with clearly associated data."""
        from stats_compass_core.eda.chi_square_tests import (
            chi_square_independence,
            ChiSquareIndependenceInput,
        )

        # Create data with strong association
        df = pd.DataFrame({
            "treatment": ["A"] * 80 + ["B"] * 80,
            "outcome": ["success"] * 60 + ["failure"] * 20 + ["failure"] * 60 + ["success"] * 20,
        })

        state = make_state_with_df(df)

        params = ChiSquareIndependenceInput(
            dataframe_name="test",
            column1="treatment",
            column2="outcome",
        )

        result = chi_square_independence(state, params)

        # Should be highly significant
        assert result.p_value < 0.001
        assert result.significant_at_05 is True
        assert result.significant_at_01 is True

    def test_chi_square_goodness_of_fit_uniform(self):
        """Test chi-square goodness of fit with uniform distribution."""
        from stats_compass_core.eda.chi_square_tests import (
            chi_square_goodness_of_fit,
            ChiSquareGoodnessOfFitInput,
        )

        # Create data close to uniform distribution
        np.random.seed(42)
        df = pd.DataFrame({
            "category": np.random.choice(["A", "B", "C", "D"], size=400),
        })

        state = make_state_with_df(df)

        params = ChiSquareGoodnessOfFitInput(
            dataframe_name="test",
            column="category",
            expected_frequencies=None,  # Assumes uniform
        )

        result = chi_square_goodness_of_fit(state, params)

        assert result.test_type == "goodness_of_fit"
        assert result.degrees_of_freedom == 3  # 4 categories - 1
        assert result.column1 == "category"
        assert result.column2 is None

    def test_chi_square_goodness_of_fit_custom_expected(self):
        """Test chi-square goodness of fit with custom expected frequencies."""
        from stats_compass_core.eda.chi_square_tests import (
            chi_square_goodness_of_fit,
            ChiSquareGoodnessOfFitInput,
        )

        # Create data that matches expected proportions
        df = pd.DataFrame({
            "grade": ["A"] * 10 + ["B"] * 30 + ["C"] * 40 + ["D"] * 20,
        })

        state = make_state_with_df(df)

        params = ChiSquareGoodnessOfFitInput(
            dataframe_name="test",
            column="grade",
            expected_frequencies=[0.1, 0.3, 0.4, 0.2],  # Expected proportions
        )

        result = chi_square_goodness_of_fit(state, params)

        # Should not be significant (data matches expected)
        assert result.p_value > 0.05

    def test_chi_square_missing_column(self):
        """Test chi-square with missing column raises error."""
        from stats_compass_core.eda.chi_square_tests import (
            chi_square_independence,
            ChiSquareIndependenceInput,
        )

        df = pd.DataFrame({"a": [1, 2, 3]})
        state = make_state_with_df(df)

        params = ChiSquareIndependenceInput(
            dataframe_name="test",
            column1="a",
            column2="nonexistent",
        )

        with pytest.raises(ValueError, match="not found"):
            chi_square_independence(state, params)


class TestDataQuality:
    """Tests for data quality analysis tools."""

    def test_analyze_missing_data_basic(self):
        """Test missing data analysis."""
        from stats_compass_core.eda.data_quality import (
            analyze_missing_data,
            AnalyzeMissingDataInput,
        )

        df = pd.DataFrame({
            "complete": [1, 2, 3, 4, 5],
            "some_missing": [1, None, 3, None, 5],
            "all_missing": [None, None, None, None, None],
        })

        state = make_state_with_df(df)

        params = AnalyzeMissingDataInput(dataframe_name="test")
        result = analyze_missing_data(state, params)

        assert result.total_rows == 5
        assert result.total_columns == 3
        assert "completely_empty_columns" in result.missing_summary
        assert "all_missing" in result.missing_summary["completely_empty_columns"]
        assert len(result.recommendations) > 0

    def test_detect_outliers_iqr(self):
        """Test outlier detection with IQR method."""
        from stats_compass_core.eda.data_quality import (
            detect_outliers,
            DetectOutliersInput,
        )

        df = pd.DataFrame({
            "normal": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "with_outliers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        })

        state = make_state_with_df(df)

        params = DetectOutliersInput(
            dataframe_name="test",
            method="iqr",
        )

        result = detect_outliers(state, params)

        assert result.outlier_summary is not None
        assert result.outlier_summary["method"] == "iqr"
        assert "with_outliers" in result.outlier_summary["columns_with_outliers"]

    def test_detect_outliers_zscore(self):
        """Test outlier detection with z-score method."""
        from stats_compass_core.eda.data_quality import (
            detect_outliers,
            DetectOutliersInput,
        )

        # Create data with clear outlier
        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        })

        state = make_state_with_df(df)

        params = DetectOutliersInput(
            dataframe_name="test",
            method="zscore",
        )

        result = detect_outliers(state, params)

        assert result.outlier_summary["method"] == "zscore"

    def test_data_quality_report_comprehensive(self):
        """Test comprehensive data quality report."""
        from stats_compass_core.eda.data_quality import (
            data_quality_report,
            DataQualityReportInput,
        )

        df = pd.DataFrame({
            "complete": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "some_missing": [1, None, 3, None, 5, 6, 7, 8, 9, 10],
            "with_outliers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
            "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        })
        # Add duplicate rows
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        state = make_state_with_df(df)

        params = DataQualityReportInput(
            dataframe_name="test",
            include_outliers=True,
            outlier_method="iqr",
        )

        result = data_quality_report(state, params)

        assert result.missing_summary is not None
        assert result.outlier_summary is not None
        assert result.quality_score is not None
        assert 0 <= result.quality_score <= 100
        assert len(result.recommendations) > 0

    def test_data_quality_report_no_outliers(self):
        """Test data quality report without outlier analysis."""
        from stats_compass_core.eda.data_quality import (
            data_quality_report,
            DataQualityReportInput,
        )

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })

        state = make_state_with_df(df)

        params = DataQualityReportInput(
            dataframe_name="test",
            include_outliers=False,
        )

        result = data_quality_report(state, params)

        assert result.outlier_summary is None
        assert result.missing_summary is not None
