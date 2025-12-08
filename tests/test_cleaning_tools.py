"""Tests for cleaning tools (drop_na, dedupe, apply_imputation)."""

import pandas as pd
import numpy as np
import pytest

from stats_compass_core.state import DataFrameState
from stats_compass_core.cleaning.dropna import drop_na, DropNAInput
from stats_compass_core.cleaning.dedupe import dedupe, DedupeInput
from stats_compass_core.cleaning.apply_imputation import apply_imputation, ApplyImputationInput


class TestDropNA:
    """Tests for drop_na tool."""

    @pytest.fixture
    def state_with_nulls(self):
        """Create state with DataFrame containing nulls."""
        state = DataFrameState()
        df = pd.DataFrame({
            "A": [1, 2, None, 4],
            "B": [None, 2, 3, 4],
            "C": [1, 2, 3, 4],
        })
        state.set_dataframe(df, name="test_data", operation="test")
        return state

    def test_drop_na_any(self, state_with_nulls):
        """Test dropping rows with any null values."""
        params = DropNAInput(how="any")

        result = drop_na(state_with_nulls, params)

        assert result.success is True
        assert result.rows_before == 4
        assert result.rows_after == 2  # Only rows 1 and 3 have no nulls
        assert result.operation == "drop_na"

    def test_drop_na_all(self):
        """Test dropping rows where all values are null."""
        state = DataFrameState()
        df = pd.DataFrame({
            "A": [1, None, None],
            "B": [None, None, 3],
        })
        state.set_dataframe(df, name="test", operation="test")
        params = DropNAInput(how="all")

        result = drop_na(state, params)

        assert result.rows_after == 2  # One row has all nulls and should be dropped

    def test_drop_na_subset(self, state_with_nulls):
        """Test dropping based on subset of columns."""
        params = DropNAInput(subset=["A"])

        result = drop_na(state_with_nulls, params)

        assert result.rows_after == 3  # Only row 2 has null in A

    def test_drop_na_dataframe_name_is_string(self, state_with_nulls):
        """Test that dataframe_name is a string, not DataFrameInfo."""
        params = DropNAInput()

        result = drop_na(state_with_nulls, params)

        assert isinstance(result.dataframe_name, str)
        assert result.dataframe_name == "test_data"

    def test_drop_na_json_serializable(self, state_with_nulls):
        """Test that result is JSON serializable."""
        params = DropNAInput()

        result = drop_na(state_with_nulls, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)


class TestDedupe:
    """Tests for dedupe tool."""

    @pytest.fixture
    def state_with_duplicates(self):
        """Create state with DataFrame containing duplicates."""
        state = DataFrameState()
        df = pd.DataFrame({
            "A": [1, 1, 2, 2, 3],
            "B": ["x", "x", "y", "z", "w"],
        })
        state.set_dataframe(df, name="test_data", operation="test")
        return state

    def test_dedupe_basic(self, state_with_duplicates):
        """Test basic deduplication."""
        params = DedupeInput()

        result = dedupe(state_with_duplicates, params)

        assert result.success is True
        assert result.rows_before == 5
        assert result.rows_after == 4  # One exact duplicate removed
        assert result.rows_affected == 1

    def test_dedupe_subset(self, state_with_duplicates):
        """Test deduplication on subset of columns."""
        params = DedupeInput(subset=["A"])

        result = dedupe(state_with_duplicates, params)

        assert result.rows_after == 3  # Duplicates in A removed

    def test_dedupe_keep_last(self, state_with_duplicates):
        """Test keeping last duplicate."""
        params = DedupeInput(keep="last")

        result = dedupe(state_with_duplicates, params)

        assert result.success is True

    def test_dedupe_dataframe_name_is_string(self, state_with_duplicates):
        """Test that dataframe_name is a string."""
        params = DedupeInput()

        result = dedupe(state_with_duplicates, params)

        assert isinstance(result.dataframe_name, str)
        assert result.dataframe_name == "test_data"

    def test_dedupe_json_serializable(self, state_with_duplicates):
        """Test that result is JSON serializable."""
        params = DedupeInput()

        result = dedupe(state_with_duplicates, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)


class TestApplyImputation:
    """Tests for apply_imputation tool."""

    @pytest.fixture
    def state_with_nulls(self):
        """Create state with DataFrame containing nulls."""
        state = DataFrameState()
        df = pd.DataFrame({
            "A": [1.0, 2.0, None, 4.0],
            "B": [10.0, None, 30.0, 40.0],
        })
        state.set_dataframe(df, name="test_data", operation="test")
        return state

    def test_imputation_mean(self, state_with_nulls):
        """Test mean imputation."""
        params = ApplyImputationInput(columns=["A"], strategy="mean")

        result = apply_imputation(state_with_nulls, params)

        assert result.success is True
        assert result.rows_affected > 0

    def test_imputation_median(self, state_with_nulls):
        """Test median imputation."""
        params = ApplyImputationInput(columns=["A", "B"], strategy="median")

        result = apply_imputation(state_with_nulls, params)

        assert result.success is True

    def test_imputation_constant(self, state_with_nulls):
        """Test constant value imputation."""
        params = ApplyImputationInput(
            columns=["A"], strategy="constant", fill_value=0
        )

        result = apply_imputation(state_with_nulls, params)

        assert result.success is True

    def test_imputation_has_required_fields(self, state_with_nulls):
        """Test that result has success and message fields."""
        params = ApplyImputationInput(columns=["A"], strategy="mean")

        result = apply_imputation(state_with_nulls, params)

        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert result.success is True
        assert isinstance(result.message, str)

    def test_imputation_json_serializable(self, state_with_nulls):
        """Test that result is JSON serializable."""
        params = ApplyImputationInput(columns=["A"], strategy="mean")

        result = apply_imputation(state_with_nulls, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)


def make_state_with_df(df: pd.DataFrame, name: str = "test") -> DataFrameState:
    """Helper to create state with a DataFrame."""
    state = DataFrameState()
    state.set_dataframe(df, name=name, operation="test_setup")
    return state


class TestOutlierHandling:
    """Tests for outlier handling tool."""

    def test_handle_outliers_cap(self):
        """Test capping outliers at percentile."""
        from stats_compass_core.cleaning.handle_outliers import (
            handle_outliers,
            HandleOutliersInput,
        )

        # Create data with outliers
        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],  # 100 is an outlier
        })

        state = make_state_with_df(df)

        params = HandleOutliersInput(
            dataframe_name="test",
            column="value",
            method="cap",
            percentile=90,
        )

        result = handle_outliers(state, params)

        assert result.success is True
        assert result.method == "cap"
        assert result.values_affected > 0
        
        # Check the data was capped
        df_after = state.get_dataframe("test")
        assert df_after["value"].max() < 100

    def test_handle_outliers_clip_iqr(self):
        """Test clipping outliers using IQR method."""
        from stats_compass_core.cleaning.handle_outliers import (
            handle_outliers,
            HandleOutliersInput,
        )

        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, -50],
        })

        state = make_state_with_df(df)

        params = HandleOutliersInput(
            dataframe_name="test",
            column="value",
            method="clip_iqr",
        )

        result = handle_outliers(state, params)

        assert result.success is True
        assert result.method == "clip_iqr"
        assert result.lower_threshold is not None
        assert result.upper_threshold is not None

    def test_handle_outliers_remove(self):
        """Test removing outlier rows."""
        from stats_compass_core.cleaning.handle_outliers import (
            handle_outliers,
            HandleOutliersInput,
        )

        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        })

        state = make_state_with_df(df)

        params = HandleOutliersInput(
            dataframe_name="test",
            column="value",
            method="remove",
            percentile=90,
        )

        result = handle_outliers(state, params)

        assert result.success is True
        assert result.rows_after < result.rows_before

    def test_handle_outliers_log_transform(self):
        """Test log transformation."""
        from stats_compass_core.cleaning.handle_outliers import (
            handle_outliers,
            HandleOutliersInput,
        )

        df = pd.DataFrame({
            "value": [1, 10, 100, 1000, 10000],
        })

        state = make_state_with_df(df)

        params = HandleOutliersInput(
            dataframe_name="test",
            column="value",
            method="log_transform",
        )

        result = handle_outliers(state, params)

        assert result.success is True
        assert result.method == "log_transform"
        
        # Check values were transformed
        df_after = state.get_dataframe("test")
        assert df_after["value"].max() < 100  # log1p(10000) ≈ 9.2

    def test_handle_outliers_log_transform_negative_fails(self):
        """Test log transformation fails with negative values."""
        from stats_compass_core.cleaning.handle_outliers import (
            handle_outliers,
            HandleOutliersInput,
        )

        df = pd.DataFrame({
            "value": [-1, 0, 1, 2, 3],
        })

        state = make_state_with_df(df)

        params = HandleOutliersInput(
            dataframe_name="test",
            column="value",
            method="log_transform",
        )

        with pytest.raises(ValueError, match="negative values"):
            handle_outliers(state, params)

    def test_handle_outliers_create_new_column(self):
        """Test creating new column instead of modifying original."""
        from stats_compass_core.cleaning.handle_outliers import (
            handle_outliers,
            HandleOutliersInput,
        )

        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        })

        state = make_state_with_df(df)

        params = HandleOutliersInput(
            dataframe_name="test",
            column="value",
            method="cap",
            percentile=90,
            create_new_column=True,
        )

        result = handle_outliers(state, params)

        assert result.result_column == "value_cleaned"
        
        df_after = state.get_dataframe("test")
        assert "value" in df_after.columns
        assert "value_cleaned" in df_after.columns
        assert df_after["value"].max() == 100  # Original unchanged
        assert df_after["value_cleaned"].max() < 100