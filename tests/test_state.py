"""Tests for DataFrameState."""

import pandas as pd
import pytest

from stats_compass_core.state import DataFrameState

# Check if sklearn is available
try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TestDataFrameState:
    """Test suite for DataFrameState."""

    def test_set_and_get_dataframe(self):
        """Test storing and retrieving a DataFrame."""
        state = DataFrameState()
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        stored_name = state.set_dataframe(df, name="test_df", operation="test")

        assert stored_name == "test_df"
        info = state.get_dataframe_info("test_df")
        assert info.shape == (3, 2)
        retrieved = state.get_dataframe("test_df")
        pd.testing.assert_frame_equal(retrieved, df)

    def test_active_dataframe(self):
        """Test active DataFrame management."""
        state = DataFrameState()
        df = pd.DataFrame({"A": [1, 2, 3]})

        state.set_dataframe(df, name="df1", operation="test")

        assert state._active_dataframe == "df1"
        # Get without name should return active
        retrieved = state.get_dataframe()
        pd.testing.assert_frame_equal(retrieved, df)

    def test_multiple_dataframes(self):
        """Test managing multiple DataFrames."""
        state = DataFrameState()
        df1 = pd.DataFrame({"A": [1, 2, 3]})
        df2 = pd.DataFrame({"X": [10, 20]})

        state.set_dataframe(df1, name="first", operation="test")
        state.set_dataframe(df2, name="second", operation="test")

        assert len(state.list_dataframes()) == 2
        pd.testing.assert_frame_equal(state.get_dataframe("first"), df1)
        pd.testing.assert_frame_equal(state.get_dataframe("second"), df2)

    def test_dataframe_not_found(self):
        """Test error when DataFrame doesn't exist."""
        state = DataFrameState()

        with pytest.raises(ValueError, match="not found"):
            state.get_dataframe("nonexistent")

    def test_remove_dataframe(self):
        """Test removing a DataFrame."""
        state = DataFrameState()
        df = pd.DataFrame({"A": [1, 2, 3]})
        state.set_dataframe(df, name="to_remove", operation="test")

        state.remove_dataframe("to_remove")

        with pytest.raises(ValueError, match="not found"):
            state.get_dataframe("to_remove")

    def test_memory_limit(self):
        """Test memory limit enforcement."""
        state = DataFrameState(memory_limit_mb=0.001)  # Very small limit
        # Create a DataFrame that exceeds the limit
        df = pd.DataFrame({"A": list(range(10000))})

        with pytest.raises(MemoryError, match="exceed memory limit"):
            state.set_dataframe(df, name="big", operation="test")

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_store_and_get_model(self):
        """Test storing and retrieving a model."""
        state = DataFrameState()
        model = LinearRegression()

        model_id = state.store_model(
            model=model,
            model_type="linear_regression",
            target_column="y",
            feature_columns=["x1", "x2"],
            source_dataframe="train_data",
        )

        assert model_id is not None
        retrieved = state.get_model(model_id)
        assert retrieved is model

    def test_get_model_not_found(self):
        """Test getting nonexistent model returns None."""
        state = DataFrameState()

        result = state.get_model("nonexistent_model")

        assert result is None

    def test_list_dataframes_returns_info(self):
        """Test list_dataframes returns DataFrameInfo objects."""
        state = DataFrameState()
        df = pd.DataFrame({"A": [1, 2, 3]})
        state.set_dataframe(df, name="test", operation="test")

        infos = state.list_dataframes()

        assert len(infos) == 1
        assert infos[0].name == "test"
        assert infos[0].shape == (3, 1)

    def test_get_total_memory_mb(self):
        """Test total memory calculation."""
        state = DataFrameState()
        df = pd.DataFrame({"A": [1, 2, 3]})
        state.set_dataframe(df, name="test", operation="test")

        memory = state.get_total_memory_mb()

        assert memory > 0
        assert isinstance(memory, float)
