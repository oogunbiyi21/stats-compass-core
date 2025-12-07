"""Tests for data tools (load_csv, get_schema, get_sample, list_dataframes)."""

import pandas as pd
import pytest
import tempfile
import os

from stats_compass_core.state import DataFrameState
from stats_compass_core.data.load_csv import load_csv, LoadCSVInput
from stats_compass_core.data.get_schema import get_schema, GetSchemaInput
from stats_compass_core.data.get_sample import get_sample, GetSampleInput
from stats_compass_core.data.list_dataframes import list_dataframes, ListDataFramesInput


class TestLoadCSV:
    """Tests for load_csv tool."""

    @pytest.fixture
    def csv_file(self):
        """Create a temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,age,salary\n")
            f.write("Alice,30,50000\n")
            f.write("Bob,25,45000\n")
            f.write("Charlie,35,60000\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_load_csv_basic(self, csv_file):
        """Test loading a CSV file."""
        state = DataFrameState()
        params = LoadCSVInput(path=csv_file)

        result = load_csv(state, params)

        assert result.success is True
        assert result.shape == (3, 3)
        assert "name" in result.columns
        assert "age" in result.columns
        assert state._active_dataframe is not None

    def test_load_csv_with_custom_name(self, csv_file):
        """Test loading CSV with custom DataFrame name."""
        state = DataFrameState()
        params = LoadCSVInput(path=csv_file, name="my_data")

        result = load_csv(state, params)

        assert result.dataframe_name == "my_data"
        assert state._active_dataframe == "my_data"

    def test_load_csv_json_serializable(self, csv_file):
        """Test that result is JSON serializable."""
        state = DataFrameState()
        params = LoadCSVInput(path=csv_file)

        result = load_csv(state, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)


class TestGetSchema:
    """Tests for get_schema tool."""

    @pytest.fixture
    def state_with_data(self):
        """Create state with test DataFrame."""
        state = DataFrameState()
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
        })
        state.set_dataframe(df, name="test_data", operation="test")
        return state

    def test_get_schema_basic(self, state_with_data):
        """Test getting schema."""
        params = GetSchemaInput()

        result = get_schema(state_with_data, params)

        assert result.shape == (3, 3)
        assert len(result.columns) == 3
        assert result.dataframe_name == "test_data"

    def test_get_schema_column_info(self, state_with_data):
        """Test column info in schema."""
        params = GetSchemaInput(sample_values=2)

        result = get_schema(state_with_data, params)

        # Check column info structure
        col_names = [c["name"] for c in result.columns]
        assert "int_col" in col_names
        assert "float_col" in col_names
        assert "str_col" in col_names

    def test_get_schema_json_serializable(self, state_with_data):
        """Test that result is JSON serializable."""
        params = GetSchemaInput()

        result = get_schema(state_with_data, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)

    def test_get_schema_with_datetime(self):
        """Test schema with datetime columns."""
        state = DataFrameState()
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "value": [1, 2, 3],
        })
        state.set_dataframe(df, name="date_data", operation="test")
        params = GetSchemaInput(sample_values=2)

        result = get_schema(state, params)
        json_str = result.model_dump_json()

        # Should not raise - datetime should be serialized
        assert isinstance(json_str, str)


class TestGetSample:
    """Tests for get_sample tool."""

    @pytest.fixture
    def state_with_data(self):
        """Create state with test DataFrame."""
        state = DataFrameState()
        df = pd.DataFrame({
            "A": range(100),
            "B": range(100, 200),
        })
        state.set_dataframe(df, name="test_data", operation="test")
        return state

    def test_get_sample_basic(self, state_with_data):
        """Test getting sample rows."""
        params = GetSampleInput(n=5)

        result = get_sample(state_with_data, params)

        assert result.sample_size == 5
        assert len(result.data) == 5
        assert result.total_rows == 100

    def test_get_sample_default(self, state_with_data):
        """Test default sample size."""
        params = GetSampleInput()

        result = get_sample(state_with_data, params)

        assert result.sample_size == 10  # Default

    def test_get_sample_json_serializable(self, state_with_data):
        """Test that result is JSON serializable."""
        params = GetSampleInput()

        result = get_sample(state_with_data, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)


class TestListDataFrames:
    """Tests for list_dataframes tool."""

    def test_list_empty(self):
        """Test listing when no DataFrames exist."""
        state = DataFrameState()
        params = ListDataFramesInput()

        result = list_dataframes(state, params)

        assert result.total_count == 0
        assert result.dataframes == []

    def test_list_multiple(self):
        """Test listing multiple DataFrames."""
        state = DataFrameState()
        state.set_dataframe(pd.DataFrame({"A": [1, 2]}), name="df1", operation="test")
        state.set_dataframe(pd.DataFrame({"B": [3, 4]}), name="df2", operation="test")
        params = ListDataFramesInput()

        result = list_dataframes(state, params)

        assert result.total_count == 2
        names = [df["name"] for df in result.dataframes]
        assert "df1" in names
        assert "df2" in names

    def test_list_json_serializable(self):
        """Test that result is JSON serializable."""
        state = DataFrameState()
        state.set_dataframe(pd.DataFrame({"A": [1]}), name="test", operation="test")
        params = ListDataFramesInput()

        result = list_dataframes(state, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)
