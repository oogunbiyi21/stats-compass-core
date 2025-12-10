"""Tests for data tools (load_csv, get_schema, get_sample, list_dataframes, merge, concat, column operations)."""

import os
import tempfile

import pandas as pd
import pytest

from stats_compass_core.data.add_column import AddColumnInput, add_column
from stats_compass_core.data.concat_dataframes import (
    ConcatDataFramesInput,
    concat_dataframes,
)
from stats_compass_core.data.drop_columns import DropColumnsInput, drop_columns
from stats_compass_core.data.get_sample import GetSampleInput, get_sample
from stats_compass_core.data.get_schema import GetSchemaInput, get_schema
from stats_compass_core.data.list_dataframes import ListDataFramesInput, list_dataframes
from stats_compass_core.data.load_csv import LoadCSVInput, load_csv
from stats_compass_core.data.merge_dataframes import (
    MergeDataFramesInput,
    merge_dataframes,
)
from stats_compass_core.data.rename_columns import RenameColumnsInput, rename_columns
from stats_compass_core.state import DataFrameState


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


class TestMergeDataFrames:
    """Tests for merge_dataframes tool."""

    @pytest.fixture
    def state_with_two_dfs(self):
        """Create state with two DataFrames for merge testing."""
        state = DataFrameState()

        # Left DataFrame: customers
        left_df = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "Diana"],
        })
        state.set_dataframe(left_df, name="customers", operation="test")

        # Right DataFrame: orders
        right_df = pd.DataFrame({
            "customer_id": [1, 2, 2, 5],
            "order_id": [101, 102, 103, 104],
            "amount": [100.0, 200.0, 150.0, 300.0],
        })
        state.set_dataframe(right_df, name="orders", operation="test")

        return state

    def test_merge_inner_join(self, state_with_two_dfs):
        """Test inner join - only matching rows."""
        params = MergeDataFramesInput(
            left_dataframe="customers",
            right_dataframe="orders",
            how="inner",
            on="customer_id",
            save_as="inner_result",
        )

        result = merge_dataframes(state_with_two_dfs, params)

        assert result.success is True
        assert result.dataframe_name == "inner_result"
        # Customer 1 has 1 order, customer 2 has 2 orders = 3 rows
        assert result.rows_after == 3

        df = state_with_two_dfs.get_dataframe("inner_result")
        assert "name" in df.columns
        assert "amount" in df.columns

    def test_merge_left_join(self, state_with_two_dfs):
        """Test left join - keep all left rows."""
        params = MergeDataFramesInput(
            left_dataframe="customers",
            right_dataframe="orders",
            how="left",
            on="customer_id",
            save_as="left_result",
        )

        result = merge_dataframes(state_with_two_dfs, params)

        assert result.success is True
        # All 4 customers, customer 2 has 2 orders = 5 rows
        # Customer 3, 4 have NaN for order columns
        df = state_with_two_dfs.get_dataframe("left_result")
        assert len(df) == 5
        assert df[df["customer_id"] == 3]["amount"].isna().all()

    def test_merge_right_join(self, state_with_two_dfs):
        """Test right join - keep all right rows."""
        params = MergeDataFramesInput(
            left_dataframe="customers",
            right_dataframe="orders",
            how="right",
            on="customer_id",
            save_as="right_result",
        )

        result = merge_dataframes(state_with_two_dfs, params)

        assert result.success is True
        # All 4 orders, customer 5 has no name
        df = state_with_two_dfs.get_dataframe("right_result")
        assert len(df) == 4
        assert df[df["customer_id"] == 5]["name"].isna().all()

    def test_merge_outer_join(self, state_with_two_dfs):
        """Test outer join - keep all rows from both."""
        params = MergeDataFramesInput(
            left_dataframe="customers",
            right_dataframe="orders",
            how="outer",
            on="customer_id",
            save_as="outer_result",
        )

        result = merge_dataframes(state_with_two_dfs, params)

        assert result.success is True
        # All customers + all orders with unmatched rows
        df = state_with_two_dfs.get_dataframe("outer_result")
        # 1 (1 order) + 2 (2 orders) + 3 (0 orders) + 4 (0 orders) + 5 (1 order, no customer) = 6
        assert len(df) == 6

    def test_merge_different_column_names(self):
        """Test merge with different column names in each DataFrame."""
        state = DataFrameState()

        left_df = pd.DataFrame({
            "id": [1, 2, 3],
            "value_a": ["a", "b", "c"],
        })
        state.set_dataframe(left_df, name="left", operation="test")

        right_df = pd.DataFrame({
            "key": [1, 2, 4],
            "value_b": ["x", "y", "z"],
        })
        state.set_dataframe(right_df, name="right", operation="test")

        params = MergeDataFramesInput(
            left_dataframe="left",
            right_dataframe="right",
            how="inner",
            left_on="id",
            right_on="key",
        )

        result = merge_dataframes(state, params)

        assert result.success is True
        assert result.rows_after == 2  # id 1 and 2 match

    def test_merge_auto_name(self, state_with_two_dfs):
        """Test auto-generated name when save_as not provided."""
        params = MergeDataFramesInput(
            left_dataframe="customers",
            right_dataframe="orders",
            how="inner",
            on="customer_id",
        )

        result = merge_dataframes(state_with_two_dfs, params)

        assert result.dataframe_name == "customers_orders_merged"

    def test_merge_invalid_column(self, state_with_two_dfs):
        """Test error when join column doesn't exist."""
        params = MergeDataFramesInput(
            left_dataframe="customers",
            right_dataframe="orders",
            how="inner",
            on="nonexistent_column",
        )

        with pytest.raises(ValueError, match="not found"):
            merge_dataframes(state_with_two_dfs, params)

    def test_merge_json_serializable(self, state_with_two_dfs):
        """Test that result is JSON serializable."""
        params = MergeDataFramesInput(
            left_dataframe="customers",
            right_dataframe="orders",
            how="inner",
            on="customer_id",
        )

        result = merge_dataframes(state_with_two_dfs, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)


class TestConcatDataFrames:
    """Tests for concat_dataframes tool."""

    @pytest.fixture
    def state_with_stackable_dfs(self):
        """Create state with DataFrames that can be stacked."""
        state = DataFrameState()

        df1 = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        })
        state.set_dataframe(df1, name="df1", operation="test")

        df2 = pd.DataFrame({
            "A": [7, 8],
            "B": [9, 10],
        })
        state.set_dataframe(df2, name="df2", operation="test")

        df3 = pd.DataFrame({
            "A": [11],
            "B": [12],
        })
        state.set_dataframe(df3, name="df3", operation="test")

        return state

    def test_concat_vertical_basic(self, state_with_stackable_dfs):
        """Test basic vertical concatenation."""
        params = ConcatDataFramesInput(
            dataframes=["df1", "df2"],
            axis=0,
            save_as="stacked",
        )

        result = concat_dataframes(state_with_stackable_dfs, params)

        assert result.success is True
        assert result.rows_after == 5  # 3 + 2

        df = state_with_stackable_dfs.get_dataframe("stacked")
        assert list(df["A"]) == [1, 2, 3, 7, 8]

    def test_concat_multiple_dfs(self, state_with_stackable_dfs):
        """Test concatenating more than two DataFrames."""
        params = ConcatDataFramesInput(
            dataframes=["df1", "df2", "df3"],
            axis=0,
            save_as="all_stacked",
        )

        result = concat_dataframes(state_with_stackable_dfs, params)

        assert result.success is True
        assert result.rows_after == 6  # 3 + 2 + 1

    def test_concat_horizontal(self):
        """Test horizontal concatenation (adding columns)."""
        state = DataFrameState()

        df1 = pd.DataFrame({"A": [1, 2, 3]})
        state.set_dataframe(df1, name="features1", operation="test")

        df2 = pd.DataFrame({"B": [4, 5, 6]})
        state.set_dataframe(df2, name="features2", operation="test")

        params = ConcatDataFramesInput(
            dataframes=["features1", "features2"],
            axis=1,
            save_as="combined_features",
        )

        result = concat_dataframes(state, params)

        assert result.success is True
        df = state.get_dataframe("combined_features")
        assert list(df.columns) == ["A", "B"]
        assert len(df) == 3

    def test_concat_outer_join(self):
        """Test outer join - keeps all columns, fills NaN."""
        state = DataFrameState()

        df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        state.set_dataframe(df1, name="df1", operation="test")

        df2 = pd.DataFrame({"A": [5, 6], "C": [7, 8]})  # Has C instead of B
        state.set_dataframe(df2, name="df2", operation="test")

        params = ConcatDataFramesInput(
            dataframes=["df1", "df2"],
            axis=0,
            join="outer",
            save_as="outer_result",
        )

        result = concat_dataframes(state, params)

        df = state.get_dataframe("outer_result")
        assert "A" in df.columns
        assert "B" in df.columns
        assert "C" in df.columns
        assert len(df) == 4

    def test_concat_inner_join(self):
        """Test inner join - keeps only common columns."""
        state = DataFrameState()

        df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        state.set_dataframe(df1, name="df1", operation="test")

        df2 = pd.DataFrame({"A": [5, 6], "C": [7, 8]})  # Has C instead of B
        state.set_dataframe(df2, name="df2", operation="test")

        params = ConcatDataFramesInput(
            dataframes=["df1", "df2"],
            axis=0,
            join="inner",
            save_as="inner_result",
        )

        result = concat_dataframes(state, params)

        df = state.get_dataframe("inner_result")
        assert list(df.columns) == ["A"]  # Only common column
        assert len(df) == 4

    def test_concat_auto_name_two(self, state_with_stackable_dfs):
        """Test auto-generated name for two DataFrames."""
        params = ConcatDataFramesInput(
            dataframes=["df1", "df2"],
            axis=0,
        )

        result = concat_dataframes(state_with_stackable_dfs, params)

        assert result.dataframe_name == "df1_df2_concat"

    def test_concat_auto_name_many(self, state_with_stackable_dfs):
        """Test auto-generated name for many DataFrames."""
        params = ConcatDataFramesInput(
            dataframes=["df1", "df2", "df3"],
            axis=0,
        )

        result = concat_dataframes(state_with_stackable_dfs, params)

        assert result.dataframe_name == "df1_and_2_others_concat"

    def test_concat_json_serializable(self, state_with_stackable_dfs):
        """Test that result is JSON serializable."""
        params = ConcatDataFramesInput(
            dataframes=["df1", "df2"],
            axis=0,
        )

        result = concat_dataframes(state_with_stackable_dfs, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)


class TestDropColumns:
    """Tests for drop_columns tool."""

    @pytest.fixture
    def state_with_df(self):
        """Create state with a test DataFrame."""
        state = DataFrameState()
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
            "D": [10, 11, 12],
        })
        state.set_dataframe(df, name="test_df", operation="test")
        return state

    def test_drop_single_column(self, state_with_df):
        """Test dropping a single column."""
        params = DropColumnsInput(
            dataframe_name="test_df",
            columns=["B"],
        )

        result = drop_columns(state_with_df, params)

        assert result.success is True
        df = state_with_df.get_dataframe("test_df")
        assert "B" not in df.columns
        assert list(df.columns) == ["A", "C", "D"]

    def test_drop_multiple_columns(self, state_with_df):
        """Test dropping multiple columns."""
        params = DropColumnsInput(
            dataframe_name="test_df",
            columns=["A", "C"],
        )

        result = drop_columns(state_with_df, params)

        assert result.success is True
        df = state_with_df.get_dataframe("test_df")
        assert list(df.columns) == ["B", "D"]
        assert result.columns_affected == ["A", "C"]

    def test_drop_column_not_found_raise(self, state_with_df):
        """Test error when column not found and errors='raise'."""
        params = DropColumnsInput(
            dataframe_name="test_df",
            columns=["nonexistent"],
            errors="raise",
        )

        with pytest.raises(KeyError) as exc_info:
            drop_columns(state_with_df, params)

        assert "nonexistent" in str(exc_info.value)

    def test_drop_column_not_found_ignore(self, state_with_df):
        """Test ignoring missing columns when errors='ignore'."""
        params = DropColumnsInput(
            dataframe_name="test_df",
            columns=["nonexistent", "A"],
            errors="ignore",
        )

        result = drop_columns(state_with_df, params)

        assert result.success is True
        df = state_with_df.get_dataframe("test_df")
        assert "A" not in df.columns
        assert len(df.columns) == 3

    def test_drop_columns_save_as(self, state_with_df):
        """Test saving result as new DataFrame."""
        params = DropColumnsInput(
            dataframe_name="test_df",
            columns=["B"],
            save_as="dropped_result",
        )

        result = drop_columns(state_with_df, params)

        assert result.dataframe_name == "dropped_result"
        # Original unchanged
        assert "B" in state_with_df.get_dataframe("test_df").columns
        # New one has dropped column
        assert "B" not in state_with_df.get_dataframe("dropped_result").columns

    def test_drop_columns_json_serializable(self, state_with_df):
        """Test that result is JSON serializable."""
        params = DropColumnsInput(
            dataframe_name="test_df",
            columns=["A"],
        )

        result = drop_columns(state_with_df, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)


class TestRenameColumns:
    """Tests for rename_columns tool."""

    @pytest.fixture
    def state_with_df(self):
        """Create state with a test DataFrame."""
        state = DataFrameState()
        df = pd.DataFrame({
            "old_a": [1, 2, 3],
            "old_b": [4, 5, 6],
            "old_c": [7, 8, 9],
        })
        state.set_dataframe(df, name="test_df", operation="test")
        return state

    def test_rename_single_column(self, state_with_df):
        """Test renaming a single column."""
        params = RenameColumnsInput(
            dataframe_name="test_df",
            mapping={"old_a": "new_a"},
        )

        result = rename_columns(state_with_df, params)

        assert result.success is True
        df = state_with_df.get_dataframe("test_df")
        assert "new_a" in df.columns
        assert "old_a" not in df.columns

    def test_rename_multiple_columns(self, state_with_df):
        """Test renaming multiple columns."""
        params = RenameColumnsInput(
            dataframe_name="test_df",
            mapping={"old_a": "A", "old_b": "B"},
        )

        result = rename_columns(state_with_df, params)

        assert result.success is True
        df = state_with_df.get_dataframe("test_df")
        assert list(df.columns) == ["A", "B", "old_c"]
        assert result.columns_affected == ["A", "B"]

    def test_rename_column_not_found_raise(self, state_with_df):
        """Test error when column to rename not found."""
        params = RenameColumnsInput(
            dataframe_name="test_df",
            mapping={"nonexistent": "something"},
            errors="raise",
        )

        with pytest.raises(KeyError) as exc_info:
            rename_columns(state_with_df, params)

        assert "nonexistent" in str(exc_info.value)

    def test_rename_column_not_found_ignore(self, state_with_df):
        """Test ignoring missing columns when errors='ignore'."""
        params = RenameColumnsInput(
            dataframe_name="test_df",
            mapping={"nonexistent": "something", "old_a": "A"},
            errors="ignore",
        )

        result = rename_columns(state_with_df, params)

        assert result.success is True
        df = state_with_df.get_dataframe("test_df")
        assert "A" in df.columns

    def test_rename_save_as(self, state_with_df):
        """Test saving result as new DataFrame."""
        params = RenameColumnsInput(
            dataframe_name="test_df",
            mapping={"old_a": "A"},
            save_as="renamed_df",
        )

        result = rename_columns(state_with_df, params)

        assert result.dataframe_name == "renamed_df"
        # Original unchanged
        assert "old_a" in state_with_df.get_dataframe("test_df").columns
        # New one has renamed column
        assert "A" in state_with_df.get_dataframe("renamed_df").columns

    def test_rename_json_serializable(self, state_with_df):
        """Test that result is JSON serializable."""
        params = RenameColumnsInput(
            dataframe_name="test_df",
            mapping={"old_a": "A"},
        )

        result = rename_columns(state_with_df, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)


class TestAddColumn:
    """Tests for add_column tool."""

    @pytest.fixture
    def state_with_df(self):
        """Create state with a test DataFrame."""
        state = DataFrameState()
        df = pd.DataFrame({
            "price": [10.0, 20.0, 30.0],
            "quantity": [2, 3, 4],
            "tax_rate": [0.1, 0.1, 0.2],
        })
        state.set_dataframe(df, name="test_df", operation="test")
        return state

    def test_add_column_with_expression(self, state_with_df):
        """Test adding a column with an expression."""
        params = AddColumnInput(
            dataframe_name="test_df",
            column_name="total",
            expression="price * quantity",
        )

        result = add_column(state_with_df, params)

        assert result.success is True
        df = state_with_df.get_dataframe("test_df")
        assert "total" in df.columns
        assert list(df["total"]) == [20.0, 60.0, 120.0]

    def test_add_column_complex_expression(self, state_with_df):
        """Test adding a column with a complex expression."""
        params = AddColumnInput(
            dataframe_name="test_df",
            column_name="total_with_tax",
            expression="price * quantity * (1 + tax_rate)",
        )

        result = add_column(state_with_df, params)

        assert result.success is True
        df = state_with_df.get_dataframe("test_df")
        assert "total_with_tax" in df.columns
        # 10*2*1.1=22, 20*3*1.1=66, 30*4*1.2=144
        expected = [22.0, 66.0, 144.0]
        assert list(df["total_with_tax"]) == expected

    def test_add_column_constant_value(self, state_with_df):
        """Test adding a column with a constant value."""
        params = AddColumnInput(
            dataframe_name="test_df",
            column_name="status",
            value="active",
        )

        result = add_column(state_with_df, params)

        assert result.success is True
        df = state_with_df.get_dataframe("test_df")
        assert all(df["status"] == "active")

    def test_add_column_constant_numeric(self, state_with_df):
        """Test adding a column with a constant numeric value."""
        params = AddColumnInput(
            dataframe_name="test_df",
            column_name="discount",
            value=0,
        )

        result = add_column(state_with_df, params)

        assert result.success is True
        df = state_with_df.get_dataframe("test_df")
        assert all(df["discount"] == 0)

    def test_add_column_overwrite_existing(self, state_with_df):
        """Test overwriting an existing column."""
        params = AddColumnInput(
            dataframe_name="test_df",
            column_name="price",  # Already exists
            expression="price * 2",
        )

        result = add_column(state_with_df, params)

        assert result.success is True
        df = state_with_df.get_dataframe("test_df")
        assert list(df["price"]) == [20.0, 40.0, 60.0]
        assert "Updated existing" in result.message

    def test_add_column_save_as(self, state_with_df):
        """Test saving result as new DataFrame."""
        params = AddColumnInput(
            dataframe_name="test_df",
            column_name="total",
            expression="price * quantity",
            save_as="with_total",
        )

        result = add_column(state_with_df, params)

        assert result.dataframe_name == "with_total"
        # Original unchanged
        assert "total" not in state_with_df.get_dataframe("test_df").columns
        # New one has the column
        assert "total" in state_with_df.get_dataframe("with_total").columns

    def test_add_column_invalid_expression(self, state_with_df):
        """Test error for invalid expression."""
        params = AddColumnInput(
            dataframe_name="test_df",
            column_name="result",
            expression="nonexistent_column * 2",
        )

        with pytest.raises(ValueError) as exc_info:
            add_column(state_with_df, params)

        assert "Invalid expression" in str(exc_info.value)

    def test_add_column_missing_both_params(self, state_with_df):
        """Test error when neither expression nor value provided."""
        params = AddColumnInput(
            dataframe_name="test_df",
            column_name="result",
        )

        with pytest.raises(ValueError) as exc_info:
            add_column(state_with_df, params)

        assert "Must provide either" in str(exc_info.value)

    def test_add_column_both_params_provided(self, state_with_df):
        """Test error when both expression and value provided."""
        params = AddColumnInput(
            dataframe_name="test_df",
            column_name="result",
            expression="price * 2",
            value=100,
        )

        with pytest.raises(ValueError) as exc_info:
            add_column(state_with_df, params)

        assert "not both" in str(exc_info.value)

    def test_add_column_json_serializable(self, state_with_df):
        """Test that result is JSON serializable."""
        params = AddColumnInput(
            dataframe_name="test_df",
            column_name="total",
            expression="price * quantity",
        )

        result = add_column(state_with_df, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)
