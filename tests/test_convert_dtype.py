"""convert_dtype: turn text columns into the type they hold, and count the cost.

Exports routinely carry numbers as text, with missing values spelled "nan",
"null" or "". Without a conversion tool these had to be coerced by hand. The
unattended question is what happens to a value that is neither a number nor a
recognised missing marker: here it becomes missing, and the result says how
many and which, rather than the column quietly losing rows of data.
"""

import numpy as np
import pandas as pd
import pytest

from stats_compass_core.cleaning.convert_dtype import ConvertDtypeInput, convert_dtype
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState


def _convert(values, to, **kwargs):
    state = DataFrameState()
    state.set_dataframe(pd.DataFrame({"col": values}), "d", operation="test")
    result = convert_dtype(state, ConvertDtypeInput(
        dataframe_name="d", columns=["col"], to=to, **kwargs
    ))
    return state, result


class TestToNumeric:
    def test_acceptance_frame(self):
        """T6.8 AC."""
        state, _ = _convert(["1", "nan", "2.5"], "numeric")
        col = state.get_dataframe("d")["col"]
        assert col.dtype == np.float64
        np.testing.assert_array_equal(col.to_numpy(), [1.0, np.nan, 2.5])

    def test_null_spellings_are_missing_not_errors(self):
        values = ["1", "", " NULL ", "None", "N/A", "nan", "2"]
        state, result = _convert(values, "numeric")
        col = state.get_dataframe("d")["col"]
        assert col.isna().sum() == 5
        assert result.columns["col"].null_tokens == 5
        assert result.columns["col"].unparseable == 0
        assert result.warnings == []

    def test_unparseable_values_are_counted_and_named(self):
        state, result = _convert(["1", "abc", "1,234", "3"], "numeric")
        col = state.get_dataframe("d")["col"]
        np.testing.assert_array_equal(col.to_numpy(), [1.0, np.nan, np.nan, 3.0])

        detail = result.columns["col"]
        assert detail.unparseable == 2
        assert set(detail.unparseable_sample) == {"abc", "1,234"}
        warning = next(w for w in result.warnings if w.code == "UNPARSEABLE_VALUES")
        assert warning.columns == ["col"]
        assert any(
            h.operation == "warning" and h.details["code"] == "UNPARSEABLE_VALUES"
            for h in state.get_history()
        )

    def test_raise_mode_refuses_and_leaves_the_frame_alone(self):
        state = DataFrameState()
        state.set_dataframe(pd.DataFrame({"col": ["1", "abc"]}), "d", operation="test")
        with pytest.raises(ValueError, match="abc"):
            convert_dtype(state, ConvertDtypeInput(
                dataframe_name="d", columns=["col"], to="numeric", errors="raise"
            ))
        assert state.get_dataframe("d")["col"].tolist() == ["1", "abc"]

    def test_a_column_with_nothing_convertible_is_refused(self):
        """Every value failing means the wrong type was asked for, not dirty data."""
        with pytest.raises(ValueError, match="none of"):
            _convert(["alice", "bob", "nan"], "numeric")

    def test_already_numeric_is_left_as_is(self):
        state, result = _convert([1, 2, 3], "numeric")
        assert state.get_dataframe("d")["col"].tolist() == [1, 2, 3]
        assert result.columns["col"].unparseable == 0


class TestToDatetime:
    def test_strings_and_null_tokens(self):
        state, result = _convert(["2024-01-05", "", "2024-02-10", "null"], "datetime")
        col = state.get_dataframe("d")["col"]
        assert pd.api.types.is_datetime64_any_dtype(col)
        assert col.isna().tolist() == [False, True, False, True]
        assert col.iloc[2] == pd.Timestamp("2024-02-10")
        assert result.warnings == []

    def test_unparseable_date_is_reported(self):
        _, result = _convert(["2024-01-05", "not a date", "2024-02-10"], "datetime")
        assert result.columns["col"].unparseable == 1
        assert [w.code for w in result.warnings] == ["UNPARSEABLE_VALUES"]

    def test_explicit_format(self):
        state, _ = _convert(
            ["05/01/2024", "10/02/2024"], "datetime", datetime_format="%d/%m/%Y"
        )
        assert state.get_dataframe("d")["col"].iloc[1] == pd.Timestamp("2024-02-10")


class TestToBool:
    def test_common_spellings(self):
        state, result = _convert(["yes", "No", "TRUE", "0", "", "maybe", "1"], "bool")
        col = state.get_dataframe("d")["col"]
        assert str(col.dtype) == "boolean"
        assert col.tolist() == [True, False, True, False, pd.NA, pd.NA, True]
        assert result.columns["col"].unparseable == 1
        assert result.columns["col"].unparseable_sample == ["maybe"]


class TestToolContract:
    def test_save_as_leaves_the_source_untouched(self):
        state = DataFrameState()
        state.set_dataframe(pd.DataFrame({"col": ["1", "2"]}), "raw", operation="test")
        result = convert_dtype(state, ConvertDtypeInput(
            dataframe_name="raw", columns=["col"], to="numeric", save_as="typed"
        ))
        assert result.dataframe_name == "typed"
        assert state.get_dataframe("raw")["col"].tolist() == ["1", "2"]
        assert state.get_dataframe("typed")["col"].tolist() == [1, 2]

    def test_missing_column_is_an_error(self):
        state = DataFrameState()
        state.set_dataframe(pd.DataFrame({"col": ["1"]}), "d", operation="test")
        with pytest.raises(ValueError, match="nope"):
            convert_dtype(state, ConvertDtypeInput(
                dataframe_name="d", columns=["nope"], to="numeric"
            ))

    def test_registered_as_a_cleaning_subtool(self):
        meta = registry.get_tool_metadata("cleaning", "convert_dtype")
        assert meta is not None
        assert meta.tier == "sub"
