"""Formula injection through headers, row labels and typed text columns (security scan F9).

sanitize_dataframe used to rewrite only cells in object-dtype columns. A header,
a row label (written when index=True), or a cell in a ``string`` or
``category`` column that starts with ``=`` reached the CSV as it was, and a
spreadsheet opened it as a live formula.
"""

import pandas as pd
import pytest

from stats_compass_core.data.save_csv import SaveCSVInput, save_csv
from stats_compass_core.state import DataFrameState
from stats_compass_core.utils.file_safety import FilePolicy
from stats_compass_core.utils.spreadsheet_safety import sanitize_dataframe

PAYLOAD = "=HYPERLINK(\"http://evil.example\",\"x\")"


def _written(tmp_path, df, index=False):
    state = DataFrameState(file_policy=FilePolicy(write_root=tmp_path))
    state.set_dataframe(df, "t", "test")
    result = save_csv(state, SaveCSVInput(dataframe_name="t", filepath="out.csv", index=index))
    return open(result["filepath"]).read()


def _no_live_formula(text):
    """No field in the file starts with a formula trigger."""
    for line in text.splitlines():
        for field in line.split(","):
            field = field.strip('"')
            assert not field.startswith(("=", "+", "@")), field


class TestEverythingWrittenIsSanitised:
    def test_a_header(self, tmp_path):
        _no_live_formula(_written(tmp_path, pd.DataFrame({PAYLOAD: [1, 2]})))

    def test_a_string_dtype_column(self, tmp_path):
        df = pd.DataFrame({"name": pd.array(["ok", PAYLOAD], dtype="string")})
        _no_live_formula(_written(tmp_path, df))

    def test_a_category_column(self, tmp_path):
        df = pd.DataFrame({"name": pd.Categorical(["ok", PAYLOAD])})
        _no_live_formula(_written(tmp_path, df))

    def test_row_labels_when_the_index_is_written(self, tmp_path):
        df = pd.DataFrame({"v": [1, 2]}, index=pd.Index(["ok", PAYLOAD], name="+label"))
        _no_live_formula(_written(tmp_path, df, index=True))

    def test_a_multiindex_header(self):
        """State refuses tuple column names, so this reaches only direct callers
        of sanitize_dataframe, which is public in stats_compass_core.utils."""
        df = pd.DataFrame([[1, 2]], columns=pd.MultiIndex.from_tuples([("a", PAYLOAD), ("b", "c")]))
        _no_live_formula(sanitize_dataframe(df).to_csv(index=False))


class TestNothingElseChanges:
    def test_the_original_is_untouched(self):
        df = pd.DataFrame({PAYLOAD: pd.Categorical(["ok", PAYLOAD])})
        sanitize_dataframe(df)
        assert list(df.columns) == [PAYLOAD]
        assert df[PAYLOAD].dtype == "category"

    def test_safe_labels_and_values_are_kept(self):
        df = pd.DataFrame(
            {"name": pd.array(["Alice", "Bob"], dtype="string"), "n": [1, 2]},
            index=pd.Index(["r1", "r2"], name="row"),
        )
        out = sanitize_dataframe(df)
        assert list(out.columns) == ["name", "n"]
        assert list(out.index) == ["r1", "r2"] and out.index.name == "row"
        assert out["name"].tolist() == ["Alice", "Bob"]
        assert out["n"].tolist() == [1, 2]

    @pytest.mark.parametrize("missing", [None, pd.NA])
    def test_missing_text_stays_missing(self, missing):
        df = pd.DataFrame({"name": pd.array(["=x", missing], dtype="string")})
        out = sanitize_dataframe(df)
        assert out["name"].iloc[0] == "'=x"
        assert pd.isna(out["name"].iloc[1])
