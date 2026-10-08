"""Caller-supplied expressions are read, never executed (security scan F1–F3, 8 Oct 2026).

filter_dataframe, add_column and inspect_data used to hand the caller's text to
pandas' own evaluator (``df.query`` / ``pd.eval``), which reaches attributes,
method calls and the calling function's locals. On the hosted MCP server that is
anyone who signs up. They now share one small evaluator that knows column names,
constants, arithmetic, comparisons, a fixed list of functions and a fixed list
of summaries, and nothing else.
"""

import numpy as np
import pandas as pd
import pytest

from stats_compass_core.data.add_column import AddColumnInput, add_column
from stats_compass_core.data.inspect_data import InspectDataInput, inspect_data
from stats_compass_core.state import DataFrameState
from stats_compass_core.transforms.filter_dataframe import (
    FilterDataFrameInput,
    filter_dataframe,
)


@pytest.fixture
def state():
    s = DataFrameState()
    s.set_dataframe(
        pd.DataFrame(
            {
                "price": [10.0, 120.0, 250.0, 80.0],
                "quantity": [1, 2, 3, 4],
                "region": ["US", "UK", "US", "DE"],
                "order date": pd.to_datetime(
                    ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
                ),
                "bathrooms": ["1", "2", "x", "3"],
            }
        ),
        name="sales",
        operation="test",
    )
    return s


@pytest.fixture
def no_pandas_eval(monkeypatch):
    """Record every call into pandas' own expression evaluator, and fail on any.

    The calls still go through, so against the old code a payload runs and the
    test shows it; the old tools also wrapped every exception in ValueError, so
    refusing here would have made those tests pass for the wrong reason.
    """
    calls = []
    for owner, name in ((pd, "eval"), (pd.DataFrame, "query"), (pd.DataFrame, "eval")):
        original = getattr(owner, name)

        def spy(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, spy)
    yield calls
    assert calls == [], f"pandas evaluator reached: {calls}"


@pytest.fixture
def probe(monkeypatch):
    """A harmless method on every frame and column: if it runs, a payload got out."""
    ran = []

    def sc_probe(self, *args, **kwargs):
        ran.append(type(self).__name__)
        return "reached"

    monkeypatch.setattr(pd.DataFrame, "sc_probe", sc_probe, raising=False)
    monkeypatch.setattr(pd.Series, "sc_probe", sc_probe, raising=False)
    yield ran
    assert ran == [], f"payload ran: {ran}"


# Shapes from the scan's exploit scenarios. Each reaches beyond the caller's own
# columns: attribute access on modules or the frame, method calls that read or
# write files or serialise the frame, dunder walks, the caller's scope.
ESCAPES = [
    "df.to_json()",
    "df.to_json() != ''",
    "pd.read_csv('/etc/hosts')",
    "np.load('/tmp/x.npy')",
    "pd.io.common",
    "df.eval('price')",
    "df.query('price > 1')",
    "price.__class__",
    "price.to_frame().to_csv('/tmp/x.csv')",
    "@state",
    "price > 1 and @params",
    "(lambda: 1)()",
    "[c for c in df]",
    "getattr(df, 'to_json')()",
    "__import__('os')",
    "df.groupby('region').apply(len)",
    "region.str.contains('(a+)+$', regex=True)",
]


REACH = [
    "df.sc_probe() == 'reached'",
    "price.sc_probe() == 'reached'",
    "@df.sc_probe() == 'reached'",
    "price > 0 and @df.sc_probe() == 'reached'",
]


class TestNothingBeyondTheColumnsIsReachable:
    @pytest.mark.parametrize("payload", REACH)
    def test_filter_dataframe_reach(self, state, no_pandas_eval, probe, payload):
        with pytest.raises(ValueError):
            filter_dataframe(state, FilterDataFrameInput(query=payload))

    @pytest.mark.parametrize("payload", REACH)
    def test_add_column_reach(self, state, no_pandas_eval, probe, payload):
        with pytest.raises(ValueError):
            add_column(state, AddColumnInput(column_name="x", expression=payload))

    @pytest.mark.parametrize("payload", REACH)
    def test_inspect_data_reach(self, state, no_pandas_eval, probe, payload):
        with pytest.raises(ValueError):
            inspect_data(state, InspectDataInput(expression=payload))

    @pytest.mark.parametrize("payload", ESCAPES)
    def test_filter_dataframe(self, state, no_pandas_eval, payload):
        with pytest.raises(ValueError):
            filter_dataframe(state, FilterDataFrameInput(query=payload))

    @pytest.mark.parametrize("payload", ESCAPES)
    def test_add_column(self, state, no_pandas_eval, payload):
        with pytest.raises(ValueError):
            add_column(state, AddColumnInput(column_name="x", expression=payload))

    @pytest.mark.parametrize("payload", ESCAPES)
    def test_inspect_data(self, state, no_pandas_eval, payload):
        with pytest.raises(ValueError):
            inspect_data(state, InspectDataInput(expression=payload))


class TestSizeIsBounded:
    """Expressions that would hang or exhaust memory are refused before they run."""

    @pytest.mark.parametrize(
        "payload",
        [
            "9 ** 9 ** 9",
            "'a' * 10 ** 9",
            "[1] * 10 ** 9",
            "region * 1000000000",
            "price + " * 400 + "price",
            # the exponent cap alone does not bound the base
            "((((2 ** 64) ** 64) ** 64) ** 64) ** 64",
            # arguments are bounded per method, by keyword and by position
            "price.value_counts(bins=10 ** 9)",
            "price.value_counts(False, True, False, 10 ** 9)",
            "region.str.contains('(a+)+$', True, 0, None, True)",
            "round(price, 10 ** 9)",
            "price.round(10 ** 9)",
            "np.round(price, decimals=10 ** 9)",
            "price.head(5, 6)",
            "np.log(price, price)",
        ],
    )
    def test_refused(self, state, no_pandas_eval, payload):
        with pytest.raises(ValueError):
            add_column(state, AddColumnInput(column_name="x", expression=payload))


OBJECT_POWERS = [
    # pre-release review F1: a column of Python objects raises each element with
    # arbitrary-precision integers, so the scalar bounds never applied
    "2 ** (index.astype('object') + 1000000000) > 0",
    "(price.astype('object') + 3) ** 64 > 0",
    "2 ** (mixed + 1000000000) > 0",
    "(mixed + 3) ** 64 ** 2 > 0",
]


class TestObjectColumnsCannotGrowWithoutBound:
    """Data with mixed types arrives as object dtype too, so the cast is not the only way in."""

    @pytest.fixture
    def mixed_state(self, state):
        df = state.get_dataframe("sales").copy()
        # whole numbers held as Python objects, as a column of mixed input can arrive
        df["mixed"] = pd.Series([10, 20, 30, 40], dtype=object)
        state.set_dataframe(df, name="sales", operation="test")
        return state

    @pytest.mark.parametrize("payload", OBJECT_POWERS)
    def test_filter_dataframe(self, mixed_state, payload):
        with pytest.raises(ValueError):
            filter_dataframe(mixed_state, FilterDataFrameInput(query=payload))

    @pytest.mark.parametrize("payload", OBJECT_POWERS)
    def test_add_column(self, mixed_state, payload):
        with pytest.raises(ValueError):
            add_column(mixed_state, AddColumnInput(column_name="x", expression=payload))

    @pytest.mark.parametrize("payload", OBJECT_POWERS)
    def test_inspect_data(self, mixed_state, payload):
        with pytest.raises(ValueError):
            inspect_data(mixed_state, InspectDataInput(expression=payload))

    def test_object_is_not_a_cast_target(self, state):
        with pytest.raises(ValueError, match="astype"):
            add_column(state, AddColumnInput(column_name="x", expression="price.astype('object')"))

    def test_text_columns_still_concatenate(self, mixed_state):
        add_column(mixed_state, AddColumnInput(column_name="x", expression="region + '!'"))
        assert mixed_state.get_dataframe("sales")["x"].tolist()[0] == "US!"


class TestFiltersStillWork:
    @pytest.mark.parametrize(
        "query, prices",
        [
            ("price > 100 and region == 'US'", [250.0]),
            ("price > 100 & (region == 'US')", [250.0]),
            ("price > 100 or quantity == 4", [120.0, 250.0, 80.0]),
            ("not price > 100", [10.0, 80.0]),
            ("~(price > 100)", [10.0, 80.0]),
            ("region in ['US', 'DE']", [10.0, 250.0, 80.0]),
            ("region not in ('US',)", [120.0, 80.0]),
            ("region == ['UK', 'DE']", [120.0, 80.0]),
            ("50 < price <= 250", [120.0, 250.0, 80.0]),
            ("`order date` >= '2024-03-01'", [250.0, 80.0]),
            ("price > price.mean()", [120.0, 250.0]),
            ("region.str.contains('U')", [10.0, 120.0, 250.0]),
            ("region.str.startswith('D')", [80.0]),
            ("price * quantity > 500", [750.0 / 3]),
            ("abs(price - 100) < 25", [120.0, 80.0]),
            ("price.between(50, 150)", [120.0, 80.0]),
            ("region.isin(['UK'])", [120.0]),
        ],
    )
    def test_query(self, state, no_pandas_eval, query, prices):
        result = filter_dataframe(state, FilterDataFrameInput(query=query))
        got = state.get_dataframe(result.dataframe_name)["price"].tolist()
        assert got == pytest.approx(prices)

    def test_a_non_condition_is_refused(self, state, no_pandas_eval):
        with pytest.raises(ValueError, match="true/false"):
            filter_dataframe(state, FilterDataFrameInput(query="price * 2"))

    def test_an_unknown_name_says_which(self, state, no_pandas_eval):
        with pytest.raises(ValueError, match="cost"):
            filter_dataframe(state, FilterDataFrameInput(query="cost > 1"))


class TestColumnsStillWork:
    @pytest.mark.parametrize(
        "expression, expected",
        [
            ("price * quantity", [10.0, 240.0, 750.0, 320.0]),
            ('df["price"] * 1.1', [11.0, 132.0, 275.0, 88.0]),
            ("np.log(price)", list(np.log([10.0, 120.0, 250.0, 80.0]))),
            ("np.where(price > 100, 1, 0)", [0, 1, 1, 0]),
            ("price / price.max()", [0.04, 0.48, 1.0, 0.32]),
            ("round(price / 3, 1)", [3.3, 40.0, 83.3, 26.7]),
            ("price ** 2", [100.0, 14400.0, 62500.0, 6400.0]),
            ("`order date`.dt.month", [1, 2, 3, 4]),
            ("2 ** 10", [1024] * 4),
        ],
    )
    def test_expression(self, state, no_pandas_eval, expression, expected):
        add_column(state, AddColumnInput(column_name="x", expression=expression))
        assert state.get_dataframe("sales")["x"].tolist() == pytest.approx(expected)

    def test_to_numeric_with_coerce(self, state, no_pandas_eval):
        add_column(
            state,
            AddColumnInput(
                column_name="x", expression="pd.to_numeric(bathrooms, errors='coerce')"
            ),
        )
        got = state.get_dataframe("sales")["x"]
        assert got.iloc[[0, 1, 3]].tolist() == [1.0, 2.0, 3.0] and got.isna().iloc[2]

    def test_string_concatenation(self, state, no_pandas_eval):
        add_column(state, AddColumnInput(column_name="x", expression="region + '-' + region"))
        assert state.get_dataframe("sales")["x"].tolist()[0] == "US-US"


class TestInspectionStillWorks:
    @pytest.mark.parametrize(
        "expression, expected",
        [
            ('df["price"].mean()', "115.0"),
            ("price.mean()", "115.0"),
            ('len(df[df["price"] > 100])', "2"),
            ("len(df)", "4"),
            ("region.nunique()", "3"),
            ("price.max() - price.min()", "240.0"),
        ],
    )
    def test_scalar(self, state, no_pandas_eval, expression, expected):
        assert inspect_data(state, InspectDataInput(expression=expression))["result"] == expected

    def test_unique(self, state, no_pandas_eval):
        out = inspect_data(state, InspectDataInput(expression='df["region"].unique()'))
        assert "US" in out["result"] and "DE" in out["result"]

    def test_value_counts(self, state, no_pandas_eval):
        out = inspect_data(state, InspectDataInput(expression="region.value_counts()"))
        assert out["result_type"] == "Series" and "US" in out["result_text"]

    def test_equality_is_allowed(self, state, no_pandas_eval):
        """The old guard refused any '=', so even '==' could not be inspected."""
        out = inspect_data(state, InspectDataInput(expression="len(df[region == 'US'])"))
        assert out["result"] == "2"
