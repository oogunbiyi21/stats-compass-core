"""describe's include/exclude are names from a fixed list (re-scan F8, 9 Oct 2026).

They were free strings, passed to pandas' dtype parsing, whose datetime pattern
backtracks quadratically on 'M8[' followed by a long run of ', '. Now each is
one of a few names pandas understands, or a short list of them.
"""

import time

import pandas as pd
import pytest
from pydantic import ValidationError

from stats_compass_core.eda.describe import DescribeInput, describe
from stats_compass_core.state import DataFrameState


@pytest.fixture
def state():
    s = DataFrameState()
    s.set_dataframe(pd.DataFrame({"n": [1.0, 2.0], "c": ["a", "b"]}), "t", "test")
    return s


@pytest.mark.parametrize("field", ["include", "exclude"])
def test_a_long_pattern_is_refused_before_pandas(field):
    started = time.time()
    with pytest.raises(ValidationError):
        DescribeInput(**{field: "M8[" + ", " * 20000})
    assert time.time() - started < 1


@pytest.mark.parametrize("bad", ["datetime64[ns, UTC]", "int64", "M8[ns]", ["number"] * 9])
def test_anything_off_the_list_is_refused(bad):
    with pytest.raises(ValidationError):
        DescribeInput(include=bad)


@pytest.mark.parametrize("include", ["all", "number", ["number", "object"], ["number", "object", "bool"]])
def test_the_listed_names_work(state, include):
    describe(state, DescribeInput(dataframe_name="t", include=include))


def test_exclude_works(state):
    describe(state, DescribeInput(dataframe_name="t", exclude=["object"]))


def test_the_expression_tools_describe_uses_the_same_list(state):
    from stats_compass_core.data.inspect_data import InspectDataInput, inspect_data

    with pytest.raises(ValueError):
        inspect_data(state, InspectDataInput(dataframe_name="t", expression="df.describe(include='M8[, , ')"))
    out = inspect_data(state, InspectDataInput(dataframe_name="t", expression="df.describe(include='all')"))
    assert out["result_type"] == "DataFrame"
