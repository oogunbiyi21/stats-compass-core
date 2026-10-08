"""Aggregation names come from a fixed list.

groupby_aggregate defined VALID_AGGS but never checked it, and pivot took any
string, so a caller could make pandas call any method of a group by name
(``plot`` ran). Found while fixing the 8 Oct 2026 security scan; same root as
F1–F3: the caller names what runs.
"""

import pandas as pd
import pytest

from stats_compass_core.state import DataFrameState
from stats_compass_core.transforms.groupby_aggregate import (
    VALID_AGGS,
    ColumnAggregation,
    GroupByAggregateInput,
    groupby_aggregate,
)
from stats_compass_core.transforms.pivot import PivotInput, pivot


@pytest.fixture
def state():
    s = DataFrameState()
    s.set_dataframe(pd.DataFrame({"g": ["a", "b", "a"], "c": ["x", "y", "x"], "v": [1, 2, 3]}), "t", "test")
    return s


@pytest.mark.parametrize("name", ["plot", "pipe", "__class__", "describe", "apply"])
def test_groupby_refuses_other_names(state, name):
    with pytest.raises(ValueError):
        groupby_aggregate(state, GroupByAggregateInput(
            dataframe_name="t", by=["g"], aggregations=[ColumnAggregation(column="v", functions=[name])],
        ))


@pytest.mark.parametrize("name", ["plot", "pipe", "__class__", "describe", "apply"])
def test_pivot_refuses_other_names(state, name):
    with pytest.raises(ValueError):
        pivot(state, PivotInput(dataframe_name="t", index="g", columns="c", values="v", aggfunc=name))


@pytest.mark.parametrize("name", VALID_AGGS)
def test_every_listed_name_works(state, name):
    groupby_aggregate(state, GroupByAggregateInput(
        dataframe_name="t", by=["g"], aggregations=[ColumnAggregation(column="v", functions=[name])],
    ))
    pivot(state, PivotInput(dataframe_name="t", index="g", columns="c", values="v", aggfunc=name))
