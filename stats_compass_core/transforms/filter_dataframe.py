"""
Tool for filtering a DataFrame with a condition on its columns.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import (
    DataFrameQueryResult,
    dataframe_to_json_safe_records,
)
from stats_compass_core.state import DataFrameState
from stats_compass_core.utils.safe_expr import evaluate


class FilterDataFrameInput(StrictToolInput):
    """Input schema for filter_dataframe tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    query: str = Field(
        description=(
            "A true/false condition on the columns, in pandas query style, e.g. "
            "`price > 100 and region == 'US'`, `region in ['US', 'UK']`, "
            "`` `order date` >= '2024-01-01' ``, `name.str.contains('Ltd')`. "
            "Columns by name (backticks if they have spaces), constants, arithmetic, "
            "comparisons, and/or/not (or & | ~). No '@' references or other methods."
        )
    )
    limit: int | None = Field(
        default=None, ge=1, description="Optional row limit after filtering"
    )
    save_as: str | None = Field(
        default=None, description="Name to save the filtered DataFrame. If None, auto-generates name."
    )


@registry.register(
    category="transforms",
    input_schema=FilterDataFrameInput,
    description="Filter a DataFrame with a condition on its columns (pandas query style)",
)
def filter_dataframe(
    state: DataFrameState, params: FilterDataFrameInput
) -> DataFrameQueryResult:
    """
    Filter a DataFrame with a condition and save the result to state.

    The condition is read by ``utils.safe_expr``, never by ``df.query``: pandas'
    own evaluator reaches attributes, calls and the caller's locals, which on a
    shared server is anyone who signs up (security scan F1, 8 Oct 2026).

    Args:
        state: DataFrameState containing the DataFrame to operate on
        params: Query string and optional row limit

    Returns:
        DataFrameQueryResult with filtered data summary

    Raises:
        ValueError: If the query fails to evaluate
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    try:
        mask = evaluate(params.query, df)
    except ValueError as exc:
        raise ValueError(f"Query failed: {exc}") from exc
    if isinstance(mask, (bool, np.bool_)):
        mask = pd.Series(bool(mask), index=df.index)
    if not (
        isinstance(mask, (pd.Series, np.ndarray))
        and pd.api.types.is_bool_dtype(mask.dtype)
        and len(mask) == len(df)
    ):
        raise ValueError(
            f"Query failed: '{params.query}' is not a true/false condition on each row."
        )
    filtered = df[mask]

    if params.limit:
        filtered = filtered.head(params.limit)

    filtered = filtered.copy()

    # Generate name for result
    result_name = params.save_as
    if result_name is None:
        # Create a deterministic hash of the query for uniqueness
        # Using SHA256 instead of hash() which is non-deterministic across processes
        query_hash = hashlib.sha256(params.query.encode()).hexdigest()[:8]
        result_name = f"{source_name}_filtered_{query_hash}"

    # Save to state
    stored_name = state.set_dataframe(filtered, name=result_name, operation="filter_dataframe")

    # Convert to JSON-safe dict (handles NaN, Timestamps, etc.)
    max_rows = 100
    data = dataframe_to_json_safe_records(filtered, max_rows=max_rows)

    return DataFrameQueryResult(
        data={"records": data, "truncated": len(filtered) > max_rows},
        shape=(len(filtered), len(filtered.columns)),
        columns=list(filtered.columns),
        dataframe_name=stored_name,
        source_dataframe=source_name,
    )
