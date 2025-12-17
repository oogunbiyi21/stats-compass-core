"""
Tool for filtering a DataFrame using a pandas query expression.
"""

from __future__ import annotations

import hashlib

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import (
    DataFrameQueryResult,
    dataframe_to_json_safe_records,
)
from stats_compass_core.state import DataFrameState


class FilterDataFrameInput(StrictToolInput):
    """Input schema for filter_dataframe tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    query: str = Field(
        description="pandas.DataFrame.query expression, e.g., `price > 100 and region == 'US'`"
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
    description="Filter a DataFrame using a pandas query expression",
)
def filter_dataframe(
    state: DataFrameState, params: FilterDataFrameInput
) -> DataFrameQueryResult:
    """
    Filter a DataFrame using pandas.query and save result to state.

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
        filtered = df.query(params.query)
    except Exception as exc:
        raise ValueError(f"Query failed: {exc}") from exc

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
