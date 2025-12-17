"""
Tool for grouping and aggregating DataFrame data.
"""

from pydantic import BaseModel, Field

from stats_compass_core.base import StrictToolInput, ToolComponent
from stats_compass_core.registry import registry
from stats_compass_core.results import (
    DataFrameQueryResult,
    dataframe_to_json_safe_records,
)
from stats_compass_core.state import DataFrameState

# Valid aggregation functions supported by this tool
# This is a deliberately constrained list to ensure deterministic behavior
# Contributors can expand this list by adding new pandas aggregation functions
VALID_AGGS = [
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "var",
    "count",
    "first",
    "last",
    "nunique",
]


class ColumnAggregation(ToolComponent):
    """Aggregation configuration for a single column."""
    column: str = Field(description="Column to aggregate")
    functions: list[str] = Field(description="List of aggregation functions (e.g. ['mean', 'sum'])")


class GroupByAggregateInput(StrictToolInput):
    """Input schema for groupby_aggregate tool."""

    dataframe_name: str | None = Field(
        default=None, description="Name of DataFrame to operate on. Uses active if not specified."
    )
    by: list[str] = Field(description="Column labels to group by")
    aggregations: list[ColumnAggregation] = Field(
        description="List of column aggregation operations"
    )
    as_index: bool = Field(default=True, description="If True, use group keys as index")
    save_as: str | None = Field(
        default=None, description="Name to save the result DataFrame. If None, auto-generates name."
    )


@registry.register(
    category="transforms",
    input_schema=GroupByAggregateInput,
    description="Group DataFrame by columns and apply aggregation functions",
)
def groupby_aggregate(state: DataFrameState, params: GroupByAggregateInput) -> DataFrameQueryResult:
    """
    Group DataFrame by specified columns and apply aggregation functions.

    Args:
        state: DataFrameState containing the DataFrame to operate on
        params: Parameters for groupby and aggregation

    Returns:
        DataFrameQueryResult with aggregated data saved to state

    Raises:
        ValueError: If group-by or aggregation columns don't exist
        TypeError: If aggregation function is not supported
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Validate group-by columns
    missing_cols = set(params.by) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Group-by columns not found: {missing_cols}")

    # Convert list of objects to dictionary for pandas
    agg_func = {agg.column: agg.functions for agg in params.aggregations}

    # Validate aggregation columns
    agg_cols = set(agg_func.keys())
    missing_agg_cols = agg_cols - set(df.columns)
    if missing_agg_cols:
        raise ValueError(f"Aggregation columns not found: {missing_agg_cols}")

    # Perform groupby and aggregation
    try:
        result_df = df.groupby(params.by, as_index=params.as_index).agg(agg_func)

        # Reset index to make it a proper DataFrame for storage
        if params.as_index:
            result_df = result_df.reset_index()

        # Flatten multi-level column names if present
        if isinstance(result_df.columns, pd.MultiIndex):
            result_df.columns = ['_'.join(col).strip('_') for col in result_df.columns.values]

    except Exception as e:
        raise TypeError(f"Aggregation failed: {str(e)}") from e

    # Generate a name for the result if not provided
    result_name = params.save_as
    if result_name is None:
        by_str = '_'.join(params.by)
        result_name = f"{source_name}_grouped_by_{by_str}"

    # Save result to state as new DataFrame
    stored_name = state.set_dataframe(result_df, name=result_name, operation="groupby_aggregate")

    # Convert to JSON-safe dict (handles NaN, Timestamps, etc.)
    max_rows = 100
    data = dataframe_to_json_safe_records(result_df, max_rows=max_rows)

    return DataFrameQueryResult(
        data={"records": data, "truncated": len(result_df) > max_rows},
        shape=(len(result_df), len(result_df.columns)),
        columns=list(result_df.columns),
        dataframe_name=stored_name,
        source_dataframe=source_name,
    )


# Import pandas here to avoid issues with forward references
import pandas as pd
