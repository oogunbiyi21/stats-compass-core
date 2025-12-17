"""
Tool for pivoting DataFrame data (reshaping from long to wide format).
"""

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import (
    DataFrameQueryResult,
    dataframe_to_json_safe_records,
)
from stats_compass_core.state import DataFrameState


class PivotInput(StrictToolInput):
    """Input schema for pivot tool."""

    dataframe_name: str | None = Field(
        default=None, description="Name of DataFrame to operate on. Uses active if not specified."
    )
    index: str | list[str] = Field(description="Column(s) to use as row index")
    columns: str | list[str] = Field(description="Column(s) to use as column headers")
    values: str | list[str] | None = Field(
        default=None,
        description="Column(s) to use for values. If None, uses all remaining columns",
    )
    aggfunc: str = Field(
        default="mean", description="Aggregation function if multiple values per group"
    )
    fill_value: float | None = Field(
        default=None, description="Value to replace missing values with"
    )
    save_as: str | None = Field(
        default=None, description="Name to save the result DataFrame. If None, auto-generates name."
    )


@registry.register(
    category="transforms",
    input_schema=PivotInput,
    description="Pivot DataFrame from long to wide format",
)
def pivot(state: DataFrameState, params: PivotInput) -> DataFrameQueryResult:
    """
    Pivot a DataFrame from long to wide format.

    Args:
        state: DataFrameState containing the DataFrame to operate on
        params: Parameters for pivoting

    Returns:
        DataFrameQueryResult with pivoted data saved to state

    Raises:
        ValueError: If specified columns don't exist
        KeyError: If pivot operation creates duplicate entries without aggregation
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Collect all column names that should exist
    cols_to_check = []
    if isinstance(params.index, str):
        cols_to_check.append(params.index)
    else:
        cols_to_check.extend(params.index)

    if isinstance(params.columns, str):
        cols_to_check.append(params.columns)
    else:
        cols_to_check.extend(params.columns)

    if params.values:
        if isinstance(params.values, str):
            cols_to_check.append(params.values)
        else:
            cols_to_check.extend(params.values)

    # Validate columns exist
    missing_cols = set(cols_to_check) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Perform pivot
    try:
        result_df = df.pivot_table(
            index=params.index,
            columns=params.columns,
            values=params.values,
            aggfunc=params.aggfunc,
            fill_value=params.fill_value,
        )

        # Reset index and flatten column names for easier storage/access
        result_df = result_df.reset_index()

        # Flatten multi-level column names if present
        if hasattr(result_df.columns, 'to_flat_index'):
            result_df.columns = ['_'.join(map(str, col)).strip('_') if isinstance(col, tuple) else str(col)
                                  for col in result_df.columns]

    except Exception as e:
        raise KeyError(f"Pivot operation failed: {str(e)}") from e

    # Generate a name for the result if not provided
    result_name = params.save_as
    if result_name is None:
        index_str = params.index if isinstance(params.index, str) else '_'.join(params.index)
        cols_str = params.columns if isinstance(params.columns, str) else '_'.join(params.columns)
        result_name = f"{source_name}_pivot_{index_str}_by_{cols_str}"

    # Save result to state as new DataFrame
    stored_name = state.set_dataframe(result_df, name=result_name, operation="pivot")

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
