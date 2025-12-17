"""
Tool for merging two DataFrames using SQL-style joins.
"""

from typing import Literal

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameMutationResult
from stats_compass_core.state import DataFrameState


class MergeDataFramesInput(StrictToolInput):
    """Input schema for merge_dataframes tool."""

    left_dataframe: str = Field(
        description="Name of the left DataFrame to merge"
    )
    right_dataframe: str = Field(
        description="Name of the right DataFrame to merge"
    )
    how: Literal["inner", "left", "right", "outer"] = Field(
        default="inner",
        description=(
            "Type of merge: 'inner' (intersection), 'left' (keep all left rows), "
            "'right' (keep all right rows), 'outer' (union of both)"
        ),
    )
    on: str | list[str] | None = Field(
        default=None,
        description=(
            "Column(s) to join on. Must exist in both DataFrames. "
            "If None, uses left_on and right_on."
        ),
    )
    left_on: str | list[str] | None = Field(
        default=None,
        description="Column(s) from left DataFrame to join on (if different from right)"
    )
    right_on: str | list[str] | None = Field(
        default=None,
        description="Column(s) from right DataFrame to join on (if different from left)"
    )
    suffixes: list[str] = Field(
        default=["_left", "_right"],
        description="Suffixes to apply to overlapping column names (must be length 2)"
    )
    save_as: str | None = Field(
        default=None,
        description=(
            "Name for the merged DataFrame. "
            "If not provided, uses '{left}_{right}_merged'"
        ),
    )
    set_active: bool = Field(
        default=True,
        description="Whether to set the merged DataFrame as active"
    )


@registry.register(
    category="data",
    input_schema=MergeDataFramesInput,
    description="Merge two DataFrames using SQL-style joins (inner, left, right, outer)",
)
def merge_dataframes(
    state: DataFrameState, params: MergeDataFramesInput
) -> DataFrameMutationResult:
    """
    Merge two DataFrames using SQL-style join operations.

    This is analogous to SQL JOIN operations:
    - inner: Only rows with matching keys in both DataFrames
    - left: All rows from left DataFrame, matched rows from right
    - right: All rows from right DataFrame, matched rows from left
    - outer: All rows from both DataFrames

    Args:
        state: DataFrameState containing the DataFrames to merge
        params: Merge configuration parameters

    Returns:
        DataFrameMutationResult with merged DataFrame info

    Raises:
        ValueError: If DataFrames not found or join columns invalid
    """
    # Get both DataFrames
    left_df = state.get_dataframe(params.left_dataframe)
    right_df = state.get_dataframe(params.right_dataframe)

    left_rows = len(left_df)
    right_rows = len(right_df)

    # Validate join columns
    if params.on is not None:
        # Using same column(s) from both DataFrames
        on_cols = [params.on] if isinstance(params.on, str) else params.on
        for col in on_cols:
            if col not in left_df.columns:
                raise ValueError(
                    f"Join column '{col}' not found in left DataFrame '{params.left_dataframe}'. "
                    f"Available: {list(left_df.columns)}"
                )
            if col not in right_df.columns:
                raise ValueError(
                    f"Join column '{col}' not found in right DataFrame '{params.right_dataframe}'. "
                    f"Available: {list(right_df.columns)}"
                )
    elif params.left_on is not None and params.right_on is not None:
        # Using different columns from each DataFrame
        left_cols = [params.left_on] if isinstance(params.left_on, str) else params.left_on
        right_cols = [params.right_on] if isinstance(params.right_on, str) else params.right_on

        for col in left_cols:
            if col not in left_df.columns:
                raise ValueError(
                    f"Join column '{col}' not found in left DataFrame. "
                    f"Available: {list(left_df.columns)}"
                )
        for col in right_cols:
            if col not in right_df.columns:
                raise ValueError(
                    f"Join column '{col}' not found in right DataFrame. "
                    f"Available: {list(right_df.columns)}"
                )
    else:
        raise ValueError(
            "Must specify either 'on' (for same column names) or both "
            "'left_on' and 'right_on' (for different column names)"
        )

    # Perform merge
    # Convert list to tuple for pandas suffixes
    suffixes_tuple = tuple(params.suffixes) if params.suffixes else ("_left", "_right")

    merged_df = pd.merge(
        left_df,
        right_df,
        how=params.how,
        on=params.on,
        left_on=params.left_on,
        right_on=params.right_on,
        suffixes=suffixes_tuple,
    )

    # Determine result name
    if params.save_as:
        result_name = params.save_as
    else:
        result_name = f"{params.left_dataframe}_{params.right_dataframe}_merged"

    # Store in state
    state.set_dataframe(
        merged_df,
        name=result_name,
        operation=f"merge_{params.how}"
    )

    if params.set_active:
        state.set_active_dataframe(result_name)

    # Build message
    join_desc = params.on if params.on else f"{params.left_on} = {params.right_on}"
    message = (
        f"Merged '{params.left_dataframe}' ({left_rows} rows) with "
        f"'{params.right_dataframe}' ({right_rows} rows) using {params.how.upper()} JOIN "
        f"on {join_desc}. Result: {len(merged_df)} rows."
    )

    return DataFrameMutationResult(
        success=True,
        operation=f"merge_{params.how}",
        rows_before=left_rows + right_rows,
        rows_after=len(merged_df),
        rows_affected=len(merged_df),
        message=message,
        dataframe_name=result_name,
        columns_affected=list(merged_df.columns),
    )
