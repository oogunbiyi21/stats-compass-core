"""
Tool for concatenating multiple DataFrames together.
"""

from typing import Literal

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameMutationResult
from stats_compass_core.state import DataFrameState


class ConcatDataFramesInput(StrictToolInput):
    """Input schema for concat_dataframes tool."""

    dataframes: list[str] = Field(
        description="List of DataFrame names to concatenate (in order)",
        min_length=2,
    )
    axis: Literal[0, 1] = Field(
        default=0,
        description=(
            "Axis to concatenate along: "
            "0 = stack vertically (add rows), "
            "1 = stack horizontally (add columns)"
        ),
    )
    join: Literal["inner", "outer"] = Field(
        default="outer",
        description=(
            "How to handle columns that don't exist in all DataFrames: "
            "'outer' = keep all columns (fill missing with NaN), "
            "'inner' = keep only common columns"
        ),
    )
    ignore_index: bool = Field(
        default=True,
        description="If True, reset the index in the result (recommended for vertical concat)"
    )
    save_as: str | None = Field(
        default=None,
        description="Name for the concatenated DataFrame. If not provided, auto-generates name."
    )
    set_active: bool = Field(
        default=True,
        description="Whether to set the concatenated DataFrame as active"
    )


@registry.register(
    category="data",
    input_schema=ConcatDataFramesInput,
    description="Concatenate multiple DataFrames vertically (stack rows) or horizontally (add columns)",
)
def concat_dataframes(
    state: DataFrameState, params: ConcatDataFramesInput
) -> DataFrameMutationResult:
    """
    Concatenate multiple DataFrames together.

    Vertical concatenation (axis=0):
        Stacks DataFrames on top of each other, adding rows.
        Useful for combining datasets with the same structure.

    Horizontal concatenation (axis=1):
        Places DataFrames side by side, adding columns.
        Useful for combining different features/attributes.

    Args:
        state: DataFrameState containing the DataFrames to concatenate
        params: Concatenation configuration parameters

    Returns:
        DataFrameMutationResult with concatenated DataFrame info

    Raises:
        ValueError: If any DataFrame not found or list is empty
    """
    # Get all DataFrames
    dfs: list[pd.DataFrame] = []
    df_info: list[tuple[str, int, int]] = []  # (name, rows, cols)

    for df_name in params.dataframes:
        df = state.get_dataframe(df_name)
        dfs.append(df)
        df_info.append((df_name, len(df), len(df.columns)))

    total_rows_before = sum(info[1] for info in df_info)

    # For horizontal concat (axis=1), ignore_index would reset column names to integers
    # which breaks downstream operations. Only use ignore_index for vertical concat.
    use_ignore_index = params.ignore_index if params.axis == 0 else False

    # Perform concatenation
    result_df = pd.concat(
        dfs,
        axis=params.axis,
        join=params.join,
        ignore_index=use_ignore_index,
    )

    # Determine result name
    if params.save_as:
        result_name = params.save_as
    else:
        # Auto-generate name based on first two DataFrames
        if len(params.dataframes) == 2:
            result_name = f"{params.dataframes[0]}_{params.dataframes[1]}_concat"
        else:
            result_name = f"{params.dataframes[0]}_and_{len(params.dataframes)-1}_others_concat"

    # Store in state
    state.set_dataframe(
        result_df,
        name=result_name,
        operation=f"concat_axis{params.axis}"
    )

    if params.set_active:
        state.set_active_dataframe(result_name)

    # Build message
    axis_desc = "vertically (rows)" if params.axis == 0 else "horizontally (columns)"
    df_list = ", ".join(f"'{name}' ({rows}×{cols})" for name, rows, cols in df_info)

    message = (
        f"Concatenated {len(params.dataframes)} DataFrames {axis_desc}: {df_list}. "
        f"Result: {len(result_df)} rows × {len(result_df.columns)} columns."
    )

    return DataFrameMutationResult(
        success=True,
        operation=f"concat_axis{params.axis}",
        rows_before=total_rows_before,
        rows_after=len(result_df),
        rows_affected=len(result_df),
        message=message,
        dataframe_name=result_name,
        columns_affected=list(result_df.columns),
    )
