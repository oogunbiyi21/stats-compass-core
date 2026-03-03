"""
Tool for splitting a numeric column into separate columns by group.

Converts long-format data into wide-format where each unique group value
becomes its own column, filled with the corresponding values from the
value column. This is useful for preparing data for two-sample tests.
"""

from __future__ import annotations

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import (
    DataFrameQueryResult,
    dataframe_to_json_safe_records,
)
from stats_compass_core.state import DataFrameState


class SplitColumnByGroupInput(StrictToolInput):
    """Input schema for split_column_by_group tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    value_column: str = Field(
        description="Numeric column whose values will be split into separate columns",
    )
    group_column: str = Field(
        description="Categorical column defining the groups (each unique value becomes a column)",
    )
    groups: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of group labels to include. "
            "If None, all unique groups are used."
        ),
    )
    save_as: str | None = Field(
        default=None,
        description="Name to save the result DataFrame. If None, auto-generates name.",
    )


@registry.register(
    category="transforms",
    input_schema=SplitColumnByGroupInput,
    description="Split a numeric column into separate columns by group values",
)
def split_column_by_group(
    state: DataFrameState, params: SplitColumnByGroupInput
) -> DataFrameQueryResult:
    """
    Split a value column into separate columns based on a grouping column.

    For example, given a DataFrame with 'IMDB Score' and 'Language' columns,
    this produces a DataFrame with columns like 'English', 'Spanish', etc.,
    each containing the IMDB scores for that language. Columns are NaN-padded
    to accommodate groups of different sizes.

    Args:
        state: DataFrameState containing the DataFrame to operate on
        params: Parameters specifying the value and group columns

    Returns:
        DataFrameQueryResult with the wide-format result saved to state

    Raises:
        ValueError: If specified columns don't exist or groups are not found
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Validate columns exist
    for col in (params.value_column, params.group_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # Determine which groups to include
    available_groups = df[params.group_column].dropna().unique().tolist()

    if params.groups is not None:
        missing = set(params.groups) - set(available_groups)
        if missing:
            raise ValueError(
                f"Groups not found in column '{params.group_column}': {missing}. "
                f"Available groups: {available_groups}"
            )
        selected_groups = params.groups
    else:
        selected_groups = sorted(str(g) for g in available_groups)

    # Build a column of values for each group, reset index so they align from 0
    group_series: dict[str, pd.Series] = {}
    for group in selected_groups:
        values = (
            df.loc[df[params.group_column] == group, params.value_column]
            .dropna()
            .reset_index(drop=True)
        )
        group_series[str(group)] = values

    result_df = pd.DataFrame(group_series)

    # Generate result name
    result_name = params.save_as
    if result_name is None:
        result_name = f"{source_name}_{params.value_column}_by_{params.group_column}"

    stored_name = state.set_dataframe(result_df, name=result_name, operation="split_column_by_group")

    max_rows = 100
    data = dataframe_to_json_safe_records(result_df, max_rows=max_rows)

    return DataFrameQueryResult(
        data={"records": data, "truncated": len(result_df) > max_rows},
        shape=(len(result_df), len(result_df.columns)),
        columns=list(result_df.columns),
        dataframe_name=stored_name,
        source_dataframe=source_name,
    )
