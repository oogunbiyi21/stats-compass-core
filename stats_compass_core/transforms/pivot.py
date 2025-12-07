"""
Tool for pivoting DataFrame data (reshaping from long to wide format).
"""

import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.registry import registry


class PivotInput(BaseModel):
    """Input schema for pivot tool."""

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


@registry.register(
    category="transforms",
    input_schema=PivotInput,
    description="Pivot DataFrame from long to wide format",
)
def pivot(df: pd.DataFrame, params: PivotInput) -> pd.DataFrame:
    """
    Pivot a DataFrame from long to wide format.

    Args:
        df: Input DataFrame
        params: Parameters for pivoting

    Returns:
        New pivoted DataFrame

    Raises:
        ValueError: If specified columns don't exist
        KeyError: If pivot operation creates duplicate entries without aggregation
    """
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
        result = df.pivot_table(
            index=params.index,
            columns=params.columns,
            values=params.values,
            aggfunc=params.aggfunc,
            fill_value=params.fill_value,
        )
        return result
    except Exception as e:
        raise KeyError(f"Pivot operation failed: {str(e)}") from e
