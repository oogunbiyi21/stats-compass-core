"""
Tool for dropping rows or columns with missing values from a DataFrame.
"""

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameMutationResult
from stats_compass_core.state import DataFrameState


class DropNAInput(StrictToolInput):
    """Input schema for drop_na tool."""

    dataframe_name: str | None = Field(
        default=None, description="Name of DataFrame to operate on. Uses active if not specified."
    )
    axis: int = Field(default=0, ge=0, le=1, description="0 for rows, 1 for columns")
    how: str = Field(default="any", pattern="^(any|all)$", description="'any' or 'all'")
    thresh: int | None = Field(
        default=None, ge=0, description="Minimum number of non-NA values"
    )
    subset: list[str] | None = Field(
        default=None, description="Column labels to consider"
    )
    save_as: str | None = Field(
        default=None,
        description="Save result as new DataFrame with this name. If not provided, modifies in place.",
    )


@registry.register(
    category="cleaning",
    input_schema=DropNAInput,
    description="Drop rows or columns with missing values",
)
def drop_na(state: DataFrameState, params: DropNAInput) -> DataFrameMutationResult:
    """
    Drop rows or columns with missing values from a DataFrame.

    Args:
        state: DataFrameState containing the DataFrame to operate on
        params: Parameters for dropping NA values

    Returns:
        DataFrameMutationResult with operation summary

    Raises:
        ValueError: If subset columns don't exist in DataFrame
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()
    rows_before = len(df)
    cols_before = len(df.columns)

    if params.subset:
        missing_cols = set(params.subset) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Build kwargs - pandas doesn't allow both how and thresh
    kwargs = {"axis": params.axis, "subset": params.subset}
    if params.thresh is not None:
        kwargs["thresh"] = params.thresh
    else:
        kwargs["how"] = params.how

    result_df = df.dropna(**kwargs)

    # Determine result name - use save_as if provided, otherwise modify in place
    result_name = params.save_as if params.save_as else source_name

    # Save DataFrame to state
    stored_name = state.set_dataframe(result_df, name=result_name, operation="drop_na")

    rows_after = len(result_df)
    cols_after = len(result_df.columns)

    # Determine what was affected
    if params.axis == 0:
        rows_affected = rows_before - rows_after
        message = f"Dropped {rows_affected} rows with missing values"
        columns_affected = params.subset
    else:
        rows_affected = cols_before - cols_after
        message = f"Dropped {rows_affected} columns with missing values"
        columns_affected = list(set(df.columns) - set(result_df.columns))

    return DataFrameMutationResult(
        success=True,
        operation="drop_na",
        rows_before=rows_before,
        rows_after=rows_after,
        rows_affected=rows_affected,
        message=message,
        dataframe_name=stored_name,
        columns_affected=columns_affected,
    )
