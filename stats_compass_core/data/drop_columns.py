"""
Tool for dropping columns from a DataFrame.
"""

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameMutationResult
from stats_compass_core.state import DataFrameState


class DropColumnsInput(StrictToolInput):
    """Input schema for drop_columns tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    columns: list[str] = Field(
        description="List of column names to drop",
        min_length=1,
    )
    errors: str = Field(
        default="raise",
        pattern="^(raise|ignore)$",
        description="'raise' to error if column not found, 'ignore' to skip missing columns",
    )
    save_as: str | None = Field(
        default=None,
        description="Save result as new DataFrame with this name. If not provided, modifies in place.",
    )
    set_active: bool = Field(
        default=True,
        description="Whether to set the result DataFrame as active",
    )


@registry.register(
    category="data",
    input_schema=DropColumnsInput,
    description="Drop one or more columns from a DataFrame",
)
def drop_columns(
    state: DataFrameState, params: DropColumnsInput
) -> DataFrameMutationResult:
    """
    Drop specified columns from a DataFrame.

    Args:
        state: DataFrameState containing the DataFrame to modify
        params: Parameters specifying which columns to drop

    Returns:
        DataFrameMutationResult with operation summary

    Raises:
        KeyError: If errors='raise' and a column doesn't exist
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    cols_before = len(df.columns)

    # Validate columns if errors='raise'
    if params.errors == "raise":
        missing = set(params.columns) - set(df.columns)
        if missing:
            raise KeyError(
                f"Columns not found in DataFrame: {sorted(missing)}. "
                f"Available columns: {list(df.columns)}"
            )

    # Drop columns
    result_df = df.drop(columns=params.columns, errors=params.errors)

    cols_after = len(result_df.columns)
    dropped_count = cols_before - cols_after

    # Determine which columns were actually dropped
    dropped_cols = [c for c in params.columns if c in df.columns]

    # Determine result name
    result_name = params.save_as if params.save_as else source_name

    # Store in state
    state.set_dataframe(result_df, name=result_name, operation="drop_columns")

    if params.set_active:
        state.set_active_dataframe(result_name)

    message = f"Dropped {dropped_count} column(s): {dropped_cols}"

    return DataFrameMutationResult(
        success=True,
        operation="drop_columns",
        rows_before=len(df),
        rows_after=len(result_df),
        rows_affected=0,
        message=message,
        dataframe_name=result_name,
        columns_affected=dropped_cols,
    )
