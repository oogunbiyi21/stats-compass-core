"""
Tool for renaming columns in a DataFrame.
"""

from pydantic import BaseModel, Field

from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameMutationResult
from stats_compass_core.state import DataFrameState


class RenameColumnsInput(BaseModel):
    """Input schema for rename_columns tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    mapping: dict[str, str] = Field(
        description="Mapping of old column names to new names, e.g. {'old_name': 'new_name'}",
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
    input_schema=RenameColumnsInput,
    description="Rename columns in a DataFrame using a mapping",
)
def rename_columns(
    state: DataFrameState, params: RenameColumnsInput
) -> DataFrameMutationResult:
    """
    Rename columns in a DataFrame using a mapping.

    Args:
        state: DataFrameState containing the DataFrame to modify
        params: Parameters specifying the column rename mapping

    Returns:
        DataFrameMutationResult with operation summary

    Raises:
        KeyError: If errors='raise' and a column to rename doesn't exist
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Validate columns if errors='raise'
    if params.errors == "raise":
        missing = set(params.mapping.keys()) - set(df.columns)
        if missing:
            raise KeyError(
                f"Columns to rename not found: {sorted(missing)}. "
                f"Available columns: {list(df.columns)}"
            )

    # Filter mapping to only existing columns if errors='ignore'
    if params.errors == "ignore":
        actual_mapping = {k: v for k, v in params.mapping.items() if k in df.columns}
    else:
        actual_mapping = params.mapping

    # Rename columns
    result_df = df.rename(columns=actual_mapping)

    # Determine result name
    result_name = params.save_as if params.save_as else source_name

    # Store in state
    state.set_dataframe(result_df, name=result_name, operation="rename_columns")

    if params.set_active:
        state.set_active_dataframe(result_name)

    renamed_list = [f"'{k}' → '{v}'" for k, v in actual_mapping.items()]
    message = f"Renamed {len(actual_mapping)} column(s): {', '.join(renamed_list)}"

    return DataFrameMutationResult(
        success=True,
        operation="rename_columns",
        rows_before=len(df),
        rows_after=len(result_df),
        rows_affected=0,
        message=message,
        dataframe_name=result_name,
        columns_affected=list(actual_mapping.values()),
    )
