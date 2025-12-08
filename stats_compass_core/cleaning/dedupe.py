"""
Tool for removing duplicate rows from a DataFrame.
"""

from pydantic import BaseModel, Field

from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState
from stats_compass_core.results import DataFrameMutationResult


class DedupeInput(BaseModel):
    """Input schema for dedupe tool."""

    dataframe_name: str | None = Field(
        default=None, description="Name of DataFrame to operate on. Uses active if not specified."
    )
    subset: list[str] | None = Field(
        default=None, description="Column labels to consider for identifying duplicates"
    )
    keep: str = Field(
        default="first",
        pattern="^(first|last|False)$",
        description="Which duplicates to keep: 'first', 'last', or False (drop all)",
    )
    ignore_index: bool = Field(
        default=False, description="If True, reset index in result"
    )


@registry.register(
    category="cleaning",
    input_schema=DedupeInput,
    description="Remove duplicate rows from DataFrame",
)
def dedupe(state: DataFrameState, params: DedupeInput) -> DataFrameMutationResult:
    """
    Remove duplicate rows from a DataFrame.

    Args:
        state: DataFrameState containing the DataFrame to operate on
        params: Parameters for deduplication

    Returns:
        DataFrameMutationResult with operation summary

    Raises:
        ValueError: If subset columns don't exist in DataFrame
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()
    rows_before = len(df)
    
    if params.subset:
        missing_cols = set(params.subset) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    keep_value = False if params.keep == "False" else params.keep

    result_df = df.drop_duplicates(
        subset=params.subset,
        keep=keep_value,  # type: ignore
        ignore_index=params.ignore_index,
    )
    
    # Update the DataFrame in state (in-place modification)
    stored_name = state.set_dataframe(result_df, name=source_name, operation="dedupe")
    
    rows_after = len(result_df)
    rows_affected = rows_before - rows_after
    
    message = f"Removed {rows_affected} duplicate rows"
    if params.subset:
        message += f" (based on columns: {', '.join(params.subset)})"
    
    return DataFrameMutationResult(
        success=True,
        operation="dedupe",
        rows_before=rows_before,
        rows_after=rows_after,
        rows_affected=rows_affected,
        message=message,
        dataframe_name=stored_name,
        columns_affected=params.subset,
    )
