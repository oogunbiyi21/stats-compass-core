"""
Tool for adding or transforming columns in a DataFrame.
"""

from typing import Any

from pydantic import BaseModel, Field

from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameMutationResult
from stats_compass_core.state import DataFrameState


class AddColumnInput(BaseModel):
    """Input schema for add_column tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    column_name: str = Field(
        description="Name of the new column to create (or existing column to overwrite)",
    )
    expression: str | None = Field(
        default=None,
        description=(
            "Pandas expression to compute the column value. "
            "Can reference existing columns by name, e.g. 'price * quantity' or 'age + 1'. "
            "Uses pandas.eval() for safe evaluation. "
            "Either expression or value must be provided."
        ),
    )
    value: Any = Field(
        default=None,
        description=(
            "Constant value to assign to all rows in the new column. "
            "Either expression or value must be provided."
        ),
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
    input_schema=AddColumnInput,
    description="Add a new column or transform an existing column using an expression or constant value",
)
def add_column(
    state: DataFrameState, params: AddColumnInput
) -> DataFrameMutationResult:
    """
    Add a new column or transform an existing column.

    Supports two modes:
    1. Expression mode: Compute column from existing columns using pandas.eval()
       Example: expression="price * quantity" creates a new column from price and quantity
    2. Constant mode: Assign the same value to all rows
       Example: value=0 or value="unknown"

    Args:
        state: DataFrameState containing the DataFrame to modify
        params: Parameters specifying the column to add

    Returns:
        DataFrameMutationResult with operation summary

    Raises:
        ValueError: If neither expression nor value is provided, or if expression is invalid
    """
    if params.expression is None and params.value is None:
        raise ValueError("Must provide either 'expression' or 'value'")

    if params.expression is not None and params.value is not None:
        raise ValueError("Provide either 'expression' or 'value', not both")

    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    result_df = df.copy()
    is_new_column = params.column_name not in df.columns

    if params.expression is not None:
        # Use pandas.eval for safe expression evaluation
        try:
            result_df[params.column_name] = result_df.eval(params.expression)
        except Exception as e:
            raise ValueError(
                f"Invalid expression '{params.expression}': {str(e)}. "
                f"Available columns: {list(df.columns)}"
            ) from e
    else:
        # Constant value assignment
        result_df[params.column_name] = params.value

    # Determine result name
    result_name = params.save_as if params.save_as else source_name

    # Store in state
    state.set_dataframe(result_df, name=result_name, operation="add_column")

    if params.set_active:
        state.set_active_dataframe(result_name)

    action = "Added new" if is_new_column else "Updated existing"
    if params.expression:
        message = f"{action} column '{params.column_name}' = {params.expression}"
    else:
        message = f"{action} column '{params.column_name}' = {repr(params.value)}"

    return DataFrameMutationResult(
        success=True,
        operation="add_column",
        rows_before=len(df),
        rows_after=len(result_df),
        rows_affected=len(result_df),
        message=message,
        dataframe_name=result_name,
        columns_affected=[params.column_name],
    )
