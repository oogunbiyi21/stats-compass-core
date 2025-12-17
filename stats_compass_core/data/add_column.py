"""
Tool for adding or transforming columns in a DataFrame.
"""

import re
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameMutationResult
from stats_compass_core.state import DataFrameState


class AddColumnInput(StrictToolInput):
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
            "Python/Pandas expression to compute the column value. "
            "You can use: "
            "1. Column names directly as variables (e.g., 'price * quantity') "
            "2. The dataframe as 'df' (e.g., 'df[\"price\"] * 1.1') "
            "3. Pandas/Numpy functions (e.g., 'pd.to_numeric(bathrooms)', 'np.log(price)') "
            "Either expression or value must be provided."
        ),
    )
    value: str | None = Field(
        default=None,
        description=(
            "Constant value to assign to all rows in the new column. "
            "If the string looks like a number (e.g. '123' or '12.5'), it will be converted. "
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
    description="Add a new column or transform an existing column using a Python expression or constant value",
)
def add_column(
    state: DataFrameState, params: AddColumnInput
) -> DataFrameMutationResult:
    """
    Add a new column or transform an existing column.

    Supports two modes:
    1. Expression mode: Compute column using a Python expression with access to df, pd, np.
       Example: expression="pd.to_numeric(bathrooms, errors='coerce')"
       Example: expression="price * quantity"
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
        # Security check for dangerous patterns
        dangerous_patterns = [
            r"\bimport\b",
            r"\bexec\b",
            r"\beval\b",
            r"\bopen\b",
            r"\.to_csv",
            r"\.to_excel",
            r"\b__.*?__\b",
            r"\bdel\b",
        ]
        if any(re.search(p, params.expression, re.IGNORECASE) for p in dangerous_patterns):
            raise ValueError("Unsafe operation detected in expression")

        # Create namespace with safe globals and column variables
        namespace = {
            "df": result_df,
            "pd": pd,
            "np": np,
            "__builtins__": {},  # Restrict builtins
        }

        # Add valid column names to namespace for convenience (e.g. price * quantity)
        for col in result_df.columns:
            if isinstance(col, str) and col.isidentifier():
                namespace[col] = result_df[col]

        try:
            # Evaluate expression in restricted namespace
            result_df[params.column_name] = eval(params.expression, namespace)
        except Exception as e:
            raise ValueError(
                f"Invalid expression '{params.expression}': {str(e)}. "
                f"Available variables: df, pd, np, and columns."
            ) from e
    else:
        # Constant value assignment
        val = params.value
        # Try to convert string to number if it looks like one
        if isinstance(val, str):
            try:
                if "." in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                pass  # Keep as string
        
        result_df[params.column_name] = val

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
