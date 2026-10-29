"""
Tool for melting a DataFrame from wide to long format (the reverse of pivot).

Retailer sell-out exports are often wide: one row per product and one column
per week. The sales tools read long data, one row per date, so this turns the
week columns into rows.
"""

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import (
    DataFrameQueryResult,
    ToolWarning,
    dataframe_to_json_safe_records,
)
from stats_compass_core.state import DataFrameState


class MeltInput(StrictToolInput):
    """Input schema for melt tool."""

    dataframe_name: str | None = Field(
        default=None, description="Name of DataFrame to operate on. Uses active if not specified."
    )
    id_columns: list[str] = Field(
        description="Columns that identify each row and are kept as they are, such as ['retailer', 'product']."
    )
    value_columns: list[str] | None = Field(
        default=None,
        description="Columns to turn into rows, such as the week columns. If None, every column not in id_columns.",
    )
    variable_name: str = Field(
        default="variable", description="Name for the new column holding the old column names (e.g. 'week_ending')."
    )
    value_name: str = Field(default="value", description="Name for the new column holding the values (e.g. 'units').")
    save_as: str | None = Field(
        default=None, description="Name to save the result DataFrame. If None, auto-generates name."
    )


@registry.register(
    category="transforms",
    input_schema=MeltInput,
    description="Melt a DataFrame from wide to long format: columns such as one per week become rows",
)
def melt(state: DataFrameState, params: MeltInput) -> DataFrameQueryResult:
    """
    Melt a DataFrame from wide to long format.

    Args:
        state: DataFrameState containing the DataFrame to operate on
        params: Parameters for melting

    Returns:
        DataFrameQueryResult with the long data saved to state

    Raises:
        ValueError: If specified columns don't exist, or a new name clashes with a kept column
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    wanted = [*params.id_columns, *(params.value_columns or [])]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        names = ", ".join(f"'{c}'" for c in missing)
        raise ValueError(
            f"Couldn't find {'the column' if len(missing) == 1 else 'the columns'} {names}. "
            f"The columns are: {', '.join(map(str, df.columns))}."
        )
    for name in (params.variable_name, params.value_name):
        if name in params.id_columns:
            raise ValueError(f"'{name}' is already a kept column; choose another name for the new column.")
    if params.variable_name == params.value_name:
        raise ValueError("variable_name and value_name must differ.")

    value_columns = params.value_columns or [c for c in df.columns if c not in params.id_columns]
    if not value_columns:
        raise ValueError("There are no columns to melt: every column is in id_columns.")
    warnings: list[ToolWarning] = []
    non_numeric = [c for c in value_columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        warnings.append(ToolWarning(
            code="NON_NUMERIC_VALUES",
            columns=non_numeric[:10],
            message=(
                f"{len(non_numeric)} melted {'column isn' if len(non_numeric) == 1 else 'columns aren'}'t numeric, so '{params.value_name}' "
                f"may hold text. Convert it with convert_dtype before analysing it."
            ),
        ))

    result_df = df.melt(
        id_vars=params.id_columns,
        value_vars=value_columns,
        var_name=params.variable_name,
        value_name=params.value_name,
    )
    result_name = params.save_as or f"{source_name}_long"
    stored_name = state.set_dataframe(result_df, name=result_name, operation="melt")
    state.record_warnings(stored_name, warnings)

    max_rows = 100
    data = dataframe_to_json_safe_records(result_df, max_rows=max_rows)
    return DataFrameQueryResult(
        data={"records": data, "truncated": len(result_df) > max_rows},
        shape=(len(result_df), len(result_df.columns)),
        columns=list(result_df.columns),
        dataframe_name=stored_name,
        source_dataframe=source_name,
        warnings=warnings,
    )
