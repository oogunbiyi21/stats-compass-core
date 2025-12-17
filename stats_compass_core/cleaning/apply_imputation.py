"""
Tool for applying simple imputation strategies to DataFrame columns.
"""

from __future__ import annotations

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameMutationResult
from stats_compass_core.state import DataFrameState


class ApplyImputationInput(StrictToolInput):
    """Input schema for apply_imputation tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    strategy: str = Field(
        default="mean",
        pattern="^(mean|median|most_frequent|constant)$",
        description="Imputation strategy to apply",
    )
    columns: list[str] | None = Field(
        default=None,
        description="Columns to impute. Defaults depend on strategy (numeric for mean/median, all for others).",
    )
    fill_value: str | None = Field(
        default=None,
        description="Constant value used when strategy='constant'. Will be converted to number if possible.",
    )
    save_as: str | None = Field(
        default=None,
        description="Save result as new DataFrame with this name. If not provided, modifies in place.",
    )


@registry.register(
    category="cleaning",
    input_schema=ApplyImputationInput,
    description="Apply simple imputation (mean/median/most_frequent/constant) to DataFrame columns",
)
def apply_imputation(
    state: DataFrameState, params: ApplyImputationInput
) -> DataFrameMutationResult:
    """
    Apply simple imputation strategies to handle missing values.

    Args:
        state: DataFrameState containing the DataFrame to operate on
        params: Imputation parameters

    Returns:
        DataFrameMutationResult with summary of changes

    Raises:
        ValueError: If required columns are missing or strategy parameters are invalid
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()
    rows_before = len(df)

    if params.columns:
        missing_cols = set(params.columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    strategy = params.strategy

    if strategy in {"mean", "median"}:
        # Default to numeric columns if none provided
        target_cols = params.columns or list(df.select_dtypes(include=["number"]).columns)
    else:
        target_cols = params.columns or list(df.columns)

    imputed_df = df.copy()
    filled_counts: dict[str, int] = {}
    total_filled = 0

    for col in target_cols:
        col_series = imputed_df[col]
        missing_mask = col_series.isna()
        missing_count = int(missing_mask.sum())

        if missing_count == 0:
            continue

        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(col_series):
                raise ValueError(f"Column '{col}' must be numeric for mean imputation")
            fill_value = col_series.mean()
        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(col_series):
                raise ValueError(f"Column '{col}' must be numeric for median imputation")
            fill_value = col_series.median()
        elif strategy == "most_frequent":
            mode_series = col_series.mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else None
        else:  # constant
            if params.fill_value is None:
                raise ValueError("fill_value is required when strategy='constant'")
            
            fill_value = params.fill_value
            # Try to convert string to number if it looks like one
            if isinstance(fill_value, str):
                try:
                    if "." in fill_value:
                        fill_value = float(fill_value)
                    else:
                        fill_value = int(fill_value)
                except ValueError:
                    pass  # Keep as string


        if fill_value is not None:
            imputed_df.loc[missing_mask, col] = fill_value
            filled_counts[col] = missing_count
            total_filled += missing_count

    # Determine result name - use save_as if provided, otherwise modify in place
    result_name = params.save_as if params.save_as else source_name

    # Save DataFrame to state
    state.set_dataframe(imputed_df, name=result_name, operation=f"apply_imputation_{strategy}")

    columns_imputed = list(filled_counts.keys())
    message = (
        f"Imputed {total_filled} values in {len(columns_imputed)} column(s) "
        f"using '{strategy}' strategy"
    )

    return DataFrameMutationResult(
        success=True,
        rows_before=rows_before,
        rows_after=len(imputed_df),
        rows_affected=total_filled,
        dataframe_name=result_name,
        operation=f"apply_imputation ({strategy})",
        message=message,
        columns_affected=columns_imputed,
    )
