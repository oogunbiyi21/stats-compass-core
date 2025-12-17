"""
Bin rare categories in categorical variables.

Groups infrequent categories into a single label (default: 'Other')
based on a frequency threshold. Helps reduce noise and improve model
performance by limiting the number of unique categories.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import BinRareCategoriesResult
from stats_compass_core.state import DataFrameState


class BinRareCategoriesInput(StrictToolInput):
    """Input schema for bin rare categories tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    categorical_columns: list[str] = Field(
        description="List of categorical column names to process"
    )
    threshold: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Frequency threshold (0-1). Categories with frequency below this are binned."
    )
    bin_label: str = Field(
        default="Other",
        description="Label to use for binned rare categories"
    )
    min_count: int | None = Field(
        default=None,
        ge=1,
        description="Alternative to threshold: minimum count. Categories with fewer occurrences are binned."
    )
    save_as: str | None = Field(
        default=None,
        description="Name to save the result DataFrame. If None, modifies in place."
    )


def _validate_categorical_column(
    df: pd.DataFrame, col: str
) -> tuple[bool, str | None]:
    """
    Validate that a column is suitable for category binning.
    
    Returns:
        Tuple of (is_valid, error_message_if_invalid)
    """
    if df[col].dtype == "object" or df[col].dtype.name == "category":
        # Check for corrupted data (mostly numeric strings)
        unique_vals = df[col].dropna().unique()
        numeric_looking = 0

        for val in unique_vals:
            try:
                float(str(val))
                numeric_looking += 1
            except (ValueError, TypeError):
                pass

        # If >80% of values are numeric, likely a corrupted numeric column
        if len(unique_vals) > 0 and numeric_looking / len(unique_vals) > 0.8:
            return False, (
                f"Column '{col}' appears to be a corrupted numeric column "
                f"(most values are numeric strings). Clean the data first."
            )

        return True, None
    else:
        return False, (
            f"Column '{col}' is not categorical (dtype: {df[col].dtype}). "
            f"Convert to string/category first."
        )


def _bin_column(
    series: pd.Series,
    threshold: float,
    min_count: int | None,
    bin_label: str,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Bin rare categories in a single column.
    
    Returns:
        Tuple of (binned_series, binning_info_dict)
    """
    value_counts = series.value_counts()
    total_count = len(series.dropna())

    # Determine which categories are rare
    if min_count is not None:
        rare_categories = value_counts[value_counts < min_count].index.tolist()
    else:
        value_frequencies = value_counts / total_count
        rare_categories = value_frequencies[value_frequencies < threshold].index.tolist()

    # Build category mapping
    category_mapping = {}
    for cat in value_counts.index:
        if cat in rare_categories:
            category_mapping[str(cat)] = bin_label
        else:
            category_mapping[str(cat)] = str(cat)

    # Apply binning
    binned_series = series.copy()
    if rare_categories:
        binned_series = binned_series.replace(rare_categories, bin_label)

    # Calculate stats
    rows_affected = 0
    if rare_categories:
        rows_affected = int(series.isin(rare_categories).sum())

    binning_info = {
        "categories_before": int(series.nunique()),
        "categories_after": int(binned_series.nunique()),
        "rare_categories": [str(c) for c in rare_categories],
        "rare_categories_count": len(rare_categories),
        "rows_affected": rows_affected,
        "total_rows": total_count,
        "percent_affected": round(rows_affected / total_count * 100, 2) if total_count > 0 else 0,
    }

    return binned_series, binning_info, category_mapping


@registry.register(
    category="transforms",
    input_schema=BinRareCategoriesInput,
    description="Bin rare categories into a single label based on frequency threshold",
)
def bin_rare_categories(
    state: DataFrameState, params: BinRareCategoriesInput
) -> BinRareCategoriesResult:
    """
    Bin rare categories in categorical columns.

    Groups infrequent categories into a single label (default: 'Other')
    based on either a frequency threshold or minimum count. This reduces
    noise from rare categories and can improve model performance.

    Args:
        state: DataFrameState containing the DataFrame to operate on
        params: Parameters for binning

    Returns:
        BinRareCategoriesResult with binning details and category mappings

    Raises:
        ValueError: If columns don't exist or aren't categorical
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Validate columns exist
    missing_cols = [col for col in params.categorical_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns not found: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate columns are categorical
    valid_columns = []
    validation_errors = []

    for col in params.categorical_columns:
        is_valid, error_msg = _validate_categorical_column(df, col)
        if is_valid:
            valid_columns.append(col)
        else:
            validation_errors.append(error_msg)

    if validation_errors and not valid_columns:
        raise ValueError(
            "No valid categorical columns to process. Errors:\n" +
            "\n".join(f"  - {e}" for e in validation_errors)
        )

    # Create working copy
    df_binned = df.copy()

    # Process each column
    binning_details: dict[str, dict[str, Any]] = {}
    category_mapping: dict[str, dict[str, str]] = {}
    columns_modified = []

    for col in valid_columns:
        binned_series, binning_info, col_mapping = _bin_column(
            df_binned[col],
            threshold=params.threshold,
            min_count=params.min_count,
            bin_label=params.bin_label,
        )

        df_binned[col] = binned_series
        binning_details[col] = binning_info
        category_mapping[col] = col_mapping

        if binning_info["rare_categories_count"] > 0:
            columns_modified.append(col)

    # Save result to state
    if params.save_as:
        result_name = params.save_as
    else:
        result_name = source_name  # Modify in place

    stored_name = state.set_dataframe(df_binned, name=result_name, operation="bin_rare_categories")

    # Build summary message
    total_categories_binned = sum(
        details["rare_categories_count"] for details in binning_details.values()
    )
    total_rows_affected = sum(
        details["rows_affected"] for details in binning_details.values()
    )

    if columns_modified:
        message = (
            f"Binned {total_categories_binned} rare categories across "
            f"{len(columns_modified)} column(s) into '{params.bin_label}'. "
            f"{total_rows_affected} rows affected. "
            f"Threshold: {params.threshold:.1%}"
            if params.min_count is None
            else f"min_count: {params.min_count}"
        )
    else:
        message = (
            f"No rare categories found in {len(valid_columns)} column(s) "
            f"using threshold {params.threshold:.1%}."
            if params.min_count is None
            else f"using min_count {params.min_count}."
        )

    return BinRareCategoriesResult(
        success=True,
        operation="bin_rare_categories",
        dataframe_name=stored_name,
        source_dataframe=source_name,
        rows_affected=len(df_binned),
        columns_processed=valid_columns,
        columns_modified=columns_modified,
        binning_details=binning_details,
        category_mapping=category_mapping,
        threshold=params.threshold,
        bin_label=params.bin_label,
        message=message,
    )
