"""
Outlier handling tools for data cleaning.

Provides multiple methods for handling outliers:
- cap: Cap values at percentile threshold
- remove: Remove rows with outlier values
- winsorize: Replace outliers with boundary values
- log_transform: Apply log transformation for right-skewed data
- clip_iqr: Cap using IQR boundaries (1.5 * IQR beyond Q1/Q3)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import OutlierHandlingResult
from stats_compass_core.state import DataFrameState


class HandleOutliersInput(StrictToolInput):
    """Input schema for outlier handling tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    column: str = Field(description="Column name to handle outliers in (must be numeric)")
    method: Literal["cap", "remove", "winsorize", "log_transform", "clip_iqr"] = Field(
        default="cap",
        description=(
            "Outlier handling method: "
            "'cap' - Cap at percentile, "
            "'remove' - Remove outlier rows, "
            "'winsorize' - Replace outliers with boundary values, "
            "'log_transform' - Apply log transformation, "
            "'clip_iqr' - Cap at IQR boundaries"
        ),
    )
    percentile: float = Field(
        default=99,
        ge=50,
        le=100,
        description="Upper percentile for capping (50-100). Used for 'cap', 'remove', 'winsorize'.",
    )
    lower_percentile: float | None = Field(
        default=None,
        ge=0,
        le=50,
        description="Lower percentile for two-sided capping (0-50). If None, only caps upper tail.",
    )
    create_new_column: bool = Field(
        default=False,
        description="If True, creates new column '{column}_cleaned' instead of modifying original",
    )
    save_as: str | None = Field(
        default=None,
        description="Save result as new DataFrame with this name. If not provided, modifies in place.",
    )


def _get_column_stats(series: pd.Series) -> dict[str, float]:
    """Get descriptive statistics for a column."""
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "median": float(series.median()),
    }


def _cap_at_percentile(
    df: pd.DataFrame,
    column: str,
    upper_pct: float,
    lower_pct: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Cap values at specified percentiles."""
    df_result = df.copy()
    original_values = df_result[column].copy()

    upper_threshold = df_result[column].quantile(upper_pct / 100)

    if lower_pct is not None:
        lower_threshold = df_result[column].quantile(lower_pct / 100)
        df_result[column] = df_result[column].clip(lower=lower_threshold, upper=upper_threshold)
        n_lower = int((original_values < lower_threshold).sum())
        n_upper = int((original_values > upper_threshold).sum())

        stats = {
            "lower_threshold": float(lower_threshold),
            "upper_threshold": float(upper_threshold),
            "n_lower_capped": n_lower,
            "n_upper_capped": n_upper,
            "total_affected": n_lower + n_upper,
        }
    else:
        df_result[column] = df_result[column].clip(upper=upper_threshold)
        n_capped = int((original_values > upper_threshold).sum())

        stats = {
            "lower_threshold": None,
            "upper_threshold": float(upper_threshold),
            "total_affected": n_capped,
        }

    return df_result, stats


def _clip_iqr(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict]:
    """Cap values using IQR method (1.5 * IQR beyond Q1/Q3)."""
    df_result = df.copy()
    original_values = df_result[column].copy()

    q1 = df_result[column].quantile(0.25)
    q3 = df_result[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df_result[column] = df_result[column].clip(lower=lower_bound, upper=upper_bound)

    n_lower = int((original_values < lower_bound).sum())
    n_upper = int((original_values > upper_bound).sum())

    stats = {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_threshold": float(lower_bound),
        "upper_threshold": float(upper_bound),
        "total_affected": n_lower + n_upper,
    }

    return df_result, stats


def _remove_outliers(
    df: pd.DataFrame,
    column: str,
    upper_pct: float,
    lower_pct: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Remove rows with outlier values."""
    upper_threshold = df[column].quantile(upper_pct / 100)

    if lower_pct is not None:
        lower_threshold = df[column].quantile(lower_pct / 100)
        mask = (df[column] >= lower_threshold) & (df[column] <= upper_threshold)
        stats = {
            "lower_threshold": float(lower_threshold),
            "upper_threshold": float(upper_threshold),
        }
    else:
        mask = df[column] <= upper_threshold
        stats = {
            "lower_threshold": None,
            "upper_threshold": float(upper_threshold),
        }

    df_result = df[mask].copy()
    stats["total_affected"] = len(df) - len(df_result)

    return df_result, stats


def _log_transform(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict]:
    """Apply log transformation (log1p to handle zeros)."""
    df_result = df.copy()
    original_values = df_result[column].copy()

    # Check for negative values
    if (original_values < 0).any():
        raise ValueError(
            f"Cannot log-transform column '{column}' - contains negative values. "
            "Consider using a different method or transforming the data first."
        )

    # Use log1p (log(1+x)) to handle zeros
    df_result[column] = np.log1p(original_values)

    stats = {
        "lower_threshold": None,
        "upper_threshold": None,
        "total_affected": len(df_result),  # All values transformed
        "n_zeros": int((original_values == 0).sum()),
    }

    return df_result, stats


@registry.register(
    category="cleaning",
    input_schema=HandleOutliersInput,
    description="Handle outliers in numeric columns using various statistical methods",
)
def handle_outliers(
    state: DataFrameState, params: HandleOutliersInput
) -> OutlierHandlingResult:
    """
    Handle outliers in a numeric column using the specified method.
    
    Methods:
    - cap: Cap values at percentile threshold (safest, preserves all rows)
    - clip_iqr: Cap using IQR method (robust, standard practice)
    - winsorize: Replace outliers with boundary values (same as cap)
    - remove: Remove rows with outliers (reduces dataset size)
    - log_transform: Log transformation for right-skewed data

    Args:
        state: DataFrameState containing the DataFrame to operate on
        params: Parameters for outlier handling

    Returns:
        OutlierHandlingResult with operation summary and statistics

    Raises:
        ValueError: If column doesn't exist, is not numeric, or method fails
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Validate column exists
    if params.column not in df.columns:
        raise ValueError(f"Column '{params.column}' not found in DataFrame")

    # Validate column is numeric
    if not pd.api.types.is_numeric_dtype(df[params.column]):
        dtype = df[params.column].dtype
        raise ValueError(
            f"Column '{params.column}' must be numeric for outlier handling (got {dtype})"
        )

    # Get stats before
    stats_before = _get_column_stats(df[params.column].dropna())
    rows_before = len(df)

    # Apply the appropriate method
    if params.method == "cap" or params.method == "winsorize":
        df_result, method_stats = _cap_at_percentile(
            df, params.column, params.percentile, params.lower_percentile
        )
    elif params.method == "clip_iqr":
        df_result, method_stats = _clip_iqr(df, params.column)
    elif params.method == "remove":
        df_result, method_stats = _remove_outliers(
            df, params.column, params.percentile, params.lower_percentile
        )
    elif params.method == "log_transform":
        df_result, method_stats = _log_transform(df, params.column)
    else:
        raise ValueError(f"Unknown method: {params.method}")

    # Handle new column creation
    if params.create_new_column:
        result_column = f"{params.column}_cleaned"
        # Keep original DataFrame, just add the new column
        original_df = df.copy()
        original_df[result_column] = df_result[params.column]
        df_result = original_df
    else:
        result_column = params.column

    # Get stats after
    stats_after = _get_column_stats(df_result[result_column].dropna())
    rows_after = len(df_result)

    values_affected = method_stats.get("total_affected", 0)
    pct_affected = (values_affected / rows_before * 100) if rows_before > 0 else 0

    # Determine result name (save_as or original)
    result_name = params.save_as if params.save_as else source_name

    # Save to state
    stored_name = state.set_dataframe(df_result, name=result_name, operation="handle_outliers")

    # Generate message
    method_names = {
        "cap": "Capping at Percentile",
        "clip_iqr": "IQR Clipping",
        "remove": "Outlier Removal",
        "winsorize": "Winsorization",
        "log_transform": "Log Transformation",
    }

    if params.method == "remove":
        message = f"{method_names[params.method]}: Removed {values_affected} rows ({pct_affected:.1f}%)"
    elif params.method == "log_transform":
        message = f"{method_names[params.method]}: Transformed all {values_affected} values"
    else:
        message = f"{method_names[params.method]}: Capped {values_affected} values ({pct_affected:.1f}%)"

    return OutlierHandlingResult(
        success=True,
        method=params.method,
        column=params.column,
        rows_before=rows_before,
        rows_after=rows_after,
        values_affected=values_affected,
        percentage_affected=pct_affected,
        lower_threshold=method_stats.get("lower_threshold"),
        upper_threshold=method_stats.get("upper_threshold"),
        result_column=result_column,
        dataframe_name=stored_name,
        stats_before=stats_before,
        stats_after=stats_after,
        message=message,
    )
