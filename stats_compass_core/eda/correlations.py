"""
Tool for computing pairwise correlation of DataFrame columns.
"""

from typing import Any

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import CorrelationsResult
from stats_compass_core.state import DataFrameState


class CorrelationsInput(StrictToolInput):
    """Input schema for correlations tool."""

    dataframe_name: str | None = Field(
        default=None, description="Name of DataFrame to analyze. Uses active if not specified."
    )
    method: str = Field(
        default="pearson",
        pattern="^(pearson|kendall|spearman)$",
        description="Correlation method: 'pearson', 'kendall', or 'spearman'",
    )
    min_periods: int | None = Field(
        default=None,
        ge=1,
        description="Minimum number of observations required per pair",
    )
    numeric_only: bool = Field(default=True, description="Include only numeric columns")
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="If provided, include list of high-correlation pairs above this threshold",
    )


@registry.register(
    category="eda",
    input_schema=CorrelationsInput,
    description="Compute pairwise correlation of DataFrame columns",
)
def correlations(state: DataFrameState, params: CorrelationsInput) -> CorrelationsResult:
    """
    Compute pairwise correlation of columns, excluding NA/null values.

    Args:
        state: DataFrameState containing the DataFrame to analyze
        params: Parameters for correlation computation

    Returns:
        CorrelationsResult with JSON-serializable correlation matrix

    Raises:
        ValueError: If no numeric columns available or computation fails
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    if params.numeric_only:
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty:
            raise ValueError("No numeric columns found in DataFrame")
        working_df = numeric_df
    else:
        working_df = df

    try:
        corr_df = working_df.corr(
            method=params.method,
            min_periods=params.min_periods,
            numeric_only=params.numeric_only,
        )

        # Convert to JSON-serializable nested dict
        correlations_dict: dict[str, dict[str, float]] = {}
        for col in corr_df.columns:
            correlations_dict[col] = {}
            for row in corr_df.index:
                value = corr_df.loc[row, col]
                if pd.isna(value):
                    correlations_dict[col][row] = None  # type: ignore
                elif hasattr(value, 'item'):
                    correlations_dict[col][row] = value.item()
                else:
                    correlations_dict[col][row] = float(value)

        # Find high correlation pairs if threshold is provided
        high_correlations: list[dict[str, Any]] | None = None
        if params.threshold is not None:
            high_correlations = []
            cols = list(corr_df.columns)
            for i, col1 in enumerate(cols):
                for col2 in cols[i+1:]:  # Upper triangle only
                    corr_value = corr_df.loc[col1, col2]
                    if pd.notna(corr_value) and abs(corr_value) >= params.threshold:
                        high_correlations.append({
                            "column_1": col1,
                            "column_2": col2,
                            "correlation": float(corr_value) if hasattr(corr_value, 'item') else corr_value,
                        })
            # Sort by absolute correlation descending
            high_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return CorrelationsResult(
            correlations=correlations_dict,
            method=params.method,
            dataframe_name=source_name,
            columns=list(corr_df.columns),
            high_correlations=high_correlations,
        )
    except Exception as e:
        raise ValueError(f"Correlation computation failed: {str(e)}") from e
