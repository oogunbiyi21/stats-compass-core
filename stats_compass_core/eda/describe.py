"""
Tool for generating descriptive statistics of DataFrame columns.
"""

from typing import Any

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DescribeResult
from stats_compass_core.state import DataFrameState


class DescribeInput(StrictToolInput):
    """Input schema for describe tool."""

    dataframe_name: str | None = Field(
        default=None, description="Name of DataFrame to analyze. Uses active if not specified."
    )
    percentiles: list[float] | None = Field(
        default=None, description="List of percentiles to include (between 0 and 1)"
    )
    include: str | list[str] | None = Field(
        default=None,
        description=(
            "Data types to include "
            "('all', 'number', 'object', 'category', 'datetime')"
        ),
    )
    exclude: str | list[str] | None = Field(
        default=None, description="Data types to exclude"
    )


@registry.register(
    category="eda",
    input_schema=DescribeInput,
    description="Generate descriptive statistics for DataFrame",
)
def describe(state: DataFrameState, params: DescribeInput) -> DescribeResult:
    """
    Generate descriptive statistics that summarize the central tendency,
    dispersion and shape of a dataset's distribution.

    Args:
        state: DataFrameState containing the DataFrame to analyze
        params: Parameters for describe operation

    Returns:
        DescribeResult containing JSON-serializable statistics

    Raises:
        ValueError: If percentiles are out of range or incompatible types specified
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Validate percentiles
    if params.percentiles:
        for p in params.percentiles:
            if not 0 <= p <= 1:
                raise ValueError(f"Percentiles must be between 0 and 1, got {p}")

    # Build kwargs for describe
    kwargs: dict[str, Any] = {}
    if params.percentiles:
        kwargs["percentiles"] = params.percentiles
    if params.include:
        kwargs["include"] = params.include
    if params.exclude:
        kwargs["exclude"] = params.exclude

    try:
        stats_df = df.describe(**kwargs)

        # Convert to JSON-serializable nested dict
        # Structure: {column: {stat_name: value}}
        statistics: dict[str, dict[str, Any]] = {}
        for col in stats_df.columns:
            statistics[col] = {}
            for stat in stats_df.index:
                value = stats_df.loc[stat, col]
                # Convert numpy types to Python native types
                if pd.isna(value):
                    statistics[col][stat] = None
                elif hasattr(value, 'item'):
                    statistics[col][stat] = value.item()
                else:
                    statistics[col][stat] = value

        # Determine included types
        include_types: list[str] | None = None
        if params.include:
            if isinstance(params.include, str):
                include_types = [params.include]
            else:
                include_types = params.include

        return DescribeResult(
            statistics=statistics,
            dataframe_name=source_name,
            columns_analyzed=list(stats_df.columns),
            include_types=include_types,
        )
    except Exception as e:
        raise ValueError(f"Describe operation failed: {str(e)}") from e
