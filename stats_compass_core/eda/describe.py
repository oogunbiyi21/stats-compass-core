"""
Tool for generating descriptive statistics of DataFrame columns.
"""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from stats_compass_core.registry import registry


class DescribeInput(BaseModel):
    """Input schema for describe tool."""

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


class DescribeResult(BaseModel):
    """Result of describe operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    statistics: pd.DataFrame


@registry.register(
    category="eda",
    input_schema=DescribeInput,
    description="Generate descriptive statistics for DataFrame",
)
def describe(df: pd.DataFrame, params: DescribeInput) -> DescribeResult:
    """
    Generate descriptive statistics that summarize the central tendency,
    dispersion and shape of a dataset's distribution.

    Args:
        df: Input DataFrame
        params: Parameters for describe operation

    Returns:
        DescribeResult containing DataFrame with descriptive statistics

    Raises:
        ValueError: If percentiles are out of range or incompatible types specified
    """
    # Validate percentiles
    if params.percentiles:
        for p in params.percentiles:
            if not 0 <= p <= 1:
                raise ValueError(f"Percentiles must be between 0 and 1, got {p}")

    # Build kwargs for describe
    kwargs = {}
    if params.percentiles:
        kwargs["percentiles"] = params.percentiles
    if params.include:
        kwargs["include"] = params.include
    if params.exclude:
        kwargs["exclude"] = params.exclude

    try:
        stats_df = df.describe(**kwargs)
        return DescribeResult(statistics=stats_df)
    except Exception as e:
        raise ValueError(f"Describe operation failed: {str(e)}") from e
