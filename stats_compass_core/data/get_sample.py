"""
Tool for getting sample rows from a DataFrame.
"""

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import (
    DataFrameSampleResult,
    dataframe_to_json_safe_records,
)
from stats_compass_core.state import DataFrameState


class GetSampleInput(StrictToolInput):
    """Input schema for get_sample tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to sample. Uses active if not specified."
    )
    n: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of rows to sample"
    )
    method: str = Field(
        default="head",
        pattern="^(head|tail|random)$",
        description="Sampling method: 'head', 'tail', or 'random'"
    )
    random_state: int | None = Field(
        default=None,
        description="Random seed for reproducibility (only used with method='random')"
    )


@registry.register(
    category="data",
    input_schema=GetSampleInput,
    description="Get sample rows from a DataFrame",
)
def get_sample(state: DataFrameState, params: GetSampleInput) -> DataFrameSampleResult:
    """
    Get sample rows from a DataFrame.

    Args:
        state: DataFrameState containing the DataFrame
        params: Parameters for sampling

    Returns:
        DataFrameSampleResult with sample rows

    Raises:
        ValueError: If no DataFrame is available
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Get sample based on method
    n = min(params.n, len(df))  # Don't sample more than available

    if params.method == "head":
        sample_df = df.head(n)
    elif params.method == "tail":
        sample_df = df.tail(n)
    else:  # random
        sample_df = df.sample(n=n, random_state=params.random_state)

    # Convert to JSON-safe list of dicts (handles NaN, Timestamps, etc.)
    data = dataframe_to_json_safe_records(sample_df)

    return DataFrameSampleResult(
        dataframe_name=source_name,
        data=data,
        total_rows=len(df),
        sample_size=len(data),
        columns=list(df.columns),
    )
