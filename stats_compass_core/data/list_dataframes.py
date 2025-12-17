"""
Tool for listing available DataFrames in the session.
"""

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameListResult
from stats_compass_core.state import DataFrameState


class ListDataFramesInput(StrictToolInput):
    """Input schema for list_dataframes tool."""

    # No inputs required - lists all DataFrames
    pass


@registry.register(
    category="data",
    input_schema=ListDataFramesInput,
    description="List all available DataFrames in the session",
)
def list_dataframes(state: DataFrameState, params: ListDataFramesInput) -> DataFrameListResult:
    """
    List all available DataFrames in the session.

    Args:
        state: DataFrameState containing all DataFrames
        params: Empty input (no parameters needed)

    Returns:
        DataFrameListResult with list of DataFrames and metadata
    """
    dataframes_info = state.list_dataframes()

    # Convert DataFrameInfo objects to dicts for JSON serialization
    dataframes_dicts = [df_info.model_dump() for df_info in dataframes_info]

    # Convert memory from MB to bytes
    total_memory_bytes = int(state.get_total_memory_mb() * 1024 * 1024)

    return DataFrameListResult(
        dataframes=dataframes_dicts,
        active_dataframe=state.get_active_dataframe_name(),
        total_count=len(dataframes_info),
        total_memory_bytes=total_memory_bytes,
    )
