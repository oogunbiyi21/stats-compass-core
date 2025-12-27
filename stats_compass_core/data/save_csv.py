import os
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState
from stats_compass_core.utils import safe_save, UnsafePathError


class SaveCSVInput(StrictToolInput):
    """Input for saving a DataFrame to a CSV file."""

    dataframe_name: str = Field(..., description="Name of the DataFrame to save")
    filepath: str = Field(..., description="Path where the CSV file will be saved")
    index: bool = Field(False, description="Whether to write row names (index)")


@registry.register(
    category="data",
    name="save_csv",
    input_schema=SaveCSVInput,
    description="Save a DataFrame to a CSV file. Never overwrites existing files - automatically adds _1, _2, etc. suffix if file exists.",
    tier="util",
)
def save_csv(state: DataFrameState, input_data: SaveCSVInput) -> dict[str, str]:
    """
    Save a DataFrame to a CSV file.

    Args:
        state: The DataFrameState manager.
        input_data: The input parameters.

    Returns:
        A dictionary with a success message and actual filepath used.
        
    Raises:
        ValueError: If DataFrame not found
        UnsafePathError: If path is in a protected location or has protected extension
    """
    df = state.get_dataframe(input_data.dataframe_name)
    if df is None:
        raise ValueError(f"DataFrame '{input_data.dataframe_name}' not found.")

    result = safe_save(df, input_data.filepath, "csv", index=input_data.index)

    return {
        "message": f"DataFrame '{input_data.dataframe_name}' saved to '{result['filepath']}'",
        "filepath": result["filepath"],
        "was_renamed": str(result["was_renamed"]),
        "rows": str(len(df)),
        "columns": str(len(df.columns)),
    }
