import os
from typing import Literal

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState


class SaveCSVInput(StrictToolInput):
    """Input for saving a DataFrame to a CSV file."""

    dataframe_name: str = Field(..., description="Name of the DataFrame to save")
    filepath: str = Field(..., description="Path where the CSV file will be saved")
    index: bool = Field(False, description="Whether to write row names (index)")
    mode: Literal["w", "a"] = Field("w", description="Write mode: 'w' for write (overwrite), 'a' for append")


@registry.register(
    category="data",
    name="save_csv",
    input_schema=SaveCSVInput,
    description="Save a DataFrame to a CSV file on the local filesystem.",
)
def save_csv(state: DataFrameState, input_data: SaveCSVInput) -> dict[str, str]:
    """
    Save a DataFrame to a CSV file.

    Args:
        state: The DataFrameState manager.
        input_data: The input parameters.

    Returns:
        A dictionary with a success message.
    """
    # Get the DataFrame from state
    df = state.get_dataframe(input_data.dataframe_name)
    if df is None:
        raise ValueError(f"DataFrame '{input_data.dataframe_name}' not found.")

    # Resolve path (handle ~)
    filepath = os.path.expanduser(input_data.filepath)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Save to CSV
    df.to_csv(filepath, index=input_data.index, mode=input_data.mode)

    return {
        "message": f"DataFrame '{input_data.dataframe_name}' saved to '{filepath}'",
        "filepath": filepath,
        "rows": str(len(df)),
        "columns": str(len(df.columns)),
    }
