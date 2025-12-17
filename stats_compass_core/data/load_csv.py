"""
Tool for loading CSV data from a local file path.
"""

import os
from typing import Any

import pandas as pd
from pydantic import ConfigDict, Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameLoadResult
from stats_compass_core.state import DataFrameState


class LoadCSVInput(StrictToolInput):
    """Input schema for load_csv tool."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    file_path: str = Field(
        description="Absolute path to the CSV file. Supports '~' expansion (e.g. ~/Downloads/data.csv). Relative paths are relative to the server's working directory.",
        alias="path",
    )
    name: str | None = Field(
        default=None,
        description="Name to assign to the DataFrame. If None, uses filename without extension."
    )
    set_active: bool = Field(
        default=True,
        description="Whether to set this as the active DataFrame"
    )
    # Common pandas read_csv parameters
    delimiter: str = Field(default=",", description="Field delimiter")
    encoding: str = Field(default="utf-8", description="File encoding")
    nrows: int | None = Field(
        default=None,
        ge=1,
        description="Number of rows to read (useful for large files)"
    )


@registry.register(
    category="data",
    input_schema=LoadCSVInput,
    description="Load data from a local CSV file. You MUST provide an absolute file path. Drag-and-drop uploads are NOT supported.",
)
def load_csv(state: DataFrameState, params: LoadCSVInput) -> DataFrameLoadResult:
    """
    Load a CSV file into the session state.

    IMPORTANT: This tool requires a valid path on the local filesystem where the
    server is running. It cannot access files uploaded directly to the chat interface
    unless they are saved to a known local path.

    Args:
        state: DataFrameState to store the loaded DataFrame
        params: Parameters for loading the CSV

    Returns:
        DataFrameLoadResult with load summary

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be parsed as CSV
    """
    # Resolve path (handle ~ for home directory)
    file_path = os.path.expanduser(params.file_path)
    
    # Validate file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {params.file_path}")

    # Determine DataFrame name
    if params.name:
        df_name = params.name
    else:
        # Use filename without extension
        df_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load the CSV
    try:
        read_kwargs: dict[str, Any] = {
            "delimiter": params.delimiter,
            "encoding": params.encoding,
        }
        if params.nrows:
            read_kwargs["nrows"] = params.nrows

        df = pd.read_csv(file_path, **read_kwargs)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {str(e)}") from e

    # Store in state (set_dataframe returns the name string)
    stored_name = state.set_dataframe(df, name=df_name, operation="load_csv")

    # Set as active if requested
    if params.set_active:
        state.set_active_dataframe(stored_name)

    # Get dtypes as strings
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return DataFrameLoadResult(
        success=True,
        dataframe_name=stored_name,
        source=file_path,
        shape=(len(df), len(df.columns)),
        columns=list(df.columns),
        dtypes=dtypes,
        message=f"Loaded {len(df)} rows and {len(df.columns)} columns from {file_path}",
    )
