"""
Tool for loading Excel data from a local file path.
"""

import os
from typing import Any

import pandas as pd
from pydantic import ConfigDict, Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameLoadResult
from stats_compass_core.state import DataFrameState


class LoadExcelInput(StrictToolInput):
    """Input schema for load_excel tool."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    file_path: str = Field(
        description="Absolute path to the Excel file. Supports '~' expansion.",
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
    sheet_name: str | int | None = Field(
        default=0,
        description="Name or index of the sheet to load. Defaults to first sheet (0). Set to None to load all sheets (not supported yet)."
    )
    header: int | None = Field(
        default=0,
        description="Row (0-indexed) to use for the column labels of the parsed DataFrame. If None, no header is assumed."
    )
    nrows: int | None = Field(
        default=None,
        ge=1,
        description="Number of rows to read"
    )


@registry.register(
    category="data",
    input_schema=LoadExcelInput,
    description="Load data from a local Excel file (.xlsx, .xls). You MUST provide an absolute file path. Drag-and-drop uploads are NOT supported.",
)
def load_excel(state: DataFrameState, params: LoadExcelInput) -> DataFrameLoadResult:
    """
    Load an Excel file into the session state.

    Args:
        state: DataFrameState to store the loaded DataFrame
        params: Parameters for loading the Excel file

    Returns:
        DataFrameLoadResult with load summary
    """
    # Expand user path
    file_path = os.path.expanduser(params.file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    # Determine DataFrame name
    if params.name:
        df_name = params.name
    else:
        base_name = os.path.basename(file_path)
        df_name = os.path.splitext(base_name)[0]
        # Simple sanitization
        df_name = "".join(c if c.isalnum() or c == "_" else "_" for c in df_name)

    # Load the Excel file
    try:
        read_kwargs: dict[str, Any] = {
            "sheet_name": params.sheet_name,
            "header": params.header,
        }
        if params.nrows:
            read_kwargs["nrows"] = params.nrows

        # Check for openpyxl
        try:
            import openpyxl
        except ImportError as e:
            raise ImportError(
                "openpyxl is required for reading Excel files. "
                "Please install it with: pip install openpyxl"
            ) from e

        df = pd.read_excel(file_path, **read_kwargs)
        
    except Exception as e:
        raise ValueError(f"Failed to parse Excel file: {str(e)}") from e

    # Store in state
    stored_name = state.set_dataframe(df, name=df_name, operation="load_excel", set_active=params.set_active)

    # Get dtypes as strings
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return DataFrameLoadResult(
        success=True,
        dataframe_name=stored_name,
        source=file_path,
        shape=df.shape,
        columns=list(df.columns),
        dtypes=dtypes,
        message=f"Successfully loaded {len(df)} rows from {os.path.basename(file_path)}",
    )
