"""
Tool for loading built-in sample datasets.
"""

from pathlib import Path

import pandas as pd
from pydantic import ConfigDict, Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameLoadResult
from stats_compass_core.state import DataFrameState

# Path to datasets directory (inside the package)
_DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def _list_available_datasets() -> list[str]:
    """List available sample datasets."""
    if not _DATASETS_DIR.exists():
        return []
    return [f.stem for f in _DATASETS_DIR.glob("*.csv")]


class LoadDatasetInput(StrictToolInput):
    """Input schema for load_dataset tool."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    name: str = Field(
        description=f"Name of the dataset to load. Available: {', '.join(_list_available_datasets())}",
    )
    set_active: bool = Field(
        default=True,
        description="Whether to set this as the active DataFrame"
    )


@registry.register(
    category="data",
    input_schema=LoadDatasetInput,
    description="Load a built-in sample dataset (e.g. Housing, TATASTEEL, Bukayo_Saka_7322)",
)
def load_dataset(state: DataFrameState, params: LoadDatasetInput) -> DataFrameLoadResult:
    """
    Load a built-in sample dataset into the session state.

    Args:
        state: The DataFrameState object.
        params: Parameters for loading the dataset.

    Returns:
        DataFrameLoadResult with details about the loaded DataFrame.
    """
    try:
        # Handle potential .csv extension in name
        name = params.name
        if name.lower().endswith(".csv"):
            name = name[:-4]

        file_path = _DATASETS_DIR / f"{name}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Add to state
        state.set_dataframe(df, name=name, operation="load_dataset", set_active=params.set_active)
        
        return DataFrameLoadResult(
            success=True,
            dataframe_name=name,
            source=str(file_path),
            shape=df.shape,
            columns=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            message=f"Loaded dataset '{name}' with {len(df)} rows and {len(df.columns)} columns.",
        )
        
    except FileNotFoundError:
        available = ", ".join(_list_available_datasets())
        raise ValueError(f"Dataset '{params.name}' not found. Available datasets: {available}")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{params.name}': {str(e)}")
