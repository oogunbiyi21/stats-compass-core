"""
Tool for loading built-in sample datasets.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from stats_compass_core.data.datasets import list_datasets, load_dataset as load_dataset_func
from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameLoadResult
from stats_compass_core.state import DataFrameState


class LoadDatasetInput(BaseModel):
    """Input schema for load_dataset tool."""
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(
        description=f"Name of the dataset to load. Available: {', '.join(list_datasets())}",
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
        # Load the dataset
        df = load_dataset_func(params.name)
        
        # Add to state
        state.add_dataframe(params.name, df, set_active=params.set_active)
        
        return DataFrameLoadResult(
            name=params.name,
            rows=len(df),
            columns=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            head=df.head().to_dict(orient="records"),
            is_active=params.set_active,
        )
        
    except FileNotFoundError:
        available = ", ".join(list_datasets())
        raise ValueError(f"Dataset '{params.name}' not found. Available datasets: {available}")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{params.name}': {str(e)}")
