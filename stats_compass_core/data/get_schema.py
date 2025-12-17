"""
Tool for getting the schema/metadata of a DataFrame.
"""

from typing import Any

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameSchemaResult
from stats_compass_core.state import DataFrameState


class GetSchemaInput(StrictToolInput):
    """Input schema for get_schema tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to get schema for. Uses active if not specified."
    )
    sample_values: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of sample values to include per column"
    )


@registry.register(
    category="data",
    input_schema=GetSchemaInput,
    description="Get the schema and metadata of a DataFrame",
)
def get_schema(state: DataFrameState, params: GetSchemaInput) -> DataFrameSchemaResult:
    """
    Get the schema and metadata of a DataFrame.

    Args:
        state: DataFrameState containing the DataFrame
        params: Parameters for schema retrieval

    Returns:
        DataFrameSchemaResult with column info and metadata

    Raises:
        ValueError: If no DataFrame is available
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Build column info
    columns: list[dict[str, Any]] = []
    for col in df.columns:
        col_info: dict[str, Any] = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "null_percent": round(df[col].isna().mean() * 100, 2),
            "unique_count": int(df[col].nunique()),
        }

        # Add sample values
        if params.sample_values > 0:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                samples = non_null.head(params.sample_values).tolist()
                # Convert to JSON-serializable Python types
                serializable_samples = []
                for v in samples:
                    if hasattr(v, 'item'):
                        # numpy types (int64, float64, etc.)
                        serializable_samples.append(v.item())
                    elif hasattr(v, 'isoformat'):
                        # pandas Timestamp, datetime, date
                        serializable_samples.append(v.isoformat())
                    elif isinstance(v, pd.Timedelta):
                        # pandas Timedelta
                        serializable_samples.append(str(v))
                    else:
                        serializable_samples.append(v)
                col_info["sample_values"] = serializable_samples
            else:
                col_info["sample_values"] = []

        # Add numeric stats if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else None
            col_info["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else None

        columns.append(col_info)

    # Get memory usage
    memory_bytes = int(df.memory_usage(deep=True).sum())

    # Get index info
    index_info: dict[str, Any] = {
        "name": df.index.name,
        "dtype": str(df.index.dtype),
        "is_unique": df.index.is_unique,
    }

    return DataFrameSchemaResult(
        dataframe_name=source_name,
        shape=(len(df), len(df.columns)),
        columns=columns,
        memory_usage_bytes=memory_bytes,
        index_info=index_info,
    )
