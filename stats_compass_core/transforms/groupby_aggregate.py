"""
Tool for grouping and aggregating DataFrame data.
"""
import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.registry import registry


class GroupByAggregateInput(BaseModel):
    """Input schema for groupby_aggregate tool."""
    
    by: list[str] = Field(
        description="Column labels to group by"
    )
    agg_func: dict[str, str | list[str]] = Field(
        description="Dictionary mapping column names to aggregation functions"
    )
    as_index: bool = Field(
        default=True,
        description="If True, use group keys as index"
    )


@registry.register(
    category="transforms",
    input_schema=GroupByAggregateInput,
    description="Group DataFrame by columns and apply aggregation functions"
)
def groupby_aggregate(df: pd.DataFrame, params: GroupByAggregateInput) -> pd.DataFrame:
    """
    Group DataFrame by specified columns and apply aggregation functions.
    
    Args:
        df: Input DataFrame
        params: Parameters for groupby and aggregation
    
    Returns:
        New aggregated DataFrame
    
    Raises:
        ValueError: If group-by or aggregation columns don't exist
        TypeError: If aggregation function is not supported
    """
    # Validate group-by columns
    missing_cols = set(params.by) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Group-by columns not found: {missing_cols}")
    
    # Validate aggregation columns
    agg_cols = set(params.agg_func.keys())
    missing_agg_cols = agg_cols - set(df.columns)
    if missing_agg_cols:
        raise ValueError(f"Aggregation columns not found: {missing_agg_cols}")
    
    # Perform groupby and aggregation
    try:
        result = df.groupby(params.by, as_index=params.as_index).agg(params.agg_func)
        return result
    except Exception as e:
        raise TypeError(f"Aggregation failed: {str(e)}") from e
