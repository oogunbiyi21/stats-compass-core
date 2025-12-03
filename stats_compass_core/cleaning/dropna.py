"""
Tool for dropping rows or columns with missing values from a DataFrame.
"""
import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.registry import registry


class DropNAInput(BaseModel):
    """Input schema for drop_na tool."""
    
    axis: int = Field(default=0, ge=0, le=1, description="0 for rows, 1 for columns")
    how: str = Field(default="any", pattern="^(any|all)$", description="'any' or 'all'")
    thresh: int | None = Field(default=None, ge=0, description="Minimum number of non-NA values")
    subset: list[str] | None = Field(default=None, description="Column labels to consider")


@registry.register(
    category="cleaning",
    input_schema=DropNAInput,
    description="Drop rows or columns with missing values"
)
def drop_na(df: pd.DataFrame, params: DropNAInput) -> pd.DataFrame:
    """
    Drop rows or columns with missing values from a DataFrame.
    
    Args:
        df: Input DataFrame
        params: Parameters for dropping NA values
    
    Returns:
        New DataFrame with NA values removed
    
    Raises:
        ValueError: If subset columns don't exist in DataFrame
    """
    if params.subset:
        missing_cols = set(params.subset) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    return df.dropna(
        axis=params.axis,
        how=params.how,
        thresh=params.thresh,
        subset=params.subset
    )
