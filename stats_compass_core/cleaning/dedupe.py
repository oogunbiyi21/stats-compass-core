"""
Tool for removing duplicate rows from a DataFrame.
"""
import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.registry import registry


class DedupeInput(BaseModel):
    """Input schema for dedupe tool."""
    
    subset: list[str] | None = Field(
        default=None,
        description="Column labels to consider for identifying duplicates"
    )
    keep: str = Field(
        default="first",
        pattern="^(first|last|False)$",
        description="Which duplicates to keep: 'first', 'last', or False (drop all)"
    )
    ignore_index: bool = Field(
        default=False,
        description="If True, reset index in result"
    )


@registry.register(
    category="cleaning",
    input_schema=DedupeInput,
    description="Remove duplicate rows from DataFrame"
)
def dedupe(df: pd.DataFrame, params: DedupeInput) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        params: Parameters for deduplication
    
    Returns:
        New DataFrame with duplicates removed
    
    Raises:
        ValueError: If subset columns don't exist in DataFrame
    """
    if params.subset:
        missing_cols = set(params.subset) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    keep_value = False if params.keep == "False" else params.keep
    
    return df.drop_duplicates(
        subset=params.subset,
        keep=keep_value,  # type: ignore
        ignore_index=params.ignore_index
    )
