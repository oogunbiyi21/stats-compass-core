"""
Tool for computing pairwise correlation of DataFrame columns.
"""
import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.registry import registry


class CorrelationsInput(BaseModel):
    """Input schema for correlations tool."""
    
    method: str = Field(
        default="pearson",
        pattern="^(pearson|kendall|spearman)$",
        description="Correlation method: 'pearson', 'kendall', or 'spearman'"
    )
    min_periods: int | None = Field(
        default=None,
        ge=1,
        description="Minimum number of observations required per pair"
    )
    numeric_only: bool = Field(
        default=True,
        description="Include only numeric columns"
    )


@registry.register(
    category="eda",
    input_schema=CorrelationsInput,
    description="Compute pairwise correlation of DataFrame columns"
)
def correlations(df: pd.DataFrame, params: CorrelationsInput) -> pd.DataFrame:
    """
    Compute pairwise correlation of columns, excluding NA/null values.
    
    Args:
        df: Input DataFrame
        params: Parameters for correlation computation
    
    Returns:
        DataFrame containing correlation matrix
    
    Raises:
        ValueError: If no numeric columns available or computation fails
    """
    if params.numeric_only:
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty:
            raise ValueError("No numeric columns found in DataFrame")
        working_df = numeric_df
    else:
        working_df = df
    
    try:
        return working_df.corr(
            method=params.method,
            min_periods=params.min_periods,
            numeric_only=params.numeric_only
        )
    except Exception as e:
        raise ValueError(f"Correlation computation failed: {str(e)}") from e
