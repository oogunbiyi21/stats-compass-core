"""
Tool for creating histogram plots from DataFrame columns.
"""
import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.registry import registry


class HistogramResult(BaseModel):
    """Result of histogram creation."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    figure: object
    column: str
    bins: int
    

class HistogramInput(BaseModel):
    """Input schema for histogram tool."""
    
    column: str = Field(
        description="Name of the column to plot"
    )
    bins: int = Field(
        default=30,
        ge=1,
        description="Number of bins for the histogram"
    )
    title: str | None = Field(
        default=None,
        description="Plot title. If None, uses column name"
    )
    xlabel: str | None = Field(
        default=None,
        description="X-axis label. If None, uses column name"
    )
    ylabel: str = Field(
        default="Frequency",
        description="Y-axis label"
    )
    figsize: tuple[float, float] = Field(
        default=(10, 6),
        description="Figure size as (width, height) in inches"
    )


@registry.register(
    category="plots",
    input_schema=HistogramInput,
    description="Create a histogram plot from DataFrame column"
)
def histogram(df: pd.DataFrame, params: HistogramInput) -> HistogramResult:
    """
    Create a histogram plot from a DataFrame column.
    
    Note: Requires matplotlib to be installed (install with 'plots' extra).
    
    Args:
        df: Input DataFrame
        params: Parameters for histogram creation
    
    Returns:
        HistogramResult containing the matplotlib figure object
    
    Raises:
        ImportError: If matplotlib is not installed
        ValueError: If column doesn't exist or is not numeric
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting tools. "
            "Install with: pip install stats-compass-core[plots]"
        ) from e
    
    # Validate column exists
    if params.column not in df.columns:
        raise ValueError(f"Column '{params.column}' not found in DataFrame")
    
    # Check if column is numeric
    if not pd.api.types.is_numeric_dtype(df[params.column]):
        raise ValueError(f"Column '{params.column}' is not numeric")
    
    # Create figure
    fig, ax = plt.subplots(figsize=params.figsize)
    
    # Plot histogram
    ax.hist(df[params.column].dropna(), bins=params.bins, edgecolor='black', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel(params.xlabel or params.column)
    ax.set_ylabel(params.ylabel)
    ax.set_title(params.title or f"Histogram of {params.column}")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return HistogramResult(
        figure=fig,
        column=params.column,
        bins=params.bins
    )
