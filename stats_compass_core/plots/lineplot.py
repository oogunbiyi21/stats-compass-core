"""
Tool for creating line plots from DataFrame columns.
"""
import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.registry import registry


class LineplotResult(BaseModel):
    """Result of lineplot creation."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    figure: object
    x_column: str | None
    y_column: str


class LineplotInput(BaseModel):
    """Input schema for lineplot tool."""
    
    x_column: str | None = Field(
        default=None,
        description="Name of the column for x-axis. If None, uses index"
    )
    y_column: str = Field(
        description="Name of the column for y-axis"
    )
    title: str | None = Field(
        default=None,
        description="Plot title"
    )
    xlabel: str | None = Field(
        default=None,
        description="X-axis label"
    )
    ylabel: str | None = Field(
        default=None,
        description="Y-axis label"
    )
    figsize: tuple[float, float] = Field(
        default=(10, 6),
        description="Figure size as (width, height) in inches"
    )
    marker: str | None = Field(
        default=None,
        description="Marker style (e.g., 'o', 's', '^')"
    )


@registry.register(
    category="plots",
    input_schema=LineplotInput,
    description="Create a line plot from DataFrame columns"
)
def lineplot(df: pd.DataFrame, params: LineplotInput) -> LineplotResult:
    """
    Create a line plot from DataFrame columns.
    
    Note: Requires matplotlib to be installed (install with 'plots' extra).
    
    Args:
        df: Input DataFrame
        params: Parameters for line plot creation
    
    Returns:
        LineplotResult containing the matplotlib figure object
    
    Raises:
        ImportError: If matplotlib is not installed
        ValueError: If specified columns don't exist
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting tools. "
            "Install with: pip install stats-compass-core[plots]"
        ) from e
    
    # Validate y column exists
    if params.y_column not in df.columns:
        raise ValueError(f"Column '{params.y_column}' not found in DataFrame")
    
    # Validate x column if specified
    if params.x_column and params.x_column not in df.columns:
        raise ValueError(f"Column '{params.x_column}' not found in DataFrame")
    
    # Create figure
    fig, ax = plt.subplots(figsize=params.figsize)
    
    # Prepare data
    if params.x_column:
        x_data = df[params.x_column]
        x_label = params.xlabel or params.x_column
    else:
        x_data = df.index
        x_label = params.xlabel or "Index"
    
    y_data = df[params.y_column]
    y_label = params.ylabel or params.y_column
    
    # Plot line
    if params.marker:
        ax.plot(x_data, y_data, marker=params.marker, linewidth=2)
    else:
        ax.plot(x_data, y_data, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(params.title or f"{params.y_column} vs {x_label}")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return LineplotResult(
        figure=fig,
        x_column=params.x_column,
        y_column=params.y_column
    )
