"""
Tool for creating line plots from DataFrame columns.
"""

import base64
from io import BytesIO
from typing import Any

from pydantic import BaseModel, Field

from stats_compass_core.registry import registry
from stats_compass_core.results import ChartResult
from stats_compass_core.state import DataFrameState


class LinePlotInput(BaseModel):
    """Input schema for lineplot tool."""

    dataframe_name: str | None = Field(
        default=None, description="Name of DataFrame to plot. Uses active if not specified."
    )
    x_column: str | None = Field(
        default=None, description="Name of the column for x-axis. If None, uses index"
    )
    y_column: str = Field(description="Name of the column for y-axis")
    title: str | None = Field(default=None, description="Plot title")
    xlabel: str | None = Field(default=None, description="X-axis label")
    ylabel: str | None = Field(default=None, description="Y-axis label")
    figsize: tuple[float, float] = Field(
        default=(10, 6), description="Figure size as (width, height) in inches"
    )
    marker: str | None = Field(
        default=None, description="Marker style (e.g., 'o', 's', '^')"
    )
    dpi: int = Field(default=100, ge=50, le=300, description="Resolution in dots per inch")


@registry.register(
    category="plots",
    input_schema=LinePlotInput,
    description="Create a line plot from DataFrame columns",
)
def lineplot(state: DataFrameState, params: LinePlotInput) -> ChartResult:
    """
    Create a line plot from DataFrame columns.

    Note: Requires matplotlib to be installed (install with 'plots' extra).

    Args:
        state: DataFrameState containing the DataFrame to plot
        params: Parameters for line plot creation

    Returns:
        ChartResult containing base64-encoded PNG image

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

    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

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
    title = params.title or f"{params.y_column} vs {x_label}"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert figure to base64 PNG
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=params.dpi, bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    # Collect metadata
    metadata: dict[str, Any] = {
        "x_column": params.x_column,
        "y_column": params.y_column,
        "data_points": len(df),
        "marker": params.marker,
        "figsize": params.figsize,
        "dpi": params.dpi,
    }

    return ChartResult(
        image_base64=image_base64,
        image_format="png",
        title=title,
        chart_type="lineplot",
        dataframe_name=source_name,
        metadata=metadata,
    )
