"""
Tool for creating line plots from DataFrame columns.
"""

import base64
import os
from io import BytesIO
from typing import Any, Literal

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import ChartResult
from stats_compass_core.state import DataFrameState


class LinePlotInput(StrictToolInput):
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
    figsize: list[float] = Field(
        default_factory=lambda: [10.0, 6.0],
        min_length=2,
        max_length=2,
        description="Figure size as [width, height] in inches",
    )
    marker: str | None = Field(
        default=None, description="Marker style (e.g., 'o', 's', '^')"
    )
    dpi: int = Field(default=100, ge=50, le=300, description="Resolution in dots per inch")
    save_path: str | None = Field(
        default=None, description="Path to save the plot image (e.g., 'plot.png')"
    )
    format: Literal["png", "json"] = Field(
        default="png", description="Output format: 'png' for image, 'json' for raw data"
    )


@registry.register(
    category="plots",
    input_schema=LinePlotInput,
    description="Create a line plot. Use format='json' to get raw data for interactive visualizations.",
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
    figsize = tuple(params.figsize)
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    if params.x_column:
        x_data = df[params.x_column]
        x_label = params.xlabel or params.x_column
    else:
        x_data = df.index
        x_label = params.xlabel or "Index"

    y_data = df[params.y_column]
    y_label = params.ylabel or params.y_column
    title = params.title or f"{params.y_column} vs {x_label}"

    # Handle JSON format
    if params.format == "json":
        # Limit data points for JSON
        MAX_POINTS = 5000
        if len(df) > MAX_POINTS:
            # Simple downsampling
            step = len(df) // MAX_POINTS
            x_data_json = x_data[::step]
            y_data_json = y_data[::step]
        else:
            x_data_json = x_data
            y_data_json = y_data

        # Convert index to list if needed
        if hasattr(x_data_json, "tolist"):
            x_list = x_data_json.tolist()
        else:
            x_list = list(x_data_json)

        chart_data = {
            "type": "line",
            "title": title,
            "xlabel": x_label,
            "ylabel": y_label,
            "x": x_list,
            "y": y_data_json.tolist(),
        }
        
        return ChartResult(
            image_base64=None,
            image_format="json",
            title=title,
            chart_type="lineplot",
            dataframe_name=source_name,
            data=chart_data,
            metadata={
                "x_column": params.x_column,
                "y_column": params.y_column,
                "data_points": len(x_data_json),
                "total_points": len(df),
                "marker": params.marker,
            }
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot line
    if params.marker:
        ax.plot(x_data, y_data, marker=params.marker, linewidth=2)
    else:
        ax.plot(x_data, y_data, linewidth=2)

    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if params.save_path:
        save_path = os.path.expanduser(params.save_path)
        fig.savefig(save_path, dpi=params.dpi, bbox_inches='tight')

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
