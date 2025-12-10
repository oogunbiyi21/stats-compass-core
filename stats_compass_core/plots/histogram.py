"""
Tool for creating histogram plots from DataFrame columns.
"""

import base64
from io import BytesIO
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.registry import registry
from stats_compass_core.results import ChartResult
from stats_compass_core.state import DataFrameState


class HistogramInput(BaseModel):
    """Input schema for histogram tool."""

    dataframe_name: str | None = Field(
        default=None, description="Name of DataFrame to plot. Uses active if not specified."
    )
    column: str = Field(description="Name of the column to plot")
    bins: int = Field(default=30, ge=1, description="Number of bins for the histogram")
    title: str | None = Field(
        default=None, description="Plot title. If None, uses column name"
    )
    xlabel: str | None = Field(
        default=None, description="X-axis label. If None, uses column name"
    )
    ylabel: str = Field(default="Frequency", description="Y-axis label")
    figsize: tuple[float, float] = Field(
        default=(10, 6), description="Figure size as (width, height) in inches"
    )
    dpi: int = Field(default=100, ge=50, le=300, description="Resolution in dots per inch")


@registry.register(
    category="plots",
    input_schema=HistogramInput,
    description="Create a histogram plot from DataFrame column",
)
def histogram(state: DataFrameState, params: HistogramInput) -> ChartResult:
    """
    Create a histogram plot from a DataFrame column.

    Note: Requires matplotlib to be installed (install with 'plots' extra).

    Args:
        state: DataFrameState containing the DataFrame to plot
        params: Parameters for histogram creation

    Returns:
        ChartResult containing base64-encoded PNG image

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

    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Validate column exists
    if params.column not in df.columns:
        raise ValueError(f"Column '{params.column}' not found in DataFrame")

    # Check if column is numeric
    if not pd.api.types.is_numeric_dtype(df[params.column]):
        raise ValueError(f"Column '{params.column}' is not numeric")

    # Create figure
    fig, ax = plt.subplots(figsize=params.figsize)

    # Plot histogram
    data = df[params.column].dropna()
    ax.hist(data, bins=params.bins, edgecolor="black", alpha=0.7)

    # Set labels and title
    title = params.title or f"Histogram of {params.column}"
    ax.set_xlabel(params.xlabel or params.column)
    ax.set_ylabel(params.ylabel)
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
        "column": params.column,
        "bins": params.bins,
        "data_points": len(data),
        "min_value": float(data.min()),
        "max_value": float(data.max()),
        "mean_value": float(data.mean()),
        "figsize": params.figsize,
        "dpi": params.dpi,
    }

    return ChartResult(
        image_base64=image_base64,
        image_format="png",
        title=title,
        chart_type="histogram",
        dataframe_name=source_name,
        metadata=metadata,
    )
