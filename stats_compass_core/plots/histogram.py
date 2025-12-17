"""
Tool for creating histogram plots from DataFrame columns.
"""

import base64
import os
from io import BytesIO
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import ChartResult
from stats_compass_core.state import DataFrameState


class HistogramInput(StrictToolInput):
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
    figsize: list[float] = Field(
        default_factory=lambda: [10.0, 6.0],
        min_length=2,
        max_length=2,
        description="Figure size as [width, height] in inches"
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
    input_schema=HistogramInput,
    description="Create a histogram plot. Use format='json' to get raw data for interactive visualizations.",
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

    data = df[params.column].dropna()
    title = params.title or f"Histogram of {params.column}"

    # Handle JSON format
    if params.format == "json":
        counts, bin_edges = np.histogram(data, bins=params.bins)
        
        # Create bin labels (e.g., "10-20")
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            bin_labels.append(f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}")
            
        chart_data = {
            "type": "histogram",
            "title": title,
            "xlabel": params.xlabel or params.column,
            "ylabel": params.ylabel,
            "bins": bin_labels,
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
        }
        
        return ChartResult(
            image_base64=None,
            image_format="json",
            title=title,
            chart_type="histogram",
            dataframe_name=source_name,
            data=chart_data,
            metadata={
                "column": params.column,
                "bins": params.bins,
                "data_points": len(data),
                "min_value": float(data.min()),
                "max_value": float(data.max()),
                "mean_value": float(data.mean()),
            }
        )

    # Create figure
    figsize = tuple(params.figsize)
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    ax.hist(data, bins=params.bins, edgecolor="black", alpha=0.7)

    # Set labels and title
    ax.set_xlabel(params.xlabel or params.column)
    ax.set_ylabel(params.ylabel)
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
